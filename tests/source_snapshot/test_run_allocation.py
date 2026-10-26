"""Exercise externally visible run contracts through the real CLI."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ENTRY = Path(__file__).resolve().parents[1] / 'run_allocation.py'


def cli(*arguments):
    return subprocess.run([sys.executable, str(ENTRY), *map(str, arguments)],
                          capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=45)


def test_cli_requires_explicit_delta(tmp_path):
    output = tmp_path / 'must_not_be_created'
    result = cli('--n', 50, '--out', output)
    assert result.returncode == 2
    assert '--delta' in result.stderr and 'required' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_cli_rejects_training_held_out_partition_before_dataset_access(tmp_path, split):
    output = tmp_path / split
    result = cli('--n', 50, '--delta', .05, '--phase', 'train', '--split', split,
                 '--dataset-manifest', tmp_path / 'not_opened.json', '--out', output)
    assert result.returncode != 0
    assert 'validation/test partitions cannot train or update weights' in result.stderr
    assert 'FileNotFoundError' not in result.stderr
    assert not output.exists()


def test_cli_policy_evaluation_requires_explicit_checkpoint(tmp_path):
    output = tmp_path / 'policy_eval'
    result = cli('--n', 50, '--delta', .05, '--methods', 'policy', '--out', output)
    assert result.returncode != 0
    assert 'policy evaluation requires an explicit checkpoint' in result.stderr
    assert not output.exists()


def test_cli_does_not_overwrite_nonempty_output_directory(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('original evidence', encoding='utf-8')
    result = cli('--n', 50, '--delta', .05, '--methods', 'reference', '--out', output)
    assert result.returncode != 0
    assert 'output directory is not empty' in result.stderr
    assert sentinel.read_text(encoding='utf-8') == 'original evidence'
    assert sorted(path.name for path in output.iterdir()) == ['keep.txt']


@pytest.mark.parametrize('changed,expected_error', [
    ({}, None),
    ({'split': 'test'}, 'training from this frozen partition registry'),
    ({'dataset_manifest_sha256': 'other-registry'}, 'training from this frozen partition registry'),
    ({'training_traffic_seed': 2}, 'traffic overlap'),
    ({'traffic_hashes': {'mobility.npz': 'evaluation-mobility'}}, 'traffic overlap'),
    ({'channel_seed': 11}, 'seeds must be independent'),
    ({'forecast_seed': 12}, 'seeds must be independent'),
    ({'forecast_seed': 11, 'forecast_seeds': [14, 15]}, 'realized forecast seeds overlap'),
    ({'physical_config_sha256': 'other-physics'}, 'configuration mismatch'),
])
def test_real_checkpoint_provenance_rejects_held_out_reuse(tmp_path, changed, expected_error):
    # Persist and reload real model weights with their provenance.  No fake
    # learner or mocked dataset reader is needed for this boundary check.
    from p3learn.masac import HybridMASAC
    from p3learn.models import ModelConfig
    from p3learn.features import FEATURE_VERSION
    from run_allocation import validate_checkpoint_provenance
    learner = HybridMASAC(ModelConfig(hidden_dim=8))
    provenance = dict(split='train', n=50, dataset_manifest_sha256='registry',
                      training_traffic_seed=1, traffic_hashes={'mobility.npz': 'training-mobility'},
                      channel_seed=1, forecast_seed=2, forecast_seeds=[4, 5], feature_version=FEATURE_VERSION,
                      physical_config_sha256='physics', source_sha256='source', run_config_sha256='run')
    provenance.update(changed)
    checkpoint = tmp_path / 'actual_model.pt'
    learner.save(checkpoint, provenance=provenance)
    restored = HybridMASAC.load(checkpoint)
    args = SimpleNamespace(n=50, phase='evaluate', split='test', channel_seed=11, forecast_seed=12,
                           start_block=2, blocks=2)
    kwargs = dict(args=args, dataset={'traffic_seed': 2},
                  traffic_hashes={'mobility.npz': 'evaluation-mobility'},
                  manifest_hash='registry', physical_hash='physics')
    if expected_error is None:
        validate_checkpoint_provenance(restored, **kwargs)
    else:
        with pytest.raises(ValueError, match=expected_error):
            validate_checkpoint_provenance(restored, **kwargs)
