import json
from unittest.mock import patch

import numpy as np
import pytest

from run_safety import validate_window, commit_block, atomic_json, load_committed, consolidate_results
from recover_saved_blocks import recover


@pytest.mark.parametrize('start,count,valid', [(2,999,False),(2,998,True),(1,999,True),
                                               (999,1,True),(1000,1,False),(2,0,False),
                                               (0,1000,True),(0,1001,False)])
def test_window(start,count,valid):
    if valid:
        result=validate_window(np.arange(1001)*.1,start,count,.1,.001)
        assert result['end_s'] <= 100 + 1e-9
    else:
        with pytest.raises(ValueError):
            validate_window(np.arange(1001)*.1,start,count,.1,.001)


def setup_manifest(tmp_path):
    atomic_json(tmp_path/'run_manifest.json', dict(source_sha256='source',config_sha256='cfg',
                window=dict(start_block=2,planned_blocks=3)))


def payload(k):
    return dict(block=k,source_sha256='source',config_sha256='cfg',
                rows=[dict(block=k,ee=123.5)],block_records=[dict(block=k,success=True)])


def test_interrupted_write_keeps_previous_commit_and_exports(tmp_path):
    setup_manifest(tmp_path)
    commit_block(tmp_path,2,payload(2),{'reference_b2_power_w':np.array([.1,.2])})
    with patch('run_safety.np.savez_compressed',side_effect=OSError('injected disk error')):
        with pytest.raises(OSError):
            commit_block(tmp_path,3,payload(3),{'reference_b3_power_w':np.array([.3,.4])})
    assert not (tmp_path/'block_checkpoints/block_000003').exists()
    assert json.loads((tmp_path/'progress.json').read_text())['last_committed_block']==2
    assert recover(tmp_path,tmp_path/'export')==1
    assert not (tmp_path/'export/summary.json').exists()
    with np.load(tmp_path/'export/fine_samples.partial.npz') as z:
        np.testing.assert_array_equal(z['reference_b2_power_w'],[.1,.2])


def test_corruption_is_rejected(tmp_path):
    setup_manifest(tmp_path)
    commit_block(tmp_path,2,payload(2),{'reference_b2_power_w':np.array([.1])})
    (tmp_path/'block_checkpoints/block_000002/records.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        recover(tmp_path,tmp_path/'export')


def test_duplicate_commit_is_rejected(tmp_path):
    setup_manifest(tmp_path)
    commit_block(tmp_path,2,payload(2),{})
    with pytest.raises(FileExistsError):
        commit_block(tmp_path,2,payload(2),{})


def test_restart_identity_and_state(tmp_path):
    setup_manifest(tmp_path)
    rng=np.random.default_rng(123)
    state=dict(rng=rng)
    commit_block(tmp_path,2,payload(2),{'reference_b2_power_w':np.array([.1])},state)
    saved, arrays, restored=load_committed(tmp_path,'source','cfg',2,3)
    np.testing.assert_array_equal(restored['rng'].normal(size=20),rng.normal(size=20))
    assert len(saved)==1
    with pytest.raises(ValueError,match='identity'):
        load_committed(tmp_path,'changed_source','cfg',2,3)


def test_pending_write_archived_on_resume(tmp_path):
    setup_manifest(tmp_path)
    commit_block(tmp_path,2,payload(2),{},dict(a=3))
    pending=tmp_path/'block_checkpoints/.block_000003.pending'
    pending.mkdir()
    (pending/'evidence.txt').write_text('unfinished')
    load_committed(tmp_path,'source','cfg',2,3)
    assert not pending.exists()
    assert (pending.with_name(pending.name+'.abandoned')/'evidence.txt').read_text()=='unfinished'
    commit_block(tmp_path,3,payload(3),{},dict(a=4))


def test_streaming_final_files(tmp_path):
    setup_manifest(tmp_path)
    for k in (2,3):
        value=payload(k)
        for field in ('training_curve','power_solver_records','literature_records','weak_link_records','credit_records'):
            value[field]=[]
        value['power_solver_records']=[dict(block=k,diagnostic=1.25)]
        commit_block(tmp_path,k,value,{f'reference_b{k}_power_w':np.array([k*.1])})
    cfg=dict(power_solver='continuous',methods=['reference'],maxmin_weak_link=False,maxmin_refine_solver='nm')
    consolidate_results(tmp_path,cfg)
    with np.load(tmp_path/'fine_samples.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['reference_b2_power_w'],[.2])
        np.testing.assert_array_equal(z['reference_b3_power_w'],[3*.1])
    assert len(json.loads((tmp_path/'continuous_power_audit.json').read_text()))==2
