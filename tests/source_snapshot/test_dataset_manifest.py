"""Dataset leakage, frozen-file integrity, and path-boundary regression checks."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3learn.data import freeze_datasets, resolve_dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


@pytest.fixture
def registry(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    manifest = root / 'development' / 'datasets.json'

    def scene(seed, n=50, label=None):
        directory = root / (label or f'seed{seed}_n{n}')
        directory.mkdir()
        # These minimal arrays test the dataset registry, not SUMO dynamics.
        # Every seed has distinct array contents, without relying on ZIP metadata.
        time = np.arange(1001) * .1
        center = np.zeros((1001, n, 2))
        center[:, :, 0] = seed + np.arange(n)[None, :] * 10 + time[:, None] * .01
        center[:, :, 1] = np.arange(n)[None, :] * 2
        np.savez_compressed(directory / 'mobility.npz', time_s=time,
                            center_xy_m=center, heading_rad=np.zeros((1001, n)),
                            speed_mps=np.full((1001, n), .01),
                            vehicle_ids=np.array([f'v{i}' for i in range(n)]))
        np.savez_compressed(directory / 'buildings.npz', boxes=np.empty((0, 6)))
        write_json(directory / 'scenario.json', dict(seed=seed, vehicles=n))
        write_json(directory / 'audit.json', dict(vehicles=n, seed=seed, collision_count=0, teleport_count=0,
                                                 building_overlap_count=0, arrived_count=0))
        return directory

    return root, manifest, scene


def test_independent_splits_freeze_resolve_and_survive_relocation(registry, tmp_path):
    root, manifest, scene = registry
    splits = {split: [scene(seed, n) for n in (50, 100)]
              for split, seed in [('train', 11), ('validation', 22), ('test', 33)]}
    value = freeze_datasets(root, splits, manifest)
    assert len(value['entries']) == 6
    assert {item['traffic_seed'] for item in value['entries']} == {11, 22, 33}
    assert not Path(value['root_relative_to_manifest']).is_absolute()
    for split, directories in splits.items():
        for n, expected in zip((50, 100), directories):
            resolved, item, manifest_digest = resolve_dataset(manifest, split, n)
            assert resolved == expected
            assert item['split'] == split and item['n'] == n
            assert manifest_digest == digest(manifest)
            assert set(item['files']) == {'mobility.npz', 'buildings.npz', 'scenario.json', 'audit.json'}
            assert all(digest(resolved / name) == expected_hash for name, expected_hash in item['files'].items())
    relocated = tmp_path / 'relocated'
    shutil.copytree(root, relocated)
    moved_manifest = relocated / 'development' / 'datasets.json'
    for split in splits:
        for n in (50, 100):
            resolved, item, moved_digest = resolve_dataset(moved_manifest, split, n)
            assert resolved.is_relative_to(relocated)
            assert resolved == relocated / item['directory']
            assert moved_digest == digest(manifest)


def test_reusing_same_scene_in_multiple_splits_is_rejected(registry):
    root, manifest, scene = registry
    directory = scene(11)
    with pytest.raises(ValueError, match='reused|disjoint|overlap'):
        freeze_datasets(root, {'train': [directory], 'test': [directory]}, manifest)
    assert not manifest.exists()


def test_same_traffic_seed_is_rejected_even_across_different_scales(registry):
    root, manifest, scene = registry
    with pytest.raises(ValueError, match='disjoint|reused|overlap'):
        freeze_datasets(root, {'train': [scene(11, 50)], 'validation': [scene(11, 100)]}, manifest)
    assert not manifest.exists()


def test_relabelling_identical_mobility_with_a_new_seed_does_not_make_it_independent(registry):
    root, manifest, scene = registry
    train, test = scene(11), scene(33)
    shutil.copyfile(train / 'mobility.npz', test / 'mobility.npz')
    with pytest.raises(ValueError, match='identical mobility'):
        freeze_datasets(root, {'train': [train], 'test': [test]}, manifest)
    assert not manifest.exists()


@pytest.mark.parametrize('audit_key', ['collision_count', 'teleport_count', 'building_overlap_count', 'arrived_count'])
def test_failed_traffic_audit_cannot_be_frozen(registry, audit_key):
    root, manifest, scene = registry
    directory = scene(11)
    audit_path = directory / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit[audit_key] = 1
    write_json(audit_path, audit)
    with pytest.raises(ValueError, match='traffic audit failed'):
        freeze_datasets(root, {'train': [directory]}, manifest)
    assert not manifest.exists()


@pytest.mark.parametrize('filename', ['mobility.npz', 'buildings.npz', 'scenario.json', 'audit.json'])
def test_every_frozen_file_is_verified_before_dataset_use(registry, filename):
    root, manifest, scene = registry
    directory = scene(11)
    freeze_datasets(root, {'train': [directory]}, manifest)
    path = directory / filename
    path.write_bytes(path.read_bytes() + b'\nfrozen-file-tampering')
    with pytest.raises(ValueError, match='frozen traffic file changed'):
        resolve_dataset(manifest, 'train', 50)


def test_resolver_rejects_seed_overlap_in_a_edited_manifest(registry):
    root, manifest, scene = registry
    value = freeze_datasets(root, {'train': [scene(11)], 'test': [scene(33)]}, manifest)
    value['entries'][1]['traffic_seed'] = value['entries'][0]['traffic_seed']
    write_json(manifest, value)
    with pytest.raises(ValueError, match='overlapping'):
        resolve_dataset(manifest, 'train', 50)


def test_resolver_rejects_ambiguous_split_scale_in_a_edited_manifest(registry):
    root, manifest, scene = registry
    value = freeze_datasets(root, {'train': [scene(11)]}, manifest)
    value['entries'].append(copy.deepcopy(value['entries'][0]))
    write_json(manifest, value)
    with pytest.raises(ValueError, match='exactly one|duplicate'):
        resolve_dataset(manifest, 'train', 50)


@pytest.mark.parametrize('absolute', [False, True])
def test_resolver_refuses_directory_escape_outside_declared_root(registry, absolute):
    root, manifest, scene = registry
    value = freeze_datasets(root, {'train': [scene(11)]}, manifest)
    outside = root.parent / 'outside'
    outside.mkdir()
    value['entries'][0]['directory'] = str(outside) if absolute else '../outside'
    write_json(manifest, value)
    with pytest.raises(ValueError, match='unsafe|path|outside'):
        resolve_dataset(manifest, 'train', 50)


def test_resolver_refuses_file_key_escape_even_when_directory_is_safe(registry):
    root, manifest, scene = registry
    value = freeze_datasets(root, {'train': [scene(11)]}, manifest)
    outside = root.parent / 'outside.txt'
    outside.write_bytes(b'outside the declared dataset root')
    value['entries'][0]['files']['../../outside.txt'] = digest(outside)
    write_json(manifest, value)
    with pytest.raises(ValueError, match='file|path|unsafe|manifest'):
        resolve_dataset(manifest, 'train', 50)


def test_resolver_cannot_skip_integrity_checks_using_empty_file_registry(registry):
    root, manifest, scene = registry
    value = freeze_datasets(root, {'train': [scene(11)]}, manifest)
    value['entries'][0]['files'] = {}
    write_json(manifest, value)
    with pytest.raises(ValueError, match='file|manifest|frozen|required'):
        resolve_dataset(manifest, 'train', 50)


def test_freeze_refuses_scenes_outside_the_declared_package_root(registry):
    root, manifest, scene = registry
    directory = scene(11)
    outside = root.parent / 'external_scene'
    shutil.copytree(directory, outside)
    with pytest.raises(ValueError):
        freeze_datasets(root, {'train': [outside]}, manifest)
    assert not manifest.exists()


def test_freeze_cannot_overwrite_an_existing_version(registry):
    root, manifest, scene = registry
    splits = {'train': [scene(11)]}
    freeze_datasets(root, splits, manifest)
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='already exists|new version|overwrite'):
        freeze_datasets(root, splits, manifest)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize('invalid_grid', ['short', 'shifted', 'irregular', 'column', 'nan'])
def test_freeze_requires_the_complete_zero_to_100_second_sample_grid(registry, invalid_grid):
    root, manifest, scene = registry
    directory = scene(11)
    with np.load(directory / 'mobility.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if invalid_grid == 'short':
        arrays['time_s'] = np.array([0., .1])
    elif invalid_grid == 'shifted':
        arrays['time_s'] += .1  # Correct spacing cannot compensate for the wrong time origin.
    elif invalid_grid == 'irregular':
        arrays['time_s'][500] += .01
    elif invalid_grid == 'column':
        arrays['time_s'] = arrays['time_s'][:, None]
    else:
        arrays['time_s'][500] = np.nan
    np.savez_compressed(directory / 'mobility.npz', **arrays)
    with pytest.raises(ValueError, match='dimensions|sample grid|invalid traffic'):
        freeze_datasets(root, {'train': [directory]}, manifest)
    assert not manifest.exists()


@pytest.mark.parametrize('field', ['center_xy_m', 'heading_rad', 'speed_mps'])
def test_freeze_rejects_incomplete_kinematic_array_dimensions(registry, field):
    root, manifest, scene = registry
    directory = scene(11)
    with np.load(directory / 'mobility.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays[field] = arrays[field][:-1]
    np.savez_compressed(directory / 'mobility.npz', **arrays)
    with pytest.raises(ValueError, match='dimensions|invalid traffic'):
        freeze_datasets(root, {'train': [directory]}, manifest)
    assert not manifest.exists()


@pytest.mark.parametrize('field', ['heading_rad', 'speed_mps'])
@pytest.mark.parametrize('invalid_value', [np.nan, np.inf])
def test_freeze_rejects_nonfinite_headings_and_speeds(registry, field, invalid_value):
    root, manifest, scene = registry
    directory = scene(11)
    with np.load(directory / 'mobility.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays[field][500, 0] = invalid_value
    np.savez_compressed(directory / 'mobility.npz', **arrays)
    with pytest.raises(ValueError, match='dimensions|invalid traffic'):
        freeze_datasets(root, {'train': [directory]}, manifest)
    assert not manifest.exists()
