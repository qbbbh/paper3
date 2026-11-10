"""Run-window preflight, committed results, and local numerical restart state."""
import hashlib
import json
import os
import gzip
import pickle
import csv
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np


def validate_window(times, start_block, blocks, block_dt, fine_dt):
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("invalid scene times")
    if start_block < 0 or blocks <= 0 or not np.isfinite([block_dt, fine_dt]).all() or min(block_dt, fine_dt) <= 0:
        raise ValueError("invalid start/count/time step")
    if not np.isclose(round(block_dt / fine_dt) * fine_dt, block_dt, rtol=0, atol=1e-10):
        raise ValueError("block duration must contain whole fine steps")
    first, end = start_block * block_dt, (start_block + blocks) * block_dt
    max_blocks = max(0, int(np.floor((times[-1] - first) / block_dt + 1e-8)))
    if first < times[0] - 1e-9 or end > times[-1] + 1e-9:
        raise ValueError(f"requested [{first:.9g}, {end:.9g}] s exceeds scene "
                         f"[{times[0]:.9g}, {times[-1]:.9g}] s; "
                         f"start_block={start_block} permits at most {max_blocks} complete blocks")
    indices = np.arange(start_block, start_block + blocks)
    if indices[-1] >= len(times) or not np.allclose(times[indices], indices * block_dt, rtol=0, atol=1e-9):
        raise ValueError("scene knot indices do not match the driver's block times")
    return dict(start_s=float(first), end_s=float(end), duration_s=float(end-first),
                planned_blocks=int(blocks), start_block=int(start_block), max_blocks=max_blocks)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def commit_block(out, block_index, payload, arrays, restart_state=None):
    """Publish a directory only after JSON/NPZ are flushed and hashed.

    An interrupted .pending directory is not a committed block. No simulation
    object or RNG is inspected/modified. Existing commits are never overwritten.
    """
    root = Path(out) / 'block_checkpoints'
    root.mkdir(exist_ok=True)
    final = root / f'block_{block_index:06d}'
    staging = root / f'.block_{block_index:06d}.pending'
    if final.exists() or staging.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    staging.mkdir()
    atomic_json(staging / 'records.json', payload)
    with (staging / 'fine_samples.npz').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    checksums = {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                 for name in ('records.json', 'fine_samples.npz')}
    if restart_state is not None:
        with (staging / 'restart.pkl.gz').open('wb') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                pickle.dump(restart_state, compressed, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        checksums['restart.pkl.gz'] = hashlib.sha256((staging / 'restart.pkl.gz').read_bytes()).hexdigest()
    atomic_json(staging / 'commit.json', dict(block=int(block_index), sha256=checksums))
    os.replace(staging, final)
    atomic_json(Path(out) / 'progress.json', dict(status='running', last_committed_block=int(block_index)))


def load_committed(out, source_hash, config_hash, start_block, planned_blocks, lightweight=False):
    """Read local trusted checkpoints, verify identity/hash/continuity before pickle.

    Restart files are executable Python serialization: only load this runner's
    own local output, never downloaded or externally supplied checkpoint files.
    """
    root = Path(out) / 'block_checkpoints'
    payloads, arrays, last = [], {}, None
    for folder in sorted(root.glob('block_*')):
        commit = json.loads((folder / 'commit.json').read_text(encoding='utf-8'))
        for name in ('records.json', 'fine_samples.npz', 'restart.pkl.gz'):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != commit['sha256'].get(name):
                raise ValueError(f'restart checksum mismatch: {folder / name}')
        value = json.loads((folder / 'records.json').read_text(encoding='utf-8'))
        expected = start_block + len(payloads)
        if value['block'] != expected or commit['block'] != expected:
            raise ValueError('restart blocks are not contiguous')
        if value['source_sha256'] != source_hash or value['config_sha256'] != config_hash:
            raise ValueError('restart source/config identity mismatch')
        if lightweight:
            payloads.append(dict(block=value['block'], block_records=value['block_records']))
        else:
            payloads.append(value)
            with np.load(folder / 'fine_samples.npz', allow_pickle=False) as z:
                if set(arrays).intersection(z.files):
                    raise ValueError('duplicate restart samples')
                arrays.update({key: z[key] for key in z.files})
        last = folder
    if len(payloads) > planned_blocks:
        raise ValueError('too many committed blocks')
    runtime = None
    if last:
        with gzip.open(last / 'restart.pkl.gz', 'rb') as stream:
            runtime = pickle.load(stream)
    # A killed write leaves a pending directory. Preserve it as forensic evidence;
    # don't consume it and don't delete it. New writes use the original name.
    for pending in root.glob('.block_*.pending'):
        suffix = 0
        archived = pending.with_name(pending.name + '.abandoned')
        while archived.exists():
            suffix += 1
            archived = pending.with_name(pending.name + f'.abandoned{suffix}')
        os.replace(pending, archived)
    return payloads, arrays, runtime


def consolidate_results(out, config, credit_elapsed_s=0.0):
    """Stream committed data to legacy final files without retaining all arrays/logs."""
    out = Path(out)
    folders = sorted((out / 'block_checkpoints').glob('block_*'))
    def values():
        for folder in folders:
            yield json.loads((folder / 'records.json').read_text(encoding='utf-8'))
    fields = dict.fromkeys(['row_type', 'tx', 'rx', 'mode'])
    for payload in values():
        for row in payload['rows']:
            fields.update(dict.fromkeys(row))
    temporary = out / 'results.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        for payload in values():
            writer.writerows(payload['rows'])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, out / 'results.csv')
    temporary = out / 'fine_samples.npz.tmp'
    names = set()
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as dest:
        for folder in folders:
            with zipfile.ZipFile(folder / 'fine_samples.npz') as source:
                for name in source.namelist():
                    if name in names:
                        raise ValueError('duplicate final NPZ key')
                    names.add(name)
                    with source.open(name) as a, dest.open(name, 'w', force_zip64=True) as b:
                        shutil.copyfileobj(a, b)
        for key, value in dict(audit_only=np.array(True), units=np.array(
                'time=s;rate=bit/s;sinr=linear;RF S/I/N=W;VLC S/I/N=A^2')).items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            dest.writestr(key+'.npy', buffer.getvalue())
    with temporary.open('rb+') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, out / 'fine_samples.npz')

    def write_list(filename, key, metadata=None):
        temporary = out / (filename + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            if metadata is not None:
                stream.write(json.dumps(metadata, ensure_ascii=False)[:-1] + ', "blocks": ')
            stream.write('[')
            first = True
            for payload in values():
                for item in payload[key]:
                    if not first:
                        stream.write(',\n')
                    json.dump(item, stream, ensure_ascii=False, allow_nan=False)
                    first = False
            stream.write(']' + ('}' if metadata is not None else ''))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out / filename)
    write_list('training_curve.json', 'training_curve')
    if config['power_solver'] == 'continuous':
        write_list('continuous_power_audit.json', 'power_solver_records')
    if set(config['methods']) & {'literature', 'aboagye_matching_qt', 'zappone_mmee_fp_fixed'}:
        write_list('literature_audit.json', 'literature_records')
    if config['maxmin_weak_link'] or config['maxmin_refine_solver'] != 'nm':
        write_list('weak_link.json', 'weak_link_records', dict(schema_version=1,
            refine_solver=config['maxmin_refine_solver'], weak_link_repair=config['maxmin_weak_link'],
            repair_limit=config['maxmin_repair_limit']))
    if config.get('credit'):
        write_list('credit.json', 'credit_records', dict(schema_version=1,
            information='conditional_trajectory_forecast_only', calibration_status='approximate_not_certified',
            max_evals=config['credit_max_evals'], focus=config['credit_focus'], elapsed_s=credit_elapsed_s))


def record_failure(out, error):
    out = Path(out)
    # Preflight failures must not create a seemingly started run.
    if (out / 'run_manifest.json').exists():
        atomic_json(out / 'failure.json', dict(status='failed', exception_type=type(error).__name__,
                                              message=str(error), partial_results='block_checkpoints'))
