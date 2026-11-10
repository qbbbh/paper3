"""Export verified committed results to a NEW partial-results directory.

This preserves computed results; it does not resume oracle/estimator/RNG state.
No summary.json is created, so an incomplete run cannot pass as a complete run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_safety import atomic_json


def recover(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('choose a new recovery directory')
    manifest = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8'))
    rows, records, arrays, ids = [], [], {}, []
    for folder in sorted((source / 'block_checkpoints').glob('block_*')):
        commit = json.loads((folder / 'commit.json').read_text(encoding='utf-8'))
        for name in ('records.json', 'fine_samples.npz'):
            actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            if actual != commit['sha256'][name]:
                raise ValueError(f'checksum mismatch: {folder / name}')
        payload = json.loads((folder / 'records.json').read_text(encoding='utf-8'))
        expected = manifest['window']['start_block'] + len(ids)
        if payload['block'] != expected or commit['block'] != expected:
            raise ValueError('noncontiguous block sequence')
        for key in ('source_sha256', 'config_sha256'):
            if payload[key] != manifest[key]:
                raise ValueError(f'{key} mismatch')
        ids.append(expected)
        rows.extend(payload['rows'])
        records.extend(payload['block_records'])
        with np.load(folder / 'fine_samples.npz', allow_pickle=False) as z:
            if set(arrays).intersection(z.files):
                raise ValueError('duplicate sample keys')
            arrays.update({key: z[key] for key in z.files})
    if not ids:
        raise ValueError('no committed blocks')
    destination.mkdir(parents=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (destination / 'results.partial.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    arrays['audit_only'] = np.array(True)
    arrays['units'] = np.array('time=s;rate=bit/s;sinr=linear;RF S/I/N=W;VLC S/I/N=A^2')
    np.savez_compressed(destination / 'fine_samples.partial.npz', **arrays)
    atomic_json(destination / 'recovery.json', dict(status='partial_export', source=str(source.resolve()),
        manifest=manifest, recovered_blocks=ids, block_records=records,
        planned_blocks=manifest['window']['planned_blocks'],
        restart_state_available=all((source/'block_checkpoints'/f'block_{k:06d}'/'restart.pkl.gz').exists() for k in ids)))
    return len(ids)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    print(f'Recovered {recover(args.source, args.destination)} committed blocks')
