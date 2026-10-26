"""Materialize the original import layout without changing scientific source bytes."""
from pathlib import Path
import hashlib
import json
import runpy
import shutil

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def contained(base, relative):
    p = (base / relative).resolve()
    if base.resolve() not in p.parents:
        raise ValueError('Path outside root: ' + relative)
    return p

def build():
    runpy.run_path(str(ROOT / 'tools/restore_data.py'), run_name='__main__')
    layout = json.loads((ROOT / 'reviews/source_layout_map.json').read_text(encoding='utf-8'))
    runtime = ROOT / '.runtime/frozen_95960f5a'
    runtime.mkdir(parents=True, exist_ok=True)
    expected = set()
    for entry in layout['files']:
        src = contained(ROOT, entry['organized'])
        if digest(src) != entry['sha256']:
            raise RuntimeError('Source changed; create a new reviewed version: ' + entry['organized'])
        dst = contained(runtime, entry['original'])
        expected.add(dst)
        if dst.exists() and digest(dst) != entry['sha256']:
            raise RuntimeError('Runtime modified; refusing overwrite: ' + str(dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): shutil.copyfile(src, dst)
    extras = set((runtime / 'code').rglob('*.py')) - expected
    if extras: raise RuntimeError('Unexpected Python sources in runtime: ' + str(extras))
    h = hashlib.sha256()
    for p in sorted((runtime / 'code').rglob('*.py')):
        h.update(p.relative_to(runtime).as_posix().encode())
        h.update(p.read_bytes())
    if h.hexdigest() != layout['source_sha256']:
        raise RuntimeError('Original source hash not reproduced')
    for n in (10, 50, 100):
        for name in ('mobility.npz', 'buildings.npz', 'scenario.json', 'audit.json'):
            src = ROOT / f'resource_allocation/n{n}/data' / name
            dst = runtime / f'SUMO_10_50_100车_建筑公园_20261023/n{n}' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and digest(dst) != digest(src):
                raise RuntimeError('Runtime data differs: ' + str(dst))
            if not dst.exists(): shutil.copyfile(src, dst)
    fixture = runtime / 'reference/geometry_expected.npz'
    fixture.parent.mkdir(exist_ok=True)
    src = ROOT / 'tests/fixtures/geometry_expected.npz'
    if fixture.exists() and digest(fixture) != digest(src):
        raise RuntimeError('Runtime fixture differs')
    if not fixture.exists(): shutil.copyfile(src, fixture)
    return runtime

if __name__ == '__main__':
    runtime = build()
    print(json.dumps({'success': True, 'runtime': str(runtime), 'source_sha256': json.loads((ROOT/'reviews/source_layout_map.json').read_text(encoding='utf-8'))['source_sha256']}))
