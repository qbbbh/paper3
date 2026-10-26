"""Verify organized materials and reconstruct the original source hash in memory."""
from pathlib import Path
import hashlib, json, sys
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'SNAPSHOT_SHA256.json').read_text(encoding='utf-8'))
bad = []
for name, expected in manifest['files'].items():
    p = (root/name).resolve()
    if root not in p.parents or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        bad.append(name)
layout = json.loads((root/'reviews/source_layout_map.json').read_text(encoding='utf-8'))
python_files = [e for e in layout['files'] if e['original'].endswith('.py')]
h = hashlib.sha256()
for item in sorted(python_files, key=lambda e:e['original']):
    p = root/item['organized']
    if not p.is_file():
        bad.append(item['organized']); continue
    data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=item['sha256']: bad.append(item['organized'])
    h.update(item['original'].encode()); h.update(data)
if h.hexdigest()!=layout['source_sha256']: bad.append('scientific_source_sha256')
expected=set(manifest['files'])
for subtree in ('channel_estimation','resource_allocation','tests','tools'):
    for p in (root/subtree).rglob('*.py'):
        if p.relative_to(root).as_posix() not in expected: bad.append('unexpected:'+p.relative_to(root).as_posix())
print(json.dumps({'success':not bad,'files':len(manifest['files']),'scientific_python_files':len(python_files),'source_sha256':h.hexdigest(),'mismatches':sorted(set(bad))},ensure_ascii=False))
sys.exit(bool(bad))
