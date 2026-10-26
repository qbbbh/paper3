"""Restore byte-identical large data files from GitHub transport parts."""
from pathlib import Path
import hashlib
import json
import os

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'data_parts/manifest.json').read_text(encoding='utf-8'))

def checked_path(name):
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError('Path outside repository: ' + name)
    return path

def digest(data):
    return hashlib.sha256(data).hexdigest()

restored = 0
for item in manifest['files']:
    destination = checked_path(item['path'])
    if destination.exists():
        if digest(destination.read_bytes()) != item['sha256']:
            raise RuntimeError('Existing file differs; refusing overwrite: ' + item['path'])
        continue
    chunks = []
    for part in item['parts']:
        content = checked_path(part['path']).read_bytes()
        if digest(content) != part['sha256']:
            raise RuntimeError('Corrupt part: ' + part['path'])
        chunks.append(content)
    data = b''.join(chunks)
    if len(data) != item['bytes'] or digest(data) != item['sha256']:
        raise RuntimeError('Restored file mismatch: ' + item['path'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.restore-tmp')
    with temporary.open('xb') as output:
        output.write(data)
    os.replace(temporary, destination)
    restored += 1
print(json.dumps({'success': True, 'restored': restored, 'verified_files': len(manifest['files'])}))
