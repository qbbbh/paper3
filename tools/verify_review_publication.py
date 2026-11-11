"""Check the published manuscript views and local navigation links."""
from pathlib import Path
import hashlib
import json
import posixpath
import re
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
published = set(json.loads((root / 'SNAPSHOT_SHA256.json').read_text(encoding='utf-8'))['files'])
directories = {name[:i] for name in published for i, char in enumerate(name) if char == '/'}
manuscripts = json.loads((root / 'manuscripts/paper3_20261111/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
errors = []
for item in manuscripts['files']:
    name = item['published_path']
    path = root / name
    if (name not in published or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != item['published_sha256']):
        errors.append('Manuscript view mismatch: ' + name)

navigation = sorted(name for name in published if name.endswith('.md') and (
    '/' not in name or name.startswith(('releases/', 'evidence/', 'manuscripts/', 'tools/'))))
link = re.compile(r'!?\[[^\]\n]*\]\((<?)([^)\n]+?)(>?)\)')
for name in navigation:
    parent = posixpath.dirname(name)
    for match in link.finditer((root / name).read_text(encoding='utf-8-sig')):
        target = match.group(2).strip().strip('<>')
        if re.match(r'^[a-z]+://', target, re.I) or target.startswith(('#', 'mailto:', 'data:')):
            continue
        local = unquote(re.split(r'[?#]', target, maxsplit=1)[0])
        if local:
            resolved = posixpath.normpath(posixpath.join(parent, local))
            if resolved not in published and resolved not in directories:
                errors.append(f'{name} -> {target}')
print(json.dumps({'success': not errors, 'manuscript_views': len(manuscripts['files']),
                  'navigation_files': len(navigation), 'errors': errors}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
