"""Verify the published files in the current Paper3 checkout."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'SNAPSHOT_SHA256.json').read_text(encoding='utf-8'))
mismatches = []
for name, expected in manifest['files'].items():
    path = (root / name).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        mismatches.append(name)
print(json.dumps({'success': not mismatches, 'files': len(manifest['files']),
                  'mismatches': mismatches}, ensure_ascii=False))
raise SystemExit(bool(mismatches))
