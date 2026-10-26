from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SNAPSHOT_SHA256.json').read_text(encoding='utf-8'))
bad=[]
for name,expected in manifest['files'].items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected: bad.append(name)
expected_py={n for n in manifest['files'] if n.startswith('code/') and n.endswith('.py')}
actual_py={p.relative_to(root).as_posix() for p in (root/'code').rglob('*.py')}
bad.extend(sorted(actual_py-expected_py))
print(json.dumps({'success':not bad,'files':len(manifest['files']),'mismatches':bad}))
sys.exit(bool(bad))
