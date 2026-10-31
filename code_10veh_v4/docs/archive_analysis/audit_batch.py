"""批量调用 audit_allocation_outputs.py，把每个 run 的结论压成一行。

用法：
  python -I audit_batch.py <code 目录> <输出 txt> <run 目录或 glob> ...

只审计已经有 summary.json 的目录（正在跑的 run 会被标成 NOT_READY）。
审计器要求 --require-current-source：源码哈希必须等于当前 code/ 的哈希。
"""
import json
import subprocess
import sys
from pathlib import Path

PY = r"D:\anaconda\envs\pymarl\python.exe"
code_dir = Path(sys.argv[1]).resolve()
out_txt = Path(sys.argv[2]).resolve()
patterns = sys.argv[3:]

runs = []
for pat in patterns:
    for p in sorted(code_dir.glob(pat)):
        if p.is_dir() and p not in runs:
            runs.append(p)

lines = []
for run in runs:
    name = run.name
    if not (run / "summary.json").exists():
        lines.append(f"{name:44s} NOT_READY (no summary.json)")
        continue
    proc = subprocess.run(
        [PY, "-B", "-I", "audit_allocation_outputs.py", "--require-current-source", str(run)],
        cwd=str(code_dir), capture_output=True, text=True, encoding="utf-8", errors="replace")
    raw = (proc.stdout or "").strip().splitlines()
    try:
        rep = json.loads(raw[-1]) if raw else {}
    except json.JSONDecodeError:
        lines.append(f"{name:44s} AUDIT_CRASH rc={proc.returncode} {raw[-1] if raw else ''}")
        continue
    per = (rep.get("runs") or [{}])[0]
    src = per.get("source") or {}
    checks = per.get("failedchecks") or []
    methods = ",".join(sorted((per.get("methods") or {}).keys()))
    lines.append(
        f"{name:44s} success={rep.get('success')} blocks={per.get('blocks')} "
        f"failed_blocks={per.get('failed_blocks')} src_match={src.get('matches')} "
        f"checks={len(checks)} methods=[{methods}]")

out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(lines)} runs -> {out_txt}")
for line in lines:
    print(line)
