"""同块不同方法是否执行了同一个动作（action_sha256 相同），以及失败块清单。

用法：python same_action.py <run> <methodA> <methodB>
"""
import json
import sys
from pathlib import Path

SEED = "2026102401"
run, ma, mb = sys.argv[1:4]
summary = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
rec = {}
for r in summary["blocks"]:
    if str(r["channel_seed"]) != SEED:
        continue
    rec.setdefault(r["method"], {})[int(r["block"])] = r

a, b = rec[ma], rec[mb]
common = sorted(set(a) & set(b))
same = [k for k in common if a[k].get("success") and b[k].get("success")
        and a[k]["action_sha256"] == b[k]["action_sha256"]]
both_ok = [k for k in common if a[k].get("success") and b[k].get("success")]
print(f"{Path(run).name}: {ma} vs {mb}")
print(f"  共同块 {len(common)}；两边都成功 {len(both_ok)}；动作完全相同 {len(same)}")
print(f"  {mb} 失败块: {sorted(k for k in common if not b[k].get('success'))}")
print(f"  {ma} 失败块: {sorted(k for k in common if not a[k].get('success'))}")
