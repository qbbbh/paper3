"""所有方法实际执行动作的"组合上界"：逐块取全部 (run, 方法) 里真值均 EE 最大的那个。

这是**动作池上界**，不是问题的最优解：它只覆盖这些方法真正执行过的动作。
用途：给出"主算法离现有最好动作还有多远"的参照，并说明它不是一个紧的上界。

用法：python portfolio_clean.py <run1> <run2> ...
"""
import json
import sys
from collections import Counter
from pathlib import Path

SEED = "2026102401"


def load(run):
    summary = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
    out = {}
    for r in summary["blocks"]:
        if str(r["channel_seed"]) != SEED or not r.get("success"):
            continue
        out[(int(r["block"]), r["method"])] = float(r["mean_pair_ee_bit_per_j"])
    return Path(run).name, out


runs = [load(p) for p in sys.argv[1:]]
blocks = sorted({b for _n, d in runs for (b, _m) in d})
best, who = {}, {}
for b in blocks:
    cand = [(d[(b, m)], n, m) for n, d in runs for (bb, m) in d if bb == b]
    if cand:
        best[b], who[b], _= (cand[0][0], cand[0][1], None) if False else (None, None, None)
        best[b], name, method = max(cand)
        who[b] = f"{name}/{method}"

print(f"块数 {len(blocks)}；组合里均 EE 最好者来自：")
for k, v in Counter(who.values()).most_common():
    print(f"  {k:50s} {v:3d} 块")
print()
for name, d in runs:
    for method in sorted({m for (_b, m) in d}):
        pairs = [(d[(b, method)], best[b]) for b in blocks if (b, method) in d]
        if not pairs:
            continue
        mine = sum(p[0] for p in pairs) / len(pairs) / 1e6
        bound = sum(p[1] for p in pairs) / len(pairs) / 1e6
        win = sum(p[0] >= p[1] * (1 - 1e-12) for p in pairs)
        print(f"{name}::{method:10s} 配对 {len(pairs):3d} 块  均 EE {mine:6.1f} M  "
              f"组合上界 {bound:6.1f} M  差距 {mine-bound:+6.1f} M ({100*(mine/bound-1):+5.1f}%)  "
              f"已是组合最好 {win}/{len(pairs)}")
