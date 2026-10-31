"""逐块配对比的分布（不只报均值）：分位数、胜/负/平、按块号列出最差与最好。

用法：python pair_detail.py <A run> <A 方法> <B run> <B 方法>
"""
import json
import sys
from pathlib import Path

SEED = "2026102401"


def blocks(run, method):
    summary = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
    out = {}
    for r in summary["blocks"]:
        if str(r["channel_seed"]) != SEED or r["method"] != method:
            continue
        if r.get("success"):
            out[int(r["block"])] = float(r["mean_pair_ee_bit_per_j"])
    return out


a_run, a_m, b_run, b_m = sys.argv[1:5]
a, b = blocks(a_run, a_m), blocks(b_run, b_m)
pairs = sorted(set(a) & set(b))
ratios = [(k, a[k] / b[k]) for k in pairs if b[k]]
ratios.sort(key=lambda x: x[1])
vals = [r for _k, r in ratios]


def q(p):
    return vals[min(len(vals) - 1, int(p * (len(vals) - 1)))]


print(f"A={Path(a_run).name}::{a_m}  B={Path(b_run).name}::{b_m}")
print(f"配对 {len(vals)} 块")
for p in (0.05, 0.25, 0.5, 0.75, 0.95):
    print(f"  p{int(p*100):02d} 比 {100*(q(p)-1):+7.2f}%")
print(f"  均值比 {100*(sum(vals)/len(vals)-1):+7.2f}%")
print(f"  A 更好 {sum(v > 1+1e-6 for v in vals)}  持平 {sum(abs(v-1) <= 1e-6 for v in vals)}  "
      f"A 更差 {sum(v < 1-1e-6 for v in vals)}")
print("最差 5 块（A/B-1）：", [(k, f"{100*(r-1):+.1f}%") for k, r in ratios[:5]])
print("最好 5 块（A/B-1）：", [(k, f"{100*(r-1):+.1f}%") for k, r in ratios[-5:]])
