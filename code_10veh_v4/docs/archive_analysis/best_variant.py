"""给文献基线取"逐块最好的变体"，再和主算法配对比较。

变体只允许换标称功率与功率求解器，不能换关联规则——这样仍是同一文献方法，
只是把它的超参调到对**对手**最有利的程度。
"""
import json
import sys
from pathlib import Path

SEED = "2026102401"


def load(run, method):
    summary = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
    out = {}
    for r in summary["blocks"]:
        if str(r["channel_seed"]) != SEED or r["method"] != method:
            continue
        out[int(r["block"])] = (float(r["mean_pair_ee_bit_per_j"]) if r.get("success") else 0.0,
                                float(r["min_pair_ee_bit_per_j"]) if r.get("success") else 0.0)
    return out


main = load(sys.argv[1], "maxmin")
variants = [load(p, "literature") for p in sys.argv[2:]]
blocks = sorted(main)
best = {b: max(v[b] for v in variants if b in v) for b in blocks if all(b in v for v in variants)}
pairs = [b for b in blocks if b in best]
ratios = [main[b][0] / best[b][0] for b in pairs if best[b][0] > 0]
print(f"主算法 maxmin vs 逐块最好的文献变体：配对 {len(ratios)} 块")
print(f"  均 EE 比：均值 {100*(sum(ratios)/len(ratios)-1):+.2f}%，"
      f"中位 {100*(sorted(ratios)[len(ratios)//2]-1):+.2f}%，"
      f"胜 {sum(r > 1 for r in ratios)}/{len(ratios)}")
mean_main = sum(main[b][0] for b in pairs) / len(pairs)
mean_best = sum(best[b][0] for b in pairs) / len(pairs)
mean_min_main = sum(main[b][1] for b in pairs) / len(pairs)
mean_min_best = sum(best[b][1] for b in pairs) / len(pairs)
print(f"  均 EE：主 {mean_main/1e6:.1f} M vs 变体最好 {mean_best/1e6:.1f} M "
      f"({100*(mean_main/mean_best-1):+.2f}%)")
print(f"  min EE：主 {mean_min_main/1e6:.1f} M vs 变体最好 {mean_min_best/1e6:.1f} M "
      f"({100*(mean_min_main/mean_min_best-1):+.2f}%)")
for name in sys.argv[2:]:
    v = load(name, "literature")
    r = [main[b][0] / v[b][0] for b in pairs if v[b][0] > 0]
    print(f"  vs {Path(name).name:34s} 均值 {100*(sum(r)/len(r)-1):+6.2f}%  "
          f"胜 {sum(x > 1 for x in r)}/{len(r)}  零块 {sum(1 for b in pairs if v[b][0] == 0)}")
