"""逐块"动作组合上界"：在同一真值下，取所有已跑方法实际执行过的动作里最好的一个。

注意这是**动作组合的上界**，不是问题本身的上界：它只覆盖被比较方法真正执行过的
动作。用来回答"主算法离现有最好动作还有多远"，不能写成"最优解"。

只统计 channel_seed=2026102401 且块区间一致的 run。
"""
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code" / "runs"
SEED = "2026102401"
RUNS = sys.argv[1:] or [
    "e5_D_chance_d010_20261031b",
    "e6_A_chance_d100_20261031b",
    "e7_ideal_ref_20261031b",
    "e7_ideal_chance_d010_20261031b",
]


def per_block(name):
    out = {}
    with io.open(ROOT / name / "results.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["row_type"] != "link" or r["channel_seed"] != SEED:
                continue
            key = (int(r["block"]), r["method"])
            out.setdefault(key, []).append(float(r["pair_ee_bit_per_j"]))
    return {k: (sum(v) / len(v), min(v)) for k, v in out.items()}


def main():
    data = {n: per_block(n) for n in RUNS}
    blocks = sorted({b for d in data.values() for (b, _m) in d})
    best_mean, best_min = {}, {}
    main_mean, main_min = {}, {}
    for b in blocks:
        cand = [(d[(b, m)][0], d[(b, m)][1], n, m)
                for n, d in data.items() for (bb, m) in d if bb == b]
        if not cand:
            continue
        best_mean[b] = max(cand, key=lambda x: x[0])
        best_min[b] = max(cand, key=lambda x: x[1])
        if ("e5_D_chance_d010_20261031b", "maxmin") in [c[2:] for c in cand]:
            pass
    main = {b: max((d[(b, "maxmin")] for d in data.values() if (b, "maxmin") in d), key=lambda x: x[0])
            for b in blocks if any((b, "maxmin") in d for d in data.values())}
    n = len(best_mean)
    gap_m = sum(best_mean[b][0] - main[b][0] for b in main if b in best_mean) / len(main)
    gap_min = sum(best_min[b][1] - main[b][1] for b in main if b in best_min) / len(main)
    win = sum(best_mean[b][0] <= main[b][0] * (1 + 1e-9) for b in main if b in best_mean)
    print(f"块数 {n}；组合里均 EE 最好者来自：")
    src = defaultdict(int)
    for b in best_mean:
        src[best_mean[b][2] + "/" + best_mean[b][3]] += 1
    for k, v in sorted(src.items(), key=lambda kv: -kv[1]):
        print(f"  {k:52s} {v:3d} 块")
    print(f"主算法（e5_D maxmin）均 EE 与组合最好的平均差 {gap_m/1e6:+.1f} M；"
          f"已在组合最好上的块 {win}/{len(main)}")
    print(f"min EE 平均差 {gap_min/1e6:+.1f} M")


if __name__ == "__main__":
    main()
