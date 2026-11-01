"""覆盖率的 block bootstrap 置信区间。

为什么要这个：9990 条链路不等于 9990 个独立样本。同一块内 10 条链路共享几何与信道实现，
相邻块之间还有时间相关。直接按链路算二项置信区间会把区间压得过窄。
这里按**块**重采样（block bootstrap），给出经验分位跌破率的区间。

不依赖 v5 源码树，只读 results.csv。

用法::

    D:\\anaconda\\envs\\pymarl\\python.exe -B -I block_bootstrap.py ^
        --run "..\\开发_10车机制定型_v4_20261027\\code\\runs\\e5_D_chance_d010_20261031h" ^
        --out out_bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def load_blocks(results_csv):
    """返回 [(block, n_used, n_violation), ...]，只统计可用于校准的行。"""
    import csv
    per_block = {}
    with open(results_csv, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("row_type") != "link":
                continue
            try:
                p05 = float(row["predicted_p05_bps"])
                mean = float(row["predicted_rate_bps"])
                actual = float(row["actual_rate_bps"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(p05) and np.isfinite(mean) and np.isfinite(actual)):
                continue
            if p05 <= 0.0 or mean <= 0.0 or mean <= p05:
                continue
            key = int(row["block"])
            n, v = per_block.get(key, (0, 0))
            per_block[key] = (n + 1, v + (1 if actual < p05 else 0))
    return [(k, n, v) for k, (n, v) in sorted(per_block.items())]


def inflate(mean, low, alpha):
    floor = 1e-9
    m = max(mean, floor)
    return m * (max(low, floor) / m) ** alpha


def violation_at_alpha(results_csv, alpha):
    """给定 alpha，返回逐块的 (n, v)。"""
    import csv
    per_block = {}
    with open(results_csv, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("row_type") != "link":
                continue
            try:
                p05 = float(row["predicted_p05_bps"])
                mean = float(row["predicted_rate_bps"])
                actual = float(row["actual_rate_bps"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(p05) and np.isfinite(mean) and np.isfinite(actual)):
                continue
            if p05 <= 0.0 or mean <= 0.0 or mean <= p05:
                continue
            key = int(row["block"])
            n, v = per_block.get(key, (0, 0))
            thr = inflate(mean, p05, alpha)
            per_block[key] = (n + 1, v + (1 if actual < thr else 0))
    return [(k, n, v) for k, (n, v) in sorted(per_block.items())]


def bootstrap(blocks, reps=4000, seed=20261101, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = np.array([b[1] for b in blocks], float)
    v = np.array([b[2] for b in blocks], float)
    if len(n) == 0:
        return None
    point = float(v.sum() / n.sum())
    idx = rng.integers(0, len(n), size=(reps, len(n)))
    rates = v[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "reps": reps,
        "n_blocks": int(len(n)),
        "n_links": int(n.sum()),
        "mean_links_per_block": float(n.mean()),
        "probability_above_nominal": float(np.mean(rates > 0.05)),
    }


def naive_binomial(blocks, alpha=0.05):
    """对照：误把链路当独立样本时的二项区间（会偏窄）。"""
    n = sum(b[1] for b in blocks)
    v = sum(b[2] for b in blocks)
    if n == 0:
        return None
    p = v / n
    se = float(np.sqrt(p * (1 - p) / n))
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return {"point": p, "ci_low": max(0.0, p - z * se), "ci_high": min(1.0, p + z * se),
            "n_links": int(n)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--alpha-targets", type=float, nargs="+", default=[1.0, 1.35, 2.0])
    parser.add_argument("--reps", type=int, default=4000)
    parser.add_argument("--out", default="out_bootstrap.json")
    args = parser.parse_args(argv)

    results_csv = os.path.join(args.run, "results.csv")
    report = {"run": args.run, "reps": args.reps, "alpha": {}}
    for a in args.alpha_targets:
        blocks = violation_at_alpha(results_csv, a)
        boot = bootstrap(blocks, reps=args.reps)
        naive = naive_binomial(blocks)
        report["alpha"][f"{a:g}"] = {"block_bootstrap": boot, "naive_binomial": naive}
        if boot:
            print(f"alpha={a:<5g} point={boot['point']:.4f} "
                  f"block-bootstrap 95% CI=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] "
                  f"P(rate>5%)={boot['probability_above_nominal']:.3f} "
                  f"blocks={boot['n_blocks']} links={boot['n_links']}")
            if naive:
                print(f"            naive binomial CI=[{naive['ci_low']:.4f}, "
                      f"{naive['ci_high']:.4f}]  (偏窄，仅供对照)")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("写出:", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
