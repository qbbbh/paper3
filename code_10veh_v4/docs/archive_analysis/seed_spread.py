"""4 个 forecast seed 的散度 vs 估计/ideal 之间的系统差。

输出每组的均 EE / min EE 均值与标准差，以及两组均值差相对"合并标准差/√n"的倍数
（Welch t，仅作量级参考，不当成正式检验——样本只有 4 个）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code" / "runs"


def series(prefix, kind, key):
    vals = []
    for s in (1, 2, 3, 4):
        p = ROOT / f"e8_seed{s}_{kind}_20261031c" / "summary.json"
        summary = json.loads(p.read_text(encoding="utf-8"))
        m = summary["methods"][prefix]
        vals.append(m[key]["mean"])
    return vals


def stats(v):
    n = len(v)
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / (n - 1)
    return mean, var ** .5


for prefix in ("reference",):
    for key in ("mean_pair_ee_including_failed_as_zero", "min_pair_ee_including_failed_as_zero",
                "mean_total_electrical_power_w"):
        est = series(prefix, "est", key)
        ide = series(prefix, "ideal", key)
        me, se = stats(est)
        mi, si = stats(ide)
        se_mean = (se ** 2 / len(est) + si ** 2 / len(ide)) ** .5
        t = (me - mi) / se_mean if se_mean else float("inf")
        unit = 1e6 if "ee" in key else 1
        print(f"{key:44s} est {me/unit:8.2f}±{se/unit:5.2f}  ideal {mi/unit:8.2f}±{si/unit:5.2f}  "
              f"差 {100*(me/mi-1):+6.2f}%  t≈{t:5.1f}（n=4，仅量级）")
