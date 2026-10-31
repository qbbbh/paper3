"""δ 曲线跨信道实现对照图：只读 runs。

主实现有 δ=0/.05/.10/.20/1.0 五点，实现② 有 δ=0/.05/.10 三点。
δ=.20 与 1.0 只有主实现，画成虚线延伸，不能读成"第二条实现也验证过"。

用法:
    python -B -I plot_delta_cross_v1.py <runs_root> <out_dir>
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(root, tag, method):
    d = root / tag
    if not (d / "summary.json").exists():
        return None
    per = defaultdict(list)
    with (d / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["row_type"] != "link" or r["selected"] != "True" or r["method"] != method:
                continue
            per[int(r["block"])].append(float(r["pair_ee_bit_per_j"]))
    if not per:
        return None
    ee = np.array([np.mean(per[k]) for k in sorted(per)])
    mn = np.array([min(per[k]) for k in sorted(per)])
    s = json.load((d / "summary.json").open(encoding="utf-8"))
    m = s["methods"][method]
    return dict(ee=float(ee.mean()), mn=float(mn.mean()),
                pel=m["mean_total_electrical_power_w"]["mean"],
                vlc=m["mean_vlc_link_fraction"]["mean"])


def main():
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    A = [("reference", "e5_D_chance_d005_20261028", "reference", 0.0),
         ("δ=.05", "e5_D_chance_d005_20261028", "maxmin", .05),
         ("δ=.10", "e5_D_chance_d010_20261028", "maxmin", .10),
         ("δ=.20", "e6_A_chance_d020_20261028", "maxmin", .20),
         ("δ=1.0", "e6_A_chance_d100_20261028", "maxmin", 1.0)]
    B = [("reference", "e6_B_ref_20261028", "reference", 0.0),
         ("δ=.05", "e6_B_d005_20261028", "maxmin", .05),
         ("δ=.10", "e6_B_d010_20261028", "maxmin", .10)]

    series = {}
    for name, spec in (("主实现", A), ("实现②", B)):
        xs, ee, mn, pel = [], [], [], []
        for label, tag, method, delta in spec:
            r = load(root, tag, method)
            if r is None:
                print("missing", name, label, tag)
                continue
            print(f"{name} {label:9s} EE={r['ee']/1e6:7.2f}M min={r['mn']/1e6:6.2f}M "
                  f"Pel={r['pel']:.3f} vlc={r['vlc']*100:.1f}%")
            xs.append(delta)
            ee.append(r["ee"] / 1e6)
            mn.append(r["mn"] / 1e6)
            pel.append(r["pel"])
        series[name] = dict(x=xs, ee=ee, mn=mn, pel=pel)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # δ 的线性轴会把 0–0.10 压成一点，改用按档位等距的分类轴。
    ticks = [0.0, .05, .10, .20, 1.0]
    ticklabels = ["ref", ".05", ".10", ".20", "1.0\n(纯 Max-Min)"]
    idx = {v: k for k, v in enumerate(ticks)}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"主实现": "C0", "实现②": "C1"}
    for name, s in series.items():
        x = np.array([idx[v] for v in s["x"]], float)
        solid = x <= 2 + 1e-12
        c = colors[name]
        for k, key in enumerate(("ee", "mn", "pel")):
            ax[k].plot(x[solid], np.array(s[key])[solid], "o-", color=c, label=name)
            ax[k].plot(x[~solid], np.array(s[key])[~solid], "o--", color=c, alpha=.6)
    for k, (title, ylab) in enumerate(
            (("(a) 均逐链路 EE", "Mbit/J"), ("(b) 块内最低 EE（100 块均值）", "Mbit/J"),
             ("(c) 全网平均电功率", "全网平均电功率 (W)"))):
        ax[k].set_xticks(range(len(ticks)))
        ax[k].set_xticklabels(ticklabels)
        ax[k].set_xlabel("δ（横轴按档位等距排列）")
        ax[k].set_ylabel(ylab)
        ax[k].set_title(title)
        ax[k].grid(alpha=.3)
        ax[k].legend(fontsize=8)
        ax[k].set_xlim(-.3, len(ticks) - .7)
    fig.tight_layout()
    fig.savefig(out / "fig_delta_cross.png", dpi=150)
    print("wrote", out / "fig_delta_cross.png")


if __name__ == "__main__":
    main()
