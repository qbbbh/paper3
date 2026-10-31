"""10 车：delta 曲线与逐块最低 EE 分布。只读 runs。

用法:
    python -B -I plot_10veh_results_v1.py <runs_dir> <out_dir>
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(root, tag, method="maxmin"):
    d = root / tag
    fp = d / "results.csv"
    if not fp.exists():
        return None
    per = defaultdict(list)
    with fp.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["row_type"] != "link" or r["selected"] != "True" or r["method"] != method:
                continue
            per[int(r["block"])].append(r)
    ks = sorted(per)
    if not ks:
        return None
    mean_ee = np.array([np.mean([float(x["pair_ee_bit_per_j"]) for x in per[k]]) for k in ks])
    min_ee = np.array([min(float(x["pair_ee_bit_per_j"]) for x in per[k]) for k in ks])
    links = [x for k in ks for x in per[k]]
    s = json.load((d / "summary.json").open(encoding="utf-8"))
    m = s["methods"].get(method, s["methods"]["reference"])
    return dict(tag=tag, ks=ks, mean_ee=mean_ee, min_ee=min_ee,
                power=m["mean_total_electrical_power_w"]["mean"],
                vlc=m["mean_vlc_link_fraction"]["mean"],
                fallback=m.get("fallback_rate", 0.),
                floor=np.mean([float(x["power_w"]) <= 1.001e-3 for x in links]),
                viol=np.mean([float(x["actual_rate_bps"]) < 1e6 for x in links]),
                n_links=len(links))


def main():
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # delta 曲线用的批次名
    groups = [
        ("Pure-EE 参考", "e5_D_chance_d005_20261028", "reference", "k"),
        ("δ=0.05", "e5_D_chance_d005_20261028", "maxmin", "C0"),
        ("δ=0.10", "e5_D_chance_d010_20261028", "maxmin", "C1"),
        ("δ=0.20", "e6_A_chance_d020_20261028", "maxmin", "C2"),
        ("δ=1.00 纯 Max-Min", "e6_A_chance_d100_20261028", "maxmin", "C3"),
    ]
    data = []
    for label, tag, method, color in groups:
        r = load(root, tag, method)
        if r is None:
            print(f"missing {label} ({tag})")
            continue
        r["label"], r["color"] = label, color
        data.append(r)
        print(f"{label:22s} meanEE={r['mean_ee'].mean()/1e6:7.2f}M "
              f"minEE={r['min_ee'].mean()/1e6:7.2f}M Pel={r['power']:.3f} "
              f"vlc={r['vlc']*100:.1f}% fb={r['fallback']:.2f}")

    if not data:
        print("no data")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in data:
        ax[0].scatter(r["mean_ee"].mean() / 1e6, r["min_ee"].mean() / 1e6,
                      s=60, color=r["color"], label=r["label"], zorder=3)
    xs = [r["mean_ee"].mean() / 1e6 for r in data]
    ys = [r["min_ee"].mean() / 1e6 for r in data]
    ax[0].plot(xs, ys, "k--", lw=.8, alpha=.5, zorder=2)
    ax[0].set_xlabel("均逐链路 EE (Mbit/J)")
    ax[0].set_ylabel("块内最低 EE 的 100 块均值 (Mbit/J)")
    ax[0].set_title("(b) 效率—公平权衡")
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8)

    for r in data:
        e = np.sort(r["min_ee"] / 1e6)
        ax[1].plot(e, np.arange(1, len(e) + 1) / len(e), color=r["color"], label=r["label"])
    ax[1].set_xlabel("块内最低逐链路 EE (Mbit/J)")
    ax[1].set_ylabel("累积比例")
    ax[1].set_title("(c) 逐块最低 EE 分布")
    ax[1].set_xlim(0, 400)
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_delta_curve.png", dpi=150)
    print("wrote", out / "fig_delta_curve.png")

    with (out / "delta_curve.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "tag", "method", "mean_ee", "min_ee_mean",
                    "min_ee_p10", "power_w", "vlc_frac", "floor_frac", "viol_frac", "fallback"])
        for r in data:
            w.writerow([r["label"], r["tag"], "maxmin" if r["label"] != "Pure-EE 参考" else "reference",
                        r["mean_ee"].mean(), r["min_ee"].mean(),
                        np.percentile(r["min_ee"], 10), r["power"], r["vlc"],
                        r["floor"], r["viol"], r["fallback"]])


if __name__ == "__main__":
    main()
