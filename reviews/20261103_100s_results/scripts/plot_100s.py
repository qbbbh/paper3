"""100 s 结果导出与绘图（v2）。

相对 v1 的修正：
1. **文献基线必须同时给两个口径**：含失败（失败块记 0）与仅成功块。
   v1 的箱线图排除了失败块，导致文献基线的箱体反而高于本文方法——那是选择偏差。
2. 图 3(a) 放大到预算边界附近（算法会把预测余量压到 0 附近，全量散点挤在原点看不清）；
   图 3(b) 改为真实预算余量的经验 CDF，直接读出违约概率。
3. 图 1 对文献基线并列两根柱（含失败 / 仅成功），并标注样本数。

产出：
- `data/dataset_blocks.csv`（逐块，含 failed 标记与预测/真实预算余量）
- `figures/fig1_main_comparison.png`、`fig2_box_min_ee.png`、`fig3_csi_margin.png`
- `figures/fig4_delta_sensitivity.png`、`fig5_csi_quality.png`（对应 run 存在时）
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["savefig.dpi"] = 300

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")

METHOD_CN = {"reference": "Pure-EE 参考", "maxmin": "本文方法", "literature": "文献基线"}
COLORS = {"reference": "#8c8c8c", "maxmin": "#c0392b", "literature": "#2e86c1"}

RUN_SPECS = [
    ("R1p", "R1p_v5A_d010_100s", "设计种子", 0.10, "estimated", True),
    ("R2p", "R2p_v5A_d010_100s_valseed", "验证种子", 0.10, "estimated", True),
    ("R3p1", "R3p1_norefine_d010_100s", "设计种子", 0.10, "estimated", False),
    ("R3p2", "R3p2_norefine_d010_100s_valseed", "验证种子", 0.10, "estimated", False),
    ("R4", "R4_d005_100s", "设计种子", 0.05, "estimated", True),
    ("R5", "R5_d020_100s", "设计种子", 0.20, "estimated", True),
    ("R6", "R6_oracle_100s", "设计种子", 0.10, "oracle", True),
]

FIELDS = ["run", "seed", "delta", "csi", "refine", "method", "block", "failed",
          "min_ee_M", "mean_ee_M", "pred_mean_ee_M", "eta_ref_pred_M", "eta_ref_true_M",
          "qos_ok", "fallback", "viol_pred", "viol_true", "loss_true_pct",
          "budget_margin_pred_M", "budget_margin_true_M",
          "total_power_W", "vlc_power_frac", "jain", "solve_s"]


def fnum(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def load_run(path, delta):
    out = {}
    with open(os.path.join(path, "results.csv"), encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["method"], int(row["block"]))
            if key in out:
                continue
            failed = row.get("row_type") == "block_failure"
            mean_ee = fnum(row.get("mean_pair_ee_bit_per_j"), 0.0)
            pred_mean = fnum(row.get("predicted_mean_pair_ee"), 0.0)
            eta_p = fnum(row.get("eta_ref"), float("nan"))
            eta_t = fnum(row.get("actual_eta_ref"), float("nan"))
            ok_p = bool(np.isfinite(eta_p))
            ok_t = bool(np.isfinite(eta_t))
            out[key] = dict(
                failed=failed, mean_ee_M=mean_ee / 1e6,
                min_ee_M=fnum(row.get("min_pair_ee_bit_per_j"), 0.0) / 1e6,
                pred_mean_ee_M=pred_mean / 1e6,
                eta_ref_pred_M=eta_p / 1e6, eta_ref_true_M=eta_t / 1e6,
                qos_ok=str(row.get("all_pairs_qos_met")).lower() in ("true", "1"),
                fallback=str(row.get("fallback")).lower() in ("true", "1"),
                viol_pred=bool(ok_p and mean_ee < (1 - delta) * eta_p),
                viol_true=bool(ok_t and mean_ee < (1 - delta) * eta_t),
                loss_true_pct=(100 * (1 - mean_ee / eta_t) if ok_t and eta_t > 0 else float("nan")),
                budget_margin_pred_M=((pred_mean - (1 - delta) * eta_p) / 1e6 if ok_p else float("nan")),
                budget_margin_true_M=((mean_ee - (1 - delta) * eta_t) / 1e6 if ok_t else float("nan")),
                total_power_W=fnum(row.get("predicted_total_electrical_power_w"), float("nan")),
                vlc_power_frac=fnum(row.get("vlc_power_fraction"), float("nan")),
                jain=fnum(row.get("jain_pair_ee"), float("nan")),
                solve_s=fnum(row.get("solve_time_s"), 0.0),
            )
    return out


def export_dataset():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows, loaded = [], {}
    for tag, dirname, seed_cn, delta, csi, refine in RUN_SPECS:
        path = os.path.join(RUNS, dirname)
        if not os.path.isfile(os.path.join(path, "results.csv")):
            print("skip  %-5s %s（结果尚未落盘）" % (tag, dirname))
            continue
        data = load_run(path, delta)
        loaded[tag] = dict(data=data, seed_cn=seed_cn, delta=delta, csi=csi, refine=refine)
        for (method, block), v in sorted(data.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            row = dict(run=tag, seed=seed_cn, delta=delta, csi=csi, refine=refine,
                       method=method, block=block)
            row.update(v)
            rows.append(row)
        print("loaded %-5s %s blocks=%d" % (tag, dirname, len({b for _m, b in data})))
    out = os.path.join(DATA, "dataset_blocks.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in FIELDS})
    print("wrote", out, len(rows), "rows")
    return loaded


def vals(loaded, tag, method, field="min_ee_M", include_failed=True):
    """逐块取值。文献基线等有失败块的方案，失败块在 include_failed 时记为 0。"""
    d = loaded[tag]["data"]
    out = []
    for (m, _b), v in sorted(d.items(), key=lambda kv: kv[0][1]):
        if m != method:
            continue
        if v["failed"]:
            if include_failed:
                out.append(0.0)
            continue
        x = v[field]
        if x is not None and np.isfinite(x):
            out.append(x)
    return np.array(out, float)


def fail_count(loaded, tag, method):
    d = loaded[tag]["data"]
    n = sum(1 for (m, _b), v in d.items() if m == method)
    f = sum(1 for (m, _b), v in d.items() if m == method and v["failed"])
    return f, n


def fig1_main(loaded):
    seeds = [t for t in ("R1p", "R2p") if t in loaded]
    groups = [
        ("reference", "Pure-EE 参考", True),
        ("maxmin", "本文方法", True),
        ("literature", "文献基线\n(含失败)", True),
        ("literature", "文献基线\n(仅成功)", False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    seed_colors = ["#4c72b0", "#dd8452"]
    for ax, field, title in zip(axes, ("min_ee_M", "mean_ee_M"),
                                ("(a) 最低通信对 EE", "(b) 平均逐链路 EE")):
        width = 0.2
        for i, (method, _lab, inc) in enumerate(groups):
            for j, tag in enumerate(seeds):
                v = vals(loaded, tag, method, field, include_failed=inc)
                if v.size == 0:
                    continue
                med = float(np.median(v))
                x = i + (j - 0.5) * width
                ax.bar(x, med, width,
                       yerr=[[med - np.percentile(v, 25)], [np.percentile(v, 75) - med]],
                       capsize=2.5, color=seed_colors[j], alpha=0.88,
                       error_kw=dict(lw=0.9, ecolor="#333"))
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[1] for g in groups], fontsize=9)
        ax.set_ylabel("Mbit/s/W")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3, ls="--")
    handles = [matplotlib.patches.Patch(color=seed_colors[0], label="设计种子"),
               matplotlib.patches.Patch(color=seed_colors[1], label="验证种子")]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle("图 1  三算法主对比（柱高为中位数，误差线 P25–P75；文献基线分两口径）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "fig1_main_comparison.png")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def fig2_box(loaded):
    seeds = [t for t in ("R1p", "R2p") if t in loaded]
    methods = ["reference", "maxmin", "literature"]
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.6 * len(seeds), 4.4), squeeze=False)
    for ax, tag in zip(axes[0], seeds):
        data = [vals(loaded, tag, m, "min_ee_M", include_failed=True) for m in methods]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("#111"); med.set_linewidth(1.6)
        f, n = fail_count(loaded, tag, "literature")
        ax.set_xticklabels([METHOD_CN[m] for m in methods])
        ax.set_ylabel("最低通信对 EE（Mbit/s/W）")
        ax.set_title("%s（n=%d）\n文献基线失败 %d 块记为 0"
                     % (loaded[tag]["seed_cn"], n, f), fontsize=10)
        ax.grid(axis="y", alpha=0.3, ls="--")
    fig.suptitle("图 2  逐块最低通信对 EE 分布（含失败块，失败记 0；未显示离群点）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(FIG, "fig2_box_min_ee.png")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def fig3_margin(loaded):
    seeds = [t for t in ("R1p", "R2p") if t in loaded]
    fig, axes = plt.subplots(2, len(seeds), figsize=(5.4 * len(seeds), 8.0), squeeze=False)
    for j, tag in enumerate(seeds):
        d = loaded[tag]["data"]
        ax = axes[0][j]
        for m, mk in (("reference", "o"), ("maxmin", "^")):
            pts = [(v["budget_margin_pred_M"], v["budget_margin_true_M"])
                   for (mm, _b), v in d.items()
                   if mm == m and not v["failed"] and np.isfinite(v["budget_margin_pred_M"])
                   and np.isfinite(v["budget_margin_true_M"])]
            if not pts:
                continue
            px, py = zip(*pts)
            ax.scatter(px, py, s=8, alpha=0.4, marker=mk, color=COLORS[m], label=METHOD_CN[m])
        ax.axhline(0, color="#333", lw=0.9, ls="--")
        ax.axvline(0, color="#333", lw=0.9, ls="--")
        ax.set_xlim(-25, 25)
        ax.set_ylim(-120, 120)
        ax.set_xlabel("预测预算余量 $\\hat{G}$（Mbit/s/W）")
        ax.set_ylabel("真实预算余量 $G$（Mbit/s/W）")
        ax.set_title("%s：边界附近散点（$|\\hat G|<25$ M）" % loaded[tag]["seed_cn"], fontsize=10)
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.25, ls="--")

        ax2 = axes[1][j]
        for m in ("reference", "maxmin"):
            g = np.array([v["budget_margin_true_M"] for (mm, _b), v in d.items()
                          if mm == m and not v["failed"]
                          and np.isfinite(v["budget_margin_true_M"])], float)
            if g.size < 10:
                continue
            gs = np.sort(g)
            ax2.plot(gs, np.arange(1, gs.size + 1) / gs.size, lw=1.8,
                     color=COLORS[m], label=METHOD_CN[m])
            viol = float(np.mean(g < 0))
            ax2.plot([0], [viol], "o", color=COLORS[m], ms=6)
            ax2.annotate("违约 %.1f%%" % (100 * viol), (0, viol), textcoords="offset points",
                         xytext=(8, -12), fontsize=8, color=COLORS[m])
        ax2.axvline(0, color="#333", lw=0.9, ls="--")
        ax2.set_xlim(-80, 80)
        ax2.set_xlabel("真实预算余量 $G$（Mbit/s/W）")
        ax2.set_ylabel("经验累积概率")
        ax2.set_title("%s：真实预算余量 ECDF" % loaded[tag]["seed_cn"], fontsize=10)
        ax2.grid(alpha=0.25, ls="--"); ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle("图 3  预测预算余量饱和于 0，而真实余量在 0 附近弥散——违约由此产生",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG, "fig3_csi_margin.png")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def fig4_delta(loaded):
    tags = [t for t in ("R4", "R1p", "R5") if t in loaded]
    if len(tags) < 3:
        print("skip fig4（缺少 δ 扫描 run）"); return
    deltas = [loaded[t]["delta"] for t in tags]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, field, title in zip(axes, ("min_ee_M", "mean_ee_M"),
                               ("(a) 最低通信对 EE", "(b) 平均逐链路 EE")):
        med = [np.median(vals(loaded, t, "maxmin", field)) for t in tags]
        allb = [vals(loaded, t, "maxmin", field) for t in tags]
        lo = [m - np.percentile(v, 25) for m, v in zip(med, allb)]
        hi = [np.percentile(v, 75) - m for m, v in zip(med, allb)]
        ax.errorbar(deltas, med, yerr=[lo, hi], fmt="o-", color=COLORS["maxmin"],
                    lw=1.9, ms=6, capsize=3, label="本文方法")
        ref = np.median(vals(loaded, "R1p", "reference", field))
        ax.axhline(ref, color=COLORS["reference"], ls="--", lw=1.5, label="Pure-EE 参考")
        ax.set_xlabel("平均 EE 损失预算 $\\delta$")
        ax.set_ylabel("Mbit/s/W"); ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=9)
    fig.suptitle("图 4  δ 敏感性（设计种子，误差线 P25–P75）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(FIG, "fig4_delta_sensitivity.png")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def fig5_csi(loaded):
    if "R6" not in loaded or "R1p" not in loaded:
        print("skip fig5（缺少 oracle run）"); return
    tags = ["R1p", "R6"]
    labels = ["估计 CSI", "Oracle CSI"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    specs = [("min_ee_M", "(a) 最低通信对 EE", "Mbit/s/W"),
             ("mean_ee_M", "(b) 平均逐链路 EE", "Mbit/s/W"),
             (None, "(c) 真实预算违约率", "%")]
    for ax, (field, title, ylab) in zip(axes, specs):
        if field is None:
            v = [100 * np.mean([x["viol_true"] for (m, _b), x in loaded[t]["data"].items()
                                if m == "maxmin" and not x["failed"]]) for t in tags]
        else:
            v = [float(np.median(vals(loaded, t, "maxmin", field))) for t in tags]
        ax.bar(labels, v, color=["#4c72b0", "#27ae60"], alpha=0.85, width=0.5)
        for i, y in enumerate(v):
            ax.text(i, y, ("%.1f" % y) if field else ("%.1f%%" % y),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3, ls="--")
    fig.suptitle("图 5  CSI 质量对本文方法的影响（设计种子）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(FIG, "fig5_csi_quality.png")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def main():
    loaded = export_dataset()
    fig1_main(loaded)
    fig2_box(loaded)
    fig3_margin(loaded)
    fig4_delta(loaded)
    fig5_csi(loaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
