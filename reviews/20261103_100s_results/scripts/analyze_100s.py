"""100 s 主实验分析（v2）：三算法主表 + 配对移动块 Bootstrap + 预算损失分布 + CSI 诊断。

相对 v1 的修订（2026-11-02 外部复核）：

1. **`reference` 不再称为"上界"**，统一叫 `Efficiency-oriented Reference (Pure-EE)`
   —— 它没有被证明是全局效率上界。
2. **两种预算违约率严格分开定义**：
   - `V_pred`：真值均 EE < (1−δ)·η̂_ref ⇒ **预测参考基准下的真实兑现偏差**；
   - `V_true`：真值均 EE < (1−δ)·η_ref^true ⇒ **真实参考预算违约率**。
3. **预算损失按完整分布报**（p50/p75/p90/p95/p99 + 违约率），不只报中位或单一比例。
4. 主表补齐：10% 分位最低 EE、Jain 指数、求解成功率、平均电功率、VLC 占比、贴边比例、耗时。
5. **`literature` 不套用相对预算门**（它没有采用该约束）：只报相对 Reference 的 EE 损失、
   QoS、失败率与"成功条件/含失败惩罚"两种口径，避免成功样本选择偏差。

用法::

    python -B -I analyze_100s.py --run-dir paper3_100s/runs/R1_v5A_d010_100s --label R1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

LABEL = {
    "reference": "Efficiency-oriented Reference (Pure-EE)",
    "maxmin": "Proposed (v5-A)",
    "literature": "Literature (equiv-SINR + fractional prog.)",
    "random": "Random lower bound",
}
# 文献基线**没有采用**本文的相对 EE 预算门，不能按"违约"评价它
NO_BUDGET_GATE = {"literature", "random"}
BOUNDS = {"RF": (0.01, 20.0), "VLC": (0.1, 30.0)}


def fnum(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def load_blocks(run_dir):
    path = os.path.join(run_dir, "results.csv")
    blocks, links = {}, defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], int(row["block"]))
            if key not in blocks:
                blocks[key] = {
                    "failed": row.get("row_type") == "block_failure",
                    "min_ee": fnum(row.get("min_pair_ee_bit_per_j"), 0.0),
                    "mean_ee": fnum(row.get("mean_pair_ee_bit_per_j"), 0.0),
                    "network_ee": fnum(row.get("network_ee_bit_per_j"), 0.0),
                    "jain": fnum(row.get("jain_pair_ee")),
                    "qos_ok": str(row.get("all_pairs_qos_met")).lower() in ("true", "1"),
                    "fallback": str(row.get("fallback")).lower() in ("true", "1"),
                    "coverage_failure": str(row.get("coverage_failure")).lower() in ("true", "1"),
                    "vlc_rate_frac": fnum(row.get("vlc_rate_fraction")),
                    "vlc_power_frac": fnum(row.get("vlc_power_fraction")),
                    "total_power": fnum(row.get("predicted_total_electrical_power_w")),
                    "eta_ref_pred": fnum(row.get("eta_ref"), float("nan")),
                    "eta_ref_true": fnum(row.get("actual_eta_ref"), float("nan")),
                    "pred_mean_ee": fnum(row.get("predicted_mean_pair_ee"), 0.0),
                    "delta": fnum(row.get("delta"), 0.10),
                    "solve_s": fnum(row.get("solve_time_s"), 0.0),
                    "qos_metric": row.get("qos_metric") or "",
                }
            mode, power = row.get("mode"), fnum(row.get("power_w"))
            if mode in BOUNDS and power is not None:
                lo, hi = BOUNDS[mode]
                links[key].append((mode, power, abs(power - lo) <= 1e-9, abs(power - hi) <= 1e-9))
    return blocks, links


def moving_block_bootstrap(values, block_len=10, reps=4000, seed=20261102):
    x = np.asarray(values, float)
    n = x.size
    if n == 0:
        return None
    L = max(1, min(int(block_len), n))
    starts = np.arange(0, n - L + 1)
    need = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for r in range(reps):
        idx = rng.choice(starts, size=need, replace=True)
        means[r] = np.concatenate([x[s:s + L] for s in idx])[:n].mean()
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_method(blocks, links, method, delta, out):
    keys = sorted([k for k in blocks if k[0] == method], key=lambda k: k[1])
    # 参考解失败的块（eta_ref 缺失）无法定义预算，单列不计入
    keys = [k for k in keys if np.isfinite(blocks[k]["eta_ref_pred"])
            and np.isfinite(blocks[k]["eta_ref_true"])]
    if not keys:
        out(f"  {LABEL.get(method, method)}: 无数据")
        return None
    a = lambda f: np.array([blocks[k][f] for k in keys], float)          # noqa: E731
    n = len(keys)
    jain = a("jain")
    jain = jain[~np.isnan(jain)]
    eta_pred = a("eta_ref_pred")
    eta_true = a("eta_ref_true")
    mean_ee = a("mean_ee")
    dlt = a("delta")
    viol_pred = mean_ee < (1.0 - dlt) * eta_pred
    viol_true = mean_ee < (1.0 - dlt) * eta_true
    loss_true = 1.0 - mean_ee / np.where(eta_true > 0, eta_true, np.nan)
    at_lo = at_hi = tot = 0
    for k in keys:
        for _m, _p, lo, hi in links.get(k, []):
            tot += 1
            at_lo += int(lo)
            at_hi += int(hi)
    qos = np.array([blocks[k]["qos_ok"] for k in keys])
    out(f"  {LABEL.get(method, method)}   n={n}")
    out(f"    平均逐链路 EE  中位 {np.median(mean_ee)/1e6:8.2f} M   均值 {mean_ee.mean()/1e6:8.2f} M")
    out(f"    最低通信对 EE  中位 {np.median(a('min_ee'))/1e6:8.2f} M   "
        f"均值 {a('min_ee').mean()/1e6:8.2f} M   **10% 分位 {np.percentile(a('min_ee'),10)/1e6:8.2f} M**")
    out(f"    真实 QoS 满足率 {qos.mean():6.1%}   回退 {np.mean([blocks[k]['fallback'] for k in keys]):5.1%}"
        f"   覆盖失败 {np.mean([blocks[k]['coverage_failure'] for k in keys]):5.1%}"
        f"   求解成功 {1-np.mean([blocks[k]['coverage_failure'] for k in keys]):5.1%}")
    out(f"    平均电功率 中位 {np.median(a('total_power')):6.3f} W   "
        f"VLC 功率占比 {np.median(a('vlc_power_frac')):5.1%}   VLC 速率占比 {np.median(a('vlc_rate_frac')):5.1%}")
    out(f"    RF/VLC 贴边  下界 {at_lo}/{tot}（{at_lo/max(tot,1):4.1%}）  上界 {at_hi}/{tot}")
    out(f"    Jain 指数 中位 {np.median(jain):.4f}" if jain.size else "    Jain 指数 缺")
    out(f"    单块耗时 中位 {np.median(a('solve_s')):6.2f} s")
    if method in NO_BUDGET_GATE:
        out("    （该基线**未采用**相对 EE 预算门，故不计算预算违约；只报相对 Reference 的 EE 损失）")
        out(f"    预算损失分布（相对真实参考的真实违背率，仅作参考）"
            f" 中位 {100*np.nanmedian(loss_true):+7.3f}%")
    else:
        out(f"    **V_pred（预测参考基准下的真实兑现偏差）** = {viol_pred.mean():6.1%}")
        out(f"    **V_true（真实参考预算违约率）**            = {viol_true.mean():6.1%}")
        qs = 100.0 * np.nanpercentile(loss_true, [50, 75, 90, 95, 99])
        out(f"    预算损失分布（1 − η_true/(η_ref_true)，%）   p50 {qs[0]:+7.3f}%  p75 {qs[1]:+7.3f}%"
            f"  p90 {qs[2]:+7.3f}%  p95 {qs[3]:+7.3f}%  p99 {qs[4]:+7.3f}%")
    if any(blocks[k]["failed"] for k in keys):
        ok = [k for k in keys if not blocks[k]["failed"]]
        nf = n - len(ok)
        out(f"    含失败惩罚口径 n={n}（失败 {nf}）= 上表；仅成功块口径 n={len(ok)}：")
        if ok:
            mo = np.array([blocks[k]["min_ee"] for k in ok]) / 1e6
            eo = np.array([blocks[k]["mean_ee"] for k in ok]) / 1e6
            out(f"      最低 EE 中位 {np.median(mo):8.2f} M  10% 分位 {np.percentile(mo,10):8.2f} M  "
                f"平均 EE 中位 {np.median(eo):8.2f} M  QoS 通过 {np.mean([blocks[k]['qos_ok'] for k in ok]):6.1%}")
    return {"keys": keys, "min_ee": a("min_ee"), "mean_ee": mean_ee,
            "viol_pred": viol_pred, "viol_true": viol_true, "loss_true": loss_true,
            "qos": qos, "eta_ref_pred": eta_pred, "eta_ref_true": eta_true}


def paired(blocks, a_name, b_name, out):
    ka = {k[1]: k for k in blocks if k[0] == a_name}
    kb = {k[1]: k for k in blocks if k[0] == b_name}
    common = sorted(set(ka) & set(kb))
    if len(common) < 2:
        out(f"  共同块 {len(common)}，无法配对（{a_name} vs {b_name}）")
        return None
    ma = np.array([blocks[ka[i]]["min_ee"] for i in common])
    mb = np.array([blocks[kb[i]]["min_ee"] for i in common])
    ea = np.array([blocks[ka[i]]["mean_ee"] for i in common])
    eb = np.array([blocks[kb[i]]["mean_ee"] for i in common])
    d_min = ma - mb
    rel_min = 100.0 * d_min / np.maximum(mb, 1e-12)
    rel_mean = 100.0 * (ea - eb) / np.maximum(eb, 1e-12)
    out(f"  {LABEL.get(a_name,a_name)} vs {LABEL.get(b_name,b_name)}  （共同块 {len(common)}）")
    out(f"    最低 EE 绝对差 中位 {np.median(d_min)/1e6:+8.2f} M   均值 {d_min.mean()/1e6:+8.2f} M")
    out(f"    相对差 中位 {np.median(rel_min):+8.2f}%（参考值极小时会爆炸，均值 {rel_min.mean():+8.2f}% 不可引用）")
    out(f"    平均 EE 相对差 中位 {np.median(rel_mean):+8.2f}%   均值 {rel_mean.mean():+8.3f}%")
    out(f"    最低 EE 改善块 {int((d_min>0).sum())}/{len(common)}（>1% {int((rel_min>1).sum())}，"
        f">10% {int((rel_min>10).sum())}）")
    out("    移动块 Bootstrap 95% CI —— 最低 EE 绝对差（M），按块长度敏感性：")
    for L in (1, 5, 10, 20, 50):
        res = moving_block_bootstrap(d_min / 1e6, block_len=L)
        if res:
            m, lo, hi = res
            out(f"      L={L:<3d} 均值 {m:+8.3f} M  CI [{lo:+8.3f}, {hi:+8.3f}]"
                + ("   *显著" if (lo > 0 or hi < 0) else "   （含 0）"))
    res = moving_block_bootstrap(rel_mean, block_len=10)
    if res:
        m, lo, hi = res
        out(f"    相对差均值 L=10  {m:+8.3f}%  CI [{lo:+8.3f}%, {hi:+8.3f}%]")
    return {"common": common, "d_min": d_min, "rel_mean": rel_mean}


def csi_diagnostic(blocks, method, delta, out):
    keys = sorted([k for k in blocks if k[0] == method], key=lambda k: k[1])
    keys = [k for k in keys if np.isfinite(blocks[k]["eta_ref_pred"])
            and np.isfinite(blocks[k]["eta_ref_true"])]
    if len(keys) < 10:
        return
    eta_p = np.array([blocks[k]["eta_ref_pred"] for k in keys])
    eta_t = np.array([blocks[k]["eta_ref_true"] for k in keys])
    # 预测侧预算余量必须用**该动作自己的预测 EE**；若两边都用 mean_ee，
    # 相减时 mean_ee 会被抵消，只剩参考解误差，与动作无关
    # （曾因此得到两种方法完全相同的虚假统计）。
    dlt = np.array([blocks[k]["delta"] for k in keys])
    pred = np.array([blocks[k]["pred_mean_ee"] - (1 - blocks[k]["delta"]) * blocks[k]["eta_ref_pred"] for k in keys])
    true = np.array([blocks[k]["mean_ee"] - (1 - blocks[k]["delta"]) * blocks[k]["eta_ref_true"] for k in keys])
    err = true - pred
    z = err / np.where(eta_p > 0, eta_p, np.nan)
    out(f"  {LABEL.get(method,method)}：预测 vs 真值预算余量（n={len(keys)}）")
    out(f"    归一化误差 (真−预测)/η̂_ref：中位 {100*np.nanmedian(z):+7.3f}%  "
        f"sd {100*np.nanstd(z):6.3f}%  p90|·| {100*np.nanpercentile(np.abs(z),90):6.3f}%")
    out(f"    Pearson r = {np.corrcoef(pred, true)[0,1]:+.4f}")
    try:
        from scipy.stats import spearmanr
        out(f"    Spearman ρ = {float(spearmanr(pred, true).statistic):+.4f}")
    except Exception:
        pass
    viol = true < 0
    edges = [-np.inf, *np.nanquantile(np.abs(z), [0.2, 0.4, 0.6, 0.8]), np.inf]
    out("    按 |归一化误差| 分位看真实预算违约率：")
    for i in range(len(edges) - 1):
        m = (np.abs(z) >= edges[i]) & (np.abs(z) < edges[i + 1])
        if np.isfinite(edges[i + 1]) or i == len(edges) - 2:
            if m.sum():
                out(f"      第{i+1}档 n={int(m.sum()):4d}  违约率 {viol[m].mean():6.1%}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="100 s 主实验分析（v2）")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    blocks, links = load_blocks(args.run_dir)
    methods = sorted({k[0] for k in blocks})
    n_blocks = len({k[1] for k in blocks})
    out(f"=== {args.label} ===")
    out(f"目录 {args.run_dir}")
    out(f"方法 {methods} | 块 {n_blocks} | δ={args.delta}")
    with open(os.path.join(args.run_dir, "results.csv"), encoding="utf-8-sig", newline="") as h:
        first = next(csv.DictReader(h))
    out(f"source_sha256 {first.get('source_sha256')}")
    out(f"config_sha256 {first.get('config_sha256')} | mobility_sha256 {first.get('mobility_sha256')}")
    out(f"channel_seed {first.get('channel_seed')} | draws {first.get('draws')}")
    out("")
    out("--- 6.2 主表 ---")
    for m in methods:
        summarise_method(blocks, links, m, args.delta, out)
        out("")
    out("--- 配对比较（移动块 Bootstrap）---")
    pairs = [("maxmin", "reference"), ("maxmin", "literature"), ("literature", "reference")]
    for a, b in pairs:
        if a in methods and b in methods:
            paired(blocks, a, b, out)
            out("")
    out("--- 6.4 CSI 预算余量诊断 ---")
    for m in methods:
        if m not in NO_BUDGET_GATE:
            csi_diagnostic(blocks, m, args.delta, out)
            out("")
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
