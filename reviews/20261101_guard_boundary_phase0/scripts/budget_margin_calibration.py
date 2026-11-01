"""预算余量的概率校准与风险决策曲线。

两个问题：

1. **预测分布标定得准不准？** 用标准化残差 z = (g_true − ĝ)/s 检验：
   各族的 sd(z) 是否接近 1、区间覆盖率是否贴合名义水平。
   若成立，则"B/C 误差大"是**合法的物理不确定性**，不是建模漏项——这时该做概率决策，不该去压误差。

2. **把硬门换成机会约束，ρ 要多大、代价多少？** 对每个候选取 32 条配对轨迹的
   G^(b) = meanEE_c^(b) − (1−δ)·meanEE_ref^(b)，用"P(G≥0) ≥ 1−ρ"替代硬门，
   扫描 ρ 并统计每块会换成哪个动作、真值上收益与预算违约代价。

**近似声明**：ρ 扫描按"每块取预测 min EE 最大的合格候选"选动作，**没有复现算法里的
顺序接受**（accept 后 `best_min` 会更新，后续候选的比较基准随之变化）。
所以扫描结果是"**若允许 p 机会约束，可达到什么**"的上界式估计，不是精确的算法重放。

用法::

    D:\\anaconda\\envs\\pymarl\\python.exe -B -I budget_margin_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DELTA = 0.10

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def load(path):
    with open(HERE / path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def family(stage):
    return stage.split("_")[0]


def main():
    p0 = load("out_phase0/phase0.json")
    tr = load("out_routeB/summary.json")

    truth = {}
    for block in tr["blocks"]:
        if block.get("mismatch") is not None or "error" in block:
            continue
        eta_t = block["eta_ref_true"]
        for cand in block["candidates"]:
            truth[(block["block"], cand["stage"])] = {
                "g_true": cand["truth_mean_ee"] - (1 - DELTA) * eta_t,
                "truth_dmin_pct": cand["d_min_vs_exec_pct"],
                "truth_dmean_pct": cand["d_mean_vs_exec_pct"],
                "truth_qos_ok": cand.get("truth_qos_ok"),
                "truth_budget_true": cand.get("truth_budget_ok_true_ref"),
                "eta_ref_true": eta_t,
                "exec_truth_min": block["executed"]["min_ee"],
                "exec_truth_mean": block["executed"]["mean_ee"],
            }

    recs = []
    for block in p0["blocks"]:
        if "error" in block:
            continue
        for cand in block["candidates"]:
            key = (block["block"], cand["stage"])
            if key not in truth:
                continue
            samples = np.asarray(cand["margin_paired_samples"], float)
            g_hat = float(samples.mean())
            s = float(samples.std(ddof=1))
            t = truth[key]
            recs.append({
                "block": block["block"], "stage": cand["stage"], "fam": family(cand["stage"]),
                "g_hat": g_hat, "s": s,
                "p_ok": float(np.mean(samples >= 0.0)),
                "g_true": t["g_true"], "eta_ref_true": t["eta_ref_true"],
                "truth_dmin_pct": t["truth_dmin_pct"],
                "truth_dmean_pct": t["truth_dmean_pct"],
                "truth_qos_ok": t["truth_qos_ok"],
                "truth_budget_true": t["truth_budget_true"],
                "pred_min_own": cand["pred_min_own"],
                "exec_pred_min": block["exec_pred_min"],
                "qos_ok_pred": cand["qos_ok_pred"],
                "eta_ref_pred": block["eta_ref_pred"],
                "exec_truth_min": t["exec_truth_min"],
                "exec_truth_mean": t["exec_truth_mean"],
            })
    print(f"对齐 {len(recs)} 个候选（预测分布 + 真值）\n")

    # ---------------------------------------------------------------- ① 标定
    print("=== ① 标准化残差 z = (g_true − ĝ)/s 的标定 ===")
    z = np.array([(r["g_true"] - r["g_hat"]) / max(r["s"], 1e-12) for r in recs])
    fams = np.array([r["fam"] for r in recs])
    print(f"  全体：mean(z) = {z.mean():+.3f}   sd(z) = {z.std(ddof=1):.3f}   "
          f"（理想 N(0,1) 为 0 与 1）")
    print(f"  {'族':<4}{'n':>5}{'mean(z)':>10}{'sd(z)':>9}{'|z|>1.96':>11}")
    for f in ("A", "B", "C", "S2"):
        m = fams == f
        if m.sum():
            print(f"  {f:<4}{int(m.sum()):>5}{z[m].mean():>+10.3f}{z[m].std(ddof=1):>9.3f}"
                  f"{100*np.mean(np.abs(z[m]) > 1.96):>10.1f}%")

    print("\n  区间覆盖率（名义 vs 实际）：")
    print(f"  {'名义':>7}{'q':>7}{'实际':>9}{'缺口(pp)':>10}")
    for nominal, q in ((0.683, 1.0), (0.799, 1.28), (0.90, 1.645), (0.95, 1.96), (0.99, 2.576)):
        actual = float(np.mean(np.abs(z) <= q))
        print(f"  {nominal:>7.3f}{q:>7.3f}{actual:>9.3f}{100*(actual-nominal):>+10.1f}")

    # ---------------------------------------------------------------- ② ρ 扫描
    print("\n=== ② 机会约束 ρ 扫描（P(G≥0) ≥ 1−ρ 替代硬门）===")
    print("  说明：每块取『预测 min EE 最大且满足该 ρ』的候选；未复现顺序接受，属估计。")
    by_block = {}
    for r in recs:
        by_block.setdefault(r["block"], []).append(r)

    print(f"\n  {'ρ':>6}{'>1−ρ':>7}{'换块数':>8}{'真值Δmin中位':>13}{'为正':>7}"
          f"{'>10%':>6}{'真值预算违约':>12}{'真值均EE中位变化':>16}")
    for rho in (0.50, 0.5625, 0.60, 0.70, 0.80, 0.90, 1.00):
        need = 1.0 - rho
        changed = []
        for block, group in by_block.items():
            ok = [r for r in group if r["qos_ok_pred"] and r["p_ok"] >= need]
            if not ok:
                continue
            best = max(ok, key=lambda r: r["pred_min_own"])
            if best["pred_min_own"] > best["exec_pred_min"] + 1e-9:
                changed.append(best)
        if not changed:
            print(f"  {rho:>6.3f}{need:>7.3f}{0:>8}          —")
            continue
        d = np.array([r["truth_dmin_pct"] for r in changed])
        dm = np.array([r["truth_dmean_pct"] for r in changed])
        viol = float(np.mean([not (r["truth_qos_ok"] and r["truth_budget_true"]) for r in changed]))
        print(f"  {rho:>6.4f}{need:>7.3f}{len(changed):>8}{np.median(d):>+13.2f}"
              f"{np.mean(d>0):>6.1%}{int((d>10).sum()):>6}{viol:>11.1%}{np.median(dm):>+16.2f}")

    # ---------------------------------------------------------------- ③ Block 23
    print("\n=== ③ Block 23 在机会约束下需要多大的 ρ ===")
    b23 = [r for r in recs if r["block"] == 23 and r["stage"].startswith("B_")]
    for r in b23:
        print(f"  {r['stage']}: ĝ = {r['g_hat']/1e6:+.2f} M, s = {r['s']/1e6:.2f} M, "
              f"P(G≥0) = {r['p_ok']:.3f} ⇒ 需 1−ρ ≤ {r['p_ok']:.3f}，即 ρ ≥ {1-r['p_ok']:.3f}")
        print(f"    真值 g = {r['g_true']/1e6:+.2f} M，z = {(r['g_true']-r['g_hat'])/max(r['s'],1e-12):+.3f}")

    # ---------------------------------------------------------------- ④ 修正口径
    print("\n=== ④ '取消预算门' 口径修正（区分『会换动作』与『换了更好』）===")
    buckets = {"gt+0.01%": 0, "近似不变": 0, "lt-0.01%": 0}
    for block, group in by_block.items():
        ok = [r for r in group if r["qos_ok_pred"]]
        if not ok:
            continue
        best = max(ok, key=lambda r: r["pred_min_own"])
        rel = 100.0 * (best["pred_min_own"] - best["exec_pred_min"]) / max(best["exec_pred_min"], 1e-12)
        if rel > 0.01:
            buckets["gt+0.01%"] += 1
        elif rel < -0.01:
            buckets["lt-0.01%"] += 1
        else:
            buckets["近似不变"] += 1
    print(f"  每块最高预测候选相对执行的预测 min EE 变化：{buckets}")
    pos = []
    for block, group in by_block.items():
        ok = [r for r in group if r["qos_ok_pred"]]
        if not ok:
            continue
        best = max(ok, key=lambda r: r["pred_min_own"])
        rel = 100.0 * (best["pred_min_own"] - best["exec_pred_min"]) / max(best["exec_pred_min"], 1e-12)
        if rel > 0.01:
            pos.append(best)
    if pos:
        d = np.array([r["truth_dmin_pct"] for r in pos])
        both = [r for r in pos if r["truth_qos_ok"] and r["truth_budget_true"]]
        print(f"  其中明确正向预测增益的 {len(pos)} 块：真值改善 {int((d>0).sum())}，"
              f">10% {int((d>10).sum())}，真值 QoS+预算都过 {len(both)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
