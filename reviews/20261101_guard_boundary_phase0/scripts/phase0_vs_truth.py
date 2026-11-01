"""Phase 0 判决：把"预测侧的候选改善"与"真值侧的实际改善"对齐。

回答的问题不是"候选有没有被拒"，而是：

1. **预测器看到的 min EE 改善，在真值上兑现了多少？**（中位数偏置、相关、分档兑现率）
2. **取消预算门会不会真的更好？**（预测会选中的候选，真值上是否更优）
3. **Block 23 在预测侧到底长什么样？**（它的自身预测 min EE 是否与执行动作几乎相同）

用法::

    D:\\anaconda\\envs\\pymarl\\python.exe -B -I phase0_vs_truth.py

输入：``out_phase0/phase0.json``（预测）与 ``out_routeB/summary.json``（真值）。
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DELTA_CONST = 0.10  # 与 counterfactual_audit / 论文一致
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def load(path):
    with open(HERE / path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def kendall(a, b):
    n = len(a)
    if n < 3:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            conc += int((da > 0) == (db > 0))
            disc += int((da > 0) != (db > 0))
    return (conc - disc) / (conc + disc) if (conc + disc) else None


def main():
    phase0 = load("out_phase0/phase0.json")
    truth = load("out_routeB/summary.json")

    pred = {}
    for block in phase0["blocks"]:
        if "error" in block:
            continue
        for cand in block["candidates"]:
            pred[(block["block"], cand["stage"])] = {
                "pred_dmin": cand["delta_min_vs_exec"],
                "pred_min_own": cand["pred_min_own"],
                "exec_pred_min": block["exec_pred_min"],
                "margin_fixed": cand["margin_fixed"],
                "margin_paired_p05": cand["margin_paired_p05"],
                "qos_ok_pred": cand["qos_ok_pred"],
                "would_select": cand["would_select_if_no_budget_gate"],
                "guard_accepted": cand["guard_accepted"],
            }
    tr = {}
    for block in truth["blocks"]:
        if block.get("mismatch") is not None or "error" in block:
            continue
        for cand in block["candidates"]:
            tr[(block["block"], cand["stage"])] = {
                "truth_mean_ee": cand["truth_mean_ee"],
                "eta_ref_true": block["eta_ref_true"],
                "truth_dmin_pct": cand["d_min_vs_exec_pct"],
                "truth_dmean_pct": cand["d_mean_vs_exec_pct"],
                "truth_min_ee": cand["truth_min_ee"],
                "exec_truth_min": block["executed"]["min_ee"],
                "truth_qos_ok": cand.get("truth_qos_ok"),
                "truth_budget_true": cand.get("truth_budget_ok_true_ref"),
            }

    eta_by_block = {b["block"]: b["eta_ref_pred"] for b in phase0["blocks"] if "error" not in b}
    keys = sorted(set(pred) & set(tr))
    print(f"预测记录 {len(pred)} 条，真值记录 {len(tr)} 条，可对齐 {len(keys)} 条")
    if not keys:
        return 1

    rows = []
    for key in keys:
        p, t = pred[key], tr[key]
        base = max(p["exec_pred_min"], 1e-12)
        eta_ref = eta_by_block.get(key[0], float("nan"))
        rows.append({
            "block": key[0], "stage": key[1],
            "pred_min_own": p["pred_min_own"],
            "exec_pred_min": p["exec_pred_min"],
            "eta_ref": eta_ref,
            "pred_dmin_pct": 100.0 * p["pred_dmin"] / base,
            "truth_dmin_pct": t["truth_dmin_pct"],
            "truth_dmean_pct": t["truth_dmean_pct"],
            "margin_fixed": p["margin_fixed"],
            "margin_paired_p05": p["margin_paired_p05"],
            "qos_ok_pred": p["qos_ok_pred"],
            "would_select": p["would_select"],
            "truth_qos_ok": t["truth_qos_ok"],
            "truth_budget_true": t["truth_budget_true"],
            "g_true": t["truth_mean_ee"] - (1.0 - DELTA_CONST) * t["eta_ref_true"],
            "eta_ref_true": t["eta_ref_true"],
        })

    pd_ = np.array([r["pred_dmin_pct"] for r in rows])
    td_ = np.array([r["truth_dmin_pct"] for r in rows])
    print("\n=== ① 预测的 min EE 改善 vs 真值改善 ===")
    print(f"  预测 Δmin%：中位 {np.median(pd_):+.2f}%  p90 {np.percentile(pd_,90):+.1f}%  最大 {pd_.max():+.1f}%")
    print(f"  真值 Δmin%：中位 {np.median(td_):+.2f}%  p90 {np.percentile(td_,90):+.2f}%  最大 {td_.max():+.1f}%")
    print(f"  **中位偏置（预测−真值）：{np.median(pd_) - np.median(td_):+.2f} 个百分点**")
    print(f"  预测为正的比例 {np.mean(pd_>0):.1%}，其中真值也为正的比例 "
          f"{np.mean(td_[pd_>0]>0):.1%}")

    print("\n  分档兑现率（预测超过阈值 T 的候选里，真值也超过 T 的比例）：")
    for thr in (1, 5, 10, 50):
        mask = pd_ > thr
        if mask.sum():
            print(f"    T={thr:>3}%  n={int(mask.sum()):3d}  真值兑现 "
                  f"{np.mean(td_[mask] > thr):.1%}（真值中位 {np.median(td_[mask]):+.2f}%）")

    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)
    taus = [kendall([c["pred_dmin_pct"] for c in group],
                    [c["truth_dmin_pct"] for c in group])
            for group in by_block.values()]
    taus = [t for t in taus if t is not None]
    if taus:
        print(f"\n  候选内「预测 Δmin 排序 vs 真值 Δmin 排序」τ：均值 {np.mean(taus):+.3f}"
              f"，正相关 {sum(1 for t in taus if t > 0)}/{len(taus)} 块")

    print("\n=== ② 取消预算门是否真的更好 ===")
    sel = [r for r in rows if r["would_select"]]
    if sel:
        ts = np.array([r["truth_dmin_pct"] for r in sel])
        print(f"  取消预算门会选中的候选：{len(sel)} 个")
        print(f"    真值 Δmin：中位 {np.median(ts):+.2f}%  为正 {np.mean(ts>0):.1%}  "
              f">10% {int((ts>10).sum())}  >50% {int((ts>50).sum())}")
        okb = [r for r in sel if r["truth_qos_ok"] and r["truth_budget_true"]]
        print(f"    真值 QoS 与真值参考预算都通过：{len(okb)} 个"
              f"（真值 Δmin 中位 {np.median([r['truth_dmin_pct'] for r in okb]):+.2f}%）"
              if okb else "    真值 QoS 与真值参考预算都通过：0 个")

    print("\n=== ②c 预算余量的预测误差（**定带宽的依据**）===")
    e_abs = np.array([r["g_true"] - r["margin_fixed"] for r in rows])
    eta = np.array([r["eta_ref"] for r in rows])
    rel_e = e_abs / np.maximum(eta, 1e-12)
    sd_pct = 100 * rel_e.std(ddof=1)
    print(f"  e = g_true − ĝ：中位 {np.median(e_abs)/1e6:+.2f} M  "
          f"均值 {e_abs.mean()/1e6:+.2f} M  sd {e_abs.std(ddof=1)/1e6:.2f} M")
    print(f"  以 η_ref 归一：中位 {100*np.median(rel_e):+.3f}%  sd {sd_pct:.3f}%  "
          f"p90|e| {100*np.percentile(abs(rel_e),90):.3f}%")
    for fam in ("A", "B", "C", "S2"):
        idx = [i for i, r in enumerate(rows) if r["stage"].startswith(fam)]
        if idx:
            print(f"    {fam:3s} n={len(idx):3d}  sd {100*rel_e[idx].std(ddof=1):6.3f}%  "
                  f"|e|中位 {100*np.median(abs(rel_e[idx])):6.3f}%")
    print(f"  Block 23 的超支 0.333% η_ref ⇒ 相当于 {0.333/sd_pct:.2f} 个 sd")

    print("\n=== ②b 边界带扫描（只放行 |预测超支| ≤ ε·η_ref 的候选）===")
    print("     每行：在该带宽下，各块按预测 min EE 选出的边界候选的真值兑现情况")
    by_block_all = {}
    for r in rows:
        by_block_all.setdefault(r["block"], []).append(r)
    for eps in (0.0025, 0.005, 0.01, 0.02, 0.05):
        picked = []
        for block, group in by_block_all.items():
            eta = group[0]["eta_ref"]
            if not np.isfinite(eta) or eta <= 0:
                continue
            band = [r for r in group
                    if r["qos_ok_pred"]
                    and -eps * eta <= r["margin_fixed"] < 0.0
                    and r["pred_min_own"] > r["exec_pred_min"]]
            if not band:
                continue
            best = max(band, key=lambda r: r["pred_min_own"])
            picked.append(best)
        if not picked:
            print(f"     ε={eps*100:4.2f}%  无候选进入边界带")
            continue
        t = np.array([r["truth_dmin_pct"] for r in picked])
        okb = [r for r in picked if r["truth_qos_ok"] and r["truth_budget_true"]]
        print(f"     ε={eps*100:4.2f}%  n={len(picked):3d}  真值Δmin 中位 {np.median(t):+7.2f}%"
              f"  为正 {np.mean(t>0):5.1%}  >10% {int((t>10).sum()):2d}  >50% {int((t>50).sum()):2d}"
              f"  | 真值三条件也过 {len(okb):2d} 个"
              + (f"（真值Δmin 中位 {np.median([r['truth_dmin_pct'] for r in okb]):+.2f}%）" if okb else ""))

    print("\n=== ③ Block 23 ===")
    b23 = [r for r in rows if r["block"] == 23]
    for r in sorted(b23, key=lambda x: -x["pred_dmin_pct"]):
        print(f"  {r['stage']:14s} predΔmin {r['pred_dmin_pct']:+8.2f}%  "
              f"truthΔmin {r['truth_dmin_pct']:+9.2f}%  "
              f"margin_fix {r['margin_fixed']/1e6:+7.2f} M  "
              f"would_select {int(r['would_select'])}")

    # ---- 判决 ----
    print("\n" + "=" * 74)
    print("Phase 0 判决输入（对照设计稿 §2 的标准）")
    cond_a = np.median(np.abs(pd_)) <= 0.1
    print(f"  标准 A（预测几乎看不到改善，|中位| ≤ 0.1%）：{'成立' if cond_a else '不成立'}"
          f"（实际中位 {np.median(pd_):+.2f}%）")
    high = pd_ > 10
    realised = np.mean(td_[high] > 10) if high.sum() else float("nan")
    print(f"  标准 B（预测看到改善且兑现）：预测>10% 的 {int(high.sum())} 个中，"
          f"真值也>10% 的占 {realised:.1%}")
    if sel:
        print(f"  标准 C（取消预算门更优）：真值为正 {np.mean(ts>0):.1%}，>10% 仅 {int((ts>10).sum())} 个")
    print("\n  结论口径：与'预测误差不可约'不同——这里要判断的是"
          "**预测看到的是否真实**，而不是**预测是否看到**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
