"""Phase 0：可恢复性审计（只补预测侧记录，不改优化器、不碰已归档结果）。

回答三个问题，**分别回答、不混为一谈**：

1. 预测器是否看到了候选的最低 EE 改善？（`pred_min_ee_own` vs 执行动作的预测 min EE）
2. 预测器是否看到了候选的预算风险？（`margin_fixed` / `margin_paired` 的分布）
3. 如果预测器已经看到了改善，**是哪一道规则**拒绝了它？
   （记录候选在预测 min EE 排序中的位置，以及"取消预算门后是否会被选中"）

另外按外部复核要求，保留**逐轨迹**的样本（不是只存均值/标准差），
并用**同轨迹配对**的预算余量 `G^(b) = meanEE_c^(b) − (1−δ)·meanEE_ref^(b)`，
以保留候选与参考动作在同一信道实现下的相关性。

用法::

    D:\\anaconda\\envs\\pymarl\\python.exe -B -I phase0_recoverability.py --blocks 100

产出：``out_phase0/phase0.json``、``out_phase0/phase0.csv``、终端判决。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

from counterfactual_audit import (                                # noqa: E402
    Session, load_archived, _patched_stage_factory, _ORIG_STAGE,
    N, DRAWS, DELTA, Q_MIN, QOS, START_BLOCK,
)
import p3learn.allocation as A                                    # noqa: E402  必须在上面之后

RUN_DIR = str(HERE.parents[1] / "代码" / "开发_10车机制定型_v5_20261101" / "code" / "runs"
              / "v5_C_ref_sca_s_v5a")


def _denominator(cfg, powers, modes):
    return np.asarray(A.link_electrical_power_w(np.asarray(powers, float), list(modes), cfg), float)


def per_trajectory_ee(forecast, cfg, powers, links):
    """返回 (逐轨迹块均 EE, 逐链路均值 EE 的最小值, 逐轨迹块内最小 EE)。

    **口径必须与项目一致**：`min_pair_ee` 是"先对每链路在轨迹上取均值，再对链路取 min"
    （`allocation._metrics`）。所以主指标用 `link_mean.min()`；
    `min_b` 是"每条轨迹内的 min 再平均"，**只是离散度诊断**，两者不可混用。
    """
    samples = np.asarray(forecast["block_rate_samples_bps"], float)   # (draws, n)
    modes = [m for _i, _j, m in links]
    denom = _denominator(cfg, powers, modes)
    ee = samples / denom[None, :]                 # (draws, n)
    mean_b = ee.mean(axis=1)                      # 逐轨迹块均逐链路 EE
    link_mean = ee.mean(axis=0)                   # 逐链路均值
    min_b = ee.min(axis=1)                        # 逐轨迹块内最小（诊断）
    return mean_b, float(link_mean.min()), min_b


def qos_ok(forecast):
    """按项目口径（chance + inflation）判断该候选是否满足预测 QoS。"""
    rates = A.predicted_qos_rates(forecast, "chance", QOS["inflation"])
    return bool(np.min(rates) + 1e-9 >= Q_MIN), float(np.min(rates))


def run(blocks, limit, out_dir):
    executed, archived_candidates, _ = load_archived(RUN_DIR)
    session = Session()
    rows = []
    t0 = time.perf_counter()

    for step in range(blocks):
        k = START_BLOCK + step
        info = session.advance(k)
        if limit and step >= limit:
            continue

        sink = []
        A.weak_link_repair_stage = _patched_stage_factory(sink)

        reference = A.pure_ee_reference(
            N, session.cfg, info["candidates"], session.business, info["ports"],
            info["bank"].evaluate, belief_gains=info["gains"],
            power_solver="continuous", search="joint", joint_rounds=8,
            joint_max_neighbours=0, q_min_bps=Q_MIN, qos_metric="chance",
            qos_fallback_mean=True, chance_rounds=2, chance_hold_mean=False,
            chance_inflation=QOS["inflation"])
        if not reference.success:
            rows.append({"block": k, "error": f"reference infeasible: {reference.reason}"})
            continue

        diag = []
        decision = A.max_min_allocate(
            N, session.cfg, info["candidates"], session.business, info["ports"],
            info["bank"].evaluate, delta=DELTA, belief_gains=info["gains"],
            reference=reference, qos=QOS, power_solver="continuous", search="joint",
            joint_rounds=8, joint_max_neighbours=0, power_refine=True,
            refine_maxiter=240, refine_restarts=2, refine_solver="sca",
            weak_link_repair=True, repair_limit=3, repair_diagnostics=diag)
        if decision.action is None:
            rows.append({"block": k, "error": f"maxmin failed: {decision.reason}"})
            continue

        eta_ref = float(decision.eta_ref)
        # 参考动作的逐轨迹均 EE（与候选用同一批轨迹）
        ref_mean_b, _, _ = per_trajectory_ee(reference.forecast, session.cfg,
                                             reference.action.powers, reference.action.links)
        # 执行动作的逐轨迹
        exec_mean_b, exec_min_pred, exec_min_b = per_trajectory_ee(
            decision.forecast, session.cfg, decision.action.powers, decision.action.links)

        cands = []
        for item in sink:
            cand = item["action"]
            fc = info["bank"].evaluate(cand)
            mean_b, min_pred, min_b = per_trajectory_ee(fc, session.cfg, cand.powers, cand.links)
            ok, min_qos_rate = qos_ok(fc)
            margin_paired_b = mean_b - (1.0 - DELTA) * ref_mean_b
            cands.append({
                "stage": "?",
                "pred_min_own": min_pred,
                "pred_min_sd_own": float(min_b.std(ddof=1)),
                "pred_mean_own": float(mean_b.mean()),
                "pred_mean_sd_own": float(mean_b.std(ddof=1)),
                "pred_mean_own_mc_se": float(mean_b.std(ddof=1) / np.sqrt(len(mean_b))),
                "margin_fixed": float(mean_b.mean() - (1.0 - DELTA) * eta_ref),
                "margin_paired_mean": float(margin_paired_b.mean()),
                "margin_paired_p05": float(np.quantile(margin_paired_b, 0.05)),
                "qos_ok_pred": ok,
                "qos_min_rate_pred": min_qos_rate,
                "delta_min_vs_exec": float(min_pred - exec_min_pred),
                "guard_accepted": bool(item["guard"].get("accepted")),
                "guard_budget_violation": float(item["guard"].get("budget_violation") or 0.0),
                # 逐轨迹样本（保留尾部/相关性）
                "mean_ee_samples": [float(x) for x in mean_b],
                "min_ee_samples": [float(x) for x in min_b],
                "margin_paired_samples": [float(x) for x in margin_paired_b],
            })
        # 用原始 stage 名对齐（sink 顺序与 weak_link.json 的 repair 顺序一致）
        arch_cands = archived_candidates.get(k, [])
        for idx, entry in enumerate(cands):
            if idx < len(arch_cands):
                entry["stage"] = arch_cands[idx]["stage"]

        # 预测 min EE 排序（1 = 最好）
        order = sorted(range(len(cands)), key=lambda i: -cands[i]["pred_min_own"])
        for rank, idx in enumerate(order, start=1):
            cands[idx]["rank_by_pred_min"] = rank

        # 取消预算门后，是否会被选中（在预测 QoS 通过的候选里取预测 min EE 最大者）
        eligible = [i for i, c in enumerate(cands) if c["qos_ok_pred"]]
        would = max(eligible, key=lambda i: cands[i]["pred_min_own"]) if eligible else None
        for idx, entry in enumerate(cands):
            entry["would_select_if_no_budget_gate"] = bool(idx == would)

        rows.append({
            "block": k,
            "eta_ref_pred": eta_ref,
            "exec_pred_min": exec_min_pred,
            "exec_pred_mean": float(exec_mean_b.mean()),
            "exec_pred_min_sd": float(exec_min_b.std(ddof=1)),
            "ref_mean_pred": float(ref_mean_b.mean()),
            "ref_mean_sd": float(ref_mean_b.std(ddof=1)),
            "candidates": cands,
        })
        best = max(cands, key=lambda c: c["delta_min_vs_exec"]) if cands else None
        print(f"  k={k:3d} cands={len(cands)} "
              f"exec_min={exec_min_pred/1e6:7.2f}M "
              f"bestΔmin={(best['delta_min_vs_exec']/1e6 if best else 0):+8.2f}M "
              f"would_select={(best['would_select_if_no_budget_gate'] if best else False)}")

    A.weak_link_repair_stage = _ORIG_STAGE
    elapsed = time.perf_counter() - t0
    block23 = [r for r in rows if r.get("block") == 23]
    summary = summarise(rows, elapsed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "phase0.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "blocks": rows, "block23": block23},
                  handle, ensure_ascii=False, indent=1)
    report(summary, block23)
    return summary


def summarise(rows, elapsed):
    ok = [r for r in rows if "error" not in r]
    all_c = [c for r in ok for c in r["candidates"]]
    s = {"blocks": len(ok), "elapsed_s": elapsed, "candidates": len(all_c)}

    def arr(key):
        return np.array([c[key] for c in all_c], float)

    if not all_c:
        return s
    recog = arr("delta_min_vs_exec")
    # 用逐块比率重算"预测看到多大改善"（绝对值不可跨块比较）
    rel = []
    for r in ok:
        base = max(r["exec_pred_min"], 1e-12)
        for c in r["candidates"]:
            rel.append(100.0 * c["delta_min_vs_exec"] / base)
    rel = np.array(rel)
    s["pred_min_improvement"] = {
        "note": "候选自身预测 min EE 相对执行动作的差（%），即'预测器是否看到改善'",
        "n_candidates": int(rel.size),
        "n_positive": int(np.sum(recog > 0)),
        "pct_median": float(np.median(rel)),
        "pct_p90": float(np.percentile(rel, 90)),
        "pct_max": float(rel.max()),
        "frac_positive": float(np.mean(rel > 0)),
        "frac_gt_10pct": float(np.mean(rel > 10)),
    }
    # 固定参考：点估计余量（与现守卫同口径）
    mf = arr("margin_fixed")
    # 逐轨迹配对：取每个候选自身分布的下 5% 分位，作为"悲观侧余量"
    mp05 = arr("margin_paired_p05")
    s["budget_margin"] = {
        "note": "负值 = 预测超支；paired_p05 是逐轨迹分布的下 5% 分位",
        "fixed_p05_p50_p95": [float(np.percentile(mf, p)) for p in (5, 50, 95)],
        "paired_p05_of_p05_p50_p95": [float(np.percentile(mp05, p)) for p in (5, 50, 95)],
        "frac_fixed_negative": float(np.mean(mf < 0)),
        "frac_paired_p05_negative": float(np.mean(mp05 < 0)),
    }
    s["n_would_select_if_no_budget_gate"] = int(sum(
        1 for r in ok for c in r["candidates"] if c["would_select_if_no_budget_gate"]))
    s["n_guard_accepted"] = int(sum(1 for c in all_c if c["guard_accepted"]))
    s["n_qos_ok_pred"] = int(sum(1 for c in all_c if c["qos_ok_pred"]))
    return s


def report(s, block23):
    print("\n" + "=" * 78)
    print("Phase 0：可恢复性审计")
    print(f"  块 {s['blocks']}，候选 {s['candidates']}，耗时 {s['elapsed_s']:.1f}s")
    if not s["candidates"]:
        return
    imp = s["pred_min_improvement"]
    print(f"\n  ① 预测器看到多少 min EE 改善（候选自身预测 − 执行动作预测）")
    print(f"     中位 {imp['pct_median']:+.3f}%  p90 {imp['pct_p90']:+.2f}%  最大 {imp['pct_max']:+.2f}%")
    print(f"     为正的比例 {imp['frac_positive']:.1%}；超过 +10% 的比例 {imp['frac_gt_10pct']:.2%}")
    bm = s["budget_margin"]
    print(f"\n  ② 预测预算余量（负=预测超支）")
    print(f"     固定参考（点估计，=现守卫口径）  p05/p50/p95 = "
          f"{bm['fixed_p05_p50_p95'][0]/1e6:+.2f} / {bm['fixed_p05_p50_p95'][1]/1e6:+.2f} / "
          f"{bm['fixed_p05_p50_p95'][2]/1e6:+.2f} M   负值占比 {bm['frac_fixed_negative']:.1%}")
    print(f"     逐轨迹配对（取各自下 5% 分位）  p05/p50/p95 = "
          f"{bm['paired_p05_of_p05_p50_p95'][0]/1e6:+.2f} / {bm['paired_p05_of_p05_p50_p95'][1]/1e6:+.2f} / "
          f"{bm['paired_p05_of_p05_p50_p95'][2]/1e6:+.2f} M   负值占比 {bm['frac_paired_p05_negative']:.1%}")
    print(f"\n  ③ 取消预算门后会被选中的候选数：{s['n_would_select_if_no_budget_gate']}"
          f"（守卫实际采纳 {s['n_guard_accepted']}，预测 QoS 通过 {s['n_qos_ok_pred']}）")

    if block23:
        r = block23[0]
        print(f"\n  ④ Block 23 明细（执行动作预测 min = {r['exec_pred_min']/1e6:.3f} M）")
        for c in sorted(r["candidates"], key=lambda x: -x["pred_mean_own"]):
            print(f"     {c['stage']:14s} pred_min {c['pred_min_own']/1e6:8.3f} M"
                  f" (rank {c['rank_by_pred_min']})"
                  f"  Δmin {(c['pred_min_own']-r['exec_pred_min'])/1e6:+8.3f} M"
                  f"  margin_fix {c['margin_fixed']/1e6:+8.2f} M"
                  f"  margin_pair {c['margin_paired_mean']/1e6:+8.2f} M"
                  f"  qos {int(c['qos_ok_pred'])}  would_sel {int(c['would_select_if_no_budget_gate'])}")
    print("\n  判决标准（设计稿 §2）：")
    print("   pred_min_own(B) − exec ≤ +0.1%  → 判不可恢复，停止本方向")
    print("   ≥ +10% 且超支仅由 0.33% 引起  → 可恢复，进入 R1")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 0 可恢复性审计")
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(HERE / "out_phase0"))
    args = parser.parse_args(argv)
    run(args.blocks, args.limit, args.out)
    print("写出:", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
