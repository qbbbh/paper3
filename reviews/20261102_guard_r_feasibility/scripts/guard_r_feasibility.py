"""Guard-R：候选可行性恢复（离线预研）。

**问题**：拒绝一个"预算违约的功率向量"，是否曾经使我们错过一个
**可通过连续功率重优化恢复可行性**的更优通信配置？

**为什么这可能是新的**（Phase A 源码事实）：
`bottleneck_sca` 只约束 QoS 代理 + 功率箱（`bottleneck_power.py:280-288`），**没有平均 EE 预算项**；
而 `maxmin_power_refine` 只在**已被采纳的拓扑**上跑。所以 B/C 离散重构候选的功率是
"瓶颈最优但预算不可行"的点，现有实现**从未**在该配置下做过带预算约束的功率求解。

**方法**：对预登记子集（B/C 类、预测 QoS 通过、预测 min EE 增益 >10%、预测预算违约），
固定其通信对/方向/模式不变，调用 `maxmin_power_refine`（它只在**可行点**里保留最优，
约束正是同一个 `(1−δ)η_ref` 与预测 QoS），看能否既恢复预算又保留 min EE 增益。

**三条对照**（计算预算必须可比）：
1. `original`：归档的执行动作；
2. `same_topology`：把同等 refine 预算用于**原拓扑**（看"多算一点"本身有没有用）；
3. `guard_r`：把同等预算用于 B/C 候选的可行性恢复。

用法::

    python -B -I guard_r_feasibility.py --run <v5 run> --channel-seed 2026102401 ^
        --phase0 out_phase0/phase0.json --out guard_r_design.json
"""
from __future__ import annotations

import argparse
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

from counterfactual_audit import (                      # noqa: E402
    Session, load_archived, _patched_stage_factory, _ORIG_STAGE,
    truth_metrics, N, DELTA, Q_MIN, QOS, START_BLOCK,
)
import p3learn.allocation as A                          # noqa: E402

GAIN_THRESHOLD = 0.10      # 预登记：预测 min EE 相对增益 > 10%
MAXITER = 240
RESTARTS = 2


def predicted_metrics(forecast, cfg, powers, links, eta_ref):
    """与 `allocation._metrics` 同口径的预测指标。"""
    rate = np.asarray(forecast["mean_rate_bps"], float)
    modes = [m for _i, _j, m in links]
    denom = np.asarray(A.link_electrical_power_w(np.asarray(powers, float), modes, cfg), float)
    ee = rate / denom
    qos_rates = A.predicted_qos_rates(forecast, "chance", QOS["inflation"])
    return {
        "mean_ee": float(ee.mean()),
        "min_ee": float(ee.min()),
        "margin": float(ee.mean() - (1.0 - DELTA) * eta_ref),
        "qos_ok": bool(np.min(qos_rates) + 1e-9 >= Q_MIN),
    }


def prefilter(phase0_path):
    """预登记筛选：不看真值。"""
    with open(phase0_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    wanted = {}
    n_bc = 0
    for block in data["blocks"]:
        if "error" in block:
            continue
        for cand in block["candidates"]:
            stage = cand["stage"]
            if not (stage.startswith("B_") or stage.startswith("C_")):
                continue
            n_bc += 1
            base = max(block["exec_pred_min"], 1e-12)
            gain = (cand["pred_min_own"] - block["exec_pred_min"]) / base
            if (cand["qos_ok_pred"] and gain > GAIN_THRESHOLD
                    and cand["margin_fixed"] < 0.0):
                wanted.setdefault(block["block"], []).append(stage)
    return wanted, n_bc


def run(args):
    wanted, n_bc = prefilter(args.phase0)
    n_cand = sum(len(v) for v in wanted.values())
    print(f"预登记筛选（不看真值）：B/C 候选 {n_bc} 个 → 命中 {n_cand} 个，"
          f"分布 {len(wanted)} 块")
    if not wanted:
        print("没有候选命中预登记条件，本轮无结论。")
        return None

    executed, archived_candidates, _ = load_archived(args.run)
    session = Session(channel_seed=args.channel_seed)
    records = []
    t0 = time.perf_counter()

    for step in range(100):
        k = START_BLOCK + step
        info = session.advance(k)
        if k not in wanted:
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
        decision = A.max_min_allocate(
            N, session.cfg, info["candidates"], session.business, info["ports"],
            info["bank"].evaluate, delta=DELTA, belief_gains=info["gains"],
            reference=reference, qos=QOS, power_solver="continuous", search="joint",
            joint_rounds=8, joint_max_neighbours=0, power_refine=True,
            refine_maxiter=MAXITER, refine_restarts=RESTARTS, refine_solver="sca",
            weak_link_repair=True, repair_limit=3)
        if decision.action is None:
            continue
        eta_ref = float(decision.eta_ref)
        window = session.truth_window(info["t"])

        # ---- original ----
        orig_pred = predicted_metrics(decision.forecast, session.cfg,
                                      decision.action.powers, decision.action.links, eta_ref)
        orig_truth = truth_metrics(session, window, decision.action, eta_ref)

        # ---- same_topology：把同等 refine 预算用于原拓扑 ----
        st_action, st_fc, st_calls = A.maxmin_power_refine(
            decision.action, decision.forecast, session.cfg, info["bank"].evaluate,
            delta=DELTA, eta_ref=eta_ref, qos=QOS, maxiter=MAXITER, restarts=RESTARTS)
        st_pred = predicted_metrics(st_fc, session.cfg, st_action.powers, st_action.links, eta_ref)
        st_truth = truth_metrics(session, window, st_action, eta_ref)

        # ---- guard_r：对每个命中候选做可行性恢复 ----
        targets = wanted[k]
        arch = archived_candidates.get(k, [])
        for idx, item in enumerate(sink):
            stage = arch[idx]["stage"] if idx < len(arch) else "?"
            if stage not in targets:
                continue
            cand = item["action"]
            cand_fc = info["bank"].evaluate(cand)
            pre = predicted_metrics(cand_fc, session.cfg, cand.powers, cand.links, eta_ref)
            restored, r_fc, r_calls = A.maxmin_power_refine(
                cand, cand_fc, session.cfg, info["bank"].evaluate,
                delta=DELTA, eta_ref=eta_ref, qos=QOS, maxiter=MAXITER, restarts=RESTARTS)
            post = predicted_metrics(r_fc, session.cfg, restored.powers, restored.links, eta_ref)
            feasible = bool(post["margin"] >= 0.0 and post["qos_ok"])
            improved = bool(feasible and post["min_ee"] > orig_pred["min_ee"] + 1e-9)
            post_truth = truth_metrics(session, window, restored, eta_ref)
            records.append({
                "block": k, "stage": stage,
                "before": pre, "after": post,
                "restored_feasible": feasible, "improved_vs_original": improved,
                "action_changed": bool(not np.allclose(np.asarray(restored.powers, float),
                                                       np.asarray(cand.powers, float))),
                "refine_calls": int(r_calls),
                "truth": {"min_ee": post_truth["min_ee"], "mean_ee": post_truth["mean_ee"],
                          "qos_ok": post_truth["min_rate_bps"] >= Q_MIN,
                          "budget_ok_true": post_truth["budget_ok"]},
                "original": {"pred": orig_pred, "truth": orig_truth},
                "same_topology": {"pred": st_pred, "truth": st_truth, "calls": int(st_calls)},
            })
            print(f"  k={k:3d} {stage:14s} before(margin {pre['margin']/1e6:+8.2f}M, "
                  f"minEE {pre['min_ee']/1e6:7.2f}M) → after(margin {post['margin']/1e6:+8.2f}M, "
                  f"minEE {post['min_ee']/1e6:7.2f}M) feasible={feasible} improved={improved} "
                  f"calls={int(r_calls)}")

    A.weak_link_repair_stage = _ORIG_STAGE
    elapsed = time.perf_counter() - t0
    out = {"run": args.run, "channel_seed": args.channel_seed,
           "prefilter": {"bc_candidates": n_bc, "matched": n_cand, "blocks": len(wanted),
                         "rule": "B/C & pred qos ok & pred min-EE gain > 10% & predicted budget violation"},
           "elapsed_s": elapsed, "records": records}
    return out, orig_pred if records else None


def summarise(payload):
    recs = payload["records"]
    if not recs:
        return
    feas = [r for r in recs if r["restored_feasible"]]
    impr = [r for r in recs if r["improved_vs_original"]]
    print("\n" + "=" * 74)
    print("Guard-R 可行性恢复（离线）")
    print(f"  候选 {len(recs)}（{payload['prefilter']['blocks']} 块），"
          f"耗时 {payload['elapsed_s']:.0f}s")
    print(f"  恢复后可行：{len(feas)}/{len(recs)}；其中优于原执行动作：{len(impr)}")
    if feas:
        d = np.array([100.0 * (r["after"]["min_ee"] - r["original"]["pred"]["min_ee"])
                      / max(r["original"]["pred"]["min_ee"], 1e-12) for r in feas])
        print(f"  可行者的预测 min EE 增益：中位 {np.median(d):+.2f}%  >10% {int((d>10).sum())}")
    if impr:
        td = np.array([100.0 * (r["truth"]["min_ee"] - r["original"]["truth"]["min_ee"])
                       / max(r["original"]["truth"]["min_ee"], 1e-12) for r in impr])
        print(f"  其中真值 min EE 增益：中位 {np.median(td):+.2f}%  最大 {td.max():+.1f}%")
        print(f"  真值预算与 QoS 都通过：{sum(1 for r in impr if r['truth']['qos_ok'] and r['truth']['budget_ok_true'])}")
    # 对照：same_topology 是否也改进了
    st_imp = [r for r in recs
              if r["same_topology"]["pred"]["min_ee"] > r["original"]["pred"]["min_ee"] + 1e-9]
    st_feas = [r for r in recs if r["same_topology"]["pred"]["margin"] >= 0.0]
    print(f"  对照 same_topology：改变动作 {st_imp} 条；预测预算可行 {len(st_feas)}/{len(recs)}")
    print(f"  额外 refine 调用：guard_r 平均 {np.mean([r['refine_calls'] for r in recs]):.1f} 次，"
          f"same_topology 平均 {np.mean([r['same_topology']['calls'] for r in recs]):.1f} 次")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Guard-R 候选可行性恢复（离线预研）")
    parser.add_argument("--run", required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
    parser.add_argument("--phase0", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = run(args)
    if not result:
        return 1
    payload = result[0]
    summarise(payload)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
