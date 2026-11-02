"""Feasible-first 联合优化 + Guard-R 审计纠正。

**要回答的问题**（外部复核提出）：
> 预算约束从候选生成阶段就参与优化，是否比"先生成高瓶颈 EE 候选、再恢复预算"
> 更容易找到可靠且有实际收益的通信配置？

**对上一轮 Guard-R 的四处纠正**（`reviews/20261102_guard_r_feasibility/`）：
1. **控制组不是等预算**：上一轮 `same_topology` 每块只跑一次，Guard-R 每候选跑一次
   （1.73–1.80 倍）。本脚本对**每个候选**都跑一次等预算的原拓扑控制。
2. **控制组判据错误**：上一轮用"min EE 是否提高"代替"动作是否改变"。本脚本两样都记。
3. **真值预算口径混用**：同时记 `(1−δ)η_ref,pred` 与 `(1−δ)η_ref,true` 两个阈值。
4. **缺少可追溯记录**：保存每个 arm 的 links / modes / powers，以及求解器诊断
   （重启终止状态、可行点数、最接近可行的余量）。

**Phase B 的做法**：对预登记的 B/C 候选，比较四种组合
（初始化 × 求解器）：

| 初始化 | 含义 |
|---|---|
| `cand` | 候选自身的功率向量（上一轮 Guard-R 用的） |
| `mapped` | 把**当前可行的参考动作**功率映射到新配置（同 (tx,rx,mode) 直接搬，换模式按新边界裁剪） |

求解器：`nm`（默认）与 `sca`（`bottleneck_sca` 起点 + NM 收敛）。

用法::

    python -B -I feasible_first_joint.py --run <v5 run> --channel-seed 2026102401 ^
        --phase0 out_phase0/phase0.json --out ff_design.json
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

GAIN_THRESHOLD = 0.10
MAXITER = 240
RESTARTS = 2


def predicted_metrics(forecast, cfg, powers, links, eta_ref):
    rate = np.asarray(forecast["mean_rate_bps"], float)
    modes = [m for _i, _j, m in links]
    denom = np.asarray(A.link_electrical_power_w(np.asarray(powers, float), modes, cfg), float)
    ee = rate / denom
    qos_rates = A.predicted_qos_rates(forecast, "chance", QOS["inflation"])
    return {
        "mean_ee": float(ee.mean()),
        "min_ee": float(ee.min()),
        "margin_pred_ref": float(ee.mean() - (1.0 - DELTA) * eta_ref),
        "qos_ok": bool(np.min(qos_rates) + 1e-9 >= Q_MIN),
        "min_qos_rate": float(np.min(qos_rates)),
    }


def map_reference_powers(ref_action, cand_links, cfg):
    """把参考动作的功率映射到新配置：同 (tx,rx,mode) 直接搬，换模式按新边界裁剪。"""
    ref_map = {}
    for (i, j, m), p in zip(ref_action.links, np.asarray(ref_action.powers, float)):
        ref_map[(int(i), int(j), m)] = float(p)
    # 同发送端的功率（换接收端/换模式时保留发射链路初值）
    by_tx = {}
    for (i, j, m), p in ref_map.items():
        by_tx.setdefault(i, []).append(p)
    out = []
    for (i, j, m) in cand_links:
        i, j, m = int(i), int(j), str(m)
        lo = cfg.rf.power_min_w if m == "RF" else cfg.vlc.power_min_w
        hi = cfg.rf.power_max_w if m == "RF" else cfg.vlc.power_max_w
        if (i, j, m) in ref_map:
            p = ref_map[(i, j, m)]
        elif by_tx.get(i):
            p = float(np.mean(by_tx[i]))
        else:
            p = float(np.sqrt(lo * hi))
        out.append(float(np.clip(p, lo, hi)))
    return np.asarray(out, float)


def action_payload(action):
    return {
        "links": [[int(i), int(j), str(m)] for (i, j, m) in action.links],
        "powers": [float(x) for x in np.asarray(action.powers, float)],
    }


def prefilter(phase0_path):
    with open(phase0_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    wanted, n_bc = {}, 0
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
            if cand["qos_ok_pred"] and gain > GAIN_THRESHOLD and cand["margin_fixed"] < 0.0:
                wanted.setdefault(block["block"], []).append(stage)
    return wanted, n_bc


def solve_arm(action, forecast, cfg, evaluator, eta_ref, solver, belief_gains=None):
    diag = {}
    out, fc, calls = A.maxmin_power_refine(
        action, forecast, cfg, evaluator, delta=DELTA, eta_ref=eta_ref, qos=QOS,
        maxiter=MAXITER, restarts=RESTARTS, diagnostics=diag, solver=solver,
        belief_gains=belief_gains)
    changed = not np.allclose(np.asarray(out.powers, float),
                             np.asarray(action.powers, float), rtol=0, atol=1e-9)
    return out, fc, int(calls), diag, bool(changed)


def run(args):
    wanted, n_bc = prefilter(args.phase0)
    max_blocks = int(getattr(args, "max_blocks", 0) or 0)
    done_blocks = 0
    n_cand = sum(len(v) for v in wanted.values())
    print(f"预登记：B/C {n_bc} → 命中 {n_cand}（{len(wanted)} 块）")
    if not wanted:
        return None

    executed, archived_candidates, _ = load_archived(args.run)
    session = Session(channel_seed=args.channel_seed)
    records = []
    t0 = time.perf_counter()
    arms = [("cand", "nm"), ("cand", "sca"), ("mapped", "nm"), ("mapped", "sca")]

    for step in range(100):
        k = START_BLOCK + step
        info = session.advance(k)
        if k not in wanted:
            continue
        if max_blocks and done_blocks >= max_blocks:
            break
        done_blocks += 1

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
        orig_pred = predicted_metrics(decision.forecast, session.cfg,
                                      decision.action.powers, decision.action.links, eta_ref)
        orig_truth = truth_metrics(session, window, decision.action, eta_ref)
        ref_truth = truth_metrics(session, window, reference.action, eta_ref)
        eta_ref_true = ref_truth["mean_ee"]

        # ---- 等预算控制：对每个候选都跑一次原拓扑 ----
        ctrl = {}
        for solver in ("nm", "sca"):
            with_diag = {}
            out, fc, calls = A.maxmin_power_refine(
                decision.action, decision.forecast, session.cfg, info["bank"].evaluate,
                delta=DELTA, eta_ref=eta_ref, qos=QOS, maxiter=MAXITER,
                restarts=RESTARTS, diagnostics=with_diag, solver=solver,
                belief_gains=info["gains"])
            ctrl[solver] = {
                "pred": predicted_metrics(fc, session.cfg, out.powers, out.links, eta_ref),
                "calls": int(calls),
                "action_changed": bool(not np.allclose(
                    np.asarray(out.powers, float), np.asarray(decision.action.powers, float),
                    rtol=0, atol=1e-9)),
                "diagnostics": with_diag,
                "action": action_payload(out),
            }

        arch = archived_candidates.get(k, [])
        for idx, item in enumerate(sink):
            stage = arch[idx]["stage"] if idx < len(arch) else "?"
            if stage not in wanted[k]:
                continue
            cand = item["action"]
            cand_fc = info["bank"].evaluate(cand)
            before = predicted_metrics(cand_fc, session.cfg, cand.powers, cand.links, eta_ref)
            mapped_powers = map_reference_powers(reference.action, cand.links, session.cfg)
            mapped = A.Action(tuple(cand.links), mapped_powers, cand.ports)
            mapped_fc = info["bank"].evaluate(mapped)
            mapped_before = predicted_metrics(mapped_fc, session.cfg, mapped.powers,
                                              mapped.links, eta_ref)
            arm_results = {}
            for init_name, solver in arms:
                start, start_fc = ((cand, cand_fc) if init_name == "cand"
                                   else (mapped, mapped_fc))
                out, fc, calls, diag, changed = solve_arm(
                    start, start_fc, session.cfg, info["bank"].evaluate, eta_ref, solver,
                    belief_gains=info["gains"])
                post = predicted_metrics(fc, session.cfg, out.powers, out.links, eta_ref)
                t = truth_metrics(session, window, out, eta_ref)
                arm_results[f"{init_name}+{solver}"] = {
                    "pred": post,
                    "action_changed": changed,
                    "calls": calls,
                    "diagnostics": diag,
                    "action": action_payload(out),
                    "truth": {"min_ee": t["min_ee"], "mean_ee": t["mean_ee"],
                              "min_rate_bps": t["min_rate_bps"],
                              "qos_ok": t["min_rate_bps"] >= Q_MIN,
                              "budget_pred_ref": t["budget_ok"],
                              "budget_true_ref": bool(
                                  t["mean_ee"] + 1e-9 >= (1.0 - DELTA) * eta_ref_true)},
                }
            records.append({
                "block": k, "stage": stage,
                "candidate": {"before": before, "action": action_payload(cand)},
                "mapped": {"before": mapped_before, "action": action_payload(mapped)},
                "original": {"pred": orig_pred, "truth": orig_truth,
                             "action": action_payload(decision.action)},
                "eta_ref_pred": eta_ref, "eta_ref_true": eta_ref_true,
                "control_same_topology": ctrl,
                "arms": arm_results,
            })
            best = max(arm_results.items(),
                       key=lambda kv: kv[1]["pred"]["min_ee"] if kv[1]["pred"]["margin_pred_ref"] >= 0
                       else -1e30)
            print(f"  k={k:3d} {stage:14s} before {before['min_ee']/1e6:7.2f}M "
                  f"(m {before['margin_pred_ref']/1e6:+8.2f}M) | "
                  f"best({best[0]}) {best[1]['pred']['min_ee']/1e6:7.2f}M "
                  f"(m {best[1]['pred']['margin_pred_ref']/1e6:+8.2f}M) "
                  f"true {best[1]['truth']['min_ee']/1e6:7.2f}M budT={best[1]['truth']['budget_true_ref']}")

    A.weak_link_repair_stage = _ORIG_STAGE
    return {"run": args.run, "channel_seed": args.channel_seed,
            "prefilter": {"bc": n_bc, "matched": n_cand, "blocks": len(wanted)},
            "elapsed_s": time.perf_counter() - t0, "records": records}


def summarise(payload):
    R = payload["records"]
    print("\n" + "=" * 78)
    print(f"Feasible-first 联合优化（{len(R)} 候选，{payload['elapsed_s']:.0f}s）")
    print(f"  {'arm':<14}{'预测可行':>8}{'改动作':>7}{'预测minEE>原':>12}"
          f"{'真值预算(pred ref)':>18}{'真值预算(true ref)':>18}{'真值minEE>原':>12}")
    for arm in ("cand+nm", "cand+sca", "mapped+nm", "mapped+sca"):
        rows = [r["arms"][arm] for r in R if arm in r["arms"]]
        if not rows:
            continue
        feas = sum(1 for r, row in zip(R, rows) if row["pred"]["margin_pred_ref"] >= 0)
        changed = sum(1 for row in rows if row["action_changed"])
        better_pred = sum(1 for r, row in zip(R, rows)
                          if row["pred"]["min_ee"] > r["original"]["pred"]["min_ee"] + 1e-9)
        budp = sum(1 for row in rows if row["truth"]["budget_pred_ref"])
        budt = sum(1 for row in rows if row["truth"]["budget_true_ref"])
        better_true = sum(1 for r, row in zip(R, rows)
                          if row["truth"]["min_ee"] > r["original"]["truth"]["min_ee"] + 1e-9)
        print(f"  {arm:<14}{feas:>8}{changed:>7}{better_pred:>12}{budp:>18}{budt:>18}{better_true:>12}")
    print("\n  控制组（等预算、每候选一次）：")
    for solver in ("nm", "sca"):
        rows = [r["control_same_topology"][solver] for r in R]
        changed = sum(1 for row in rows if row["action_changed"])
        calls = sum(row["calls"] for row in rows)
        print(f"    same_topology+{solver}: 改动作 {changed}/{len(rows)}，总评价调用 {calls}")
    gr_calls = {a: sum(r["arms"][a]["calls"] for r in R if a in r["arms"])
                for a in ("cand+nm", "cand+sca", "mapped+nm", "mapped+sca")}
    print(f"    各 arm 总评价调用：{gr_calls}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Feasible-first 联合优化 + 等预算控制")
    parser.add_argument("--run", required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
    parser.add_argument("--phase0", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-blocks", type=int, default=0, dest="max_blocks",
                        help="只处理前 N 个命中块（冒烟）")
    args = parser.parse_args(argv)
    payload = run(args)
    if not payload:
        return 1
    summarise(payload)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
