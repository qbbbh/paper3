"""Guard-R Final Audit v1.

独立离线审计脚本：
  - 复现 v5 的 reference / maxmin / weak-link candidate 生成；
  - 逐块与归档 results.csv / weak_link.json 做复现校验；
  - 对每个预登记 B/C 候选运行：
      1) legacy Guard-R: 原 maxmin_power_refine；
      2) legacy same-topology: 同样求解预算、原拓扑；
      3) strict feasible-first: QoS -> mean-EE budget -> max-min；
      4) strict same-topology control: 同一 feasible-first 求解器与预算；
  - 所有最终动作在同一块真实信道上重新评价；
  - 分别报告预测参考预算与真值参考预算。

本脚本不修改 p3base/p3learn，不替换连续功率，不将局部搜索失败解释为数学不可行。
建议放在分析_三条主线_20261101\\scripts\\，与 counterfactual_audit.py 同目录。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from feasible_first_solver import SolverConfig, solve_feasible_first


GAIN_THRESHOLD = 0.10
ACTION_ATOL = 1e-6
ACTION_CHANGE_ATOL = 1e-9


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path) -> tuple[str, list[dict[str, str]]]:
    rows = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or "runs" in path.parts:
            continue
        digest = _sha256_file(path)
        if digest:
            rows.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
    h = hashlib.sha256()
    for row in rows:
        h.update(row["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(row["sha256"].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest(), rows


def _load_project_modules():
    """Lazy import: --help and unit-level use do not require the v5 tree."""
    import counterfactual_audit as CA
    import p3learn.allocation as A
    return CA, A


def _load_phase0(path: Path, gain_threshold: float):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    wanted: dict[int, list[str]] = {}
    n_bc = 0
    total_phase0 = 0
    for block in data.get("blocks", []):
        if "error" in block:
            continue
        total_phase0 += 1
        if "block" not in block or "exec_pred_min" not in block:
            raise ValueError("phase0 JSON lacks block/exec_pred_min fields")
        base = max(float(block["exec_pred_min"]), 1e-12)
        for cand in block.get("candidates", []):
            stage = str(cand.get("stage", "?"))
            if not (stage.startswith("B_") or stage.startswith("C_")):
                continue
            n_bc += 1
            gain = (float(cand["pred_min_own"]) - float(block["exec_pred_min"])) / base
            if (bool(cand.get("qos_ok_pred"))
                    and gain > gain_threshold
                    and float(cand.get("margin_fixed", 0.0)) < 0.0):
                wanted.setdefault(int(block["block"]), []).append(stage)
    return wanted, {
        "phase0_blocks": total_phase0,
        "bc_candidates": n_bc,
        "selected_candidates": sum(len(v) for v in wanted.values()),
        "selected_blocks": len(wanted),
        "rule": (
            "stage B_/C_ AND predicted chance-QoS passes AND predicted min-EE gain "
            f">{gain_threshold:.3%} AND fixed-reference predicted mean-EE budget margin < 0"
        ),
    }


def _action_payload(action) -> dict[str, Any]:
    return {
        "links": [[int(i), int(j), str(m)] for i, j, m in action.links],
        "powers": [float(x) for x in np.asarray(action.powers, dtype=float)],
    }


def _action_hash(action) -> str:
    payload = _action_payload(action)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _action_changed(a, b) -> bool:
    if list(a.links) != list(b.links):
        return True
    pa = np.asarray(a.powers, dtype=float)
    pb = np.asarray(b.powers, dtype=float)
    return pa.shape != pb.shape or not np.allclose(pa, pb, rtol=0.0, atol=ACTION_CHANGE_ATOL)


def _compare_action(action, archived_rows, atol=ACTION_ATOL):
    if archived_rows is None:
        return "missing archived action rows"
    rows = sorted(archived_rows, key=lambda x: int(x[0]))
    links = list(action.links)
    powers = np.asarray(action.powers, dtype=float)
    if len(rows) != len(links) or len(rows) != powers.size:
        return f"link count mismatch reconstructed={len(links)} archived={len(rows)}"
    for idx, row in enumerate(rows):
        _link_index, tx, rx, mode, power = row
        if (int(tx), int(rx), str(mode)) != tuple(links[idx]):
            return f"link[{idx}] identity mismatch reconstructed={links[idx]} archived={(tx, rx, mode)}"
        if abs(float(power) - float(powers[idx])) > atol:
            return (f"link[{idx}] power mismatch reconstructed={powers[idx]:.10f} "
                    f"archived={float(power):.10f}")
    return None


def _prefilter_stage_map(phase0_data: dict, wanted: dict[int, list[str]]) -> dict[int, dict[str, dict]]:
    out: dict[int, dict[str, dict]] = {}
    for block in phase0_data.get("blocks", []):
        k = int(block.get("block", -1))
        if k not in wanted:
            continue
        for cand in block.get("candidates", []):
            stage = str(cand.get("stage", "?"))
            if stage in wanted[k]:
                out.setdefault(k, {})[stage] = {
                    "pred_min_own": float(cand["pred_min_own"]),
                    "exec_pred_min": float(block["exec_pred_min"]),
                    "pred_gain_pct": 100.0 * (
                        float(cand["pred_min_own"]) - float(block["exec_pred_min"])
                    ) / max(float(block["exec_pred_min"]), 1e-12),
                    "margin_fixed": float(cand["margin_fixed"]),
                    "margin_paired_p05": cand.get("margin_paired_p05"),
                    "qos_ok_pred": bool(cand.get("qos_ok_pred")),
                }
    return out


def _map_reference_powers(ref_action, target_links, cfg):
    ref_map = {}
    by_tx: dict[int, list[float]] = {}
    for (i, j, mode), power in zip(ref_action.links, np.asarray(ref_action.powers, float)):
        ref_map[(int(i), int(j), str(mode))] = float(power)
        by_tx.setdefault(int(i), []).append(float(power))
    out = []
    for i, j, mode in target_links:
        i, j, mode = int(i), int(j), str(mode)
        mode_cfg = cfg.rf if mode == "RF" else cfg.vlc
        if (i, j, mode) in ref_map:
            value = ref_map[(i, j, mode)]
        elif by_tx.get(i):
            value = float(np.mean(by_tx[i]))
        else:
            value = math.sqrt(float(mode_cfg.power_min_w) * float(mode_cfg.power_max_w))
        out.append(float(np.clip(value, mode_cfg.power_min_w, mode_cfg.power_max_w)))
    return np.asarray(out, dtype=float)


def _power_bounds(links, cfg):
    lo, hi = [], []
    for _i, _j, mode in links:
        mode_cfg = cfg.rf if mode == "RF" else cfg.vlc
        lo.append(float(mode_cfg.power_min_w))
        hi.append(float(mode_cfg.power_max_w))
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _make_starts(action, reference_action, cfg):
    lo, hi = _power_bounds(action.links, cfg)
    starts = [
        np.asarray(action.powers, dtype=float),
        _map_reference_powers(reference_action, action.links, cfg),
        np.sqrt(lo * hi),
        lo.copy(),
        hi.copy(),
    ]
    unique = []
    seen = set()
    for p in starts:
        p = np.clip(np.asarray(p, dtype=float), lo, hi)
        key = tuple(np.round(p, 12))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return lo, hi, unique


def _predicted_metrics(A, action, forecast, cfg, q_min_bps, qos_metric, inflation, eta_ref, delta):
    rates = np.asarray(forecast["mean_rate_bps"], dtype=float)
    modes = [m for _i, _j, m in action.links]
    denom = np.asarray(A.link_electrical_power_w(np.asarray(action.powers, float), modes, cfg), float)
    ee = rates / denom
    qos_rates = np.asarray(A.predicted_qos_rates(forecast, qos_metric, inflation), dtype=float)
    budget_threshold = (1.0 - float(delta)) * eta_ref
    return {
        "mean_ee": float(np.mean(ee)),
        "min_ee": float(np.min(ee)),
        "per_link_ee": [float(x) for x in ee],
        "min_qos_rate_bps": float(np.min(qos_rates)) if qos_rates.size else None,
        "qos_ok": bool(qos_rates.size == 0 or np.min(qos_rates) + 1e-9 >= q_min_bps),
        "budget_margin_vs_pred_ref": float(np.mean(ee) - budget_threshold),
        "budget_ok_vs_pred_ref": bool(np.mean(ee) + 1e-9 >= budget_threshold),
    }


def _truth_payload(truth, eta_ref_pred, eta_ref_true, q_min_bps, baseline_min_ee, delta):
    mean_ee = float(truth["mean_ee"])
    min_ee = float(truth["min_ee"])
    threshold_pred = (1.0 - float(delta)) * float(eta_ref_pred)
    threshold_true = (1.0 - float(delta)) * float(eta_ref_true)
    gain_pct = 100.0 * (min_ee - baseline_min_ee) / max(baseline_min_ee, 1e-12)
    loss_vs_true_ref_pct = 100.0 * (1.0 - mean_ee / max(eta_ref_true, 1e-12))
    qos_ok = bool(float(truth["min_rate_bps"]) + 1e-9 >= q_min_bps)
    budget_pred_ok = bool(mean_ee + 1e-9 >= threshold_pred)
    budget_true_ok = bool(mean_ee + 1e-9 >= threshold_true)
    return {
        "mean_ee": mean_ee,
        "min_ee": min_ee,
        "min_rate_bps": float(truth["min_rate_bps"]),
        "network_ee": float(truth["network_ee"]),
        "total_tx_power_w": float(truth["total_power_w"]),
        "qos_ok": qos_ok,
        "budget_ok_vs_pred_ref": budget_pred_ok,
        "budget_ok_vs_true_ref": budget_true_ok,
        "both_budget_gates_ok": bool(budget_pred_ok and budget_true_ok),
        "all_true_gates_ok": bool(qos_ok and budget_pred_ok and budget_true_ok),
        "min_ee_gain_pct_vs_original": float(gain_pct),
        "mean_ee_loss_pct_vs_true_reference": float(loss_vs_true_ref_pct),
        "legitimate_improvement": bool(
            qos_ok and budget_pred_ok and budget_true_ok and min_ee > baseline_min_ee + 1e-9
        ),
    }


def _make_power_evaluator(A, template_action, evaluator, cfg, q_min_bps, qos_metric, inflation):
    links = tuple(template_action.links)
    modes = [m for _i, _j, m in links]

    def evaluate_powers(powers):
        candidate = A.Action(links, np.asarray(powers, dtype=float), template_action.ports)
        forecast = A._forecast(evaluator, candidate)
        rates = np.asarray(forecast["mean_rate_bps"], dtype=float)
        denom = np.asarray(A.link_electrical_power_w(np.asarray(powers, dtype=float), modes, cfg), dtype=float)
        ee = rates / denom
        qos_rates = np.asarray(A.predicted_qos_rates(forecast, qos_metric, inflation), dtype=float)
        return {"ee": ee, "qos_rates": qos_rates}

    return evaluate_powers


def _action_from_selected(A, template, selected):
    if not selected:
        return None
    return A.Action(tuple(template.links), np.asarray(selected["powers"], dtype=float), template.ports)


def _legacy_refine(A, action, forecast, cfg, evaluator, eta_ref, qos, gains,
                   maxiter, restarts, solver):
    diagnostics = {}
    out, out_fc, calls = A.maxmin_power_refine(
        action, forecast, cfg, evaluator,
        delta=qos["delta"], eta_ref=eta_ref, qos=qos["qos"],
        maxiter=maxiter, restarts=restarts, diagnostics=diagnostics,
        solver=solver, belief_gains=gains,
    )
    return out, out_fc, int(calls), diagnostics


def _summarise(records, elapsed):
    verified = [r for r in records if r.get("reconstruction_mismatch") is None and not r.get("error")]
    excluded = [r for r in records if r.get("reconstruction_mismatch") is not None or r.get("error")]
    summary: dict[str, Any] = {
        "candidate_records_total": len(records),
        "candidate_records_verified": len(verified),
        "candidate_records_excluded": len(excluded),
        "elapsed_s": float(elapsed),
        "excluded": [
            {"block": r.get("block"), "stage": r.get("stage"),
             "reason": r.get("reconstruction_mismatch") or r.get("error")}
            for r in excluded
        ],
    }
    arms = (
        "legacy_guard_r_candidate",
        "legacy_same_topology_control",
        "strict_feasible_first_candidate",
        "strict_same_topology_control",
    )
    arm_summary = {}
    for arm in arms:
        rows = [r[arm] for r in verified if r.get(arm) is not None]
        truth_rows = [x["truth"] for x in rows if x.get("truth")]
        pred_rows = [x["predicted"] for x in rows if x.get("predicted")]
        gains = [x["min_ee_gain_pct_vs_original"] for x in truth_rows]
        arm_summary[arm] = {
            "n": len(rows),
            "arm_errors": sum(bool(x.get("error")) for x in rows),
            "predicted_qos_pass": sum(bool(x["qos_ok"]) for x in pred_rows),
            "predicted_budget_pass": sum(bool(x["budget_ok_vs_pred_ref"]) for x in pred_rows),
            "truth_qos_pass": sum(bool(x["qos_ok"]) for x in truth_rows),
            "truth_budget_pred_ref_pass": sum(bool(x["budget_ok_vs_pred_ref"]) for x in truth_rows),
            "truth_budget_true_ref_pass": sum(bool(x["budget_ok_vs_true_ref"]) for x in truth_rows),
            "truth_all_gates_pass": sum(bool(x["all_true_gates_ok"]) for x in truth_rows),
            "legitimate_improvement_count": sum(bool(x["legitimate_improvement"]) for x in truth_rows),
            "truth_min_gain_pct_median": float(np.median(gains)) if gains else None,
            "truth_min_gain_pct_p10": float(np.percentile(gains, 10)) if gains else None,
            "truth_min_gain_pct_p90": float(np.percentile(gains, 90)) if gains else None,
            "truth_min_gain_positive_count": sum(g > 0 for g in gains),
            "truth_min_gain_gt_1pct_count": sum(g > 1 for g in gains),
            "truth_min_gain_gt_10pct_count": sum(g > 10 for g in gains),
            "mean_evaluator_calls": (
                float(np.mean([x.get("calls", 0) for x in rows])) if rows else None
            ),
        }
    summary["arms"] = arm_summary

    paired = []
    for r in verified:
        cand = r.get("strict_feasible_first_candidate")
        ctrl = r.get("strict_same_topology_control")
        if not cand or not ctrl or not cand.get("truth") or not ctrl.get("truth"):
            continue
        ct = cand["truth"]
        st = ctrl["truth"]
        paired.append({
            "block": r["block"], "stage": r["stage"],
            "candidate_minus_control_min_ee": float(ct["min_ee"] - st["min_ee"]),
            "candidate_minus_control_min_gain_pct": float(
                ct["min_ee_gain_pct_vs_original"] - st["min_ee_gain_pct_vs_original"]
            ),
            "candidate_all_true_gates_ok": bool(ct["all_true_gates_ok"]),
            "control_all_true_gates_ok": bool(st["all_true_gates_ok"]),
            "candidate_legitimate_improvement": bool(ct["legitimate_improvement"]),
        })
    diffs = [x["candidate_minus_control_min_gain_pct"] for x in paired]
    summary["strict_feasible_first_paired_vs_same_topology"] = {
        "n_pairs": len(paired),
        "candidate_better_count": sum(x > 0 for x in diffs),
        "candidate_not_worse_count": sum(x >= 0 for x in diffs),
        "median_difference_in_min_gain_pct": float(np.median(diffs)) if diffs else None,
        "pairs_both_true_feasible": sum(
            x["candidate_all_true_gates_ok"] and x["control_all_true_gates_ok"] for x in paired
        ),
        "candidate_legitimate_improvement_count": sum(
            x["candidate_legitimate_improvement"] for x in paired
        ),
        "details": paired,
    }
    summary["interpretation_guardrail"] = (
        "A found feasible point is a numerical witness. Failure to find a feasible point "
        "under multistart local SLSQP is NOT a proof of infeasibility or global nonexistence."
    )
    return summary


def _write_csv(path: Path, records: list[dict]):
    columns = [
        "block", "stage", "reconstruction_mismatch",
        "candidate_pred_gain_pct", "candidate_raw_truth_min_gain_pct",
        "legacy_guard_r_status", "legacy_guard_r_truth_min_gain_pct",
        "legacy_guard_r_truth_all_gates_ok", "legacy_guard_r_calls",
        "legacy_control_truth_min_gain_pct", "legacy_control_truth_all_gates_ok",
        "legacy_control_calls",
        "strict_ff_candidate_status", "strict_ff_candidate_truth_min_gain_pct",
        "strict_ff_candidate_truth_all_gates_ok", "strict_ff_candidate_calls",
        "strict_ff_control_status", "strict_ff_control_truth_min_gain_pct",
        "strict_ff_control_truth_all_gates_ok", "strict_ff_control_calls",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {
                "block": r.get("block"),
                "stage": r.get("stage"),
                "reconstruction_mismatch": r.get("reconstruction_mismatch"),
                "candidate_pred_gain_pct": r.get("prefilter", {}).get("pred_gain_pct"),
                "candidate_raw_truth_min_gain_pct": r.get("candidate_raw", {}).get("truth", {}).get("min_ee_gain_pct_vs_original"),
            }
            for key, prefix in (
                ("legacy_guard_r_candidate", "legacy_guard_r"),
                ("legacy_same_topology_control", "legacy_control"),
                ("strict_feasible_first_candidate", "strict_ff_candidate"),
                ("strict_same_topology_control", "strict_ff_control"),
            ):
                arm = r.get(key) or {}
                solver = arm.get("solver") or {}
                truth = arm.get("truth") or {}
                row[prefix + "_status"] = solver.get("status") if solver else arm.get("solver_name", "legacy")
                row[prefix + "_truth_min_gain_pct"] = truth.get("min_ee_gain_pct_vs_original")
                row[prefix + "_truth_all_gates_ok"] = truth.get("all_true_gates_ok")
                row[prefix + "_calls"] = arm.get("calls", solver.get("evaluation_calls", 0))
            writer.writerow(row)


def run(args):
    CA, A = _load_project_modules()
    run_dir = Path(args.run).resolve()
    phase0_path = Path(args.phase0).resolve()
    out_dir = Path(args.out).resolve()
    if not (run_dir / "results.csv").is_file():
        raise FileNotFoundError(f"run directory lacks results.csv: {run_dir}")
    if not phase0_path.is_file():
        raise FileNotFoundError(f"phase0 JSON not found: {phase0_path}")
    if int(args.phase0_seed) != int(args.channel_seed):
        raise ValueError(
            f"phase0 seed attestation ({args.phase0_seed}) differs from reconstruction seed "
            f"({args.channel_seed}); regenerate phase0 with the same channel seed"
        )

    phase0_data = json.loads(phase0_path.read_text(encoding="utf-8-sig"))
    wanted, prefilter_info = _load_phase0(phase0_path, GAIN_THRESHOLD)
    stage_info = _prefilter_stage_map(phase0_data, wanted)
    if not wanted:
        raise RuntimeError("No pre-registered B/C candidates matched the frozen Phase-0 rule.")

    executed, archived_candidates, archived_ee = CA.load_archived(str(run_dir))
    session = CA.Session(channel_seed=int(args.channel_seed))
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    selected_blocks_done = 0
    max_blocks = int(args.max_blocks or 0)
    max_candidates_per_block = int(args.max_candidates_per_block or 0)

    code_root = Path(CA.CODE)
    code_hash, source_manifest = _sha256_tree(code_root)
    input_hashes = {
        "phase0_json": _sha256_file(phase0_path),
        "run_results_csv": _sha256_file(run_dir / "results.csv"),
        "run_weak_link_json": _sha256_file(run_dir / "weak_link.json"),
        "run_summary_json": _sha256_file(run_dir / "summary.json"),
    }

    qos = dict(CA.QOS)
    solver_cfg = SolverConfig(
        delta=float(CA.DELTA),
        q_min_bps=float(CA.Q_MIN),
        qos_metric=str(qos["metric"]),
        qos_inflation=float(qos["inflation"]),
        max_evals=int(args.max_evals_per_arm),
        max_starts=int(args.starts),
        maxiter_qos=int(args.maxiter_qos),
        maxiter_mean=int(args.maxiter_mean),
        maxiter_maxmin=int(args.maxiter_maxmin),
    )

    print(f"run={run_dir}")
    print(f"phase0={phase0_path}")
    print(f"seed={args.channel_seed}; selected={prefilter_info['selected_candidates']} "
          f"candidates in {prefilter_info['selected_blocks']} blocks")
    print(f"strict feasible-first cap={solver_cfg.max_evals} unique evaluator calls per arm")

    original_stage_fn = A.weak_link_repair_stage
    try:
        for step in range(int(args.blocks)):
            k = int(CA.START_BLOCK) + step
            info = session.advance(k)
            if k not in wanted:
                continue
            if max_blocks and selected_blocks_done >= max_blocks:
                break
            selected_blocks_done += 1

            sink = []
            A.weak_link_repair_stage = CA._patched_stage_factory(sink)
            try:
                reference = A.pure_ee_reference(
                    CA.N, session.cfg, info["candidates"], session.business, info["ports"],
                    info["bank"].evaluate, belief_gains=info["gains"],
                    power_solver="continuous", search="joint", joint_rounds=8,
                    joint_max_neighbours=0, q_min_bps=CA.Q_MIN, qos_metric="chance",
                    qos_fallback_mean=True, chance_rounds=2, chance_hold_mean=False,
                    chance_inflation=CA.QOS["inflation"],
                )
                if not reference.success:
                    records.append({
                        "block": k, "error": f"reference infeasible: {reference.reason}",
                        "reconstruction_mismatch": None,
                    })
                    continue

                repair_diag = []
                refine_diag = {}
                decision = A.max_min_allocate(
                    CA.N, session.cfg, info["candidates"], session.business, info["ports"],
                    info["bank"].evaluate, delta=CA.DELTA, belief_gains=info["gains"],
                    reference=reference, qos=CA.QOS, power_solver="continuous", search="joint",
                    joint_rounds=8, joint_max_neighbours=0, power_refine=True,
                    refine_maxiter=240, refine_restarts=2, refine_solver="sca",
                    refine_diagnostics=refine_diag,
                    weak_link_repair=True, repair_limit=3,
                    repair_diagnostics=repair_diag,
                )
            finally:
                A.weak_link_repair_stage = original_stage_fn

            if decision.action is None:
                records.append({
                    "block": k, "error": f"maxmin failed: {decision.reason}",
                    "reconstruction_mismatch": None,
                })
                continue

            # ---- Exact reconstruction gate: do not draw conclusions from mismatched blocks.
            mismatch = _compare_action(reference.action, executed.get((k, "reference")))
            if mismatch is None:
                mismatch = _compare_action(decision.action, executed.get((k, "maxmin")))
            arch_cands = archived_candidates.get(k, [])
            if mismatch is None:
                if len(arch_cands) != len(sink):
                    mismatch = f"candidate count mismatch reconstructed={len(sink)} archived={len(arch_cands)}"
                else:
                    for idx, (got, want) in enumerate(zip(sink, arch_cands)):
                        gp = np.asarray(got["action"].powers, dtype=float)
                        wp = np.asarray(want["powers"], dtype=float)
                        if gp.shape != wp.shape or not np.allclose(gp, wp, rtol=0.0, atol=ACTION_ATOL):
                            mismatch = f"candidate[{idx}] {want.get('stage')} power mismatch"
                            break
                if mismatch is None:
                    archived_stages = {str(x.get("stage", "?")) for x in arch_cands}
                    missing_stages = sorted(set(wanted[k]) - archived_stages)
                    if missing_stages:
                        mismatch = f"Phase-0 selected stages absent from archive: {missing_stages}"
            ref_ee = archived_ee.get(k)
            if mismatch is None and ref_ee:
                # The archived values are the maxmin action's truth EE; this is a strict
                # cross-check of the true scoring denominator/aggregation.
                window = session.truth_window(info["t"])
                original_truth = CA.truth_metrics(session, window, decision.action, decision.eta_ref)
                ee_dev = max(
                    abs(original_truth["min_ee"] - ref_ee[0]) / max(abs(ref_ee[0]), 1e-12),
                    abs(original_truth["mean_ee"] - ref_ee[1]) / max(abs(ref_ee[1]), 1e-12),
                )
                if ee_dev > 1e-6:
                    mismatch = f"truth EE archive deviation {ee_dev:.3e}"
            else:
                window = session.truth_window(info["t"])
                original_truth = CA.truth_metrics(session, window, decision.action, decision.eta_ref)
                ee_dev = None

            if mismatch is not None:
                records.append({
                    "block": k, "reconstruction_mismatch": mismatch,
                    "archive_truth_ee_relative_deviation": ee_dev,
                    "candidate_count_reconstructed": len(sink),
                    "candidate_count_archived": len(arch_cands),
                    "error": None,
                })
                print(f"  block={k:3d} EXCLUDED: {mismatch}")
                continue

            eta_ref_pred = float(decision.eta_ref)
            reference_truth = CA.truth_metrics(session, window, reference.action, eta_ref_pred)
            eta_ref_true = float(reference_truth["mean_ee"])
            original_pred_fc = A._forecast(info["bank"].evaluate, decision.action)
            original_pred = _predicted_metrics(
                A, decision.action, original_pred_fc, session.cfg,
                CA.Q_MIN, CA.QOS["metric"], CA.QOS["inflation"], eta_ref_pred, CA.DELTA,
            )
            original_truth_payload = _truth_payload(
                original_truth, eta_ref_pred, eta_ref_true, CA.Q_MIN,
                float(original_truth["min_ee"]), CA.DELTA,
            )

            # Keep candidate-stage mapping from weak_link.json and Phase-0 pre-registration.
            selected_stages = set(wanted[k])
            candidate_items = []
            for idx, item in enumerate(sink):
                stage = str(arch_cands[idx]["stage"])
                if stage not in selected_stages:
                    continue
                candidate_items.append((stage, item["action"], item.get("guard", {})))
            if max_candidates_per_block:
                candidate_items = candidate_items[:max_candidates_per_block]

            for stage, candidate, guard_record in candidate_items:
                prefilter_row = stage_info.get(k, {}).get(stage)
                if prefilter_row is None:
                    records.append({
                        "block": k, "stage": stage,
                        "reconstruction_mismatch": "selected stage absent from phase0 candidate map",
                    })
                    continue

                candidate_fc = A._forecast(info["bank"].evaluate, candidate)
                candidate_pred = _predicted_metrics(
                    A, candidate, candidate_fc, session.cfg,
                    CA.Q_MIN, CA.QOS["metric"], CA.QOS["inflation"], eta_ref_pred, CA.DELTA,
                )
                candidate_truth_raw = CA.truth_metrics(session, window, candidate, eta_ref_pred)
                candidate_truth = _truth_payload(
                    candidate_truth_raw, eta_ref_pred, eta_ref_true, CA.Q_MIN,
                    float(original_truth["min_ee"]), CA.DELTA,
                )

                row: dict[str, Any] = {
                    "block": int(k),
                    "stage": stage,
                    "reconstruction_mismatch": None,
                    "prefilter": prefilter_row,
                    "original": {
                        "action": _action_payload(decision.action),
                        "predicted": original_pred,
                        "truth": original_truth_payload,
                        "fallback": bool(decision.fallback),
                        "reason": str(decision.reason),
                        "refine_diagnostics": refine_diag,
                        "repair_diagnostics": repair_diag,
                    },
                    "reference": {
                        "action": _action_payload(reference.action),
                        "eta_ref_pred": eta_ref_pred,
                        "eta_ref_true": eta_ref_true,
                        "truth_mean_ee": float(reference_truth["mean_ee"]),
                    },
                    "candidate_raw": {
                        "action": _action_payload(candidate),
                        "action_sha256": _action_hash(candidate),
                        "predicted": candidate_pred,
                        "truth": candidate_truth,
                        "guard_verdict": guard_record,
                    },
                }

                # Legacy Guard-R: same legacy local refinement on candidate and original topology.
                legacy_cand, legacy_cand_fc, legacy_cand_calls, legacy_cand_diag = _legacy_refine(
                    A, candidate, candidate_fc, session.cfg, info["bank"].evaluate,
                    eta_ref_pred, {"delta": CA.DELTA, "qos": CA.QOS},
                    info["gains"], args.legacy_maxiter, args.legacy_restarts, args.legacy_solver,
                )
                legacy_ctrl_diag = {}
                legacy_ctrl, legacy_ctrl_fc, legacy_ctrl_calls = A.maxmin_power_refine(
                    decision.action, original_pred_fc, session.cfg, info["bank"].evaluate,
                    delta=CA.DELTA, eta_ref=eta_ref_pred, qos=CA.QOS,
                    maxiter=args.legacy_maxiter, restarts=args.legacy_restarts,
                    diagnostics=legacy_ctrl_diag, solver=args.legacy_solver,
                    belief_gains=info["gains"],
                )
                legacy_cand_truth_raw = CA.truth_metrics(session, window, legacy_cand, eta_ref_pred)
                legacy_ctrl_truth_raw = CA.truth_metrics(session, window, legacy_ctrl, eta_ref_pred)
                row["legacy_guard_r_candidate"] = {
                    "solver_name": str(args.legacy_solver),
                    "calls": int(legacy_cand_calls),
                    "diagnostics": legacy_cand_diag,
                    "action_changed": _action_changed(candidate, legacy_cand),
                    "action": _action_payload(legacy_cand),
                    "predicted": _predicted_metrics(
                        A, legacy_cand, legacy_cand_fc, session.cfg,
                        CA.Q_MIN, CA.QOS["metric"], CA.QOS["inflation"], eta_ref_pred, CA.DELTA,
                    ),
                    "truth": _truth_payload(
                        legacy_cand_truth_raw, eta_ref_pred, eta_ref_true, CA.Q_MIN,
                        float(original_truth["min_ee"]), CA.DELTA,
                    ),
                }
                row["legacy_same_topology_control"] = {
                    "solver_name": str(args.legacy_solver),
                    "calls": int(legacy_ctrl_calls),
                    "diagnostics": legacy_ctrl_diag,
                    "action_changed": _action_changed(decision.action, legacy_ctrl),
                    "action": _action_payload(legacy_ctrl),
                    "predicted": _predicted_metrics(
                        A, legacy_ctrl, legacy_ctrl_fc, session.cfg,
                        CA.Q_MIN, CA.QOS["metric"], CA.QOS["inflation"], eta_ref_pred, CA.DELTA,
                    ),
                    "truth": _truth_payload(
                        legacy_ctrl_truth_raw, eta_ref_pred, eta_ref_true, CA.Q_MIN,
                        float(original_truth["min_ee"]), CA.DELTA,
                    ),
                }

                # Strict feasible-first candidate arm.
                for arm_name, template in (
                    ("strict_feasible_first_candidate", candidate),
                    ("strict_same_topology_control", decision.action),
                ):
                    lo, hi, starts = _make_starts(template, reference.action, session.cfg)
                    evaluate_powers = _make_power_evaluator(
                        A, template, info["bank"].evaluate, session.cfg,
                        CA.Q_MIN, CA.QOS["metric"], CA.QOS["inflation"],
                    )
                    result = solve_feasible_first(
                        evaluate_powers=evaluate_powers,
                        lo=lo, hi=hi, starts=starts,
                        eta_ref=eta_ref_pred,
                        config=solver_cfg,
                    )
                    out_action = _action_from_selected(A, template, result.get("selected"))
                    arm = {"solver": result}
                    if out_action is not None:
                        out_fc = A._forecast(info["bank"].evaluate, out_action)
                        out_truth_raw = CA.truth_metrics(session, window, out_action, eta_ref_pred)
                        arm.update({
                            "calls": int(result["evaluation_calls"]),
                            "action_changed": _action_changed(template, out_action),
                            "action": _action_payload(out_action),
                            "action_sha256": _action_hash(out_action),
                            "predicted": _predicted_metrics(
                                A, out_action, out_fc, session.cfg,
                                CA.Q_MIN, CA.QOS["metric"], CA.QOS["inflation"], eta_ref_pred, CA.DELTA,
                            ),
                            "truth": _truth_payload(
                                out_truth_raw, eta_ref_pred, eta_ref_true, CA.Q_MIN,
                                float(original_truth["min_ee"]), CA.DELTA,
                            ),
                        })
                    else:
                        arm.update({
                            "calls": int(result["evaluation_calls"]),
                            "action_changed": False,
                            "action": None,
                            "predicted": None,
                            "truth": None,
                        })
                    row[arm_name] = arm

                records.append(row)
                print(
                    f"  block={k:3d} {stage:14s} "
                    f"rawΔmin={candidate_truth['min_ee_gain_pct_vs_original']:+7.2f}% | "
                    f"legacyΔ={row['legacy_guard_r_candidate']['truth']['min_ee_gain_pct_vs_original']:+7.2f}% "
                    f"trueOK={int(row['legacy_guard_r_candidate']['truth']['all_true_gates_ok'])} | "
                    f"FF={row['strict_feasible_first_candidate']['solver']['status']} "
                    f"FFtrue={None if not row['strict_feasible_first_candidate'].get('truth') else round(row['strict_feasible_first_candidate']['truth']['min_ee_gain_pct_vs_original'],2)}%"
                )

    finally:
        A.weak_link_repair_stage = original_stage_fn

    elapsed = time.perf_counter() - t0
    summary = _summarise(records, elapsed)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": "guard-r-final-audit-v1",
        "role": args.role,
        "run_dir": str(run_dir),
        "phase0_path": str(phase0_path),
        "channel_seed": int(args.channel_seed),
        "phase0_seed_attested_by_cli": int(args.phase0_seed),
        "delta": float(CA.DELTA),
        "q_min_bps": float(CA.Q_MIN),
        "qos_metric": str(CA.QOS["metric"]),
        "qos_inflation": float(CA.QOS["inflation"]),
        "draws": int(CA.DRAWS),
        "start_block": int(CA.START_BLOCK),
        "blocks_scanned": int(args.blocks),
        "max_selected_blocks": int(args.max_blocks),
        "max_candidates_per_block": int(args.max_candidates_per_block),
        "legacy_solver": args.legacy_solver,
        "legacy_maxiter": int(args.legacy_maxiter),
        "legacy_restarts": int(args.legacy_restarts),
        "strict_solver_config": solver_cfg.__dict__,
        "prefilter": prefilter_info,
        "source_tree_sha256": code_hash,
        "source_tree_manifest": source_manifest,
        "input_sha256": input_hashes,
        "python": sys.version,
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit_script_sha256": _sha256_file(Path(__file__).resolve()),
        "solver_module_sha256": _sha256_file(Path(__file__).with_name("feasible_first_solver.py")),
    }
    payload = {"meta": meta, "summary": summary, "records": records}
    (out_dir / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(out_dir / "candidates.csv", records)
    (out_dir / "report.txt").write_text(_format_report(meta, summary), encoding="utf-8")
    print(_format_report(meta, summary))
    print(f"output: {out_dir}")
    return payload


def _format_report(meta, summary):
    lines = [
        "=" * 88,
        f"Guard-R Final Audit v1 | role={meta.get('role')} | seed={meta.get('channel_seed')}",
        f"delta={meta.get('delta')} | QoS={meta.get('q_min_bps')} bps / {meta.get('qos_metric')} "
        f"/ inflation={meta.get('qos_inflation')}",
        f"records verified={summary.get('candidate_records_verified')} / "
        f"{summary.get('candidate_records_total')} | excluded={summary.get('candidate_records_excluded')}",
        "",
        "Arm results (all counts are candidate-level; only reconstruction-verified rows):",
    ]
    for name, row in summary.get("arms", {}).items():
        lines.append(
            f"- {name}: n={row['n']}, true-all-gates={row['truth_all_gates_pass']}, "
            f"legitimate-improvement={row['legitimate_improvement_count']}, "
            f"median true min-EE gain={row['truth_min_gain_pct_median']}"
        )
    pair = summary.get("strict_feasible_first_paired_vs_same_topology", {})
    lines.extend([
        "",
        "Strict feasible-first candidate vs equal-budget same-topology control:",
        f"- pairs={pair.get('n_pairs')}; candidate better={pair.get('candidate_better_count')}; "
        f"median paired difference in min-EE gain (%)={pair.get('median_difference_in_min_gain_pct')}",
        "",
        "Interpretation restriction:",
        summary.get("interpretation_guardrail", ""),
        "=" * 88,
    ])
    return "\n".join(lines) + "\n"


def build_parser():
    p = argparse.ArgumentParser(description="Guard-R Final Audit v1 (read-only v5 integration)")
    p.add_argument("--run", required=True, help="v5 run directory containing results.csv and weak_link.json")
    p.add_argument("--phase0", required=True, help="phase0.json generated with the same channel seed")
    p.add_argument("--channel-seed", required=True, type=int)
    p.add_argument("--phase0-seed", required=True, type=int,
                   help="explicit attestation: must equal --channel-seed")
    p.add_argument("--role", choices=("design", "validation", "smoke"), default="design")
    p.add_argument("--out", required=True)
    p.add_argument("--blocks", type=int, default=100)
    p.add_argument("--max-blocks", type=int, default=0,
                   help="maximum selected blocks; 0 means all selected blocks")
    p.add_argument("--max-candidates-per-block", type=int, default=0,
                   help="maximum selected candidates per block; 0 means all")
    p.add_argument("--max-evals-per-arm", type=int, default=600)
    p.add_argument("--starts", type=int, default=5)
    p.add_argument("--maxiter-qos", type=int, default=60)
    p.add_argument("--maxiter-mean", type=int, default=80)
    p.add_argument("--maxiter-maxmin", type=int, default=100)
    p.add_argument("--legacy-solver", choices=("nm", "sca"), default="nm")
    p.add_argument("--legacy-maxiter", type=int, default=240)
    p.add_argument("--legacy-restarts", type=int, default=2)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.blocks < 1:
        raise SystemExit("--blocks must be >= 1")
    if args.max_blocks < 0 or args.max_candidates_per_block < 0:
        raise SystemExit("max block/candidate limits must be nonnegative")
    if args.max_evals_per_arm < 30:
        raise SystemExit("--max-evals-per-arm must be >= 30")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
