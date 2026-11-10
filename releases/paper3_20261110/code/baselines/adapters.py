"""把两条基线接到冻结评价流：共同预测器、QoS 复验、AllocationResult。

不启用 δ，不调用本文公平搜索。失败/回退口径与 ``pure_ee_reference`` 对齐。
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any, Callable

import numpy as np

from baselines._path import METHOD_ABOAGYE, METHOD_ZAPPONE, ensure_frozen_on_path
from baselines.aboagye_matching_qt import aboagye_matching_qt
from baselines.zappone_mmee_fp import zappone_mmee_fp_fixed

ensure_frozen_on_path()

from p3learn.allocation import (  # noqa: E402
    Action, AllocationResult, CoverageFailure, ReferencePolicy, _domain,
    _failure, _forecast, _legal, _metrics, _optimize_action, _raise_targets,
    copy_action, inflate_low,
)


def _qos_rates(forecast, metric, chance_inflation):
    key = "low_rate_bps" if metric == "chance" else "mean_rate_bps"
    rates = np.asarray(forecast[key], float)
    if metric == "chance" and chance_inflation != 1.0:
        rates = inflate_low(rates, forecast["mean_rate_bps"], chance_inflation)
    return rates


def _ok_result(action, forecast, cfg, *, start, calls, method_id, audit: dict[str, Any],
               chance_fallback=False, q_min_bps=0.0, qos_metric="chance",
               chance_inflation=1.0):
    network, mean_pair, weakest = _metrics(action, forecast, cfg)
    elapsed = perf_counter() - start
    return AllocationResult(
        copy_action(action), forecast, float(mean_pair), float(network), float(weakest),
        reason=f"{method_id}_ok",
        mean_pair_ee=float(mean_pair),
        solve_time_s=float(elapsed), evaluator_calls=int(calls),
        reference_solve_time_s=float(elapsed), reference_evaluator_calls=int(calls),
        attempted_candidates=int(calls),
        fallback=bool(chance_fallback),
        policy_config={
            "method": method_id,
            "q_min_bps": float(q_min_bps),
            "qos_metric": qos_metric,
            "chance_inflation": float(chance_inflation),
            "chance_fallback": bool(chance_fallback),
            "delta": None,
            "audit": audit,
        },
    )


def _qos_fail(forecast, *, start, calls, method_id, audit, q_min_bps, qos_metric,
              chance_inflation, policy):
    failure = _failure(f"qos_not_found: {method_id}", start, calls, policy=policy)
    return replace(failure, policy_config={
        "method": method_id,
        "q_min_bps": float(q_min_bps),
        "qos_metric": qos_metric,
        "chance_inflation": float(chance_inflation),
        "audit": audit,
        "best_predicted_rates_bps": (None if forecast is None
                                     else np.asarray(forecast["mean_rate_bps"]).tolist()),
        "best_predicted_low_rates_bps": (None if forecast is None
                                         else np.asarray(forecast["low_rate_bps"]).tolist()),
    })


def _repair_qos(action, belief_gains, cfg, evaluator, *, q_min_bps, qos_metric,
                chance_inflation, chance_rounds, calls):
    """有限预算信念抬升 + 预测复验。返回 (action, forecast, calls, feasible, log)."""
    log = []
    forecast = _forecast(evaluator, action)
    calls += 1
    rates = _qos_rates(forecast, qos_metric, chance_inflation)
    if q_min_bps <= 0 or bool(np.all(rates >= q_min_bps * (1 - 1e-8))):
        return action, forecast, calls, True, log
    for repair in range(int(chance_rounds)):
        short = rates < q_min_bps * (1 - 1e-8)
        tgt = np.where(short, q_min_bps, 0.0)
        new_p = _raise_targets(action.links, belief_gains, action.ports, cfg,
                               action.powers, tgt if np.any(tgt) else q_min_bps)
        action = Action(action.links, new_p, action.ports)
        forecast = _forecast(evaluator, action)
        calls += 1
        rates = _qos_rates(forecast, qos_metric, chance_inflation)
        log.append(dict(repair=repair + 1,
                        shortfall=float(np.maximum(q_min_bps - rates, 0).sum())))
        if bool(np.all(rates >= q_min_bps * (1 - 1e-8))):
            return action, forecast, calls, True, log
    return action, forecast, calls, False, log


def solve_aboagye_matching_qt(
    n, cfg, candidates, business, ports, evaluator: Callable, *,
    belief_gains=None, policy=None,
    power_solver="dinkelbach", literature_nominal_w=None, matching_rounds=3,
    q_min_bps=0.0, qos_metric="chance", qos_fallback_mean=True,
    chance_rounds=2, chance_inflation=1.35,
):
    """Matching-QT 完整块求解：交替匹配后，用共同预测器复验 QoS。"""
    policy = policy or ReferencePolicy()
    start = perf_counter()
    calls = 0
    if belief_gains is None:
        return _failure("aboagye_missing_belief_gains", start, calls, policy=policy)
    masks, checked_ports = _domain(n, cfg, candidates, business, ports)
    nominal = (float(literature_nominal_w) if literature_nominal_w is not None
               else float(np.median(np.asarray(policy.powers_w, float))))
    solver_log: list[dict] = []
    try:
        links, powers, out_ports, minfo = aboagye_matching_qt(
            n, cfg, candidates, business, checked_ports, belief_gains,
            power_solver=power_solver, nominal_w=nominal,
            rounds=matching_rounds, q_min_bps=q_min_bps, solver_log=solver_log)
    except CoverageFailure as exc:
        return _failure(str(exc), start, calls, coverage=True, policy=policy)

    proposal = Action(links, np.asarray(powers, float), out_ports)
    try:
        action = _optimize_action(proposal, cfg, belief_gains, checked_ports, masks,
                                  power_solver=power_solver, solver_log=solver_log,
                                  q_min_bps=q_min_bps)
    except (ValueError, KeyError, IndexError) as exc:
        return _failure(f"aboagye_power_failure: {exc}", start, calls, policy=policy)

    try:
        action, forecast, calls, feasible, repair_log = _repair_qos(
            action, belief_gains, cfg, evaluator,
            q_min_bps=q_min_bps, qos_metric=qos_metric,
            chance_inflation=chance_inflation, chance_rounds=chance_rounds, calls=calls)
    except (ValueError, FloatingPointError) as exc:
        return _failure(f"aboagye_prediction_failure: {exc}", start, calls, policy=policy)

    chance_fallback = False
    if not feasible and qos_fallback_mean and qos_metric == "chance":
        mean_rates = np.asarray(forecast["mean_rate_bps"], float)
        if bool(np.all(mean_rates >= q_min_bps * (1 - 1e-8))):
            feasible = True
            chance_fallback = True

    audit = dict(matching=minfo.to_dict(), solver_log=solver_log, qos_repair=repair_log)
    if not feasible:
        return _qos_fail(forecast, start=start, calls=calls, method_id=METHOD_ABOAGYE,
                         audit=audit, q_min_bps=q_min_bps, qos_metric=qos_metric,
                         chance_inflation=chance_inflation, policy=policy)
    return _ok_result(action, forecast, cfg, start=start, calls=calls,
                      method_id=METHOD_ABOAGYE, audit=audit,
                      chance_fallback=chance_fallback, q_min_bps=q_min_bps,
                      qos_metric=qos_metric, chance_inflation=chance_inflation)


def solve_zappone_mmee_fp_fixed(
    n, cfg, candidates, business, ports, evaluator: Callable, *,
    belief_gains=None, reference_action: Action | None = None,
    policy=None, q_min_bps=0.0, qos_metric="chance",
    qos_fallback_mean=True, chance_rounds=2, chance_inflation=1.35,
    max_outer=20, max_sca=6,
):
    """固定参考配置的 Max-Min EE 分式功率。无 δ。"""
    policy = policy or ReferencePolicy()
    start = perf_counter()
    calls = 0
    if belief_gains is None:
        return _failure("zappone_missing_belief_gains", start, calls, policy=policy)
    if reference_action is None:
        return _failure("zappone_missing_reference_action", start, calls, policy=policy)
    try:
        _legal(reference_action, n, cfg, candidates, business)
    except ValueError as exc:
        return _failure(f"zappone_illegal_reference_action: {exc}", start, calls, policy=policy)

    try:
        ref_forecast = _forecast(evaluator, reference_action)
    except (ValueError, FloatingPointError) as exc:
        return _failure(f"zappone_reference_prediction_failure: {exc}", start, calls, policy=policy)
    calls += 1
    ref_rates = _qos_rates(ref_forecast, qos_metric, chance_inflation)
    ref_qos_ok = bool(np.all(ref_rates >= q_min_bps * (1 - 1e-8))) if q_min_bps > 0 else True
    _n, ref_mean, ref_min = _metrics(reference_action, ref_forecast, cfg)

    powers, fpinfo = zappone_mmee_fp_fixed(
        reference_action.links, belief_gains, reference_action.ports, cfg,
        p_start=np.asarray(reference_action.powers, float),
        q_min_bps=q_min_bps, max_outer=max_outer, max_sca=max_sca)
    action = Action(reference_action.links, np.asarray(powers, float), reference_action.ports)
    if q_min_bps > 0:
        action = Action(action.links,
                        _raise_targets(action.links, belief_gains, action.ports, cfg,
                                       action.powers, q_min_bps),
                        action.ports)

    try:
        action, forecast, calls, feasible, repair_log = _repair_qos(
            action, belief_gains, cfg, evaluator,
            q_min_bps=q_min_bps, qos_metric=qos_metric,
            chance_inflation=chance_inflation, chance_rounds=chance_rounds, calls=calls)
    except (ValueError, FloatingPointError) as exc:
        return _failure(f"zappone_prediction_failure: {exc}", start, calls, policy=policy)

    chance_fallback = False
    if not feasible and qos_fallback_mean and qos_metric == "chance":
        mean_rates = np.asarray(forecast["mean_rate_bps"], float)
        if bool(np.all(mean_rates >= q_min_bps * (1 - 1e-8))):
            feasible = True
            chance_fallback = True

    audit = dict(fp=fpinfo.to_dict(), qos_repair=repair_log,
                 reference_qos_ok=ref_qos_ok,
                 reference_mean_pair_ee=float(ref_mean),
                 reference_min_pair_ee=float(ref_min))
    # The initial reference is a known candidate of this fixed-configuration
    # power problem. Retain it if the proxy search loses feasibility or worsens
    # the actual conditional objective. This is not a delta/fairness shield.
    candidate_min = float(_metrics(action, forecast, cfg)[2])
    keep_reference = ref_qos_ok and (
        not feasible or chance_fallback or candidate_min < float(ref_min))
    audit.update(candidate_min_pair_ee=candidate_min,
                 retained_initial_action=bool(keep_reference))
    if keep_reference:
        action, forecast, feasible = copy_action(reference_action), ref_forecast, True
        chance_fallback = False
    if not feasible:
        return _qos_fail(forecast, start=start, calls=calls, method_id=METHOD_ZAPPONE,
                         audit=audit, q_min_bps=q_min_bps, qos_metric=qos_metric,
                         chance_inflation=chance_inflation, policy=policy)
    return _ok_result(action, forecast, cfg, start=start, calls=calls,
                      method_id=METHOD_ZAPPONE, audit=audit,
                      chance_fallback=chance_fallback, q_min_bps=q_min_bps,
                      qos_metric=qos_metric, chance_inflation=chance_inflation)
