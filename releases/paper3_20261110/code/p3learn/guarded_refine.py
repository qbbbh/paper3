"""Structured power proposals checked against the complete conditional forecast.

The existing allocator remains the incumbent. The additional search changes
only powers on its chosen links and ports. It never reads execution outcomes.
An accepted proposal must satisfy the original predicted mean-EE budget and
calibrated service requirement and improve the predicted minimum link EE.
Keeping an incumbent that came from mean fallback does NOT certify feasibility.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from p3base.energy import link_electrical_power_w
from p3learn.bottleneck_power import bottleneck_sca


VERSION = "guarded-bottleneck-sca-v1"


def improve_fair_power(action, forecast, cfg, evaluator, *, delta, eta_ref,
                       qos=None, belief_gains=None):
    """Return (action, forecast, diagnostics), without changing physical inputs.

    Proposals use only posterior mean gains. Log-power interpolation explores
    a fixed set of steps towards the proposal; feasibility need not be monotone
    along that path. Every evaluated point is independently checked using the
    original trajectory distribution. Bounds/constraints are never relaxed.
    """
    from p3learn.allocation import (
        Action, _forecast, _raise_targets, copy_action, predicted_qos_rates,
    )

    start = perf_counter()
    if not 0 <= float(delta) <= 1 or not np.isfinite(eta_ref):
        raise ValueError("invalid delta or reference EE")
    modes = [m for _, _, m in action.links]
    lo = np.array([getattr(cfg, m.lower()).power_min_w for m in modes])
    hi = np.array([getattr(cfg, m.lower()).power_max_w for m in modes])
    p0 = np.asarray(action.powers, float).copy()
    if p0.shape != lo.shape or not np.all(np.isfinite(p0)):
        raise ValueError("invalid incumbent powers")
    if np.any(p0 < lo) or np.any(p0 > hi):
        raise ValueError("incumbent powers outside the physical domain")
    qos = qos or {}
    q = float(qos.get("q_min_bps", 0.0))
    metric = qos.get("metric", "mean")
    inflation = float(qos.get("inflation", 1.0))
    budget = (1.0 - float(delta)) * float(eta_ref)

    def metrics(p, f):
        rates = np.asarray(f["mean_rate_bps"], float)
        service = np.asarray(predicted_qos_rates(f, metric, inflation), float)
        if rates.shape != p0.shape or service.shape != p0.shape:
            raise ValueError("forecast shape does not match action")
        if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(service)):
            raise ValueError("nonfinite forecast")
        ee = rates / link_electrical_power_w(p, modes, cfg)
        mean_ee, min_ee = float(ee.mean()), float(ee.min())
        return dict(mean_ee=mean_ee, min_ee=min_ee,
                    min_service_bps=float(service.min()),
                    budget_ok=bool(mean_ee >= budget),
                    service_ok=bool(q <= 0 or np.all(service >= q)),
                    feasible=bool(mean_ee >= budget and
                                  (q <= 0 or np.all(service >= q))))

    original = metrics(p0, forecast)
    best_action, best_forecast, best_metrics = copy_action(action), dict(forecast), original
    evaluations = []
    errors = []
    calls = 0
    seen = {p0.tobytes()}

    def consider(p, label):
        nonlocal best_action, best_forecast, best_metrics, calls
        p = np.asarray(p, float)
        if p.shape != p0.shape or not np.all(np.isfinite(p)):
            errors.append(dict(stage=label, error="invalid proposed powers"))
            return None
        p = np.clip(p, lo, hi)
        key = p.tobytes()
        if key in seen:
            return None
        seen.add(key)
        proposed = Action(action.links, p.copy(), action.ports)
        calls += 1
        f = _forecast(evaluator, proposed)
        m = metrics(p, f)
        accepted = bool(m["feasible"] and
                        m["min_ee"] > best_metrics["min_ee"] +
                        1e-10 * max(1.0, abs(best_metrics["min_ee"])))
        evaluations.append(dict(proposal=label, accepted=accepted, **m))
        if accepted:
            best_action, best_forecast, best_metrics = copy_action(proposed), f, m
        return f

    solver_info = None
    if belief_gains is not None and len(p0):
        # Fixed budgets; no parameters are selected from observed execution EE.
        try:
            proposal, info = bottleneck_sca(
                action.links, belief_gains, action.ports, cfg, p_start=p0,
                q_min_bps=q, max_bisect=6, max_sca=3)
            solver_info = info.to_dict()
            proposal = np.clip(np.asarray(proposal, float), lo, hi)
            # Include the full proposal and deterministic intermediate steps.
            for alpha in (1.0, 0.75, 0.5, 0.25, 0.125):
                p = np.clip(np.exp((1.0-alpha)*np.log(p0) +
                                   alpha*np.log(proposal)), lo, hi)
                consider(p, f"sca_step_{alpha:g}")
            # Analytic service repair is a proposal only, never a certificate.
            if q > 0:
                repaired = _raise_targets(action.links, belief_gains, action.ports,
                                          cfg, proposal, q)
                for alpha in (1.0, 0.5, 0.25):
                    p = np.clip(np.exp((1-alpha)*np.log(p0) +
                                       alpha*np.log(np.clip(repaired, lo, hi))), lo, hi)
                    consider(p, f"service_repair_step_{alpha:g}")
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            # An unavailable extra proposal cannot discard the prior action.
            errors.append(dict(stage="structured_proposal", error=f"{type(exc).__name__}: {exc}"))

    diagnostic = dict(
        version=VERSION, evaluator_calls=calls, evaluations=evaluations,
        original=original, final=best_metrics, budget=budget, q_min_bps=q,
        metric=metric, inflation=inflation, solver=solver_info, errors=errors,
        accepted=bool(best_metrics["min_ee"] > original["min_ee"]),
        retained_incumbent=bool(np.array_equal(best_action.powers, p0)),
        final_original_constraints_passed=best_metrics["feasible"],
        elapsed_s=perf_counter()-start,
    )
    return best_action, best_forecast, diagnostic


def augment_result(result, cfg, evaluator, *, delta, qos, belief_gains,
                   diagnostics=None):
    """Apply the same post-stage in full runs and saved-input replay."""
    from p3learn.allocation import _metrics, predicted_qos_rates
    if not result.success:
        return result
    action, forecast, audit = improve_fair_power(
        result.action, result.forecast, cfg, evaluator, delta=delta,
        eta_ref=result.eta_ref, qos=qos, belief_gains=belief_gains)
    if diagnostics is not None:
        diagnostics["guarded_post_stage"] = audit
    network, mean, weakest = _metrics(action, forecast, cfg)
    policy = dict(result.policy_config or {})
    policy["guarded_post_stage"] = audit
    final_qos = predicted_qos_rates(forecast, (qos or {}).get("metric", "mean"),
                                    float((qos or {}).get("inflation", 1.0)))
    violation_count = int(np.sum(final_qos < float((qos or {}).get("q_min_bps", 0.0))))
    return replace(result, action=action, forecast=forecast,
                   network_ee=network, mean_pair_ee=mean, min_pair_ee=weakest,
                   qos_violation_links=violation_count,
                   solve_time_s=result.solve_time_s+audit["elapsed_s"],
                   evaluator_calls=result.evaluator_calls+audit["evaluator_calls"],
                   fallback=(result.fallback and not audit["accepted"]),
                   reason=("guarded_power_improved" if audit["accepted"] else result.reason),
                   policy_config=policy)
