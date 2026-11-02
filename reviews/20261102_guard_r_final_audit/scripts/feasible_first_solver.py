"""Lexicographic feasibility-first continuous-power solver used only by the Guard-R final audit.

This module is deliberately independent of p3base/p3learn.  The caller supplies an
evaluator that returns per-link EE and QoS rates for a power vector.

Stages:
  A. Find a QoS-feasible point by minimizing the maximum normalized QoS violation.
  B. Maximize mean per-link EE subject to QoS; this is a feasibility search for the
     frozen mean-EE budget, not a claim of global optimality.
  C. Starting from a found budget-feasible point, maximize minimum per-link EE
     subject to BOTH QoS and mean-EE budget constraints.

A failed multistart search is reported as "not found", never as a proof of
mathematical infeasibility.  All power variables are optimized continuously in
log space; no power grid is used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Any

import numpy as np
from scipy.optimize import minimize


class EvaluationBudgetExceeded(RuntimeError):
    """Raised internally when the per-arm evaluator-call cap is reached."""


@dataclass
class SolverConfig:
    delta: float = 0.10
    q_min_bps: float = 1.0e6
    qos_metric: str = "chance"
    qos_inflation: float = 1.35
    max_evals: int = 600
    max_starts: int = 5
    maxiter_qos: int = 60
    maxiter_mean: int = 80
    maxiter_maxmin: int = 100
    tol_budget_rel: float = 1.0e-8
    tol_qos_rel: float = 1.0e-8
    feasibility_buffer_rel: float = 1.0e-7
    ftol: float = 1.0e-9

    def validate(self) -> None:
        if not (0.0 <= self.delta < 1.0):
            raise ValueError("delta must be in [0, 1)")
        if self.q_min_bps < 0:
            raise ValueError("q_min_bps must be nonnegative")
        if self.qos_metric not in ("mean", "chance"):
            raise ValueError("qos_metric must be 'mean' or 'chance'")
        if self.qos_inflation <= 0:
            raise ValueError("qos_inflation must be positive")
        if self.max_evals < 30:
            raise ValueError("max_evals must be >= 30 for three-stage optimization")
        if self.max_starts < 1:
            raise ValueError("max_starts must be >= 1")


def _unique_starts(starts: Iterable[np.ndarray], lo: np.ndarray, hi: np.ndarray,
                   max_starts: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for raw in starts:
        p = np.clip(np.asarray(raw, dtype=float), lo, hi)
        if p.shape != lo.shape or not np.isfinite(p).all():
            continue
        key = tuple(np.round(p, 12))
        if key not in seen:
            seen.add(key)
            out.append(p)
        if len(out) >= max_starts:
            break
    if not out:
        out.append(np.sqrt(lo * hi))
    return out


def solve_feasible_first(
    evaluate_powers: Callable[[np.ndarray], dict[str, Any]],
    lo: np.ndarray,
    hi: np.ndarray,
    starts: Iterable[np.ndarray],
    eta_ref: float,
    config: SolverConfig | None = None,
) -> dict[str, Any]:
    """Run QoS-feasibility -> mean-EE budget-feasibility -> max-min stages.

    `evaluate_powers(p)` must return:
      {"ee": 1-D per-link EE array, "qos_rates": 1-D per-link QoS rates}
    All EE values must use the project's electrical denominator P_c + p.
    """
    cfg = config or SolverConfig()
    cfg.validate()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.ndim != 1 or hi.shape != lo.shape or lo.size == 0:
        raise ValueError("lo/hi must be nonempty one-dimensional arrays of equal shape")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo <= 0) or np.any(hi < lo):
        raise ValueError("invalid power bounds")
    if not np.isfinite(eta_ref) or eta_ref <= 0:
        raise ValueError("eta_ref must be finite and positive")

    starts_u = _unique_starts(starts, lo, hi, cfg.max_starts)
    log_lo, log_hi = np.log(lo), np.log(hi)
    power_bounds = list(zip(log_lo.tolist(), log_hi.tolist()))
    budget_threshold = (1.0 - cfg.delta) * float(eta_ref)

    # Reserve equal, explicit evaluation budgets for the three logical stages.
    q1 = max(1, int(round(cfg.max_evals * 0.30)))
    q2 = max(1, int(round(cfg.max_evals * 0.35)))
    q3 = max(1, cfg.max_evals - q1 - q2)
    stage_limits = {"qos_phase1": q1, "mean_phase1": q2, "maxmin_phase2": q3}
    stage_calls = {k: 0 for k in stage_limits}
    total_calls = 0
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    terminations: dict[str, list[dict[str, Any]]] = {k: [] for k in stage_limits}
    budget_exhausted: list[str] = []
    last_error: str | None = None

    def metrics(p: np.ndarray, stage: str) -> dict[str, Any]:
        nonlocal total_calls, last_error
        p = np.clip(np.asarray(p, dtype=float), lo, hi)
        key = tuple(np.round(p, 12))
        if key in cache:
            return cache[key]
        if total_calls >= cfg.max_evals or stage_calls[stage] >= stage_limits[stage]:
            raise EvaluationBudgetExceeded(stage)
        try:
            raw = evaluate_powers(p.copy())
            ee = np.asarray(raw["ee"], dtype=float)
            qos_rates = np.asarray(raw.get("qos_rates", []), dtype=float)
            if ee.shape != lo.shape or not np.isfinite(ee).all() or np.any(ee < 0):
                raise ValueError("evaluate_powers returned invalid per-link EE")
            if cfg.q_min_bps > 0:
                if qos_rates.shape != lo.shape or not np.isfinite(qos_rates).all():
                    raise ValueError("evaluate_powers returned invalid per-link QoS rates")
                qos_margin = (qos_rates - cfg.q_min_bps) / cfg.q_min_bps
            else:
                qos_margin = np.zeros(0, dtype=float)
            mean_ee = float(np.mean(ee))
            min_ee = float(np.min(ee))
            budget_margin = (mean_ee - budget_threshold) / float(eta_ref)
            result = {
                "powers": p.copy(),
                "ee": ee.copy(),
                "qos_rates": qos_rates.copy(),
                "qos_margin": qos_margin.copy(),
                "mean_ee": mean_ee,
                "min_ee": min_ee,
                "budget_margin": float(budget_margin),
                "qos_ok": bool(qos_margin.size == 0 or np.min(qos_margin) >= -cfg.tol_qos_rel),
                "budget_ok": bool(budget_margin >= -cfg.tol_budget_rel),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            raise
        cache[key] = result
        total_calls += 1
        stage_calls[stage] += 1
        return result

    def safe_eval(p: np.ndarray, stage: str) -> dict[str, Any]:
        return metrics(p, stage)

    # ---- Stage A: QoS feasibility only; no budget/QoS weighted-sum tradeoff. ----
    qos_points: dict[tuple[float, ...], dict[str, Any]] = {}
    if cfg.q_min_bps <= 0:
        for p0 in starts_u:
            try:
                m = safe_eval(p0, "qos_phase1")
                qos_points[tuple(np.round(m["powers"], 12))] = m
            except EvaluationBudgetExceeded:
                budget_exhausted.append("qos_phase1")
                break
    else:
        for restart, p0 in enumerate(starts_u):
            try:
                m0 = safe_eval(p0, "qos_phase1")
                if m0["qos_ok"]:
                    qos_points[tuple(np.round(m0["powers"], 12))] = m0
                qmargin0 = m0["qos_margin"]
                t0 = max(0.0, float(-np.min(qmargin0))) if qmargin0.size else 0.0
                x0 = np.r_[np.log(m0["powers"]), t0]

                def qos_constraint(x):
                    p = np.exp(np.clip(x[:-1], log_lo, log_hi))
                    return safe_eval(p, "qos_phase1")["qos_margin"] + float(x[-1])

                res = minimize(
                    fun=lambda x: float(x[-1]),
                    x0=x0,
                    method="SLSQP",
                    bounds=power_bounds + [(0.0, 2.0)],
                    constraints=[{"type": "ineq", "fun": qos_constraint}],
                    options={"maxiter": cfg.maxiter_qos, "ftol": cfg.ftol, "disp": False},
                )
                p_res = np.exp(np.clip(res.x[:-1], log_lo, log_hi))
                m_res = safe_eval(p_res, "qos_phase1")
                terminations["qos_phase1"].append({
                    "restart": restart, "success": bool(res.success),
                    "status": int(res.status), "message": str(res.message),
                    "nit": int(getattr(res, "nit", -1)),
                    "nfev": int(getattr(res, "nfev", -1)),
                    "epigraph_t": float(res.x[-1]),
                    "qos_min_margin": float(np.min(m_res["qos_margin"])) if m_res["qos_margin"].size else 0.0,
                })
                if m_res["qos_ok"]:
                    qos_points[tuple(np.round(m_res["powers"], 12))] = m_res
            except EvaluationBudgetExceeded:
                budget_exhausted.append("qos_phase1")
                break
            except Exception as exc:
                terminations["qos_phase1"].append({
                    "restart": restart, "success": False,
                    "status": "exception", "message": f"{type(exc).__name__}: {exc}",
                })

    qos_feasible = list(qos_points.values())
    qos_feasible.sort(key=lambda x: (x["mean_ee"], x["min_ee"]), reverse=True)
    best_qos = qos_feasible[0] if qos_feasible else None

    # ---- Stage B: maximize mean pair-EE subject to QoS; budget is checked after. ----
    budget_points: dict[tuple[float, ...], dict[str, Any]] = {}
    if qos_feasible:
        for restart, m0 in enumerate(qos_feasible[:cfg.max_starts]):
            p0 = m0["powers"]
            try:
                # Preserve a feasible starting point even if SLSQP later fails.
                budget_points[tuple(np.round(p0, 12))] = m0 if m0["budget_ok"] else None
                if budget_points[tuple(np.round(p0, 12))] is None:
                    del budget_points[tuple(np.round(p0, 12))]
                res = minimize(
                    fun=lambda u: -safe_eval(np.exp(np.clip(u, log_lo, log_hi)),
                                             "mean_phase1")["mean_ee"] / float(eta_ref),
                    x0=np.log(p0),
                    method="SLSQP",
                    bounds=power_bounds,
                    constraints=[{
                        "type": "ineq",
                        "fun": lambda u: safe_eval(
                            np.exp(np.clip(u, log_lo, log_hi)), "mean_phase1"
                        )["qos_margin"] - cfg.feasibility_buffer_rel,
                    }],
                    options={"maxiter": cfg.maxiter_mean, "ftol": cfg.ftol, "disp": False},
                )
                p_res = np.exp(np.clip(res.x, log_lo, log_hi))
                m_res = safe_eval(p_res, "mean_phase1")
                terminations["mean_phase1"].append({
                    "restart": restart, "success": bool(res.success),
                    "status": int(res.status), "message": str(res.message),
                    "nit": int(getattr(res, "nit", -1)),
                    "nfev": int(getattr(res, "nfev", -1)),
                    "mean_ee": float(m_res["mean_ee"]),
                    "budget_margin_rel": float(m_res["budget_margin"]),
                    "qos_ok": bool(m_res["qos_ok"]),
                })
                if m_res["qos_ok"] and m_res["budget_ok"]:
                    budget_points[tuple(np.round(p_res, 12))] = m_res
            except EvaluationBudgetExceeded:
                budget_exhausted.append("mean_phase1")
                break
            except Exception as exc:
                terminations["mean_phase1"].append({
                    "restart": restart, "success": False,
                    "status": "exception", "message": f"{type(exc).__name__}: {exc}",
                })

    # A QoS-feasible point may already satisfy the budget, even if Phase B had no calls.
    for m in qos_feasible:
        if m["budget_ok"]:
            budget_points[tuple(np.round(m["powers"], 12))] = m

    budget_feasible = list(budget_points.values())
    budget_feasible.sort(key=lambda x: (x["min_ee"], x["mean_ee"]), reverse=True)
    best_budget = budget_feasible[0] if budget_feasible else None

    # ---- Stage C: maximize min pair-EE subject to BOTH frozen constraints. ----
    maxmin_points: dict[tuple[float, ...], dict[str, Any]] = {}
    if budget_feasible:
        for restart, m0 in enumerate(budget_feasible[:cfg.max_starts]):
            p0 = m0["powers"]
            z0 = float(m0["min_ee"] / eta_ref)
            try:
                x0 = np.r_[np.log(p0), z0]
                z_hi = max(1.0, 2.0 * z0 + 1.0)

                def maxmin_constraints(x):
                    p = np.exp(np.clip(x[:-1], log_lo, log_hi))
                    m = safe_eval(p, "maxmin_phase2")
                    return np.r_[
                        m["budget_margin"] - cfg.feasibility_buffer_rel,
                        m["qos_margin"] - cfg.feasibility_buffer_rel,
                        m["ee"] / float(eta_ref) - float(x[-1]) + 1e-10,
                    ]

                res = minimize(
                    fun=lambda x: -float(x[-1]),
                    x0=x0,
                    method="SLSQP",
                    bounds=power_bounds + [(0.0, z_hi)],
                    constraints=[{"type": "ineq", "fun": maxmin_constraints}],
                    options={"maxiter": cfg.maxiter_maxmin, "ftol": cfg.ftol, "disp": False},
                )
                p_res = np.exp(np.clip(res.x[:-1], log_lo, log_hi))
                m_res = safe_eval(p_res, "maxmin_phase2")
                terminations["maxmin_phase2"].append({
                    "restart": restart, "success": bool(res.success),
                    "status": int(res.status), "message": str(res.message),
                    "nit": int(getattr(res, "nit", -1)),
                    "nfev": int(getattr(res, "nfev", -1)),
                    "z_reported": float(res.x[-1]),
                    "min_ee": float(m_res["min_ee"]),
                    "budget_margin_rel": float(m_res["budget_margin"]),
                    "qos_ok": bool(m_res["qos_ok"]),
                    "budget_ok": bool(m_res["budget_ok"]),
                })
                if m_res["qos_ok"] and m_res["budget_ok"]:
                    maxmin_points[tuple(np.round(p_res, 12))] = m_res
            except EvaluationBudgetExceeded:
                budget_exhausted.append("maxmin_phase2")
                break
            except Exception as exc:
                terminations["maxmin_phase2"].append({
                    "restart": restart, "success": False,
                    "status": "exception", "message": f"{type(exc).__name__}: {exc}",
                })

    candidates = list(budget_feasible) + list(maxmin_points.values())
    candidates.sort(key=lambda x: (x["min_ee"], x["mean_ee"]), reverse=True)
    final = candidates[0] if candidates else (best_qos if best_qos is not None else None)

    if not qos_feasible:
        status = "qos_feasible_point_not_found"
    elif not budget_feasible:
        status = "qos_feasible_budget_point_not_found"
    elif maxmin_points:
        status = "maxmin_local_solution_found"
    else:
        status = "budget_feasible_point_found_maxmin_not_improved_or_not_completed"

    def pack(m):
        if m is None:
            return None
        return {
            "powers": [float(x) for x in m["powers"]],
            "ee": [float(x) for x in m["ee"]],
            "qos_rates": [float(x) for x in m["qos_rates"]],
            "mean_ee": float(m["mean_ee"]),
            "min_ee": float(m["min_ee"]),
            "budget_margin_rel": float(m["budget_margin"]),
            "qos_min_margin_rel": (
                float(np.min(m["qos_margin"])) if m["qos_margin"].size else None
            ),
            "qos_ok": bool(m["qos_ok"]),
            "budget_ok": bool(m["budget_ok"]),
        }

    return {
        "status": status,
        "solver_claim": "multistart local SLSQP; found feasible points are witnesses; failure to find is not an infeasibility proof",
        "qos_feasible_found": bool(qos_feasible),
        "budget_feasible_found": bool(budget_feasible),
        "maxmin_local_solution_found": bool(maxmin_points),
        "selected": pack(final),
        "best_qos_feasible": pack(best_qos),
        "best_budget_feasible": pack(best_budget),
        "evaluation_calls": int(total_calls),
        "evaluation_calls_by_stage": {k: int(v) for k, v in stage_calls.items()},
        "stage_evaluation_limits": stage_limits,
        "cache_size": int(len(cache)),
        "budget_exhausted_stages": budget_exhausted,
        "last_error": last_error,
        "terminations": terminations,
        "config": asdict(cfg),
    }
