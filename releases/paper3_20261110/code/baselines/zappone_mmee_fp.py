"""MMEE-FP：Zappone 等 TSP 2016 最低个体 EE 功率控制，对本模型的固定配置适配。

原文：A. Zappone et al., Energy-Efficient Power Control: A Look at 5G Wireless
Technologies, IEEE TSP 64(7):1668–1683, 2016 (arXiv:1503.04609).

选用原文**网络中心最低个体 EE**分支（式 (29) / Algorithm 1 的 Minimum EE
路径），不移植 GEE 总比值或分布式自利博弈。

适配组织（见 docs/ALGORITHM_MAP.md）：

* 固定共同参考配置的配对/模式/端口；只优化连续功率；**无 δ 预算**。
* 外层用广义分式规划：给定 λ，处理
  ``max_{p,t} t`` s.t. ``R_k(p) − λ (Pc_k + p_k) ≥ t``，再令
  ``λ ← min_k R_k(p)/(Pc_k+p_k)``。
* 内层速率非凹：采用与项目一致的二次变换 SCA 凹下界（Shen & Yu），
  **不是**原文式 (37) 的 log-SINR 下界——因为本模型含 VLC 平方律，SINR
  不落在原文式 (1) 的 α/φ/ω 族。因此**不继承**原文 KKT/全局收敛证明。
* 代理 QoS（块初信念瞬时速率）可在内层加入；机会约束 p05 由 adapters
  用共同 ``predicted_qos_rates`` 复验与有限抬升修复。

本模块**不**调用 ``maxmin_power_refine`` / ``protect_budget`` / 本文配置搜索。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from baselines._path import METHOD_ZAPPONE, ensure_frozen_on_path

ensure_frozen_on_path()

from p3learn.bottleneck_power import (  # noqa: E402
    _SLSQP_FTOL, _SLSQP_MAXITER, _linear_rates, link_ee,
)
from p3learn.dinkelbach_sca import _Physics  # noqa: E402


@dataclass
class MMEEFPInfo:
    method: str = METHOD_ZAPPONE
    outer_rounds: int = 0
    sca_rounds: int = 0
    lambda_trace: list = field(default_factory=list)
    residual: float = float("nan")
    start_min_ee: float = float("nan")
    final_min_ee: float = float("nan")
    status: str = ""
    improved_over_start: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sca_max_min_surplus(phys, p, lam, q, trust):
    """固定 λ：最大化最小裕量 t，约束 r~_k(p) − λ(Pc_k+p_k) ≥ t。"""
    bound = _linear_rates(phys, p)
    k = phys.k
    scale = np.maximum(p, 0.01)
    # 归一化避免 SLSQP 数值爆炸（与 bottleneck_power 同策略）。
    cnorm = np.maximum(np.abs(lam * (phys.static + p) + 1.0), 1.0)
    qnorm = None if q is None else np.maximum(np.abs(q), 1.0)

    def constraints(w):
        pw = w[:k] * scale
        r, _jac = bound(pw)
        rows = [(r - lam * (phys.static + pw) - w[k]) / cnorm]
        if q is not None:
            rows.append((r - q) / qnorm)
        return np.concatenate(rows)

    def constraint_jac(w):
        pw = w[:k] * scale
        _r, jac = bound(pw)  # jac[c, link] = dr_link / dp_c
        j_lp = jac.T  # (link, param)
        # d/dw_c [(r - lam*(Pc+p) - t)/cnorm] = (dr/dp_c * scale_c - lam*scale_c)/cnorm
        d_p = (j_lp * scale[None, :] - lam * np.diag(scale)) / cnorm[:, None]
        row = np.c_[d_p, -1.0 / cnorm]
        rows = [row]
        if q is not None:
            rows.append(np.c_[j_lp * scale[None, :] / qnorm[:, None], np.zeros(k)])
        return np.vstack(rows)

    def objective(w):
        return -float(w[k]), np.r_[np.zeros(k), -1.0]

    x0 = np.r_[p / scale, 0.0]
    # 用当前真实裕量初始化 t
    rates0 = phys.rates(p)
    surplus0 = rates0 - lam * (phys.static + p)
    x0[k] = float(np.min(surplus0))
    low = np.maximum(phys.lo, p / trust)
    high = np.minimum(phys.hi, p * trust)
    bounds = list(zip(low / scale, np.maximum(high / scale, low / scale))) + [(None, None)]
    try:
        res = minimize(objective, x0, jac=True, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": constraints,
                                     "jac": constraint_jac}],
                       options=dict(maxiter=_SLSQP_MAXITER, ftol=_SLSQP_FTOL))
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return None
    cand = np.asarray(res.x[:k], float) * scale
    if not np.isfinite(cand).all():
        return None
    return np.clip(cand, phys.lo, phys.hi)


def _inner_generalized_fp(phys, p0, lam, q, max_sca, trust=2.0):
    """对固定 λ 做若干轮 SCA，返回改进后的功率与 SCA 轮数。"""
    p = np.clip(np.asarray(p0, float), phys.lo, phys.hi)
    best_p = p.copy()
    best_t = float(np.min(phys.rates(p) - lam * (phys.static + p)))
    rounds = 0
    for _ in range(max(int(max_sca), 1)):
        rounds += 1
        cand = _sca_max_min_surplus(phys, p, lam, q, trust)
        if cand is None:
            break
        # 用原始速率复验裕量（不认线性下界）
        t = float(np.min(phys.rates(cand) - lam * (phys.static + cand)))
        if q is not None and float(np.min(phys.rates(cand))) < float(np.min(q)) * (1 - 1e-9):
            # QoS 未满足：若最弱 EE 仍改善则小步接受，否则停
            ee_c = float(link_ee(phys, cand).min())
            ee_p = float(link_ee(phys, p).min())
            if ee_c > ee_p + 1e-12:
                p = cand
                continue
            break
        if t > best_t + 1e-12:
            best_t, best_p = t, cand.copy()
            p = cand
        elif float(np.max(np.abs(cand - p))) <= 1e-9 * max(float(np.max(p)), 1e-9):
            break
        else:
            # 无改进则停止，避免空转
            break
    return best_p, rounds, best_t


def zappone_mmee_fp_fixed(links, gains, ports, cfg, *, p_start=None,
                          q_min_bps=0.0, max_outer=20, max_sca=6, tol=1e-5,
                          trust=2.0):
    """固定配置下最大化最低逐链路 EE。返回 ``(powers, MMEEFPInfo)``。

    无 δ；``q_min_bps`` 仅为信念域瞬时速率代理约束。
    """
    notes = [
        "branch=network_centric_minimum_individual_EE",
        "outer=generalized_Dinkelbach_on_min_ratio",
        "inner=quadratic_transform_SCA_not_paper_eq37",
        "SINR_model=project_RF_VLC_square_law_not_paper_eq1",
        "no_delta_budget",
        "fixed_configuration",
    ]
    if not links:
        info = MMEEFPInfo(status="empty", notes=notes)
        return np.zeros(0), info
    phys = _Physics(links, gains, ports, cfg)
    k = phys.k
    if p_start is None:
        p = 0.5 * (phys.lo + phys.hi)
    else:
        p = np.clip(np.asarray(p_start, float), phys.lo, phys.hi)
    if p.shape != (k,):
        raise ValueError("p_start must have one power per link")

    q = None
    if q_min_bps is not None and float(np.max(np.asarray(q_min_bps, float))) > 0:
        q_arr = np.asarray(q_min_bps, float)
        q = np.full(k, float(q_arr)) if q_arr.ndim == 0 else q_arr.astype(float)

    start_min = float(link_ee(phys, p).min())
    lam = max(start_min, 0.0)
    info = MMEEFPInfo(start_min_ee=start_min, notes=notes)
    info.lambda_trace.append(dict(outer=0, lambda_=lam, min_ee=start_min,
                                  residual=float("nan")))

    best_p = p.copy()
    best_min = start_min
    total_sca = 0
    status = "max_outer"
    for outer in range(1, max(int(max_outer), 1) + 1):
        cand, sca_r, surplus = _inner_generalized_fp(phys, best_p, lam, q, max_sca, trust)
        total_sca += sca_r
        ee = link_ee(phys, cand)
        new_lam = float(ee.min())
        residual = abs(new_lam - lam) / max(abs(lam), 1.0)
        info.lambda_trace.append(dict(outer=outer, lambda_=new_lam, min_ee=new_lam,
                                      surplus=surplus, residual=residual, sca=sca_r))
        if new_lam > best_min + 1e-12:
            best_min, best_p = new_lam, cand.copy()
        info.outer_rounds = outer
        if residual < tol and surplus >= -1e-6 * max(abs(lam), 1.0):
            status = "stationary_residual"
            lam = new_lam
            break
        if new_lam <= lam + 1e-15 and surplus < 0:
            status = "no_progress"
            break
        lam = new_lam

    info.sca_rounds = total_sca
    info.final_min_ee = float(link_ee(phys, best_p).min())
    info.residual = abs(info.final_min_ee - info.start_min_ee)
    info.improved_over_start = info.final_min_ee > start_min + 1e-12
    info.status = status
    # 未改善则退回起点，避免把结果改坏
    if not info.improved_over_start:
        return np.asarray(p_start if p_start is not None else p, float).copy(), info
    return best_p, info
