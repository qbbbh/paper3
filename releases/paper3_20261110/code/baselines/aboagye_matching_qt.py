"""Matching-QT：Aboagye 等 TWC 2023 匹配+二次变换功率交替，对本车联网模型的适配。

原文：S. Aboagye et al., Energy-Efficient Resource Allocation for Aggregated
RF/VLC Systems, IEEE TWC 22(10):6624–6640, 2023 (arXiv:2206.01567).

保留：匹配分配与二次变换功率优化**交替**的骨架；全过程保留信念域上最好的
均逐链路 EE 可行解。

因模型而改动（必须在论文中写明，见 docs/ALGORITHM_MAP.md）：

* 原文 AP 一对多 + 子信道；此处退化为车辆**一对一完整配对**、每对单模式，
  **省略子信道步**。
* 初始匹配用最大权完美匹配（匈牙利），这是适配初始化，**不等价**于原文
  Algorithm 1 的提议–保留稳定匹配。
* 后续 refinement 用 2-交换稳定（四端点同时严格变好），比原文稳定匹配弱。
* QoS 使用本项目机会约束接口在外层复验；交替环内功率子问题默认不做 QoS，
  与原文"QoS 放在 PA 步"的精神一致，但最终由 adapters 调用共同
  ``predicted_qos_rates`` 复验与有限预算修复。
* 目标对齐本项目**算术平均逐链路 EE**（非原文系统总比值 EE）。

本模块**不**调用本文公平搜索器（``max_min_allocate`` / ``protect_budget`` /
``maxmin_power_refine``），功率子问题复用共享的 ``_power_solve``（二次变换）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from baselines._path import METHOD_ABOAGYE, ensure_frozen_on_path

ensure_frozen_on_path()

from p3base.energy import static_power_w  # noqa: E402
from p3base.noise import vlc_noise_linear_w  # noqa: E402
from p3learn.allocation import (  # noqa: E402
    MODES, CoverageFailure, _domain, _flat_gain, _vlc_optical_sinr,
    decode_complete_pairing, _power_solve, _proxy_scores, _rates_and_ee,
    _link_gain_tables,
)

_TIE = 1e-12
_INFEASIBLE = -1e6


@dataclass
class MatchingQTInfo:
    method: str = METHOD_ABOAGYE
    rounds_budget: int = 0
    rounds_run: int = 0
    stop_reason: str = ""
    swap_local_stable: bool = False
    swap_budget_exhausted: bool = False
    matching_changes: list = field(default_factory=list)
    objective_trace: list = field(default_factory=list)
    power_solver_calls: int = 0
    best_mean_ee_belief: float = float("nan")
    nominal_power_w: float = float("nan")
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feasible_edges(n, cfg, ports, masks, belief_gains, q_min_bps):
    """单链路上限功率、无干扰下的乐观 QoS 筛边（适配项，非原文）。"""
    ok = {}
    for mode in MODES:
        model = cfg.rf if mode == "RF" else cfg.vlc
        gain = _flat_gain(n, mode, belief_gains[mode], ports)
        p = float(model.power_max_w)
        if mode == "RF":
            snr = cfg.rf.pa_efficiency * p * gain / cfg.rf.noise_w
        else:
            snr = _vlc_optical_sinr(p, gain, cfg)
        rate = model.bandwidth_hz * np.log2(1.0 + snr / model.gap)
        ok[mode] = (rate >= float(q_min_bps)) & masks[mode]
    return ok


def edge_ee_scores(n, cfg, ports, belief_gains, grid, feasible=None):
    best = {mode: np.full((n, n), -np.inf) for mode in MODES}
    for p in grid:
        scores = _proxy_scores(n, cfg, ports, belief_gains, float(p))
        for mode in MODES:
            best[mode] = np.maximum(best[mode], scores[mode])
    if feasible is not None:
        for mode in MODES:
            best[mode] = np.where(feasible[mode], best[mode], _INFEASIBLE)
    return best


def interference_scores(n, cfg, ports, masks, belief_gains, links, powers, p_by_mode):
    """固定当前匹配与功率后，候选有向边的含干扰逐链路 EE。

    换接收者/方向/模式时，旧链路功率不参与干扰、新候选用 ``p_by_mode``，
    受影响接收端的交叉干扰在此函数内按当前 active 集合重算。
    """
    rf = _flat_gain(n, "RF", belief_gains["RF"], ports)
    vlc_grid = np.asarray(belief_gains["VLC"], float)
    eta_rf = float(cfg.rf.pa_efficiency)
    rho = float(cfg.vlc.responsivity)
    out = {}
    for mode in MODES:
        active = [(tx, rx, float(p)) for (tx, rx, m), p in zip(links, powers) if m == mode]
        for i in range(n):
            for j in range(n):
                if i == j or not masks[mode][i, j]:
                    continue
                if mode == "RF":
                    inter = eta_rf * sum(p * rf[u, j] for u, _v, p in active if u != i)
                    signal = eta_rf * float(p_by_mode[mode]) * rf[i, j]
                    sinr = signal / (cfg.rf.noise_w + max(inter, 0.0))
                    rate = cfg.rf.bandwidth_hz * np.log2(1.0 + sinr / cfg.rf.gap)
                    denom = static_power_w("RF", cfg) + float(p_by_mode[mode])
                else:
                    power = float(p_by_mode[mode])
                    rx_port = int(ports["rx"][i, j])
                    if vlc_grid.ndim == 4:
                        h_self = vlc_grid[i, j, int(ports["tx"][i, j]), rx_port]
                        terms = [cfg.vlc.eo_efficiency * p
                                 * vlc_grid[u, j, int(ports["tx"][u, v]), rx_port]
                                 for u, v, p in active if u != i]
                    else:
                        h_self = vlc_grid[i, j]
                        terms = [cfg.vlc.eo_efficiency * p * vlc_grid[u, j]
                                 for u, _v, p in active if u != i]
                    optical_self = cfg.vlc.eo_efficiency * power * h_self
                    optical_other = float(sum(terms))
                    signal = (rho * optical_self) ** 2
                    inter_sq = (rho ** 2) * float(sum(t * t for t in terms))
                    noise = vlc_noise_linear_w(optical_self + optical_other, cfg.vlc)
                    sinr = signal / (noise + inter_sq)
                    rate = cfg.vlc.bandwidth_hz * np.log2(1.0 + sinr / cfg.vlc.gap)
                    denom = static_power_w("VLC", cfg) + power
                out[(mode, i, j)] = float(rate / denom)
    return out


def preference_from_scores(n, scores):
    value = np.full((n, n), -np.inf)
    choice = {}
    for i in range(n):
        for j in range(i + 1, n):
            best = None
            for tx, rx in ((i, j), (j, i)):
                for mode in MODES:
                    score = scores.get((mode, tx, rx))
                    if score is None or score <= _INFEASIBLE:
                        continue
                    key = (float(score), -tx, -rx, mode == "RF")
                    if best is None or key > best[0]:
                        best = (key, (tx, rx, mode))
            if best is not None:
                value[i, j] = value[j, i] = best[0][0]
                choice[(i, j)] = choice[(j, i)] = best[1]
    return value, choice


def swap_refine(edges, value, max_rounds=50):
    """2-交换稳定 refinement。返回 ``(edges, info)``。

    ``info['local_stable']``：在预算内遍历邻域后无改进交换。
    ``info['budget_exhausted']``：达到 ``max_rounds`` 仍可能有未检查的改进。
    """
    edges = [tuple(sorted(edge)) for edge in edges]
    local_stable = False
    rounds_used = 0
    for _ in range(max_rounds):
        rounds_used += 1
        best_gain, best_move = 0.0, None
        for a in range(len(edges)):
            for b in range(a + 1, len(edges)):
                i, j = edges[a]
                k, l = edges[b]
                old = {i: j, j: i, k: l, l: k}
                for new in (((i, k), (j, l)), ((i, l), (j, k))):
                    pairs = [tuple(sorted(x)) for x in new]
                    if any(not np.isfinite(value[x, y]) for x, y in pairs):
                        continue
                    if not all(value[u, v] > value[u, old[u]] + _TIE
                               for x, y in pairs for u, v in ((x, y), (y, x))):
                        continue
                    gain = (value[pairs[0]] + value[pairs[1]]
                            - value[i, j] - value[k, l])
                    if gain > best_gain + _TIE:
                        best_gain, best_move = gain, (a, b, pairs)
        if best_move is None:
            local_stable = True
            break
        a, b, pairs = best_move
        for idx in sorted((a, b), reverse=True):
            edges.pop(idx)
        edges.extend(pairs)
        edges = sorted(edges)
    budget_exhausted = (not local_stable) and rounds_used >= max_rounds
    return edges, dict(local_stable=local_stable, budget_exhausted=budget_exhausted,
                       rounds_used=rounds_used)


def _per_mode_power(links, powers, cfg, fallback_w):
    out = {}
    for mode in MODES:
        model = cfg.rf if mode == "RF" else cfg.vlc
        vals = [float(p) for (_i, _j, m), p in zip(links, powers) if m == mode]
        p = float(np.mean(vals)) if vals else float(fallback_w)
        out[mode] = float(np.clip(p, model.power_min_w, model.power_max_w))
    return out


def _belief_mean_ee(links, powers, belief_gains, ports, cfg) -> float:
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, belief_gains, ports, cfg)
    _rates, ee = _rates_and_ee(np.asarray(powers, float), links, cfg, s_gain, h_rf, h_vlc, h_opt)
    return float(np.mean(ee))


def aboagye_matching_qt(n, cfg, candidates, business, ports, belief_gains, *,
                        power_solver="dinkelbach", nominal_w=0.3, rounds=3,
                        grid=(0.01, 0.3, 3.0), q_min_bps=0.0, swap_max_rounds=50,
                        solver_log=None):
    """匹配→二次变换功率→含干扰偏好→2-交换→再功率，交替若干轮。

    返回 ``(links, powers, ports, MatchingQTInfo)``。
    环内 ``_power_solve(..., q_min_bps=0)``；``q_min_bps`` 仅用于乐观边筛。
    全过程按**信念域均逐链路 EE**保留最佳解（最终预测评价在 adapters）。
    """
    info = MatchingQTInfo(rounds_budget=int(rounds),
                          notes=["init=maximum_weight_perfect_matching_not_gale_shapley",
                                 "refine=2swap_stable_weaker_than_paper_stable_matching",
                                 "no_subchannel_step",
                                 "objective=mean_per_link_EE_belief_during_search"])
    masks, port_copy = _domain(n, cfg, candidates, business, ports)
    nominal = float(np.clip(nominal_w, min(cfg.rf.power_min_w, cfg.vlc.power_min_w),
                            max(cfg.rf.power_max_w, cfg.vlc.power_max_w)))
    info.nominal_power_w = nominal
    grid = tuple(float(p) for p in grid)
    feasible = feasible_edges(n, cfg, port_copy, masks, belief_gains, q_min_bps)
    scores = edge_ee_scores(n, cfg, port_copy, belief_gains, grid, feasible=feasible)
    # 适配初始化：最大权完美匹配，不是原文提议–保留。
    start = decode_complete_pairing(n, cfg, candidates, business, port_copy, scores, nominal)
    links = [tuple(link) for link in start.links]
    powers = _power_solve(links, belief_gains, port_copy, cfg, power_solver,
                          solver_log=solver_log, q_min_bps=0.0)
    info.power_solver_calls += 1
    best_links = [tuple(x) for x in links]
    best_powers = np.asarray(powers, float).copy()
    best_ee = _belief_mean_ee(best_links, best_powers, belief_gains, port_copy, cfg)
    info.objective_trace.append(dict(round=0, mean_ee_belief=best_ee,
                                     links=[list(x) for x in best_links]))
    info.best_mean_ee_belief = best_ee

    stop_reason = "rounds_exhausted"
    swap_local = True
    swap_budget = False
    for r in range(max(int(rounds), 0)):
        info.rounds_run = r + 1
        pbar = _per_mode_power(links, powers, cfg, nominal)
        edge_scores = interference_scores(n, cfg, port_copy, masks, belief_gains,
                                          links, powers, pbar)
        edge_scores = {(m, i, j): v for (m, i, j), v in edge_scores.items()
                       if feasible[m][i, j]}
        value, choice = preference_from_scores(n, edge_scores)
        edges = [tuple(sorted((tx, rx))) for tx, rx, _mode in links]
        refined, swap_info = swap_refine(edges, value, max_rounds=swap_max_rounds)
        swap_local = bool(swap_info["local_stable"])
        swap_budget = bool(swap_info["budget_exhausted"])
        new_links = sorted((choice[edge] for edge in refined),
                           key=lambda link: (min(link[:2]), max(link[:2])))
        changed = new_links != links
        info.matching_changes.append(dict(round=r + 1, changed=changed,
                                          swap=swap_info,
                                          links_before=[list(x) for x in links],
                                          links_after=[list(x) for x in new_links]))
        if not changed:
            stop_reason = "matching_unchanged"
            break
        links = [tuple(link) for link in new_links]
        powers = _power_solve(links, belief_gains, port_copy, cfg, power_solver,
                              solver_log=solver_log, q_min_bps=0.0)
        info.power_solver_calls += 1
        mean_ee = _belief_mean_ee(links, powers, belief_gains, port_copy, cfg)
        info.objective_trace.append(dict(round=r + 1, mean_ee_belief=mean_ee,
                                         links=[list(x) for x in links]))
        if mean_ee > best_ee + 1e-12:
            best_ee = mean_ee
            best_links = [tuple(x) for x in links]
            best_powers = np.asarray(powers, float).copy()
            info.best_mean_ee_belief = best_ee

    info.stop_reason = stop_reason
    info.swap_local_stable = swap_local
    info.swap_budget_exhausted = swap_budget
    if 2 * len(best_links) != n:
        raise CoverageFailure("coverage_failure: matching did not cover every endpoint")
    if solver_log is not None:
        solver_log.append(dict(stage="aboagye_matching_qt", **info.to_dict()))
    return best_links, best_powers, port_copy, info
