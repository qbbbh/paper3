"""Aboagye 等人的"匹配 + 二次变换"EE 资源分配在本问题上的复现。

原文：S. Aboagye, T. M. N. Ngatched, O. A. Dobre, H. V. Poor,
*Energy-Efficient Resource Allocation for Aggregated RF/VLC Systems*,
IEEE TWC 22(10):6624-6640, 2023（arXiv:2206.01567）。

复现的三步（原文的交替优化骨架）：

1. **AP 分配 = 一对多双边匹配**。用户按 EE 偏好向 AP 提议，AP 按功率配额
   保留最优的若干个、拒绝其余；被拒用户改投次优 AP，直到没有用户还能提议。
   偏好用"预定功率下的孤立链路 EE"（原文式 13-14），不用速率。
2. **子信道分配**：原文按信道增益贪心分配子信道。
3. **功率分配 = 二次变换**（原文式 19-22 与 Algorithm 2：以等功率启动，
   交替更新辅助变量与凸功率子问题）。这里的二次变换用本仓库已有的
   ``dinkelbach_sum_ee``（Shen & Yu 变换 + SCA）。
4. 三步**交替**：功率解出来后重算偏好，再匹配，直到匹配不再变化。

与本问题的差异（论文里必须逐条写明，不能含糊）：

* 原文的 AP 可以服务多个用户，配额由功预算 ``|μ(k)|·P̄_k ≤ P_k`` 决定。
  我们的每条链路独占一个收发端口、不允许一对多，**配额恒为 1**，一对多
  因此退化为"一对一匹配"。
* 角色不预先固定（任何车都可以当发射端），所以没有固定的二分结构，不能直接
  用 Gale-Shapley 的提议-保留过程；这里用等价的**交换阻塞对**判据：
  若两对端点的重新组合能让四个端点全部变好，就交换，直到不存在这样的交换。
  这是"2-交换稳定"，比原文的稳定匹配弱，代码里如实按这个口径命名。
* 没有子信道维度（每条链路独占整段带宽），第 2 步不做。
* QoS 与原文一致，**不进匹配**，只在最后的功率子问题里用单调抬升满足。
"""
import numpy as np

from p3base.energy import static_power_w
from p3base.noise import vlc_noise_linear_w

from .allocation import (MODES, CoverageFailure, _domain, _flat_gain,
                         _vlc_optical_sinr, decode_complete_pairing, _power_solve,
                         _proxy_scores)

__all__ = ["preference_tables", "interference_scores", "preference_from_scores",
           "swap_refine", "matching_assignment"]

_TIE = 1e-12
# 合法但功率上限也达不到最低速率的边：给一个**有限**的极小分数。
# decode_complete_pairing 要求所有合法边分数有限，所以不能用 -inf。
_INFEASIBLE = -1e6


def feasible_edges(n, cfg, ports, masks, belief_gains, q_min_bps):
    """单链路在**功率上限**下能否达到最低速率；达不到的边不给偏好值。

    这一层在原问题里由功率子问题的约束满足保证（原文把 QoS 放在 PA 步），
    但我们的模型里 RF 的静态功率低、VLC 容量大，若只按孤立 EE 排序，匹配会
    系统性避开 VLC，选出一批"EE 高但功率上限也达不到 1 Mbit/s"的 RF 配对，
    最后整块不可行。所以复现时必须加这一步筛除，并在论文里写明它是适配项、
    不是原文内容。筛除只看单链路上限（无干扰），因此是乐观筛除。
    """
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
    """每个 (i, j, mode) 在参考功率网格上的**最优孤立链路 EE**。

    原文式 13-14 的偏好是"预定功率下的孤立链路 EE"；原文的预定功率是
    "AP 总功率 / 子信道数"，本问题是每条链路独占整段带宽，没有子信道可除，
    所以把"预定功率"取成参考网格上对该链路最好的那一档。这不是原文的做法，
    是复现时为了不让偏好被单一标称功率卡死而做的适配，必须写进论文。
    """
    best = {mode: np.full((n, n), -np.inf) for mode in MODES}
    for p in grid:
        scores = _proxy_scores(n, cfg, ports, belief_gains, float(p))
        for mode in MODES:
            best[mode] = np.maximum(best[mode], scores[mode])
    if feasible is not None:
        for mode in MODES:
            best[mode] = np.where(feasible[mode], best[mode], _INFEASIBLE)
    return best


def preference_tables(n, cfg, ports, masks, belief_gains, powers_by_mode, grid=None):
    """每个无序端点对 (i, j) 的偏好值与对应的有向+模式选择。

    偏好值 = **孤立链路 EE**（原文式 13-14）。``grid`` 给出时取参考网格上的
    最优档（推荐）；否则两种模式各用 ``powers_by_mode`` 里的单一预定功率。
    ``value[i, j]`` 在不可行时为 ``-inf``；``choice[(i, j)]`` 记下取得该值的
    ``(tx, rx, mode)``，方向与模式都在比较之内。
    """
    if grid is not None:
        scores = edge_ee_scores(n, cfg, ports, belief_gains, grid)
    else:
        scores = {}
        for mode in MODES:
            model = cfg.rf if mode == "RF" else cfg.vlc
            p = float(np.clip(powers_by_mode[mode], model.power_min_w, model.power_max_w))
            # _proxy_scores 的每个模式只读自己的功率，取两次调用中对应模式的那一半。
            scores[mode] = _proxy_scores(n, cfg, ports, belief_gains, p)[mode]
    value = np.full((n, n), -np.inf)
    choice = {}
    for i in range(n):
        for j in range(i + 1, n):
            best = None
            for tx, rx in ((i, j), (j, i)):
                for mode in MODES:
                    if not masks[mode][tx, rx]:
                        continue
                    score = float(scores[mode][tx, rx])
                    if not np.isfinite(score) or score <= _INFEASIBLE:
                        continue
                    # 稳定平局规则：先按分数，再按端点序号、再 RF 优先。
                    key = (score, -tx, -rx, mode == "RF")
                    if best is None or key > best[0]:
                        best = (key, (tx, rx, mode))
            if best is not None:
                value[i, j] = value[j, i] = best[0][0]
                choice[(i, j)] = choice[(j, i)] = best[1]
    return value, choice


def interference_scores(n, cfg, ports, masks, belief_gains, links, powers, p_by_mode):
    """固定当前功率时，每条候选有向边的**含干扰**逐链路 EE。

    这是交替优化的第二半：原文用"AP 分配固定 -> 求功率 -> 再分配"。给定当前
    功率后，其它链路造成的干扰是已知的，于是候选边的 EE 可以用真实的 SINR
    算，而不是孤立链路的理想值。候选边的功率取该模式的预定功率 ``p_by_mode``。

    返回 ``{(mode, tx, rx): ee}``，只含合法边。
    """
    rf = _flat_gain(n, "RF", belief_gains["RF"], ports)
    vlc_grid = np.asarray(belief_gains["VLC"], float)
    eta_rf = float(cfg.rf.pa_efficiency)
    rho = float(cfg.vlc.responsivity)
    out = {}
    # 当前链路逐条累加的接收端干扰（按"发送端 -> 接收端"取增益）。
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
                    else:                      # 退回旧的 (n,n) 增益表
                        h_self = vlc_grid[i, j]
                        terms = [cfg.vlc.eo_efficiency * p * vlc_grid[u, j]
                                 for u, _v, p in active if u != i]
                    optical_self = cfg.vlc.eo_efficiency * power * h_self
                    optical_other = float(sum(terms))
                    # 平方律：信号与各条干扰各自平方后相加，不是先求和再平方。
                    signal = (rho * optical_self) ** 2
                    inter_sq = (rho ** 2) * float(sum(t * t for t in terms))
                    noise = vlc_noise_linear_w(optical_self + optical_other, cfg.vlc)
                    sinr = signal / (noise + inter_sq)
                    rate = cfg.vlc.bandwidth_hz * np.log2(1.0 + sinr / cfg.vlc.gap)
                    denom = static_power_w("VLC", cfg) + power
                out[(mode, i, j)] = float(rate / denom)
    return out


def preference_from_scores(n, scores):
    """把逐边分数压成端点对的偏好值与有向+模式选择（与 prefer_table 同口径）。"""
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
    """2-交换稳定：只要存在"四端点全部变好"的重连就执行，直到没有。

    ``edges`` 是互不相交的无序端点对列表（覆盖全部节点）。每一轮取增益最大
    的那一次交换，保证目标单调上升因而有限步终止。返回端点对列表（已排序）。

    它比原文的稳定匹配弱：原文保证"不存在阻塞对"，这里只保证"不存在让四个
    端点同时受益的 2-交换"。名字按实际口径叫 2-交换稳定，不冒充稳定匹配。
    """
    edges = [tuple(sorted(edge)) for edge in edges]
    for _ in range(max_rounds):
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
                    # 四个端点各自比较"新伙伴"与"旧伙伴"的偏好值，全部严格变好才交换。
                    if not all(value[u, v] > value[u, old[u]] + _TIE
                               for x, y in pairs for u, v in ((x, y), (y, x))):
                        continue
                    gain = (value[pairs[0]] + value[pairs[1]]
                            - value[i, j] - value[k, l])
                    if gain > best_gain + _TIE:
                        best_gain, best_move = gain, (a, b, pairs)
        if best_move is None:
            break
        a, b, pairs = best_move
        for idx in sorted((a, b), reverse=True):
            edges.pop(idx)
        edges.extend(pairs)
        edges = sorted(edges)
    return edges


def _per_mode_power(links, powers, cfg, fallback_w):
    """各模式的预定功率（原文的 P̄_k）：该模式下现有链路的平均功率。"""
    out = {}
    for mode in MODES:
        model = cfg.rf if mode == "RF" else cfg.vlc
        vals = [float(p) for (_i, _j, m), p in zip(links, powers) if m == mode]
        p = float(np.mean(vals)) if vals else float(fallback_w)
        out[mode] = float(np.clip(p, model.power_min_w, model.power_max_w))
    return out


def matching_assignment(n, cfg, candidates, business, ports, belief_gains, *,
                        power_solver="dinkelbach", nominal_w=0.3, rounds=3,
                        grid=(0.01, 0.3, 3.0), q_min_bps=0., solver_log=None):
    """原文的"匹配 -> 二次变换功率 -> 重算偏好 -> 再匹配"交替骨架。

    起点用最大权完全匹配（等价于原文在初始偏好表上的提议-保留结果），功率
    随后交给二次变换；每轮用上一轮的逐模式平均功率重算偏好并做 2-交换，
    匹配不变即停。QoS 不参与这一层（与原文一致），由调用方的
    ``_optimize_action`` 在最后一次功率求解里抬升。

    返回 ``(links, powers, ports)``；图不连通时抛 ``CoverageFailure``。
    """
    masks, port_copy = _domain(n, cfg, candidates, business, ports)
    nominal = float(np.clip(nominal_w, min(cfg.rf.power_min_w, cfg.vlc.power_min_w),
                            max(cfg.rf.power_max_w, cfg.vlc.power_max_w)))
    grid = tuple(float(p) for p in grid)
    feasible = feasible_edges(n, cfg, port_copy, masks, belief_gains, q_min_bps)
    scores = edge_ee_scores(n, cfg, port_copy, belief_gains, grid, feasible=feasible)
    start = decode_complete_pairing(n, cfg, candidates, business, port_copy, scores, nominal)
    links = [tuple(link) for link in start.links]
    powers = _power_solve(links, belief_gains, port_copy, cfg, power_solver,
                          solver_log=solver_log, q_min_bps=0.0)
    converged = True
    rounds_run = 0
    for _ in range(max(int(rounds), 0)):
        rounds_run += 1
        pbar = _per_mode_power(links, powers, cfg, nominal)
        # 交替优化第二半：功率已知后，偏好换成**含干扰**的逐链路 EE。
        scores = interference_scores(n, cfg, port_copy, masks, belief_gains,
                                     links, powers, pbar)
        scores = {(m, i, j): v for (m, i, j), v in scores.items() if feasible[m][i, j]}
        value, choice = preference_from_scores(n, scores)
        edges = [tuple(sorted((tx, rx))) for tx, rx, _mode in links]
        refined = swap_refine(edges, value)
        new_links = sorted((choice[edge] for edge in refined),
                           key=lambda link: (min(link[:2]), max(link[:2])))
        converged = new_links == links
        if converged:
            break
        links = [tuple(link) for link in new_links]
        powers = _power_solve(links, belief_gains, port_copy, cfg, power_solver,
                              solver_log=solver_log, q_min_bps=0.0)
    if solver_log is not None:
        solver_log.append(dict(stage="matching_alternation", rounds_run=rounds_run,
                               converged=bool(converged), nominal_power_w=nominal))
    if 2 * len(links) != n:
        raise CoverageFailure("coverage_failure: matching did not cover every endpoint")
    return links, powers, port_copy
