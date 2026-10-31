"""均匀随机合法动作下界（待并入 `code/p3learn/allocation.py`）。

为什么要它：审稿人会问"max-min 比什么都不做强多少"。随机下界是唯一不需要任何
信道信息、也不需要任何优化的对照，主算法必须显著优于它，否则增益无法归因。

设计约束：
* 动作必须**合法**（完整配对、块初几何候选、业务资格、端口、逐模式功率域），
  否则它就是不可执行动作，比较没有意义；
* 抽样对"方向/模式/配对/功率"是一致随机的，不偏向任何模式；
* 功率在 log 域均匀，使"贴下界"和"顶上限"两侧都不会被系统性高估；
* 每个块只抽 **一个** 动作。取多个再挑最好的会变成带选择偏差的"随机搜索"，
  不是下界；多抽几次再平均也不行，因为行级输出只能记录一个动作。

并入方式（等当前批量跑完、`source_sha256` 不再被占用后）：
1. 把本文件的 ``random_legal_action`` 放进 ``code/p3learn/allocation.py``；
2. ``run_allocation.py`` 增加 ``--methods random``、``--random-seed``；
3. 决策分支：抽一个动作 → 用同一个 ``evaluator`` 求一次预测 → 不走
   ``protect_budget``（下界不享受预算守卫，也不要求 QoS 达标，违约要如实记录）；
4. ``audit_allocation_outputs.py`` 的 method 集合与 ``SUMMARY`` 计数同步加上 ``random``。
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from p3learn.allocation import MODES, Action, CoverageFailure, _domain


def random_legal_action(n, cfg, candidates, business, ports, rng, *, modes=MODES):
    """在块初合法域上均匀抽一个完整配对动作。

    ``rng`` 必须是 ``np.random.Generator``（不要用全局状态），这样同块同种子可复现。
    """
    masks, port_copy = _domain(n, cfg, candidates, business, ports)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            options = [(tx, rx, mode) for tx in (i, j) for rx in (i, j) if tx != rx
                       for mode in modes if masks[mode][tx, rx]]
            if options:
                pick = options[int(rng.integers(len(options)))]
                graph.add_edge(i, j, weight=float(rng.random()), link=pick)
    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    if 2 * len(matching) != n:
        raise CoverageFailure("coverage_failure: no complete matching for the random baseline")
    links = sorted((tuple(graph.edges[i, j]["link"]) for i, j in matching),
                   key=lambda link: min(link[:2]))
    powers = []
    for _i, _j, mode in links:
        m = getattr(cfg, mode.lower())
        lo, hi = np.log10(m.power_min_w), np.log10(m.power_max_w)
        powers.append(float(10.0 ** rng.uniform(lo, hi)))
    return Action(links, np.asarray(powers, float), port_copy)
