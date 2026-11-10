"""匹配基线的契约：可行性筛除、2-交换稳定性、偏好表的方向/模式选择。

这些测试只覆盖 ``p3learn/matching_ee.py`` 里可以脱离场景构造的部分；
整块行为的冒烟由 run_allocation 的 2 块运行负责。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from p3base.config import FoundationConfig
from p3learn.matching_ee import (
    _INFEASIBLE, edge_ee_scores, feasible_edges, preference_from_scores,
    preference_tables, swap_refine, matching_assignment,
)


def domain(n=4, gain_rf=1e-3, gain_vlc=1e-5):
    cfg = FoundationConfig()
    business = ~np.eye(n, dtype=bool)
    candidates = {mode: business.copy() for mode in ("RF", "VLC")}
    ports = {name: np.zeros((n, n), dtype=int) for name in ("tx", "rx")}
    gains = {"RF": np.full((n, n), gain_rf), "VLC": np.full((n, n), gain_vlc)}
    masks = {mode: business.copy() for mode in ("RF", "VLC")}
    return cfg, candidates, business, ports, gains, masks


def test_feasible_edges_rejects_links_that_cannot_reach_qos_at_max_power():
    cfg, _c, _b, ports, gains, masks = domain()
    # 极小 VLC 增益：功率上限下也到不了 1 Mbit/s；RF 增益正常，能到。
    gains["VLC"] = np.full((4, 4), 1e-30)
    ok = feasible_edges(4, cfg, ports, masks, gains, 1e6)
    assert not ok["VLC"].any()
    assert ok["RF"][masks["RF"]].all()


def test_infeasible_edges_get_finite_bad_score_not_minus_inf():
    cfg, _c, _b, ports, gains, masks = domain()
    gains["VLC"] = np.full((4, 4), 1e-30)
    ok = feasible_edges(4, cfg, ports, masks, gains, 1e6)
    best = edge_ee_scores(4, cfg, ports, gains, (0.01, 0.3, 3.0), feasible=ok)
    assert np.isfinite(best["VLC"]).all()          # decode_complete_pairing 要求有限
    assert (best["VLC"] <= _INFEASIBLE).all()


def test_preference_picks_best_direction_and_mode():
    # (0,1) 上 VLC 更好但只有 1->0 合法；(2,3) 上 RF 更好且 3->2 合法。
    scores = np.full((4, 4), -np.inf)
    scores[0, 1] = 5.0          # RF 1->0
    scores[1, 0] = 50.0
    scores[2, 3] = 9.0          # VLC 3->2
    scores[3, 2] = 9.0
    value, choice = preference_from_scores(
        4, {("RF", 1, 0): 50.0, ("VLC", 0, 1): 5.0, ("VLC", 3, 2): 9.0, ("VLC", 2, 3): 1.0})
    assert choice[(0, 1)] == (1, 0, "RF")
    assert choice[(2, 3)] == (3, 2, "VLC")
    assert value[0, 1] == pytest.approx(50.0)
    assert value[2, 3] == pytest.approx(9.0)


def test_preference_skips_infeasible_entries():
    value, choice = preference_from_scores(
        4, {("RF", 0, 1): _INFEASIBLE, ("VLC", 0, 1): 3.0})
    assert choice[(0, 1)] == (0, 1, "VLC")


def test_swap_refine_only_accepts_four_way_improvement_and_is_stable():
    n = 4
    value = np.zeros((n, n))
    # 端点对 (0,1)+(2,3) 的当前值较低；重连成 (0,2)+(1,3) 后四个端点都更高。
    v = {(0, 1): 1.0, (2, 3): 1.0, (0, 2): 5.0, (1, 3): 5.0, (0, 3): 0.0, (1, 2): 0.0}
    for (i, j), s in v.items():
        value[i, j] = value[j, i] = s
    edges = swap_refine([(0, 1), (2, 3)], value)
    assert sorted(edges) == [(0, 2), (1, 3)]
    # 再跑一次不应再有变化（已稳定）。
    assert sorted(swap_refine(edges, value)) == [(0, 2), (1, 3)]


def test_swap_refine_rejects_three_way_improvement():
    n = 4
    value = np.zeros((n, n))
    # (0,2)+(1,3) 只有三个端点变好，第 4 个变差 -> 不接受。
    v = {(0, 1): 1.0, (2, 3): 1.0, (0, 2): 5.0, (1, 3): 5.0, (0, 3): 0.0, (1, 2): 0.0}
    for (i, j), s in v.items():
        value[i, j] = value[j, i] = s
    value[2, 3] = value[3, 2] = 100.0     # 端点 2、3 不愿离开原配对
    edges = swap_refine([(0, 1), (2, 3)], value)
    assert sorted(edges) == [(0, 1), (2, 3)]


def test_matching_assignment_returns_complete_matching_without_mutating_inputs():
    cfg, candidates, business, ports, gains, _masks = domain(gain_rf=1e-2, gain_vlc=1e-4)
    ports_before = {k: v.copy() for k, v in ports.items()}
    cand_before = {k: v.copy() for k, v in candidates.items()}
    links, powers, out_ports = matching_assignment(
        4, cfg, candidates, business, ports, gains, power_solver="dinkelbach",
        rounds=2, q_min_bps=0.0)
    assert len(links) == 2
    assert len(powers) == 2
    endpoints = sorted(x for tx, rx, _m in links for x in (tx, rx))
    assert endpoints == [0, 1, 2, 3]
    assert np.isfinite(powers).all()
    for key in ("tx", "rx"):
        assert np.array_equal(ports[key], ports_before[key])
    for key in ("RF", "VLC"):
        assert np.array_equal(candidates[key], cand_before[key])
    assert out_ports is not ports
