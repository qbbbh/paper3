"""小规模验收：物理对拍、干扰更新、分式残差、目标身份、2-交换契约。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines._path import ensure_frozen_on_path, METHOD_ABOAGYE, METHOD_ZAPPONE
from baselines.aboagye_matching_qt import (
    _INFEASIBLE, aboagye_matching_qt, interference_scores, preference_from_scores,
    swap_refine, feasible_edges, edge_ee_scores,
)
from baselines.zappone_mmee_fp import zappone_mmee_fp_fixed

ensure_frozen_on_path()
from p3base.config import FoundationConfig
from p3learn.allocation import Action, _link_gain_tables, _rates_and_ee


def domain(n=4, gain_rf=1e-2, gain_vlc=1e-4):
    cfg = FoundationConfig()
    business = ~np.eye(n, dtype=bool)
    candidates = {mode: business.copy() for mode in ("RF", "VLC")}
    ports = {name: np.zeros((n, n), dtype=int) for name in ("tx", "rx")}
    gains = {"RF": np.full((n, n), float(gain_rf)), "VLC": np.full((n, n), float(gain_vlc))}
    masks = {mode: business.copy() for mode in ("RF", "VLC")}
    return cfg, candidates, business, ports, gains, masks


def test_feasible_edges_and_finite_infeasible_scores():
    cfg, _c, _b, ports, gains, masks = domain()
    gains["VLC"] = np.full((4, 4), 1e-30)
    ok = feasible_edges(4, cfg, ports, masks, gains, 1e6)
    assert not ok["VLC"].any()
    best = edge_ee_scores(4, cfg, ports, gains, (0.01, 0.3, 3.0), feasible=ok)
    assert np.isfinite(best["VLC"]).all()
    assert (best["VLC"] <= _INFEASIBLE).all()


def test_swap_refine_reports_local_stable():
    value = np.zeros((4, 4))
    for (i, j), s in {(0, 1): 1.0, (2, 3): 1.0, (0, 2): 5.0, (1, 3): 5.0}.items():
        value[i, j] = value[j, i] = s
    edges, info = swap_refine([(0, 1), (2, 3)], value)
    assert sorted(edges) == [(0, 2), (1, 3)]
    assert info["local_stable"] is True
    assert info["budget_exhausted"] is False


def test_interference_updates_when_power_changes():
    cfg, candidates, business, ports, gains, masks = domain(gain_rf=1e-2)
    links = [(0, 1, "RF"), (2, 3, "RF")]
    p_lo = np.array([0.05, 0.05])
    p_hi = np.array([0.05, 5.0])
    pbar = {"RF": 0.3, "VLC": 0.3}
    s0 = interference_scores(4, cfg, ports, masks, gains, links, p_lo, pbar)
    s1 = interference_scores(4, cfg, ports, masks, gains, links, p_hi, pbar)
    # 链路2功率增大后，对接收端1的干扰应上升，从而 (RF,0,1) 的 EE 下降或不变
    assert s1[("RF", 0, 1)] <= s0[("RF", 0, 1)] + 1e-12


def test_aboagye_matching_covers_all_endpoints_and_keeps_inputs():
    cfg, candidates, business, ports, gains, _m = domain()
    ports_before = {k: v.copy() for k, v in ports.items()}
    links, powers, out_ports, info = aboagye_matching_qt(
        4, cfg, candidates, business, ports, gains,
        power_solver="dinkelbach", rounds=2, q_min_bps=0.0)
    assert 2 * len(links) == 4
    assert np.isfinite(powers).all()
    assert info.method == METHOD_ABOAGYE
    assert info.power_solver_calls >= 1
    assert np.array_equal(ports["tx"], ports_before["tx"])
    # 连续功率：不应全落在标称网格三档（允许偶然相等，但至少有限）
    assert powers.min() >= cfg.rf.power_min_w - 1e-12


def test_zappone_single_link_improves_or_keeps_min_ee():
    cfg, candidates, business, ports, gains, _m = domain(n=2, gain_rf=5e-3)
    links = [(0, 1, "RF")]
    p0 = np.array([cfg.rf.power_max_w * 0.9])  # 过高功率通常 EE 差
    p1, info = zappone_mmee_fp_fixed(links, gains, ports, cfg, p_start=p0,
                                     q_min_bps=0.0, max_outer=12, max_sca=5)
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    ee0 = float(_rates_and_ee(p0, links, cfg, s_gain, h_rf, h_vlc, h_opt)[1].min())
    ee1 = float(_rates_and_ee(p1, links, cfg, s_gain, h_rf, h_vlc, h_opt)[1].min())
    assert info.method == METHOD_ZAPPONE
    assert ee1 + 1e-9 >= ee0  # 允许持平（退回起点），不允许变差
    assert cfg.rf.power_min_w - 1e-12 <= p1[0] <= cfg.rf.power_max_w + 1e-12


def test_zappone_two_link_fixed_config_identity():
    """MMEE-FP 优化最低 EE，不改匹配。"""
    cfg, candidates, business, ports, gains, _m = domain(gain_rf=1e-2)
    # 非对称增益：让最弱链路明确
    gains["RF"] = np.full((4, 4), 1e-2)
    gains["RF"][0, 1] = 2e-3
    links = [(0, 1, "RF"), (2, 3, "RF")]
    p0 = np.array([1.0, 1.0])
    p1, info = zappone_mmee_fp_fixed(links, gains, ports, cfg, p_start=p0,
                                     q_min_bps=0.0, max_outer=15)
    assert len(p1) == 2
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    ee0 = _rates_and_ee(p0, links, cfg, s_gain, h_rf, h_vlc, h_opt)[1]
    ee1 = _rates_and_ee(p1, links, cfg, s_gain, h_rf, h_vlc, h_opt)[1]
    assert float(ee1.min()) + 1e-9 >= float(ee0.min())
    assert "no_delta_budget" in info.notes
    assert "fixed_configuration" in info.notes


def test_preference_skips_infeasible():
    value, choice = preference_from_scores(
        4, {("RF", 0, 1): _INFEASIBLE, ("VLC", 0, 1): 3.0})
    assert choice[(0, 1)] == (0, 1, "VLC")
