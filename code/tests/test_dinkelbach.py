"""分式和 Dinkelbach(二次变换)+SCA：与共享物理一致、雅可比正确、单调不回退。"""
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from p3base.config import FoundationConfig
from p3learn.allocation import (_link_gain_tables, _power_solve, _rates_and_ee,
                                 _rates_and_ee_batch)
from p3learn.dinkelbach_sca import _Physics, dinkelbach_sum_ee


def scenario(n=8):
    """2 RF + 2 VLC，非等功率，交叉增益不对称（与 allocation 的端口用例同源）。"""
    cfg = FoundationConfig()
    links = [(0, 1, "RF"), (2, 3, "RF"), (4, 5, "VLC"), (6, 7, "VLC")]
    ports = {"tx": np.zeros((n, n), int), "rx": np.zeros((n, n), int)}
    ports["tx"][4, 5], ports["rx"][4, 5] = 0, 1
    ports["tx"][6, 7], ports["rx"][6, 7] = 1, 0
    rf = np.zeros((n, n))
    rf[0, 1], rf[2, 3] = 2.2e-7, 1.6e-7
    rf[0, 3], rf[2, 1] = 1.36e-8, 1.0e-10
    vlc = np.zeros((n, n, 2, 2))
    vlc[4, 5, 0, 1] = 3.0e-5
    vlc[6, 7, 1, 0] = 1.1e-5
    vlc[6, 5, 1, 1] = 4.0e-6
    vlc[4, 7, 0, 0] = 1.5e-7
    return cfg, links, ports, {"RF": rf, "VLC": vlc}


@pytest.mark.parametrize("model", ["linear", "square_law"])
def test_rates_and_mean_ee_use_the_shared_physics(model):
    cfg, links, ports, gains = scenario()
    cfg = cfg.with_(vlc=replace(cfg.vlc, sinr_model=model))
    p = np.array([.077, .001, .3, .05])
    s, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    rates, ee = _rates_and_ee(p, links, cfg, s, h_rf, h_vlc, h_opt)
    phys = _Physics(links, gains, ports, cfg)
    assert np.allclose(phys.rates(p), rates, rtol=1e-12, atol=1e-9)
    assert np.mean(ee) == pytest.approx(phys.mean_ee(p), rel=1e-12)


@pytest.mark.parametrize("model", ["linear", "square_law"])
def test_batched_rates_match_the_scalar_path(model):
    """批量版只是性能优化，逐位必须与 ``_rates_and_ee`` 一致。"""
    cfg, links, ports, gains = scenario()
    cfg = cfg.with_(vlc=replace(cfg.vlc, sinr_model=model))
    s, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    p = np.array([.077, .001, .3, .05])
    grid = np.stack([p, p * 2.0, p * .5, np.full(4, .7)])
    rate_b, ee_b = _rates_and_ee_batch(grid, links, cfg, s, h_rf, h_vlc, h_opt)
    for row in range(len(grid)):
        rate, ee = _rates_and_ee(grid[row], links, cfg, s, h_rf, h_vlc, h_opt)
        assert np.allclose(rate_b[row], rate, rtol=1e-13, atol=1e-9), model
        assert np.allclose(ee_b[row], ee, rtol=1e-13, atol=1e-9), model


@pytest.mark.parametrize("model", ["linear", "square_law"])
def test_rate_jacobian_matches_finite_differences(model):
    """雅可比错一位就会让 SCA 子问题朝错误方向走，必须逐元素对拍。"""
    cfg, links, ports, gains = scenario()
    cfg = cfg.with_(vlc=replace(cfg.vlc, sinr_model=model))
    phys = _Physics(links, gains, ports, cfg)
    p = np.array([.05, .02, .6, .15])
    analytic = phys.rate_jac(p)
    numeric = np.zeros_like(analytic)
    h = 1e-6
    for c in range(len(links)):
        d = np.zeros(len(links))
        d[c] = h
        numeric[c] = (phys.rates(p + d) - phys.rates(p - d)) / (2 * h)
    assert np.allclose(analytic, numeric, rtol=2e-4, atol=1e-3), model


def test_solver_never_returns_a_power_vector_worse_than_coordinate_ascent():
    """SCA 只保证稳定点；``dinkelbach`` 分支必须保留较好者，否则会回退。"""
    from p3learn.allocation import power_ascent

    for model in ("linear", "square_law"):
        cfg, links, ports, gains = scenario()
        cfg = cfg.with_(vlc=replace(cfg.vlc, sinr_model=model))
        s, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
        base = power_ascent(links, gains, ports, cfg, sweeps=2, points=8)
        got = _power_solve(links, gains, ports, cfg, "dinkelbach")
        ee_base = np.mean(_rates_and_ee(base, links, cfg, s, h_rf, h_vlc, h_opt)[1])
        ee_got = np.mean(_rates_and_ee(got, links, cfg, s, h_rf, h_vlc, h_opt)[1])
        assert ee_got >= ee_base * (1 - 1e-12), model
        caps = np.array([min(getattr(cfg, m.lower()).power_max_w, 2.0) for _i, _j, m in links])
        assert np.all(got >= 0) and np.all(got <= caps + 1e-12)


def test_outer_loop_improves_or_holds_and_reports_a_stationary_point():
    cfg, links, ports, gains = scenario()
    cfg = cfg.with_(vlc=replace(cfg.vlc, sinr_model="square_law"))
    start = np.full(len(links), 0.05)
    powers, info = dinkelbach_sum_ee(links, gains, ports, cfg, p_start=start)
    assert info.status in ("stationary", "iteration_budget")
    assert info.mean_ee >= _Physics(links, gains, ports, cfg).mean_ee(start) * (1 - 1e-12)
    assert info.outer_rounds >= 1 and info.sca_rounds >= 1
    assert np.all(np.isfinite(powers))


def test_empty_matching_is_not_an_error():
    cfg = FoundationConfig()
    gains = {"RF": np.zeros((2, 2)), "VLC": np.zeros((2, 2, 2, 2))}
    ports = {"tx": np.zeros((2, 2), int), "rx": np.zeros((2, 2), int)}
    powers, info = dinkelbach_sum_ee([], gains, ports, cfg)
    assert powers.size == 0 and info.status == "empty"
