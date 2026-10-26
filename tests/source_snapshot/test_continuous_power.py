"""Test continuous power, not just monotonicity of a grid fallback."""
import numpy as np
import pytest
from scipy.optimize import brentq
from dataclasses import replace
from test_dinkelbach import scenario as original_scenario

def scenario():
    cfg,links,ports,gains = original_scenario()
    return cfg.with_(vlc=replace(cfg.vlc,sinr_model="square_law")),links,ports,gains
from p3learn.allocation import power_ascent, _power_solve
from p3learn.continuous_power import continuous_sum_ee, surrogate
from p3learn.dinkelbach_sca import _Physics


def test_single_rf_reaches_independently_computed_interior_optimum():
    cfg, _, ports, gains = scenario()
    links = [(0, 1, 'RF')]
    gains['RF'][0, 1] = 30*cfg.rf.noise_w*cfg.rf.gap
    c, pc = 30., cfg.rf.circuit_power_w
    root = brentq(lambda p: c*(pc+p)/(1+c*p)-np.log1p(c*p), .001, 2.)
    base = power_ascent(links, gains, ports, cfg, sweeps=2, points=8)
    got, info = continuous_sum_ee(links, gains, ports, cfg, p_start=base, max_rounds=100)
    assert got[0] == pytest.approx(root, rel=.003)
    assert info.final_mean_ee > info.start_mean_ee*1.001
    grid = np.logspace(-3, np.log10(2), 8)
    assert np.min(abs(got[0]-grid)) > .001


def test_unclipped_bound_is_tight_lower_and_has_correct_jacobian():
    cfg, links, ports, gains = scenario()
    phys = _Physics(links, gains, ports, cfg)
    p = np.array([.05, .02, .6, .15])
    bound = surrogate(phys, p)
    value, jac = bound(p)
    assert np.allclose(value, phys.x(p)**2/(phys.gap*phys.D(p)), rtol=1e-12)
    for j in range(len(p)):
        d = np.zeros(len(p)); d[j] = 1e-7
        assert np.allclose(jac[:, j], (bound(p+d)[0]-bound(p-d)[0])/2e-7, rtol=1e-5, atol=1e-7)
    rng = np.random.default_rng(7)
    for _ in range(50):
        a, b = rng.uniform(.001, 2, (2, len(p)))
        av, bv = bound(a)[0], bound(b)[0]
        assert np.all(bound((a+b)/2)[0] >= (av+bv)/2-1e-8)
        assert np.all(av <= phys.x(a)**2/(phys.gap*phys.D(a))+1e-8)


def test_mixed_link_solver_moves_and_keeps_original_objective():
    cfg, links, ports, gains = scenario()
    base = power_ascent(links, gains, ports, cfg, sweeps=2, points=8)
    got, info = continuous_sum_ee(links, gains, ports, cfg, p_start=base)
    assert info.final_mean_ee >= info.start_mean_ee
    assert np.all(np.diff(info.history) > 0)
    assert np.max(abs(got-base)) > 1e-4
    assert np.all(got >= .001-1e-12) and np.all(got <= 2+1e-12)
    assert all(x['scaled_constraint_violation'] <= 1e-7 for x in info.subproblems if x['accepted'])


def test_dispatch_records_numerical_solver_status():
    cfg, links, ports, gains = scenario()
    log = []
    p = _power_solve(links, gains, ports, cfg, 'continuous', solver_log=log)
    assert len(log) == 1 and log[0]['subproblems']
    assert log[0]['final_mean_ee'] >= log[0]['start_mean_ee']
    assert np.isfinite(p).all()


def test_failed_subproblem_is_reported_and_not_accepted(monkeypatch):
    from types import SimpleNamespace
    import p3learn.continuous_power as module
    cfg, links, ports, gains = scenario()
    base = np.full(4, .01)
    monkeypatch.setattr(module, 'minimize', lambda f,x0,**kw: SimpleNamespace(
        x=x0, success=False, status=9, message='forced failure', nit=1))
    p, info = continuous_sum_ee(links, gains, ports, cfg, p_start=base)
    assert np.array_equal(p,base)
    assert info.status == 'subproblem_failed'
    assert not info.subproblems[0]['accepted']


def test_full_search_preserves_grid_forecast_incumbent_and_uses_continuous():
    from test_allocation import gain_domain, prediction
    from p3learn.allocation import pure_ee_reference
    cfg, candidates, business, ports, gains = gain_domain()
    def evaluate(action):
        phys = _Physics(action.links, gains, ports, cfg)
        # An intentionally perturbed forecast makes acceptance use the actual
        # injected evaluator, rather than assuming the proxy and forecast agree.
        return prediction(phys.rates(action.powers)*np.linspace(.5, 1.5, len(action.links)))
    kw = dict(belief_gains=gains, search='joint', joint_rounds=2)
    base = pure_ee_reference(6,cfg,candidates,business,ports,evaluate,power_solver='ascent',**kw)
    got = pure_ee_reference(6,cfg,candidates,business,ports,evaluate,power_solver='continuous',**kw)
    assert base.success and got.success
    assert got.mean_pair_ee >= base.mean_pair_ee
    assert got.policy_config['grid_incumbent_mean_ee'] == base.mean_pair_ee
    assert got.policy_config['continuous_solver_log']
    assert '+continuous-qt-sca-v1' in got.reference_version


def test_qos_restores_power_and_unreachable_qos_is_not_success():
    cfg, _, ports, gains = scenario()
    links = [(0,1,'RF')]
    gains['RF'][0,1] = 30*cfg.rf.noise_w*cfg.rf.gap
    phys = _Physics(links,gains,ports,cfg)
    q = float(phys.rates(np.array([.5]))[0])
    p, info = continuous_sum_ee(links,gains,ports,cfg,p_start=np.array([.001]),q_min_bps=q)
    assert phys.rates(p)[0] >= q*(1-1e-8)
    assert p[0] >= .5*(1-1e-8)
    p, info = continuous_sum_ee(links,gains,ports,cfg,p_start=np.array([.001]),q_min_bps=1e12)
    assert info.status == 'qos_not_found'


def test_final_forecast_qos_is_checked_not_only_proxy():
    from test_allocation import gain_domain, prediction
    from p3learn.allocation import pure_ee_reference
    cfg,candidates,business,ports,gains = gain_domain()
    got = pure_ee_reference(6,cfg,candidates,business,ports,
                           lambda action:prediction(np.zeros(len(action.links))),
                           belief_gains=gains,power_solver='continuous',q_min_bps=1e6)
    assert not got.success
    assert 'qos_not_found' in got.reason


def test_chance_targets_only_escalate_short_links():
    from p3learn.allocation import chance_targets
    low = np.array([80e6, 0.5e6])
    mean = np.array([100e6, 2e6])
    fair = chance_targets(low, mean, 1e6)
    # The healthy link is held at the constraint it was judged against, not at
    # its own mean; the short link is escalated by Q/low times its mean.
    assert fair[0] == pytest.approx(1e6)
    assert fair[1] == pytest.approx(4e6)
    pinned = chance_targets(low, mean, 1e6, hold_mean=True)
    assert pinned[0] == pytest.approx(100e6)
    assert pinned[1] == pytest.approx(4e6)
    # Ratio is capped so a near-zero prediction cannot demand unbounded rate.
    assert chance_targets(np.array([1.0]), np.array([2e6]), 1e6)[0] == pytest.approx(200e6)
