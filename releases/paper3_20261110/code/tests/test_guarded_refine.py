"""Behavioral checks for the prediction-only post-stage and its hard guards."""
from types import SimpleNamespace

import numpy as np
import pytest

from p3base.config import FoundationConfig
from p3base.energy import link_electrical_power_w
from p3learn.allocation import Action
from p3learn import guarded_refine as gr


def _case():
    cfg = FoundationConfig()
    ports = {"tx": np.zeros((4, 4), int), "rx": np.zeros((4, 4), int)}
    action = Action([(0, 1, "RF"), (2, 3, "VLC")], np.array([0.04, 0.5]), ports)
    return cfg, action


def _forecast_for_ee(action, cfg, ee, spread=0.1):
    rates = np.asarray(ee) * link_electrical_power_w(action.powers, ["RF", "VLC"], cfg)
    samples = np.stack([rates * (1.0-spread), rates * (1.0+spread)])
    return dict(mean_rate_bps=rates, block_rate_samples_bps=samples,
                low_rate_bps=np.quantile(samples, .05, axis=0),
                high_rate_bps=np.quantile(samples, .95, axis=0))


def _patch_proposer(monkeypatch, powers):
    info = SimpleNamespace(to_dict=lambda: {"status": "test_proposal"})
    monkeypatch.setattr(gr, "bottleneck_sca", lambda *a, **kw: (powers.copy(), info))
    # Repairs must remain independently forecast-checked; keep this test stub
    # equal to the proposal to avoid dependence on unrelated analytic physics.
    monkeypatch.setattr("p3learn.allocation._raise_targets",
                        lambda links, gains, ports, cfg, p, q: p.copy())


def _run(cfg, action, forecast, evaluator, **kwargs):
    return gr.improve_fair_power(action, forecast, cfg, evaluator,
                                delta=0.1, eta_ref=55e6, belief_gains={}, **kwargs)


def test_all_intermediate_steps_checked_and_best_prediction_retained(monkeypatch):
    cfg, action = _case()
    original_power = action.powers.copy()
    original = _forecast_for_ee(action, cfg, [10e6, 100e6])
    _patch_proposer(monkeypatch, action.powers * 4)
    seen = []

    def evaluator(proposed):
        assert proposed.links == action.links
        np.testing.assert_array_equal(proposed.ports["tx"], action.ports["tx"])
        np.testing.assert_array_equal(proposed.ports["rx"], action.ports["rx"])
        u = np.log(proposed.powers[0] / original_power[0]) / np.log(4.0)
        seen.append(float(u))
        # Maximum at the midpoint. A stop-at-first-feasible implementation
        # would miss it, and no execution/truth quantity enters this function.
        return _forecast_for_ee(proposed, cfg, [(10+20*np.sin(np.pi*u))*1e6, 100e6])

    out, forecast, audit = _run(cfg, action, original, evaluator)
    np.testing.assert_allclose(out.powers, original_power * 2, rtol=1e-12)
    np.testing.assert_array_equal(action.powers, original_power)
    assert len(seen) == 5
    assert audit["accepted"] and audit["final_original_constraints_passed"]
    assert audit["final"]["min_ee"] == pytest.approx(30e6)
    # A returned improvement must own its links and ports too, not just powers.
    out.links.append((1, 0, "RF"))
    out.ports["tx"][0, 1] = 1
    assert len(action.links) == 2 and action.ports["tx"][0, 1] == 0


@pytest.mark.parametrize("failure", ["mean_budget", "calibrated_service", "minimum_ee"])
def test_improving_proxy_cannot_bypass_original_constraints(monkeypatch, failure):
    cfg, action = _case()
    original = _forecast_for_ee(action, cfg, [10e6, 100e6])
    _patch_proposer(monkeypatch, action.powers * 2)
    ee = {"mean_budget": [20e6, 30e6],
          "calibrated_service": [30e6, 90e6],
          "minimum_ee": [5e6, 120e6]}[failure]
    spread = 1.0 if failure == "calibrated_service" else 0.1

    def evaluator(proposed):
        return _forecast_for_ee(proposed, cfg, ee, spread=spread)

    out, _, audit = _run(cfg, action, original, evaluator,
                        qos={"q_min_bps": 1e6, "metric": "chance", "inflation": 1.35})
    np.testing.assert_array_equal(out.powers, action.powers)
    assert not audit["accepted"] and audit["retained_incumbent"]
    assert audit["final_original_constraints_passed"]
    if failure == "mean_budget":
        assert all(not record["budget_ok"] for record in audit["evaluations"])
    elif failure == "calibrated_service":
        assert all(not record["service_ok"] for record in audit["evaluations"])


def test_retaining_mean_fallback_does_not_certify_chance_feasibility(monkeypatch):
    cfg, action = _case()
    original = _forecast_for_ee(action, cfg, [10e6, 100e6], spread=1.0)
    _patch_proposer(monkeypatch, action.powers * 2)
    out, _, audit = _run(
        cfg, action, original,
        lambda p: _forecast_for_ee(p, cfg, [15e6, 100e6], spread=1.0),
        qos={"q_min_bps": 1e6, "metric": "chance", "inflation": 1.35})
    np.testing.assert_array_equal(out.powers, action.powers)
    assert audit["retained_incumbent"]
    assert not audit["original"]["service_ok"]
    assert not audit["final_original_constraints_passed"]


@pytest.mark.parametrize("stage", ["solver", "evaluator"])
def test_extra_search_failure_keeps_incumbent(monkeypatch, stage):
    cfg, action = _case()
    original = _forecast_for_ee(action, cfg, [10e6, 100e6])
    _patch_proposer(monkeypatch, action.powers * 2)

    def fail(*args, **kwargs):
        raise FloatingPointError("controlled numerical failure")

    if stage == "solver":
        monkeypatch.setattr(gr, "bottleneck_sca", fail)
    out, _, audit = _run(cfg, action, original, fail)
    np.testing.assert_array_equal(out.powers, action.powers)
    assert audit["retained_incumbent"] and audit["errors"]


def test_no_posterior_means_means_no_extra_search(monkeypatch):
    cfg, action = _case()
    original = _forecast_for_ee(action, cfg, [10e6, 100e6])

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing-posterior branch must not evaluate new actions")

    monkeypatch.setattr(gr, "bottleneck_sca", forbidden)
    out, _, audit = gr.improve_fair_power(
        action, original, cfg, forbidden, delta=.1, eta_ref=55e6, belief_gains=None)
    np.testing.assert_array_equal(out.powers, action.powers)
    assert audit["evaluator_calls"] == 0 and audit["retained_incumbent"]


def test_augment_updates_final_service_count_preserves_initial_fallback(monkeypatch):
    from p3learn.allocation import AllocationResult, _metrics

    cfg, action = _case()
    original = _forecast_for_ee(action, cfg, [10e6, 100e6], spread=1.0)
    network, mean, minimum = _metrics(action, original, cfg)
    result = AllocationResult(action, original, mean, network, minimum,
                              mean_pair_ee=mean, qos_violation_links=1,
                              fallback=True, delta=.1, evaluator_calls=12,
                              policy_config={"chance_fallback": True})
    _patch_proposer(monkeypatch, action.powers*2)
    out = gr.augment_result(
        result, cfg, lambda a: _forecast_for_ee(a, cfg, [30e6, 100e6]),
        delta=.1, qos={"q_min_bps": 1e6, "metric": "chance", "inflation": 1.35},
        belief_gains={})
    assert out.qos_violation_links == 0 and not out.fallback
    assert out.policy_config["chance_fallback"]  # original reference history
    assert out.policy_config["guarded_post_stage"]["final_original_constraints_passed"]
    assert result.qos_violation_links == 1 and result.fallback
    assert out.evaluator_calls > result.evaluator_calls


@pytest.mark.parametrize("allocator_name", ["max_min_allocate", "same_topo_maxmin_allocate"])
def test_complete_allocator_uses_old_nm_then_same_post_stage_as_replay(monkeypatch, allocator_name):
    import p3learn.allocation as alloc

    cfg, reference_action = _case()
    base_power = reference_action.powers.copy()
    old_nm_action = alloc.copy_action(reference_action)
    old_nm_action.powers[:] *= 2
    business = ~np.eye(4, dtype=bool)
    candidates = {mode: business.copy() for mode in ("RF", "VLC")}
    events = []

    def evaluator(action):
        position = np.log(action.powers[0]/base_power[0])/np.log(2.0)
        return _forecast_for_ee(action, cfg, [(10+5*position)*1e6, 100e6])

    reference_forecast = evaluator(reference_action)
    network, mean, minimum = alloc._metrics(reference_action, reference_forecast, cfg)
    reference = alloc.AllocationResult(
        reference_action, reference_forecast, mean, network, minimum,
        mean_pair_ee=mean, evaluator_id=alloc._evaluator_identity(evaluator))

    def old_nm(action, forecast, cfg, evaluator, **kwargs):
        assert kwargs["solver"] == "nm"
        events.append("old_nm")
        return alloc.copy_action(old_nm_action), evaluator(old_nm_action), 1

    def structured(links, gains, ports, cfg, *, p_start, **kwargs):
        np.testing.assert_array_equal(p_start, old_nm_action.powers)
        events.append("structured_from_old_nm")
        return p_start*2, SimpleNamespace(to_dict=lambda: {"status": "test_proposal"})

    monkeypatch.setattr(alloc, "maxmin_power_refine", old_nm)
    monkeypatch.setattr(alloc, "_proxy_scores",
                        lambda *a, **kw: {m: np.ones((4, 4)) for m in ("RF", "VLC")})
    monkeypatch.setattr(alloc, "decode_complete_pairing",
                        lambda *a, **kw: alloc.copy_action(reference_action))
    monkeypatch.setattr(alloc, "rebalance_proposal",
                        lambda *a, **kw: alloc.copy_action(reference_action))
    monkeypatch.setattr(gr, "bottleneck_sca", structured)
    kwargs = dict(delta=.1, reference=reference, belief_gains={}, power_refine=True)
    allocator = getattr(alloc, allocator_name)
    old = allocator(4, cfg, candidates, business, reference_action.ports,
                    evaluator, refine_solver="nm", **kwargs)
    assert old.success
    np.testing.assert_array_equal(old.action.powers, old_nm_action.powers)
    events.clear()
    full = allocator(4, cfg, candidates, business, reference_action.ports,
                     evaluator, refine_solver="guarded", **kwargs)
    assert events == ["old_nm", "structured_from_old_nm"]
    replay_action, replay_forecast, replay_audit = gr.improve_fair_power(
        old.action, old.forecast, cfg, evaluator, delta=.1, eta_ref=reference.eta_ref,
        belief_gains={})
    np.testing.assert_array_equal(full.action.powers, replay_action.powers)
    np.testing.assert_array_equal(full.forecast["mean_rate_bps"], replay_forecast["mean_rate_bps"])
    assert full.min_pair_ee == pytest.approx(replay_audit["final"]["min_ee"])
    assert full.policy_config["guarded_post_stage"]["accepted"]
