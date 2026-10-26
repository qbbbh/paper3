"""Counterfactual credit must compare only legal, complete replacements."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3base.block import validate_action
from p3base.config import FoundationConfig
from p3learn.allocation import Action, protect_budget, pure_ee_reference
from p3learn.credit import credit_report, legal_replacements

N = 6


def domain(n=N):
    cfg = FoundationConfig()
    index = np.indices((n, n))
    candidates = {'RF': ~np.eye(n, dtype=bool), 'VLC': ~np.eye(n, dtype=bool)}
    business = ~np.eye(n, dtype=bool)
    ports = dict(tx=index[0] % 2, rx=index[1] % 2)
    return cfg, candidates, business, ports


def evaluator():
    """Deterministic rate model; not a physical predictor, only a test double."""
    calls = []

    def evaluate(action):
        calls.append(len(action.links))
        rates = np.array([1e6 * (1 + (tx + rx) % 3) * (1.2 if mode == 'VLC' else 1.)
                          for tx, rx, mode in action.links], float)
        return dict(mean_rate_bps=rates, block_rate_samples_bps=np.stack([rates, rates]),
                    low_rate_bps=rates * .99, high_rate_bps=rates * 1.01, draws=2)

    evaluate.calls = calls
    return evaluate


def reference_action():
    cfg, candidates, business, ports = domain()
    result = pure_ee_reference(N, cfg, candidates, business, ports, evaluator())
    assert result.success
    return result


def test_replacements_preserve_complete_coverage_and_legality():
    cfg, candidates, business, ports = domain()
    result = reference_action()
    found = legal_replacements(result.action, n=N, cfg=cfg, candidates=candidates,
                               business=business, ports_master=ports)
    assert found.items and found.generated >= len(found.items)
    assert {'mode', 'direction', 'partner', 'power'} <= {item.kind for item in found.items}
    for item in found.items:
        action = item.action
        validate_action(action.links, action.powers, action.ports, N, cfg,
                        candidates=candidates, business=business)
        endpoints = sorted(v for tx, rx, _ in action.links for v in (tx, rx))
        assert endpoints == list(range(N))
        for tx, rx, mode in action.links:
            if mode == 'VLC':
                assert action.ports['tx'][tx, rx] == ports['tx'][tx, rx]
                assert action.ports['rx'][tx, rx] == ports['rx'][tx, rx]


def test_actor_style_zero_ports_are_replaced_by_the_master_map():
    # An actor-sampled Action leaves every non-VLC port entry at zero.  Reusing
    # it would silently evaluate VLC replacements at port pair (0, 0).
    cfg, candidates, business, ports = domain()
    sampled = reference_action().action
    sampled = Action(list(sampled.links), sampled.powers.copy(),
                     {'tx': np.zeros((N, N), int), 'rx': np.zeros((N, N), int)})
    found = legal_replacements(sampled, n=N, cfg=cfg, candidates=candidates,
                               business=business, ports_master=ports)
    vlc_links = [(tx, rx) for item in found.items for tx, rx, mode in item.action.links
                 if mode == 'VLC']
    assert vlc_links
    for item in found.items:
        for tx, rx, mode in item.action.links:
            if mode == 'VLC':
                assert (item.action.ports['tx'][tx, rx], item.action.ports['rx'][tx, rx]) == (
                    ports['tx'][tx, rx], ports['rx'][tx, rx])


def test_power_replacements_stay_inside_limits_and_change_one_slot():
    cfg, candidates, business, ports = domain()
    action = reference_action().action
    found = legal_replacements(action, n=N, cfg=cfg, candidates=candidates,
                               business=business, ports_master=ports)
    power_items = [item for item in found.items if item.kind == 'power']
    assert power_items
    for item in power_items:
        changed = np.flatnonzero(np.abs(item.action.powers - action.powers) > 1e-12)
        assert len(changed) == 1
        index = int(changed[0])
        cap = cfg.rf.power_max_w if action.links[index][2] == 'RF' else cfg.vlc.power_max_w
        assert .001 <= item.action.powers[index] <= cap


def test_cap_truncates_deterministically_and_counts_evaluations():
    cfg, candidates, business, ports = domain()
    result = reference_action()
    first, calls_first = credit_report(
        result.action, result.forecast, n=N, cfg=cfg, candidates=candidates,
        business=business, ports_master=ports, evaluator=evaluator(),
        block=3, forecast_seed=99, delta=.1, eta_ref=result.eta_ref, max_evals=3)
    second, calls_second = credit_report(
        result.action, result.forecast, n=N, cfg=cfg, candidates=candidates,
        business=business, ports_master=ports, evaluator=evaluator(),
        block=3, forecast_seed=99, delta=.1, eta_ref=result.eta_ref, max_evals=3)
    assert first == second
    assert calls_first == calls_second == 3
    assert first['counts']['truncated'] and len(first['replacements']) == 3
    assert first['information'] == 'conditional_trajectory_forecast_only'


def test_credit_does_not_change_budget_guard_or_input_action():
    cfg, candidates, business, ports = domain()
    result = reference_action()
    powers_before = result.action.powers.copy()
    before = protect_budget(result.action, result, delta=.1, n=N, cfg=cfg,
                            candidates=candidates, business=business, evaluator=evaluator())
    credit_report(result.action, result.forecast, n=N, cfg=cfg, candidates=candidates,
                  business=business, ports_master=ports, evaluator=evaluator(),
                  block=2, forecast_seed=7, delta=.1, eta_ref=result.eta_ref, max_evals=5)
    after = protect_budget(result.action, result, delta=.1, n=N, cfg=cfg,
                           candidates=candidates, business=business, evaluator=evaluator())
    assert before.network_ee == after.network_ee
    assert before.fallback == after.fallback
    np.testing.assert_array_equal(before.action.powers, after.action.powers)
    np.testing.assert_array_equal(result.action.powers, powers_before)


def test_proposal_is_recorded_even_when_illegal():
    cfg, candidates, business, ports = domain()
    result = reference_action()
    executed = result.action
    illegal = Action([executed.links[0], executed.links[0]], executed.powers.copy(),
                     {key: executed.ports[key].copy() for key in ('tx', 'rx')})
    record, calls = credit_report(
        executed, result.forecast, n=N, cfg=cfg, candidates=candidates, business=business,
        ports_master=ports, evaluator=evaluator(), block=1, forecast_seed=5, delta=.1,
        eta_ref=result.eta_ref, proposal=illegal, max_evals=2)
    assert calls == 3  # two replacements plus the proposal
    assert record['proposal'] == dict(legal=False, network_ee=None, mean_pair_ee=None,
                                      min_pair_ee=None, budget_violation=None,
                                      reason='proposal_illegal')
