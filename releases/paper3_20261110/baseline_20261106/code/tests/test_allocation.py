"""Allocation contracts: full domain, trajectory means and explicit budget guard."""
from dataclasses import replace
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from p3base.block import validate_action
from p3base.config import FoundationConfig
from p3base.energy import link_electrical_power_w
from p3learn.allocation import (
    Action, CoverageFailure, ReferencePolicy, decode_complete_pairing,
    has_perfect_matching, max_min_allocate, protect_budget, pure_ee_reference,
    random_allocate, random_legal_action, repair_unusable_modes,
)


def domain(n=4):
    business = ~np.eye(n, dtype=bool)
    candidates = {mode: business.copy() for mode in ("RF", "VLC")}
    ports = {name: np.zeros((n, n), dtype=int) for name in ("tx", "rx")}
    return FoundationConfig(), candidates, business, ports


def electrical_ee(action, cfg):
    """逐链路 EE = 速率 / (静态功耗 + p)，再取算术平均（目标本身）。"""
    rates = np.asarray(ee_evaluator(action)["mean_rate_bps"], float)
    denom = link_electrical_power_w(action.powers,
                                    [mode for _i, _j, mode in action.links], cfg)
    return float(np.mean(rates / denom))


def gain_domain(n=6):
    """非均匀信念增益：让两类候选分数生成不同的匹配，而不是全部重合。"""
    cfg, candidates, business, ports = domain(n)
    rng = np.random.default_rng(3)
    gain = {"RF": 1e-9 * (1.0 + rng.random((n, n))),
            "VLC": 1e-5 * (1.0 + rng.random((n, n)))}
    return cfg, candidates, business, ports, gain


def prediction(mean, samples=None):
    mean = np.asarray(mean, float)
    samples = np.stack([mean, mean]) if samples is None else np.asarray(samples, float)
    return dict(mean_rate_bps=mean, block_rate_samples_bps=samples,
                low_rate_bps=np.quantile(samples, .05, axis=0),
                high_rate_bps=np.quantile(samples, .95, axis=0))


def ee_evaluator(action):
    """Mean EE peaks at .03 W; lower quantile misleadingly favors .3 W."""
    p = action.powers
    ee = np.where(p < .01, 1., np.where(p < .1, 10., 5.))
    mean = p * ee
    if np.all((p >= .01) & (p < .1)):
        return prediction(mean, np.stack([np.zeros_like(mean), 2 * mean]))
    return prediction(mean)


def test_full_matching_uses_business_direction_and_all_vehicles():
    cfg, candidates, business, ports = domain(6)
    business[:] = False
    business[1, 0] = business[2, 5] = business[4, 3] = True
    # An attractive geometric edge forbidden by business must not be selected.
    scores = {mode: np.full((6, 6), 100.) for mode in candidates}
    scores["VLC"][business] = 200.
    action = decode_complete_pairing(6, cfg, candidates, business, ports, scores)
    assert set(action.links) == {(1, 0, "VLC"), (2, 5, "VLC"), (4, 3, "VLC")}
    assert sorted(endpoint for link in action.links for endpoint in link[:2]) == list(range(6))
    validate_action(action.links, action.powers, action.ports, 6, cfg, candidates, business)


def test_matching_preserves_weak_low_rank_edge_needed_for_complete_coverage():
    cfg, candidates, business, ports = domain()
    business[:] = False
    business[0, 1] = business[0, 2] = business[1, 3] = True
    scores = {mode: np.zeros((4, 4)) for mode in candidates}
    scores["RF"][0, 1] = 1000.
    scores["RF"][0, 2] = scores["RF"][1, 3] = -100.
    action = decode_complete_pairing(4, cfg, candidates, business, ports, scores)
    assert {(i, j) for i, j, _ in action.links} == {(0, 2), (1, 3)}


def test_random_lower_bound_is_legal_reproducible_and_not_budget_guarded():
    """随机下界必须合法、可复现，且**不能**被预算守卫改写成 reference。

    它是"不做优化"的对照：一旦被守卫接管，就退化成 reference，下界失去意义。
    """
    cfg, candidates, business, ports = domain(6)
    a = random_legal_action(6, cfg, candidates, business, ports, np.random.default_rng(11))
    b = random_legal_action(6, cfg, candidates, business, ports, np.random.default_rng(11))
    assert a.links == b.links and np.array_equal(a.powers, b.powers)
    validate_action(a.links, a.powers, a.ports, 6, cfg, candidates, business)
    floors = np.array([getattr(cfg, m.lower()).power_min_w for _i, _j, m in a.links])
    caps = np.array([getattr(cfg, m.lower()).power_max_w for _i, _j, m in a.links])
    assert np.all(a.powers >= floors) and np.all(a.powers <= caps)
    # 同一域上抽多次必须真的不同，否则"随机"只是固定配对。
    links = {tuple(random_legal_action(6, cfg, candidates, business, ports,
                                        np.random.default_rng(s)).links) for s in range(12)}
    assert len(links) > 1
    reference = pure_ee_reference(6, cfg, candidates, business, ports, ee_evaluator)
    result = random_allocate(6, cfg, candidates, business, ports, ee_evaluator,
                             delta=0.05, seed=np.random.SeedSequence([1, 2, 3]),
                             reference=reference)
    assert result.success and not result.fallback
    assert result.reason == "uniform_random_legal_action"
    assert result.evaluator_calls == 1
    # 抽到的动作通常达不到 (1-delta)*eta_ref；此时也必须原样返回，不许换 reference。
    assert result.action.links == random_legal_action(
        6, cfg, candidates, business, ports,
        np.random.default_rng(np.random.SeedSequence([1, 2, 3]))).links


def test_no_complete_matching_is_failure_and_never_partial_result():
    cfg, candidates, business, ports = domain()
    business[:, 3] = business[3, :] = False
    with pytest.raises(CoverageFailure):
        decode_complete_pairing(4, cfg, candidates, business, ports)
    result = pure_ee_reference(4, cfg, candidates, business, ports, ee_evaluator)
    assert not result.success and result.coverage_failure
    assert result.evaluator_calls == 0


def test_completion_certificate_and_exact_fallback_agree_with_bruteforce():
    rng = np.random.default_rng(7)

    def brute(mask):
        if not len(mask):
            return True
        for j in range(1, len(mask)):
            if mask[0, j] or mask[j, 0]:
                keep = [k for k in range(len(mask)) if k not in (0, j)]
                if brute(mask[np.ix_(keep, keep)]):
                    return True
        return False

    assert has_perfect_matching(np.empty((0, 0), bool))
    assert not has_perfect_matching(np.ones((3, 3), bool))
    for n in (2, 4, 6, 8):
        for _ in range(30):
            mask = rng.random((n, n)) < .25
            assert has_perfect_matching(mask) == brute(mask)


def test_reference_is_bounded_and_selects_best_mean_ee_of_its_candidates():
    """候选最多 6 个（与 N 无关）；最终由均逐链路 EE 决定，而不是候选分数。"""
    cfg, candidates, business, ports, gain = gain_domain(6)
    seen = []

    def recording(action):
        seen.append((list(action.links), np.array(action.powers, float)))
        return ee_evaluator(action)

    result = pure_ee_reference(6, cfg, candidates, business, ports, recording,
                               belief_gains=gain)
    assert result.success
    assert 1 <= result.evaluator_calls <= 6
    assert result.evaluator_calls == result.reference_evaluator_calls == len(seen)
    assert result.evaluator_calls == result.policy_config["matching_candidates"]
    assert result.reference_solve_time_s >= 0
    assert result.policy_config["score_scope"] == "conditional_trajectory_mean_block_rate"
    # 选中的必须是评估器见过的最好均 EE；等效 SINR 只负责"提出"候选。
    scores = [electrical_ee(Action(links, powers, ports), cfg) for links, powers in seen]
    assert result.eta_ref == pytest.approx(max(scores))
    assert np.allclose(result.forecast["mean_rate_bps"],
                       np.mean(result.forecast["block_rate_samples_bps"], axis=0))
    # 只按速率排序的 5% 分位会指向别的候选；均值口径下不得被采用。
    assert result.eta_ref >= min(scores)
    again = pure_ee_reference(6, cfg, candidates, business, ports, ee_evaluator,
                              belief_gains=gain)
    assert again.action.links == result.action.links


def test_search_layer_matches_execution_layer_on_asymmetric_interference():
    """搜索层（`_rates_and_ee`）与执行层（`link_eval.evaluate`）必须是同一套物理。

    这里同时盯住三类曾经出错的写法：
    * 干扰矩阵方向：``H[b, a]`` 必须按 ``H.T @ p`` 用，写成 ``H @ p`` 会到 1e0 量级；
    * VLC 交叉项的端口：``h_cross`` 要用 **b 的发射端口 + a 的接收端口**，
      用 ``_flat_gain`` 压出来的 ``vlc[u, j]``（b、a 各自配对的端口）是错的；
    * 平方律的自身项：不能"先乘再减对角"，两项差十几个数量级会灾难性相消。
    因此需要至少两条 VLC 链路、端口互不相同、交叉增益不对称。
    """
    from p3base import link_eval
    from p3learn.allocation import _link_gain_tables, _rates_and_ee

    n = 8
    links = [(0, 1, "RF"), (2, 3, "RF"), (4, 5, "VLC"), (6, 7, "VLC")]
    p = np.array([.077, .001, .3, .05])
    rf = np.zeros((n, n))
    rf[0, 1], rf[2, 3] = 2.2e-7, 1.6e-7        # 期望
    rf[0, 3], rf[2, 1] = 1.36e-8, 1.0e-11      # 交叉，差三个数量级
    vlc4 = np.zeros((n, n, 2, 2))
    ports = {"tx": np.zeros((n, n), int), "rx": np.zeros((n, n), int)}
    ports["tx"][4, 5], ports["rx"][4, 5] = 0, 1     # 与另一条 VLC 链路端口相反
    ports["tx"][6, 7], ports["rx"][6, 7] = 1, 0
    vlc4[4, 5, 0, 1] = 3.0e-5                  # 期望
    vlc4[6, 7, 1, 0] = 1.1e-5
    vlc4[6, 5, 1, 1] = 4.0e-6                  # 6 干扰 5（强）
    vlc4[4, 7, 0, 0] = 1.5e-7                  # 4 干扰 7（弱）
    for model in ("linear", "square_law"):
        cfg = FoundationConfig().with_(vlc=replace(FoundationConfig().vlc, sinr_model=model))
        gains = {"RF": rf, "VLC": vlc4}
        s, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
        # 交叉项必须真的非零，否则本用例退化成单链路、什么也测不出。
        assert h_vlc[2, 3] > 0 and h_vlc[3, 2] > 0
        assert h_vlc[2, 3] != pytest.approx(h_vlc[3, 2])
        search = _rates_and_ee(p, links, cfg, s, h_rf, h_vlc, h_opt)[0]
        executed = link_eval.evaluate(links, p, ports, rf, vlc4, cfg)["link_rates_bps"]
        assert np.allclose(search, executed, rtol=1e-10, atol=1e-6), model


def test_unusable_link_is_switched_instead_of_sacrificed():
    """守卫：低于 1e4 bps 的链路先尝试换模式，而不是留在 0 速率上。"""
    cfg, candidates, business, ports = domain(2)
    gains = {"RF": np.zeros((2, 2)), "VLC": np.zeros((2, 2))}
    gains["VLC"][0, 1] = 1e-4
    links, _powers, changed = repair_unusable_modes(
        [(0, 1, "RF")], np.array([.001]), gains, ports, cfg, candidates)
    assert changed and links == [(0, 1, "VLC")]


def test_unusable_link_stays_when_the_other_mode_is_not_legal():
    cfg, candidates, business, ports = domain(2)
    candidates["VLC"][:] = False
    gains = {"RF": np.zeros((2, 2)), "VLC": np.zeros((2, 2))}
    links, _powers, changed = repair_unusable_modes(
        [(0, 1, "RF")], np.array([.001]), gains, ports, cfg, candidates)
    assert not changed and links == [(0, 1, "RF")]


def test_budget_violation_falls_back_to_frozen_legal_reference():
    cfg, candidates, business, ports, gain = gain_domain(6)
    reference = pure_ee_reference(6, cfg, candidates, business, ports, ee_evaluator,
                                  belief_gains=gain)
    # v4：两种模式下界不同（RF 0.01 W、VLC 0.1 W），"贴着下界"的合法提议要按模式取。
    proposed = Action(list(reference.action.links),
                      np.array([cfg.rf.power_min_w if m == "RF" else cfg.vlc.power_min_w
                                for _i, _j, m in reference.action.links]), ports)
    result = protect_budget(proposed, reference, delta=.1, n=6, cfg=cfg,
                            candidates=candidates, business=business, evaluator=ee_evaluator)
    assert result.success and result.fallback
    assert np.all(result.action.powers == reference.action.powers)
    assert result.mean_pair_ee == pytest.approx(reference.eta_ref)
    expected = electrical_ee(proposed, cfg)
    assert expected < .9 * reference.eta_ref   # 这个提议确实违约，否则用例无效
    assert result.budget_violation == pytest.approx(.9 * reference.eta_ref - expected)
    assert result.proposal_budget_violation == pytest.approx(.9 * reference.eta_ref - expected)
    assert result.executed_budget_violation == 0
    assert result.evaluator_calls == 1 and result.rejected_candidates == 1
    assert "budget_violation" in result.reason
    proposed.powers[:] = 1.
    assert np.all(result.action.powers == reference.action.powers)


def test_legal_budget_passing_actor_is_kept_even_without_auxiliary_fairness_gate():
    cfg, candidates, business, ports = domain()
    reference = pure_ee_reference(4, cfg, candidates, business, ports, ee_evaluator)
    proposed = Action(list(reference.action.links), np.full(2, .3), ports)
    result = protect_budget(proposed, reference, delta=.6, n=4, cfg=cfg,
                            candidates=candidates, business=business, evaluator=ee_evaluator)
    assert result.success and not result.fallback
    assert np.all(result.action.powers == .3)


def test_invalid_proposal_falls_back_but_invalid_reference_fails_entire_block():
    cfg, candidates, business, ports = domain()
    reference = pure_ee_reference(4, cfg, candidates, business, ports, ee_evaluator)
    bad = Action([(0, 1, "RF"), (1, 2, "RF")], np.ones(2), ports)
    arguments = dict(delta=.1, n=4, cfg=cfg, candidates=candidates,
                     business=business, evaluator=ee_evaluator)
    result = protect_budget(bad, reference, **arguments)
    assert result.success and result.fallback and result.evaluator_calls == 0
    result = protect_budget(reference.action, replace(reference, action=bad), **arguments)
    assert not result.success and result.evaluator_calls == 0
    assert "reference_failure" in result.reason


def test_reference_and_proposal_require_same_conditional_evaluator():
    cfg, candidates, business, ports = domain()
    reference = pure_ee_reference(4, cfg, candidates, business, ports, ee_evaluator)
    result = protect_budget(reference.action, reference, delta=.1, n=4, cfg=cfg,
                            candidates=candidates, business=business,
                            evaluator=lambda action: ee_evaluator(action))
    assert not result.success
    assert "same conditional evaluator" in result.reason


def test_repeated_bound_method_lookup_preserves_conditional_evaluator_identity():
    cfg, candidates, business, ports = domain()

    class Evaluator:
        def evaluate(self, action):
            return ee_evaluator(action)

    evaluator = Evaluator()
    reference = pure_ee_reference(4, cfg, candidates, business, ports, evaluator.evaluate)
    result = protect_budget(reference.action, reference, delta=.1, n=4, cfg=cfg,
                            candidates=candidates, business=business, evaluator=evaluator.evaluate)
    assert result.success and not result.fallback


def test_inconsistent_trajectory_mean_invalidates_reference():
    cfg, candidates, business, ports = domain()

    def mismatched(action):
        value = prediction(np.ones(len(action.links)))
        value["mean_rate_bps"] *= 2
        return value

    result = pure_ee_reference(4, cfg, candidates, business, ports, mismatched)
    assert not result.success and result.evaluator_calls == 1
    assert "trajectory rates" in result.reason


def test_delta_is_required_and_finite_and_reference_policy_version_is_frozen():
    assert inspect.signature(protect_budget).parameters["delta"].default is inspect.Parameter.empty
    assert inspect.signature(max_min_allocate).parameters["delta"].default is inspect.Parameter.empty
    cfg, candidates, business, ports = domain()
    for delta in (None, np.nan, np.inf, -.1, 1.1):
        with pytest.raises(ValueError):
            max_min_allocate(4, cfg, candidates, business, ports, ee_evaluator, delta=delta)
    with pytest.raises(ValueError, match="new version"):
        ReferencePolicy(powers_w=(.01, .03, .1))


def test_max_min_search_has_bounded_cost_and_preserves_budget():
    cfg, candidates, business, ports = domain(50)
    gain = {"RF": np.full((50, 50), 1e-10), "VLC": np.full((50, 50, 2, 2), 1e-5)}
    result = max_min_allocate(50, cfg, candidates, business, ports, ee_evaluator,
                              delta=.1, belief_gains=gain)
    assert result.success and len(result.action.links) == 25
    assert result.evaluator_calls <= 4 and 1 <= result.reference_evaluator_calls <= 6
    assert result.network_ee >= .9 * result.eta_ref
    assert result.attempted_candidates == 4
    # 预算守的是均逐链路 EE：无论选中的是改进候选还是回退参考，都必须满足。
    assert result.mean_pair_ee >= (1. - .1) * result.eta_ref - 1e-9
    assert result.fallback == (result.reason == "reference_fallback_no_better_feasible_action")


def test_invalid_finite_parameters_are_not_silently_removed_from_domain():
    cfg, candidates, business, ports = domain()
    bad_scores = {mode: np.zeros((4, 4)) for mode in candidates}
    bad_scores["RF"][0, 1] = np.nan
    with pytest.raises(ValueError, match="legal edge scores"):
        decode_complete_pairing(4, cfg, candidates, business, ports, bad_scores)
    with pytest.raises(ValueError, match="power_w"):
        decode_complete_pairing(4, cfg, candidates, business, ports, power_w=np.inf)


def test_rebalance_cuts_by_electrical_ee_not_by_transmit_power():
    """VLC 的 R/p 高但 R/(static+p) 并不高，不能被当成"最强"先砍。"""
    from p3learn.allocation import rebalance_proposal
    cfg = FoundationConfig()
    links = ((0, 1, "VLC"), (2, 3, "RF"))
    action = Action(links, np.array([.5, .01]), {n: np.zeros((4, 4), int) for n in ("tx", "rx")})
    forecast = {"mean_rate_bps": np.array([1e8, 2e6])}
    # R/p:  2.0e8 与 2.0e8 持平 -> 旧口径一个都不降；R/(static+p): 4.45e7 > 1.59e7。
    out = rebalance_proposal(action, forecast, cfg)
    assert out.powers[0] == pytest.approx(.5 * .25)
    assert out.powers[1] == pytest.approx(.01)
    assert action.powers[0] == .5          # 不改输入


def _smooth_evaluator(gain):
    """连续可导的速率-功率响应，供连续 Max-Min 搜索使用（不是阶梯函数）。"""
    gain = np.asarray(gain, float)

    def evaluator(action):
        p = np.asarray(action.powers, float)
        return prediction(1e6 * np.log1p(gain * p))
    return evaluator


def _harness(gain):
    cfg, candidates, business, ports = domain(2 * len(gain))
    links = tuple((2 * k, 2 * k + 1, "RF") for k in range(len(gain)))
    action = Action(links, np.full(len(links), .01), ports)
    return cfg, action, _smooth_evaluator(gain)


def test_power_refine_only_changes_power_and_respects_budget():
    from p3learn.allocation import maxmin_power_refine
    cfg, action, evaluator = _harness([1e2, 1e5, 1e8])
    forecast = evaluator(action)
    eta_ref = float(np.mean(np.asarray(forecast["mean_rate_bps"]) /
                            link_electrical_power_w(action.powers,
                                                    [m for _i, _j, m in action.links], cfg)))
    refined, _, calls = maxmin_power_refine(action, forecast, cfg, evaluator,
                                            delta=.1, eta_ref=eta_ref, maxiter=120)
    assert refined.links == action.links and refined.ports == action.ports
    powers = np.asarray(refined.powers, float)
    assert np.all(powers >= cfg.rf.power_min_w - 1e-12) and np.all(powers <= cfg.rf.power_max_w + 1e-12)
    out = evaluator(refined)
    ee = np.asarray(out["mean_rate_bps"]) / link_electrical_power_w(
        powers, [m for _i, _j, m in refined.links], cfg)
    assert float(ee.mean()) >= (1. - .1) * eta_ref - 1e-6      # 预算与守卫同口径
    assert float(ee.min()) >= float(np.min(np.asarray(forecast["mean_rate_bps"]) /
                                           link_electrical_power_w(
                                               action.powers,
                                               [m for _i, _j, m in action.links], cfg))) - 1e-6
    assert calls > 0
    assert action.powers.tolist() == [.01, .01, .01]            # 不改输入


def test_power_refine_with_infeasible_budget_returns_input_action():
    from p3learn.allocation import maxmin_power_refine
    cfg, action, evaluator = _harness([1e2, 1e5, 1e8])
    forecast = evaluator(action)
    # eta_ref 抬到任何可行功率都达不到的高度 -> 没有可行点，必须原样返回。
    refined, _, _ = maxmin_power_refine(action, forecast, cfg, evaluator,
                                        delta=.001, eta_ref=1e12, maxiter=40)
    assert np.allclose(np.asarray(refined.powers, float), np.asarray(action.powers, float))


def test_maxmin_with_power_refine_counts_the_extra_candidate_and_keeps_budget():
    cfg, candidates, business, ports = domain(50)
    gain = {"RF": np.full((50, 50), 1e-10), "VLC": np.full((50, 50, 2, 2), 1e-5)}
    result = max_min_allocate(50, cfg, candidates, business, ports, ee_evaluator,
                              delta=.1, belief_gains=gain, power_refine=True,
                              refine_maxiter=30)
    assert result.success and result.attempted_candidates == 5
    assert result.mean_pair_ee >= (1. - .1) * result.eta_ref - 1e-9


def test_power_refine_reports_termination_status_instead_of_dropping_it():
    """求解器返回状态必须留下，"没找到"才解释得清。"""
    from p3learn.allocation import maxmin_power_refine
    cfg, action, evaluator = _harness([1e2, 1e5, 1e8])
    forecast = evaluator(action)
    eta_ref = float(np.mean(np.asarray(forecast["mean_rate_bps"]) /
                            link_electrical_power_w(action.powers,
                                                    [m for _i, _j, m in action.links], cfg)))
    diag = {}
    maxmin_power_refine(action, forecast, cfg, evaluator, delta=.1, eta_ref=eta_ref,
                        maxiter=40, restarts=2, diagnostics=diag)
    assert len(diag["terminations"]) == 3                      # 1 个起点 + 2 次重启
    assert all({"nit", "nfev", "status", "message"} <= set(t) for t in diag["terminations"])
    assert diag["distinct_points"] == diag["objective_evaluations"]
    assert diag["feasible_points"] > 0 and diag["improved"] is True
    assert diag["best_min_ee"] >= diag["input_min_ee"] - 1e-9


def test_infeasible_search_records_how_far_it_missed():
    """搜索内无可行点时要记下最接近约束的余量，而不是只说"无解"。"""
    from p3learn.allocation import maxmin_power_refine
    cfg, action, evaluator = _harness([1e2, 1e5, 1e8])
    forecast = evaluator(action)
    diag = {}
    refined, _, _ = maxmin_power_refine(action, forecast, cfg, evaluator, delta=.001,
                                        eta_ref=1e12, maxiter=40, diagnostics=diag)
    assert diag["feasible_points"] == 0 and diag["improved"] is False
    assert diag["best_min_ee"] is None
    assert diag["best_infeasible_slack"] is not None           # 差多少看得见
    assert diag["best_infeasible_slack"] < 0.
    assert np.allclose(np.asarray(refined.powers, float), np.asarray(action.powers, float))


def _qos_split_evaluator():
    """EE 峰在 p≈0.108 W；离散度随功率升，p05 在峰附近已经跌破门限。"""
    def evaluator(action):
        p = np.asarray(action.powers, float)
        rate = 1.928e7 * p / (p + .1)
        low = rate * (1. - np.minimum(12. * p, .9))
        return prediction(rate, np.stack([low, 2. * rate - low]))
    return evaluator


def test_chance_qos_rejects_candidate_whose_mean_meets_the_gate():
    """均值达标、p05 不达标的候选，在机会口径下必须被拒。"""
    from p3learn.allocation import maxmin_power_refine, qos_shortfall
    cfg = FoundationConfig()
    ports = {name: np.zeros((4, 4), int) for name in ("tx", "rx")}
    action = Action(((0, 1, "RF"),), np.array([.05]), ports)
    evaluator = _qos_split_evaluator()
    forecast = evaluator(action)
    eta_ref = float(np.mean(np.asarray(forecast["mean_rate_bps"]) /
                            link_electrical_power_w(action.powers, ["RF"], cfg)))
    qos = dict(q_min_bps=2e6, inflation=1.0)
    mean_action, _, _ = maxmin_power_refine(action, forecast, cfg, evaluator, delta=.99,
                                            eta_ref=eta_ref, maxiter=200,
                                            qos=dict(qos, metric="mean"))
    chance_action, _, _ = maxmin_power_refine(action, forecast, cfg, evaluator, delta=.99,
                                              eta_ref=eta_ref, maxiter=200,
                                              qos=dict(qos, metric="chance"))
    ok_mean, _, bad_mean = qos_shortfall(evaluator(mean_action), q_min_bps=2e6, metric="chance")
    ok_chance, _, bad_chance = qos_shortfall(evaluator(chance_action), q_min_bps=2e6,
                                             metric="chance")
    # 只看均值的搜索会走到 p05 已经跌破门限的功率；机会口径必须把它压回来。
    assert bad_mean >= 1 and not ok_mean
    assert bad_chance == 0 and ok_chance


def _interference_evaluator():
    """两条 VLC 链路：弱链路被强链路的功率压制，但自身功率仍提高它的速率。"""
    def evaluator(action):
        p = np.asarray(action.powers, float)
        return prediction(np.array([1e7 * p[0] / (1. + 4. * p[1]), 1e7 * p[1]]))
    return evaluator


def test_weak_vlc_link_at_cap_can_still_gain_by_lowering_its_interferer():
    """弱链路已在上限 ≠ 功率优化没用：降同模式干扰者的功率能抬高它。"""
    from p3learn.allocation import maxmin_power_refine
    cfg = FoundationConfig()
    ports = {name: np.zeros((4, 4), int) for name in ("tx", "rx")}
    links = ((0, 1, "VLC"), (2, 3, "VLC"))
    action = Action(links, np.array([cfg.vlc.power_max_w, cfg.vlc.power_max_w]), ports)
    evaluator = _interference_evaluator()
    forecast = evaluator(action)
    modes = [mode for _i, _j, mode in links]
    before = np.asarray(forecast["mean_rate_bps"]) / link_electrical_power_w(action.powers,
                                                                             modes, cfg)
    refined, _, _ = maxmin_power_refine(action, forecast, cfg, evaluator, delta=.95,
                                        eta_ref=float(before.mean()), maxiter=300)
    powers = np.asarray(refined.powers, float)
    after = np.asarray(evaluator(refined)["mean_rate_bps"]) / link_electrical_power_w(
        powers, modes, cfg)
    assert powers[0] > cfg.vlc.power_max_w - 1e-3          # 弱链路仍顶在上限
    assert powers[1] < cfg.vlc.power_max_w - 1e-3          # 动的是干扰者
    assert float(after.min()) > 1.5 * float(before.min())


def test_literature_reproduction_is_one_association_plus_fractional_power():
    """文献基线：关联只做一次，不按评估器挑候选；功率走分式规划。

    这条测试盯住"复现不得借用主算法自由度"：评估次数必须恰好 1（没有多起点、
    没有候选间挑选），版本串要能区分出功率求解器，且网格上升被禁止（它没有
    满足 QoS 的手段）。
    """
    cfg, candidates, business, ports, gain = gain_domain(6)
    seen = []

    def recording(action):
        seen.append(list(action.links))
        return ee_evaluator(action)

    result = pure_ee_reference(6, cfg, candidates, business, ports, recording,
                               belief_gains=gain, search="literature",
                               power_solver="dinkelbach", q_min_bps=0.)
    assert result.success
    assert result.evaluator_calls == len(seen) == 1
    assert result.policy_config["matching_candidates"] == 1
    assert result.policy_config["search"] == "literature_equ_sinr_association+power_dinkelbach"
    assert result.reference_version.endswith("+fractional-programming-sca-v1")
    association = [log for log in result.policy_config["continuous_solver_log"]
                   if log.get("stage") == "literature_association"]
    assert len(association) == 1 and association[0]["rule"] == "equivalent_sinr"
    with pytest.raises(ValueError, match="分式规划"):
        pure_ee_reference(6, cfg, candidates, business, ports, recording,
                          belief_gains=gain, search="literature", power_solver="ascent")
