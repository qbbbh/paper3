"""Selection-rule and fixed-power branch contracts for the diagnostic probes.

These are pure-logic checks: no block, no bank, no training.  They pin down the
comparison the audit is allowed to make, because the earlier one compared the
critic against an infeasible-inclusive oracle and could not support any
conclusion about ranking quality.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip('torch')
from p3learn.features import EDGE_DIM, NODE_DIM
from p3learn.masac import HybridMASAC, SACConfig
from p3learn.models import ModelConfig
from probe_bc_feedback import SELECTORS, pair_concordance, pin_floor_power, select_candidates

torch.set_num_threads(1)


def test_training_label_and_feasible_best_are_different_questions():
    # Candidate 1 has the highest raw mean per-link EE but violates the budget,
    # and its cost penalty is smaller than its EE gain, so it is also the argmax
    # of the critic's own fitted target.  The feasible question has a different
    # answer.
    qs = [0.9, 0.8, 0.7]
    ees = [10.0, 12.0, 11.0]
    costs = [0.0, 0.5, 0.0]
    picks = select_candidates(qs, ees, costs, reward_scale=1.0)

    assert picks["critic"] == 0
    assert picks["training_label"] == 1
    assert picks["forecast_ignoring_budget"] == 1
    assert picks["feasible_forecast"] == 2
    assert picks["feasible"] == [0, 2]
    assert picks["best_feasible_ee"] == pytest.approx(11.0)
    assert picks["regret_predicted"]["critic"] == pytest.approx(1.0)
    assert picks["regret_predicted"]["feasible_forecast"] == pytest.approx(0.0)
    # The old rule is allowed a negative regret: it is not competing on the
    # feasible objective at all.
    assert picks["regret_predicted"]["forecast_ignoring_budget"] == pytest.approx(-1.0)
    assert set(picks["regret_predicted"]) == set(SELECTORS)


def test_no_feasible_candidate_is_reported_not_silently_replaced():
    picks = select_candidates([0.1, 0.2], [5.0, 4.0], [0.3, 0.1], reward_scale=1.0)
    assert picks["feasible"] == []
    assert picks["feasible_forecast"] is None
    assert picks["best_feasible_ee"] is None
    assert picks["regret_predicted"]["feasible_forecast"] is None
    assert picks["regret_predicted"]["critic"] is None


def test_reward_scale_matches_the_critic_target():
    # target = ee / reward_scale - cost.  With scale 1e9 the cost term
    # dominates, so the cheapest feasible candidate wins even with lower EE.
    picks = select_candidates([0.0, 0.0], [9e8, 10e8], [0.0, 0.2], reward_scale=1e9)
    assert picks["training_label"] == 0
    assert picks["forecast_ignoring_budget"] == 1
    assert picks["feasible_forecast"] == 0


def test_pair_concordance_ignores_ties_and_detects_reversal():
    assert pair_concordance([1., 2., 3.], [.1, .5, .9]) == pytest.approx(1.0)
    assert pair_concordance([1., 2., 3.], [.9, .5, .1]) == pytest.approx(0.0)
    assert pair_concordance([1., 2.], [.5, .5]) is None
    assert pair_concordance([1., 2., 3.], [.5, .5, .9]) == pytest.approx(2 / 2)


def test_pinned_power_channel_is_state_independent_and_discrete_head_survives():
    model = ModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=16, encoder="mlp")
    learner = HybridMASAC(model, SACConfig(seed=0))
    learner.actor.head[-1].weight.data.normal_(0, .1)
    learner.actor.head[-1].bias.data.normal_(0, .1)
    discrete_before = learner.actor.head[-1].weight[1].detach().clone()

    pin_floor_power(learner)
    width = 2 * model.hidden_dim + model.edge_dim + 2
    first = learner.actor.head(torch.zeros(4, width))
    second = learner.actor.head(torch.ones(4, width) * 7.0)

    assert torch.allclose(first[..., 0], second[..., 0])
    assert torch.allclose(first[..., 2], second[..., 2])
    assert float(first[0, 0]) == pytest.approx(-12.0)
    assert float(first[0, 2]) == pytest.approx(-5.0)
    assert np.allclose(learner.actor.head[-1].weight[0].detach().numpy(), 0.0)
    assert not torch.allclose(first[..., 1], second[..., 1])
    assert torch.allclose(discrete_before, learner.actor.head[-1].weight[1].detach())


def test_fixed_power_branch_config_removes_continuous_entropy():
    sac = SACConfig(seed=0)
    sac.alpha_continuous = 0.0
    learner = HybridMASAC(ModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=8),
                          sac)
    assert learner.config.alpha_continuous == 0.0
