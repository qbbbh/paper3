"""Meaningful legality, hybrid-gradient and execution/replay contract checks."""
import copy
import math
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip('torch')
from p3base.block import validate_action
from p3base.config import FoundationConfig
from p3learn.features import EDGE_DIM, NODE_DIM
from p3learn.models import FeatureBatch, ModelConfig, critic_power_feature
from p3learn.masac import (HybridMASAC, SACConfig, PolicySample, latent_from_power,
                           power_from_latent)

torch.set_num_threads(1)


def features(n=6, blocked=()):
    """Dense causal fixture; ``blocked`` pairs create non-candidate edges.

    A blocked pair is non-candidate but still inside the interference window
    only when the edge features are constant, so ``edge`` is flat in that case.
    """
    rng = np.random.default_rng(90 + n)
    mask = np.ones((n, n, 2), dtype=bool)
    mask[np.arange(n), np.arange(n)] = False
    for i, j in blocked:
        mask[i, j] = mask[j, i] = False
    ports = rng.integers(0, 2, (n, n, 2, 2))
    ports[:, :, 0] = -1
    edge = (np.full((n, n, 2, EDGE_DIM), .1, dtype=np.float32) if blocked
            else rng.normal(size=(n, n, 2, EDGE_DIM)))
    return FeatureBatch(rng.normal(size=(n, NODE_DIM)), edge, mask, ports)


def trainer(encoder='mlp', seed=8):
    return HybridMASAC(ModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=16,
                                   encoder=encoder, message_layers=1),
                       SACConfig(seed=seed, batch_size=2))


def check_legal(batch, action):
    validate_action(action.links, action.powers, action.ports, batch.n, FoundationConfig(),
                    candidates={'RF': batch.mode_mask[:, :, 0], 'VLC': batch.mode_mask[:, :, 1]})
    assert len(action.links) == batch.n // 2


def test_completion_mask_excludes_locally_valid_dead_end():
    batch = features(4)
    batch.mode_mask[:] = False
    # 0--1 is locally eligible but prevents complete coverage; the only
    # undirected matching is 0--2, 1--3.  Direction and mode remain sampled.
    for i, j in [(0, 1), (0, 2), (1, 3)]:
        batch.mode_mask[i, j] = batch.mode_mask[j, i] = True
    learner = trainer()
    for _ in range(12):
        sample = learner.select_action(batch)
        check_legal(batch, sample.action)
        assert all({tx, rx} != {0, 1} for tx, rx, _ in sample.action.links)
        assert all({tx, rx} != {0, 1} for tx, rx, _ in sample.trace[0]['legal_choices'])


def test_joint_discrete_log_probability_is_legal_sequential_distribution():
    batch, learner = features(4), trainer()
    for parameter in learner.actor.parameters():
        parameter.data.zero_()
    sample = learner.select_action(batch)
    # 12 legal directed/mode choices at anchor 0, then 4 at the final pair.
    assert len(sample.trace[0]['legal_choices']) == 12
    assert len(sample.trace[1]['legal_choices']) == 4
    assert sample.log_prob_discrete == pytest.approx(-math.log(48), abs=1e-6)
    assert sum(step['log_probability'] for step in sample.trace) == pytest.approx(sample.log_prob_discrete)
    assert np.isfinite(sample.log_prob_continuous)


@pytest.mark.parametrize('encoder', ['mlp', 'hetero_graph', 'hetero_graph_v2'])
def test_real_critic_and_hybrid_actor_update_changes_parameters(encoder):
    # v2 needs non-candidate edges, otherwise both interference relations are
    # empty by construction and their parameters receive no gradient.
    batch = features(blocked=[(0, 1), (2, 3)]) if encoder == 'hetero_graph_v2' else features()
    learner = trainer(encoder)
    before_actor = [p.detach().clone() for p in learner.actor.parameters()]
    before_critics = [p.detach().clone() for p in learner.critics.parameters()]
    for reward, cost in [(5e8, 0), (2e8, .2)]:
        proposal = learner.select_action(batch)
        check_legal(batch, proposal.action)
        learner.observe(batch, proposal.action, reward, cost=cost, proposal=proposal,
                        label_source='proposal_forecast')
    report = learner.update()
    assert report['updated'] and report['gamma'] == 0
    assert report['actor_gradient_norm'] > 0 and report['critic_gradient_norm'] > 0
    assert np.isfinite(report['critic_loss']) and np.isfinite(report['actor_loss'])
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, learner.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_critics, learner.critics.parameters()))
    # Both power and categorical outputs have actual, nonzero gradients.
    final_head = learner.actor.head[-1]
    assert final_head.weight.grad[0].abs().sum() > 0
    assert final_head.weight.grad[1:].abs().sum() > 0
    if encoder in ('hetero_graph', 'hetero_graph_v2'):
        relations = learner.actor.encoder.layers[0].relations
        assert all(sum(float(p.grad.abs().sum()) for p in relation.parameters()) > 0
                   for relation in relations)


def test_v2_empty_interference_relations_give_finite_zero_gradient():
    # A complete candidate graph has no non-candidate edge: both interference
    # relations are empty, which must stay finite and gradient-free, not NaN.
    learner, batch = trainer('hetero_graph_v2'), features()
    for reward, cost in [(5e8, 0), (2e8, .2)]:
        proposal = learner.select_action(batch)
        learner.observe(batch, proposal.action, reward, cost=cost, proposal=proposal,
                        label_source='proposal_forecast')
    report = learner.update()
    assert report['updated'] and np.isfinite(report['actor_loss'])
    relations = learner.actor.encoder.layers[0].relations
    for index in (1, 3):
        total = sum(0. if p.grad is None else float(p.grad.abs().sum())
                    for p in relations[index].parameters())
        assert total == 0
    assert all(torch.isfinite(p).all() for p in learner.actor.parameters())


def test_proposal_is_stored_with_its_own_label_after_a_fallback():
    # The shield executed something else, but the critic must learn from the
    # proposal and its own predicted gap, not from the reference reward.
    batch, learner = features(4), trainer()
    proposal = learner.select_action(batch)
    executed = copy.deepcopy(proposal.action)
    executed.powers[:] = .001
    with pytest.raises(ValueError, match='fallback'):
        learner.observe(batch, executed, 1e8, cost=.9, proposal=proposal)
    proposal.log_prob_discrete = proposal.log_prob_continuous = float('nan')
    for _ in range(2):
        learner.observe(batch, proposal.action, 1e7, cost=.9, fallback=True,
                        proposal=proposal, label_source='proposal_forecast')
    np.testing.assert_array_equal(learner.replay[0].action.powers, proposal.action.powers)
    assert learner.replay[0].fallback and learner.replay[0].matches_proposal
    assert learner.replay[0].label_source == 'proposal_forecast'
    assert learner.replay[0].cost == pytest.approx(.9)
    with pytest.raises(ValueError, match='proposal action itself'):
        learner.observe(batch, executed, 1e7, fallback=True, proposal=proposal,
                        label_source='proposal_forecast')
    recorded_critic_powers = []
    for critic in learner.critics:
        original = critic.forward
        def wrapper(features_arg, action_arg, powers=None, original=original):
            if powers is None:
                recorded_critic_powers.append(np.asarray(action_arg.powers).copy())
            return original(features_arg, action_arg, powers)
        critic.forward = wrapper
    report = learner.update()
    assert report['fallback_samples'] == 2 and report['actor_on_fallback_states'] == 2
    assert report['fresh_actor_samples'] == 2 and report['actor_gradient_norm'] > 0
    assert np.isfinite(report['actor_loss'])
    assert len(recorded_critic_powers) == 4
    assert all(np.array_equal(p, proposal.action.powers) for p in recorded_critic_powers)


def test_reference_forecast_is_a_distinct_label_source():
    # The fixed-state probe stores the reference action with its own forecast;
    # that is neither an executed action nor a proposal sample.
    batch, learner = features(4), trainer()
    action = learner.select_action(batch).action
    learner.observe(batch, action, 1e9, label_source='reference_forecast')
    assert learner.replay[0].label_source == 'reference_forecast'
    assert learner.replay[0].cost == 0.0 and learner.replay[0].matches_proposal is None
    with pytest.raises(ValueError, match='Unknown label source'):
        learner.observe(batch, action, 1e9, label_source='reference')


def test_log_uniform_power_reaches_the_one_milliwatt_floor():
    # The old `0.001 + 1.999*sigmoid(clamp(z, -5, 5))` head could not go below
    # 14.38 mW, while the reference action uses exactly 1 mW.
    low, high = 1e-3, 2.0
    power, log_jacobian = power_from_latent(torch.tensor(-12.), low, high)
    assert float(power) == pytest.approx(1.0000467e-3, rel=1e-5)
    assert float(power) < 1.01e-3
    assert torch.isfinite(log_jacobian)
    for latent in (-100., -10., 0., 10., 100.):
        value, jacobian = power_from_latent(torch.tensor(latent), low, high)
        assert low*(1-1e-6) <= float(value) <= high*(1+1e-6)
        assert torch.isfinite(jacobian)


def test_deterministic_sampler_can_emit_near_floor_power():
    batch, learner = features(4), trainer()
    last = learner.actor.head[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        last.bias[1] = -12.  # power latent mean, now unbounded above/below
    sample = learner.select_action(batch, deterministic=True)
    assert float(np.max(sample.action.powers)) < 1.01e-3


def test_critic_sees_power_on_a_log_scale_near_the_floor():
    # `power/2` mapped 1 mW and 1.05 mW to values 2.5e-5 apart, which the
    # critic could not resolve; log10 power keeps a 0.021 gap.
    values = critic_power_feature(torch.tensor([1e-3, 1.05e-3, 1e-2, 2.0]))
    assert float(values[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(values[1] - values[0]) > 0.02
    assert float(values[2]) == pytest.approx(1.0, abs=1e-6)
    assert float(values[3]) == pytest.approx(3.30103, abs=1e-4)
    assert torch.isfinite(values).all()


def test_latent_power_inverse_is_finite_at_the_floor():
    # The forward map never reaches 1 mW at a finite latent, so the supervised
    # target is nudged in by the documented tolerance instead of using -inf.
    latent = float(latent_from_power(1e-3, 1e-3, 2.0))
    assert np.isfinite(latent) and latent < -8
    power, _ = power_from_latent(torch.tensor(latent), 1e-3, 2.0)
    assert float(power) == pytest.approx(1e-3, rel=2e-3)
    for value in (1e-3, 1e-2, .1, 1.0, 2.0):
        recovered, _ = power_from_latent(
            torch.tensor(float(latent_from_power(value, 1e-3, 2.0))), 1e-3, 2.0)
        assert float(recovered) == pytest.approx(value, rel=2e-3)


def test_imitation_replays_the_sampler_order_and_recovers_the_demonstration():
    batch, learner = features(6), trainer()
    target = learner.select_action(batch).action
    report = learner.imitation_loss(batch, target)
    assert report['steps'] == 3 and np.isfinite(float(report['loss']))
    optimizer = torch.optim.Adam(learner.actor.parameters(), lr=1e-2)
    for _ in range(400):
        optimizer.zero_grad()
        loss = learner.imitation_loss(batch, target)['loss']
        loss.backward()
        optimizer.step()
    final = learner.imitation_loss(batch, target)
    assert float(final['discrete_loss']) < 1e-2
    produced = learner.select_action(batch, deterministic=True).action
    assert produced.links == target.links
    np.testing.assert_allclose(produced.powers, target.powers, rtol=2e-3)


def test_imitation_rejects_a_malformed_demonstration():
    batch, learner = features(4), trainer()
    target = learner.select_action(batch).action
    target.links[1] = target.links[0]
    with pytest.raises(ValueError, match='duplicate link'):
        learner.imitation_loss(batch, target)


def test_default_model_config_matches_feature_width():
    # A default ModelConfig paired with build_features must not fail late at
    # the first _check_features call.
    assert ModelConfig().node_dim == NODE_DIM
    assert ModelConfig().edge_dim == EDGE_DIM


@pytest.mark.parametrize('encoder', ['hetero_graph', 'hetero_graph_v2'])
def test_checkpoint_restores_weights_optimizers_rng_and_optional_actual_replay(
        tmp_path, encoder):
    learner, batch = trainer(encoder), features(4)
    for _ in range(2):
        proposal = learner.select_action(batch)
        learner.observe(batch, proposal.action, 1e8, proposal=proposal,
                        label_source='proposal_forecast')
    learner.update()
    checkpoint = tmp_path / 'dev.pt'
    provenance = dict(split='train', n=4, channel_seed=123, training_traffic_seed=456,
                      feature_version='fixture_v1', traffic_hashes={'mobility.npz': 'frozen'})
    learner.save(checkpoint, include_replay=True, provenance=provenance)
    restored = HybridMASAC.load(checkpoint)
    assert restored.provenance == provenance
    assert restored.updates == learner.updates
    assert len(restored.replay) == len(learner.replay)
    assert restored.actor_optimizer.state_dict()['state']
    for a, b in zip(learner.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(a, b)
    expected, actual = learner.select_action(batch), restored.select_action(batch)
    assert expected.action.links == actual.action.links
    np.testing.assert_array_equal(expected.action.powers, actual.action.powers)
    assert expected.log_prob_discrete == actual.log_prob_discrete
    assert restored.update()['updated']
    # Re-saving a loaded checkpoint preserves provenance unless explicitly
    # replaced by a new, recorded continuation-training provenance.
    continued_checkpoint = tmp_path / 'continued.pt'
    restored.save(continued_checkpoint)
    assert HybridMASAC.load(continued_checkpoint).provenance == provenance


def test_infeasibility_and_invalid_action_are_explicit():
    learner, batch = trainer(), features(4)
    batch.mode_mask[:] = False
    batch.mode_mask[0, 1, 0] = True
    with pytest.raises(ValueError, match='coverage_failure'):
        learner.select_action(batch)
    with pytest.raises(ValueError, match='gamma=0'):
        SACConfig(gamma=.9)
    batch = features(4)
    action = learner.select_action(batch).action
    action.links[1] = action.links[0]
    with pytest.raises(ValueError, match='endpoints'):
        learner.observe(batch, action, 1e8)
