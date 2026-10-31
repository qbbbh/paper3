"""Fixed-state learning probe: can the critic rank and the actor move?

One real 50/100-vehicle block, one frozen state, one reference action and a
handful of policy proposals.  The reference action and every proposal are
labelled with their *own* forecast, then the learner is trained on this fixed
replay.  This is a capacity/plumbing diagnostic, not a training result: it says
nothing about convergence or about ranking algorithms.

Outputs ``probe.json`` with the critic gap before/after and the proposal's
predicted min EE / budget gap before/after.  The program refuses a non-empty
output directory, so previous diagnostics are never overwritten.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from p3base import block, channel
from p3base.config import FoundationConfig
from p3base.estimator import ChannelEstimator
from p3base.fading import link_coherence_s
from p3base.scene import Scene, find_scene_root
from p3learn.allocation import copy_action, pure_ee_reference, protect_budget
from p3learn.features import build_features, NODE_DIM, EDGE_DIM
from p3learn.forecast import ConditionalTrajectoryBank
from p3learn.masac import HybridMASAC, SACConfig
from p3learn.models import ModelConfig
from run_foundation import TruthOracle

ROOT = Path(__file__).resolve().parents[1]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def evaluate_action(bank, action, eta_ref, delta):
    """Objective = mean of each link's own rate over its own power."""
    forecast = bank.evaluate(copy_action(action))
    rate = np.asarray(forecast["mean_rate_bps"], dtype=float)
    power = np.asarray(action.powers, dtype=float)
    network_ee = float(rate.sum() / power.sum())
    pair_ee = rate / power
    mean_ee, min_ee = float(pair_ee.mean()), float(pair_ee.min())
    cost = max(0.0, 1.0 - delta - mean_ee / eta_ref) if eta_ref else 0.0
    return dict(mean_ee=mean_ee, min_ee=min_ee, network_ee=network_ee, cost=cost)


def critic_values(learner, features, action):
    with torch.no_grad():
        return [float(q(features, copy_action(action))) for q in learner.critics]


def main(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("probe output must be fresh")
    torch.set_num_threads(args.threads)
    cfg = FoundationConfig()
    directory = find_scene_root(ROOT, args.n)
    scene = Scene(directory, cfg.vehicle)
    if scene.n != args.n or args.block >= len(scene.time_s):
        raise ValueError("invalid scale or block")
    context = block.GeometryContext.from_scene(scene)
    oracle = TruthOracle(scene, cfg, args.channel_seed)
    estimator = ChannelEstimator(scene.n, cfg)
    business = ~np.eye(scene.n, dtype=bool)
    t = args.block * cfg.estimation.block_dt_s
    oracle.advance_to(t)
    state = scene.knot(args.block)
    rg, vg = (channel.rf_geometry(context, state, cfg.rf),
              channel.vlc_geometry(context, context.ports(state), cfg.vlc))
    v = state["speed_mps"][:, None] * np.stack(
        [np.cos(state["heading_rad"]), np.sin(state["heading_rad"])], -1)
    rel = np.linalg.norm(v[:, None] - v[None], axis=-1)
    sigma = np.where(rg["los"], cfg.rf.los_shadow_sigma_db, cfg.rf.nlos_shadow_sigma_db)
    prior = np.exp((sigma * np.log(10.) / 10.) ** 2 / 2.)
    estimator.advance(t, oracle, rf_prior_var=prior, rf_tau=link_coherence_s(rel, cfg.rf))
    belief = estimator.gain_belief(rg["gain"], vg["gain"], t,
                                   rf_candidate=rg["candidate"], vlc_candidate=vg["candidate"])
    ports = channel.select_ports(belief["vlc_gain_mean"])
    candidates = {"RF": rg["candidate"], "VLC": vg["candidate"]}
    gains = {"RF": belief["rf_gain_mean"], "VLC": belief["vlc_gain_mean"]}
    features = build_features(state, belief, rg, vg, ports, business)
    bank = ConditionalTrajectoryBank(estimator, state, context, t, draws=args.draws,
                                     seed=args.forecast_seed + args.block)
    reference = pure_ee_reference(scene.n, cfg, candidates, business, ports, bank.evaluate,
                                  belief_gains=gains)
    if not reference.success:
        raise RuntimeError(f"reference infeasible: {reference.reason}")
    ref_label = evaluate_action(bank, reference.action, reference.eta_ref, args.delta)

    # The replay holds exactly one reference plus ``samples`` proposals; the
    # batch cannot exceed it or update() would refuse to train.
    batch_size = max(1, min(args.batch_size, args.samples + 1))
    learner = HybridMASAC(ModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                                      hidden_dim=args.hidden_dim, encoder=args.encoder),
                          SACConfig(seed=args.learning_seed, batch_size=batch_size))
    learner.observe(features, reference.action, ref_label["mean_ee"], cost=0.0,
                    label_source="reference_forecast")
    proposals = []
    for _ in range(args.samples):
        sample = learner.select_action(features)
        label = evaluate_action(bank, sample.action, reference.eta_ref, args.delta)
        learner.observe(features, sample.action, label["mean_ee"], cost=label["cost"],
                        fallback=False, proposal=sample, label_source="proposal_forecast")
        proposals.append((sample.action, label))
    initial = learner.select_action(features, deterministic=True)
    before = dict(deterministic=evaluate_action(bank, initial.action, reference.eta_ref, args.delta),
                  deterministic_mean_power_w=float(np.mean(initial.action.powers)),
                  proposal_powers=[float(np.mean(a.powers)) for a, _ in proposals],
                  proposal_costs=[l["cost"] for _, l in proposals],
                  proposal_mean_ee=[l["mean_ee"] for _, l in proposals],
                  proposal_min_pair_ee=[l["min_ee"] for _, l in proposals],
                  critic_gap=[np.mean(critic_values(learner, features, reference.action))
                              - np.mean(critic_values(learner, features, a))
                              for a, _ in proposals])
    curve = []
    for step in range(args.updates):
        report = learner.update()
        if step % max(1, args.updates // 6) == 0 or step == args.updates - 1:
            curve.append(dict(update=step, critic_loss=report["critic_loss"],
                              actor_loss=report["actor_loss"],
                              actor_gradient_norm=report["actor_gradient_norm"]))
    final_q_ref = float(np.mean(critic_values(learner, features, reference.action)))
    final_q_prop = [float(np.mean(critic_values(learner, features, a))) for a, _ in proposals]

    deterministic = learner.select_action(features, deterministic=True)
    det_label = evaluate_action(bank, deterministic.action, reference.eta_ref, args.delta)
    stochastic_labels = [evaluate_action(bank, learner.select_action(features).action,
                                         reference.eta_ref, args.delta)
                         for _ in range(args.samples)]
    protected = []
    for _ in range(args.samples):
        sample = learner.select_action(features)
        decision = protect_budget(sample.action, reference, delta=args.delta, n=scene.n, cfg=cfg,
                                  candidates=candidates, business=business, evaluator=bank.evaluate)
        protected.append(decision.fallback)
    result = dict(
        scope="fixed-state diagnostic; not training convergence, not an algorithm comparison",
        n=args.n, block=args.block, draws=args.draws, updates=args.updates,
        reference=ref_label, eta_ref=reference.eta_ref,
        before=before,
        after=dict(final_q_reference=final_q_ref, final_q_proposals=final_q_prop,
                   final_critic_gap=final_q_ref - float(np.mean(final_q_prop)),
                   deterministic=det_label,
                   deterministic_mean_power_w=float(np.mean(deterministic.action.powers)),
                   stochastic_costs=[l["cost"] for l in stochastic_labels],
                   stochastic_mean_ee=[l["mean_ee"] for l in stochastic_labels],
                   stochastic_min_pair_ee=[l["min_ee"] for l in stochastic_labels],
                   protected_fallback_rate=float(np.mean(protected))),
        curve=curve, learner=dict(updates=learner.updates, replay=len(learner.replay)),
        model=asdict(learner.model_config), sac=asdict(learner.config))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(50, 100), default=50)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--encoder", choices=("mlp", "hetero_graph", "hetero_graph_v2"), default="mlp")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--channel-seed", type=int, default=2026102401)
    parser.add_argument("--forecast-seed", type=int, default=2026102500)
    parser.add_argument("--learning-seed", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
