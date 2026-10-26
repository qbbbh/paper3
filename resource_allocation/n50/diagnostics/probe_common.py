"""Shared setup for single-block learning probes.

The probe code is diagnostic, not part of the training entry point: it builds
one frozen block, the causal features and one shared forecast bank, so every
proposal at that state is scored by exactly the same predictor.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from p3base import block, channel
from p3base.config import FoundationConfig
from p3base.estimator import ChannelEstimator
from p3base.fading import link_coherence_s
from p3base.scene import Scene, find_scene_root
from p3learn.features import build_features
from p3learn.forecast import ConditionalTrajectoryBank
from run_foundation import TruthOracle

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FrozenBlock:
    cfg: FoundationConfig
    scene: Scene
    context: object
    features: object
    bank: ConditionalTrajectoryBank
    reference: object
    candidates: dict
    business: np.ndarray
    ports: dict
    channel_seed: int = 0
    block_index: int = 0
    t: float = 0.0


def build_frozen_block(n, block_index, draws, channel_seed, forecast_seed):
    cfg = FoundationConfig()
    directory = find_scene_root(ROOT, n)
    scene = Scene(directory, cfg.vehicle)
    if scene.n != n or block_index >= len(scene.time_s):
        raise ValueError('invalid scale or block')
    context = block.GeometryContext.from_scene(scene)
    oracle = TruthOracle(scene, cfg, channel_seed)
    estimator = ChannelEstimator(scene.n, cfg)
    business = ~np.eye(scene.n, dtype=bool)
    t = block_index * cfg.estimation.block_dt_s
    oracle.advance_to(t)
    state = scene.knot(block_index)
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
    bank = ConditionalTrajectoryBank(estimator, state, context, t, draws=draws,
                                     seed=forecast_seed + block_index)
    # Imported here to keep this module free of circular imports.
    from p3learn.allocation import pure_ee_reference
    reference = pure_ee_reference(n, cfg, candidates, business, ports, bank.evaluate,
                                  belief_gains=gains)
    if not reference.success:
        raise RuntimeError(f'reference infeasible: {reference.reason}')
    return FrozenBlock(cfg=cfg, scene=scene, context=context, features=features,
                       bank=bank, reference=reference, candidates=candidates,
                       business=business, ports=ports, channel_seed=channel_seed,
                       block_index=block_index, t=t)


def action_metrics(bank, action, eta_ref, delta):
    """Forecast-only metrics of one complete action; no truth is sampled.

    ``mean_ee`` is each link's own rate over its own power, averaged: the
    objective.  ``min_ee`` and ``network_ee`` are diagnostics only, and the
    budget is measured on the same quantity as the objective.
    """
    from p3learn.allocation import copy_action
    forecast = bank.evaluate(copy_action(action))
    rate = np.asarray(forecast["mean_rate_bps"], dtype=float)
    power = np.asarray(action.powers, dtype=float)
    pair_ee = rate / power
    network_ee = float(rate.sum() / power.sum())
    mean_ee = float(pair_ee.mean())
    min_ee = float(pair_ee.min())
    cost = max(0.0, 1.0 - delta - mean_ee / eta_ref) if eta_ref else 0.0
    return dict(mean_ee=mean_ee, min_ee=min_ee, network_ee=network_ee, cost=cost,
                mean_power_w=float(power.mean()), min_power_w=float(power.min()),
                max_power_w=float(power.max()), powers=power.tolist())


def truth_window(frozen):
    """Read the block's own per-millisecond physical truth.

    Validation only: callers must run this *after* every decision for the block
    is fixed, and must never feed the result back into training. The truth
    oracle is rebuilt from the same channel seed, so the returned trajectory is
    exactly the one every method is scored against at this block.
    """
    cfg = frozen.cfg
    dt = float(cfg.estimation.fine_dt_s)
    steps = int(round(float(cfg.estimation.block_dt_s) / dt))
    oracle = TruthOracle(frozen.scene, cfg, frozen.channel_seed)
    rf, vlc = [], []
    for offset in range(steps):
        oracle.advance_to(frozen.t + offset * dt)
        truth = oracle.truth()
        rf.append(truth['rf'])
        vlc.append(truth['vlc'])
    return dict(rf=np.stack(rf), vlc=np.stack(vlc), dt_s=dt)


def truth_metrics(frozen, window, action, eta_ref, delta):
    """Executed metrics of one complete action on the block's true channel."""
    fine = block.sample_metrics(action.links, action.powers, action.ports,
                                window['rf'], window['vlc'], frozen.cfg)
    summary = block.summarize_block(action.links, action.powers, fine,
                                    window['dt_s'], frozen.scene.n)
    power = np.asarray(action.powers, dtype=float)
    network_ee = float(summary['network_ee_bit_per_j'])
    mean_ee = float(summary['mean_pair_ee_bit_per_j'])
    cost = max(0.0, 1.0 - delta - mean_ee / eta_ref) if eta_ref else 0.0
    rate = np.asarray(summary['mean_rate_bps'], dtype=float)
    return dict(mean_ee=mean_ee, min_ee=float(summary['min_pair_ee_bit_per_j']),
                network_ee=network_ee, cost=cost,
                jain_pair_ee=float(summary['jain_pair_ee']),
                mean_rate_bps=float(rate.mean()), min_rate_bps=float(rate.min()),
                mean_power_w=float(power.mean()), powers=power.tolist())


def power_summary(powers):
    values = np.asarray(powers, dtype=float).ravel()
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), max=float(values.max()),
                near_floor_fraction=float(np.mean(values <= 1.01e-3)),
                above_half_watt_fraction=float(np.mean(values >= .5)))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
