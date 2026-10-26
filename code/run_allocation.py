"""Short development training/evaluation with fixed 100 ms complete actions.

All decisions finish before future truth is sampled. Predicted trajectories are
shared across reference and proposals; true fine samples are audit-only output.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from p3base import block, channel
from p3base.config import FoundationConfig
from p3base.energy import link_electrical_power_w
from p3base.estimator import ChannelEstimator
from p3base.fading import link_coherence_s
from p3base.scene import Scene, find_scene_root
from p3learn.allocation import copy_action, pure_ee_reference, max_min_allocate, protect_budget
from p3learn.credit import credit_report
from p3learn.data import sha256, resolve_dataset
from p3learn.features import build_features, FEATURE_VERSION, NODE_DIM, EDGE_DIM
from p3learn.forecast import ConditionalTrajectoryBank
from p3learn.models import ModelConfig
from p3learn.masac import HybridMASAC, SACConfig
from run_foundation import TruthOracle

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "code").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def action_hash(action):
    if action is None:
        return ""
    values = [[int(i), int(j), mode, float(p),
               int(action.ports["tx"][i, j]) if mode == "VLC" else -1,
               int(action.ports["rx"][i, j]) if mode == "VLC" else -1]
              for (i, j, mode), p in zip(action.links, action.powers)]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def validate_checkpoint_provenance(learner, *, args, dataset, traffic_hashes, manifest_hash, physical_hash):
    provenance = getattr(learner, "provenance", {})
    if not provenance:
        raise ValueError("checkpoint has no training provenance")
    required = {"split", "n", "dataset_manifest_sha256", "training_traffic_seed", "traffic_hashes",
                "channel_seed", "forecast_seed", "forecast_seeds", "feature_version",
                "physical_config_sha256", "source_sha256", "run_config_sha256"}
    if not required <= set(provenance):
        raise ValueError("checkpoint training provenance is incomplete")
    if (not isinstance(provenance["forecast_seeds"], list)
            or not provenance["forecast_seeds"]
            or not all(isinstance(seed, int) for seed in provenance["forecast_seeds"])):
        raise ValueError("checkpoint must list realized training forecast seeds")
    if (provenance.get("n") != args.n or provenance.get("feature_version") != FEATURE_VERSION
            or provenance.get("physical_config_sha256") != physical_hash):
        raise ValueError("checkpoint scale/features/physical configuration mismatch")
    if args.phase == "evaluate" and args.split in ("validation", "test"):
        if provenance.get("split") != "train" or provenance.get("dataset_manifest_sha256") != manifest_hash:
            raise ValueError("independent evaluation requires training from this frozen partition registry")
        if (provenance.get("training_traffic_seed") == dataset["traffic_seed"]
                or provenance.get("traffic_hashes", {}).get("mobility.npz") == traffic_hashes["mobility.npz"]):
            raise ValueError("training/evaluation traffic overlap")
        if (provenance.get("channel_seed") == args.channel_seed
                or provenance.get("forecast_seed") == args.forecast_seed):
            raise ValueError("training/evaluation channel and forecast seeds must be independent")
        evaluation_forecast_seeds = {args.forecast_seed+k for k in range(args.start_block, args.start_block+args.blocks)}
        if evaluation_forecast_seeds & set(provenance.get("forecast_seeds", [])):
            raise ValueError("training/evaluation realized forecast seeds overlap")
    if args.phase == "train":
        if provenance.get("split") not in ("train", "development"):
            raise ValueError("cannot continue training from a validation/test checkpoint")
        if args.split == "train" and provenance.get("split") != "train":
            raise ValueError("development data weights cannot enter frozen independent training")
        if (provenance.get("dataset_manifest_sha256") != manifest_hash
                or provenance.get("training_traffic_seed") != dataset["traffic_seed"]
                or provenance.get("traffic_hashes") != traffic_hashes
                or provenance.get("channel_seed") != args.channel_seed
                or provenance.get("forecast_seed") != args.forecast_seed):
            raise ValueError("continuation requires the same frozen training data and seeds; mixed-history training is unsupported")


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def dump_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _current_geometry(context, state, cfg):
    # This function is deliberately independent of the truth provider.
    return (channel.rf_geometry(context, state, cfg.rf),
            channel.vlc_geometry(context, context.ports(state), cfg.vlc))


def run(args):
    if args.n == 10 and args.vlc_sinr is None:
        args.vlc_sinr = "square_law"
    if args.power_solver is None:
        args.power_solver = "continuous" if args.n == 10 else "ascent"
    if args.q_min_bps is None:
        args.q_min_bps = 1e6 if args.n == 10 else 0.
    if not np.isfinite(args.q_min_bps) or args.q_min_bps < 0:
        raise ValueError("q-min-bps must be finite and nonnegative")
    if args.q_min_bps > 0:
        if args.power_solver != "continuous" or args.methods != ["reference"]:
            raise ValueError("QoS-constrained development currently requires continuous solver and methods reference")
        if args.q_eval_bps is None:
            args.q_eval_bps = args.q_min_bps
    if args.qos_metric != "mean" and args.q_min_bps <= 0:
        raise ValueError("a non-mean QoS metric requires --q-min-bps > 0")
    if args.qos_chance_rounds < 0:
        raise ValueError("qos chance rounds must be nonnegative")
    if not np.isfinite(args.delta) or not 0 <= args.delta <= 1:
        raise ValueError("delta must be explicitly supplied in [0,1]")
    if args.blocks < 1 or args.draws < 2 or args.start_block < 1 or args.batch_size < 1:
        raise ValueError("require positive blocks/batch, >=2 draws and start-block>=1")
    if args.q_eval_bps is not None and (not np.isfinite(args.q_eval_bps) or args.q_eval_bps < 0):
        raise ValueError("q-eval-bps must be finite and nonnegative")
    if args.threads < 1 or args.updates_per_block < 0 or len(set(args.methods)) != len(args.methods):
        raise ValueError("require positive threads, nonnegative updates and unique methods")
    if args.credit_max_evals < 0 or args.credit_focus < 0:
        raise ValueError("credit caps must be nonnegative")
    if args.phase == "train" and args.split not in ("train", "development"):
        raise ValueError("validation/test partitions cannot train or update weights")
    if args.phase == "evaluate" and "policy" in args.methods and args.checkpoint is None:
        raise ValueError("policy evaluation requires an explicit checkpoint")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("output directory is not empty; choose a fresh run directory")
    if args.dataset_manifest:
        directory, dataset, manifest_hash = resolve_dataset(args.dataset_manifest, args.split, args.n)
    else:
        if args.split != "development":
            raise ValueError("independent train/validation/test runs require --dataset-manifest")
        directory = find_scene_root(ROOT, args.n)
        scenario_path = directory / "scenario.json"
        traffic_seed = json.loads(scenario_path.read_text(encoding="utf-8")).get("seed") if scenario_path.exists() else None
        dataset, manifest_hash = dict(traffic_seed=traffic_seed, directory=str(directory)), None
    torch.set_num_threads(args.threads)
    cfg = FoundationConfig()
    if args.estimator == "complex_ar1":
        cfg = cfg.with_(estimation=replace(cfg.estimation, rf_state="complex_ar1"))
    if args.vlc_sinr or args.vlc_noise:
        cfg = cfg.with_(vlc=replace(cfg.vlc,
                                    sinr_model=args.vlc_sinr or cfg.vlc.sinr_model,
                                    noise_model=args.vlc_noise or cfg.vlc.noise_model))
    scene = Scene(directory, cfg.vehicle)
    if scene.n != args.n or args.start_block + args.blocks > len(scene.time_s):
        raise ValueError("scene scale or requested block range is invalid")
    context = block.GeometryContext.from_scene(scene)
    oracle = TruthOracle(scene, cfg, args.channel_seed)
    estimator = ChannelEstimator(scene.n, cfg)
    business = ~np.eye(scene.n, dtype=bool)  # Shared-information unicast eligibility, all other vehicles.
    encoder = args.encoder or ("hetero_graph_v2" if args.n == 100 else "mlp")
    checkpoint_hash = sha256(args.checkpoint) if args.checkpoint else None
    traffic_hashes = {name: sha256(directory / name) for name in ("mobility.npz", "buildings.npz")}
    physical_hash = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
    learner = None
    if "policy" in args.methods:
        if args.checkpoint:
            learner = HybridMASAC.load(args.checkpoint)
            if learner.model_config.encoder != encoder:
                raise ValueError("checkpoint encoder differs from requested encoder")
            validate_checkpoint_provenance(learner, args=args, dataset=dataset, traffic_hashes=traffic_hashes,
                                           manifest_hash=manifest_hash, physical_hash=physical_hash)
        else:
            learner = HybridMASAC(ModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=args.hidden_dim,
                                              encoder=encoder),
                                   SACConfig(seed=args.learning_seed, batch_size=args.batch_size,
                                             replay_capacity=args.replay_capacity))
    args.out.mkdir(parents=True, exist_ok=True)
    code_hash = source_hash()
    config = dict(physical=cfg.to_dict(), delta=args.delta, q_eval_bps=args.q_eval_bps,
                  q_min_bps=args.q_min_bps, qos_metric=args.qos_metric,
                  qos_fallback_mean=bool(args.qos_fallback_mean),
                  qos_chance_rounds=args.qos_chance_rounds,
                  qos_chance_hold_mean=bool(args.qos_chance_hold_mean),
                  methods=args.methods, estimator=args.estimator, encoder=encoder, phase=args.phase,
                  power_solver=args.power_solver, reference_search=args.reference_search,
                  joint_rounds=args.joint_rounds, joint_max_neighbours=args.joint_max_neighbours,
                  feature_version=FEATURE_VERSION, draws=args.draws, start_block=args.start_block,
                  blocks=args.blocks, threads=args.threads, split=args.split,
                  channel_seed=args.channel_seed, forecast_seed=args.forecast_seed,
                  learning_seed=args.learning_seed, batch_size=args.batch_size,
                  updates_per_block=args.updates_per_block, replay_capacity=args.replay_capacity,
                  training_labels=args.training_labels,
                  model=asdict(learner.model_config) if learner else None,
                  sac=asdict(learner.config) if learner else None)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    rows, raw, block_records, training_curve = [], {}, [], []
    power_solver_records = []
    credit_records, credit_elapsed_s = [], 0.
    started = time.perf_counter()
    for episode_block in range(args.blocks):
        k = args.start_block + episode_block
        t = k * cfg.estimation.block_dt_s
        oracle.advance_to(t)
        # The estimator consumes only arrived observations; geometry and relative
        # speed below are derived from the currently recorded knot.
        state = scene.knot(k)
        rg, vg = _current_geometry(context, state, cfg)
        v = state["speed_mps"][:, None] * np.stack([np.cos(state["heading_rad"]), np.sin(state["heading_rad"])], -1)
        rel = np.linalg.norm(v[:, None] - v[None], axis=-1)
        sigma = np.where(rg["los"], cfg.rf.los_shadow_sigma_db, cfg.rf.nlos_shadow_sigma_db)
        prior = np.exp((sigma * np.log(10.) / 10.) ** 2 / 2.)
        estimator.advance(t, oracle, rf_prior_var=prior, rf_tau=link_coherence_s(rel, cfg.rf))
        belief = estimator.gain_belief(rg["gain"], vg["gain"], t,
                                       rf_candidate=rg["candidate"], vlc_candidate=vg["candidate"])
        ports = channel.select_ports(belief["vlc_gain_mean"])
        candidates = {"RF": rg["candidate"], "VLC": vg["candidate"]}
        gains = {"RF": belief["rf_gain_mean"], "VLC": belief["vlc_gain_mean"]}
        features = build_features(state, belief, rg, vg, ports, business) if learner else None
        decision_start = time.perf_counter()
        bank = ConditionalTrajectoryBank(estimator, state, context, t, draws=args.draws,
                                          seed=args.forecast_seed + k)
        evaluator = bank.evaluate  # Keep a stable callable identity for the frozen reference.
        reference = pure_ee_reference(scene.n, cfg, candidates, business, ports, evaluator,
                                      belief_gains=gains, power_solver=args.power_solver,
                                      search=args.reference_search,
                                      joint_rounds=args.joint_rounds,
                                      joint_max_neighbours=args.joint_max_neighbours,
                                      q_min_bps=args.q_min_bps,
                                      qos_metric=args.qos_metric,
                                      qos_fallback_mean=bool(args.qos_fallback_mean),
                                      chance_rounds=args.qos_chance_rounds,
                                      chance_hold_mean=bool(args.qos_chance_hold_mean))
        if args.power_solver == "continuous":
            power_solver_records.append(dict(block=k, success=reference.success,
                                             reason=reference.reason,
                                             policy_config=reference.policy_config))
        decisions, policy_sample, inference_s = {}, None, 0.
        if "reference" in args.methods:
            decisions["reference"] = reference
        if "maxmin" in args.methods:
            decisions["maxmin"] = max_min_allocate(scene.n, cfg, candidates, business, ports, evaluator,
                                                   delta=args.delta, belief_gains=gains, reference=reference)
        if "policy" in args.methods:
            inference_start = time.perf_counter()
            try:
                policy_sample = learner.select_action(features, deterministic=args.phase == "evaluate")
                proposal = policy_sample.action
            except ValueError as exc:
                if "coverage_failure" not in str(exc):
                    raise
                proposal = None
            inference_s = time.perf_counter() - inference_start
            decisions["policy"] = protect_budget(proposal, reference, delta=args.delta, n=scene.n,
                                                   cfg=cfg, candidates=candidates, business=business, evaluator=evaluator)
        decisions_wall_s = time.perf_counter() - decision_start
        # Credit is a forecast-only diagnostic.  It runs after every decision is
        # frozen (so it cannot influence them) and before truth is sampled, and
        # never reaches the learner, the action set or the audit outputs.
        if args.credit:
            subject = decisions.get("policy") or decisions.get("maxmin") or decisions.get("reference")
            if subject is not None and subject.success:
                credit_start = time.perf_counter()
                calls_before = getattr(bank, "evaluator_calls", 0)
                try:
                    record, _ = credit_report(
                        subject.action, subject.forecast, n=scene.n, cfg=cfg,
                        candidates=candidates, business=business, ports_master=ports,
                        evaluator=evaluator, block=k,
                        forecast_seed=args.forecast_seed + k, delta=args.delta,
                        eta_ref=reference.eta_ref,
                        proposal=None if policy_sample is None else policy_sample.action,
                        fallback=subject.fallback, max_evals=args.credit_max_evals,
                        focus=args.credit_focus)
                    record["evaluator_calls"] = int(getattr(bank, "evaluator_calls", 0)) - calls_before
                    credit_records.append(record)
                except Exception as exc:  # a diagnostic must never fail a block
                    credit_records.append(dict(block=k, error=f"{type(exc).__name__}: {exc}"))
                credit_elapsed_s += time.perf_counter() - credit_start
        # Proposal labels for learning.  When the shield executed the reference,
        # the proposal still keeps its own forecast label: the critic must see
        # that a rejected proposal is bad, rather than inheriting the reference
        # reward.  The executed action stays in the evaluation rows only.
        training_forecast, training_skip = None, ""
        if (args.phase == "train" and args.training_labels == "proposal"
                and policy_sample is not None):
            subject = decisions.get("policy")
            if subject is not None and subject.success and subject.fallback:
                try:
                    training_forecast = bank.evaluate(copy_action(policy_sample.action))
                except Exception as exc:  # never label a rejected proposal with the reference
                    training_skip = f"proposal_label_unavailable: {type(exc).__name__}: {exc}"
        # No decisions or feature construction occur below this point until the
        # next block. Truth arrays never enter the learner; only scalar rewards do.
        actions = {action_hash(result.action): result.action for result in [reference, *decisions.values()] if result.success}
        fine_by_action = {key: {field: [] for field in ("rate_bps", "sinr", "signal", "interference", "noise")}
                          for key in actions}
        duration, dt = cfg.estimation.block_dt_s, cfg.estimation.fine_dt_s
        steps = int(round(duration / dt))
        evaluation_started = time.perf_counter()
        for offset in range(steps):
            oracle.advance_to(t + offset * dt)
            truth = oracle.truth()
            for key, action in actions.items():
                fine = block.sample_metrics(action.links, action.powers, action.ports, truth["rf"], truth["vlc"], cfg)
                for field in fine:
                    fine_by_action[key][field].append(fine[field])
        fine_by_action = {key: {field: np.asarray(values) for field, values in fine.items()}
                          for key, fine in fine_by_action.items()}
        actual = {key: block.summarize_block(action.links, action.powers, fine_by_action[key], dt, scene.n,
                                             q_eval_bps=args.q_eval_bps, cfg=cfg) for key, action in actions.items()}
        actual_reference = actual.get(action_hash(reference.action))
        evaluation_wall_s = time.perf_counter() - evaluation_started
        plan = estimator.plan
        pilot_per_vehicle_j = duration * (plan.rf_energy_per_vehicle_per_s + plan.vlc_energy_per_vehicle_per_s)
        for method, decision in decisions.items():
            method_solve_s = bank.build_time_s + reference.solve_time_s
            if method != "reference":
                method_solve_s += decision.solve_time_s
            if method == "policy":
                method_solve_s += inference_s
            common = dict(schema_version=1, method=method, ablation="pure_mean_pair_ee" if method == "reference" else "none",
                          information_version="causal_conditional_trajectory_mean", estimator=args.estimator,
                          architecture=encoder if method == "policy" else "matching_heuristic", phase=args.phase,
                          split=args.split, scenario=f"n{scene.n}", n=scene.n, episode=0, block=k,
                          start_time_s=t, duration_s=duration, traffic_seed=dataset["traffic_seed"],
                          channel_seed=args.channel_seed, forecast_seed=args.forecast_seed+k, learning_seed=args.learning_seed,
                          source_sha256=code_hash, config_sha256=config_hash, mobility_sha256=traffic_hashes["mobility.npz"],
                          buildings_sha256=traffic_hashes["buildings.npz"], dataset_manifest_sha256=manifest_hash,
                          input_checkpoint_sha256=checkpoint_hash, reference_version=reference.reference_version,
                          delta=args.delta, q_eval_bps=args.q_eval_bps, fallback=decision.fallback,
                          coverage_failure=decision.coverage_failure, decision_reason=decision.reason,
                          failure_reason=decision.reason if not decision.success else "",
                          fallback_reason=decision.reason if decision.fallback else "",
                          solve_time_s=method_solve_s, online_within_100ms=method_solve_s <= duration,
                          actor_and_decoder_time_s=inference_s if method == "policy" else None,
                          forecast_build_time_s=bank.build_time_s, forecast_memory_bytes=bank.bytes,
                          reference_solve_time_s=reference.solve_time_s, all_methods_decision_wall_s=decisions_wall_s,
                          truth_evaluation_wall_s=evaluation_wall_s, eta_ref=reference.eta_ref,
                          # eta_ref is a mean per-link EE, so the actual counterpart
                          # must be the same quantity; network EE stays separate.
                          actual_eta_ref=(actual_reference["mean_pair_ee_bit_per_j"]
                                          if actual_reference else None),
                          actual_reference_network_ee=(actual_reference["network_ee_bit_per_j"]
                                                       if actual_reference else None),
                          rf_pilot_overhead=plan.rf_overhead, vlc_pilot_overhead=plan.vlc_overhead,
                          block_pilot_energy_j=scene.n * pilot_per_vehicle_j,
                          proposal_budget_violation=decision.budget_violation,
                          action_sha256=action_hash(decision.action))
            if not decision.success:
                rows.append(dict(common, row_type="block_failure", selected=False, active=False))
                block_records.append(dict(common, success=False, delivered_bits=0., data_energy_j=0.,
                                          min_pair_ee_bit_per_j=0., mean_pair_ee_bit_per_j=0.,
                                          mean_pair_ee_ratio_vs_actual_reference=0.,
                                          network_ee_bit_per_j=0.))
                continue
            action = decision.action
            measured = actual[common["action_sha256"]]
            forecast = decision.forecast
            predicted_network = float(forecast["mean_rate_bps"].sum() / action.powers.sum())
            predicted_electric = link_electrical_power_w(
                action.powers, [mode for _i, _j, mode in action.links], cfg)
            predicted_pair_ee = np.asarray(forecast["mean_rate_bps"], float) / predicted_electric
            predicted_mean_pair = float(predicted_pair_ee.mean())
            budget = (1-args.delta) * reference.eta_ref
            actual_eta = measured["network_ee_bit_per_j"]
            actual_mean_pair = measured["mean_pair_ee_bit_per_j"]
            actual_ref_eta = actual_reference["network_ee_bit_per_j"]
            actual_ref_mean_pair = actual_reference["mean_pair_ee_bit_per_j"]
            selected_vlc = np.array([mode == "VLC" for _, _, mode in action.links])
            common.update(predicted_network_ee=predicted_network,
                          predicted_mean_pair_ee=predicted_mean_pair,
                          # Both guard the averaged per-link EE, which is the objective.
                          predicted_budget_violation=max(0., budget-predicted_mean_pair),
                          actual_budget_violation_vs_predicted_reference=max(0., budget-actual_mean_pair),
                          actual_budget_violation_vs_actual_reference=max(0., (1-args.delta)*actual_ref_mean_pair-actual_mean_pair),
                          actual_network_loss_fraction_vs_actual_reference=1-actual_eta/actual_ref_eta if actual_ref_eta else None,
                          actual_mean_pair_loss_fraction_vs_actual_reference=(
                              1-actual_mean_pair/actual_ref_mean_pair if actual_ref_mean_pair else None),
                          mean_pair_ee_bit_per_j=actual_mean_pair,
                          mean_pair_ee_ratio_vs_actual_reference=(actual_mean_pair/actual_ref_mean_pair
                                                                  if actual_ref_mean_pair else None),
                          network_ee_bit_per_j=actual_eta, min_pair_ee_bit_per_j=measured["min_pair_ee_bit_per_j"],
                          jain_pair_ee=measured["jain_pair_ee"], vlc_link_fraction=measured["vlc_link_fraction"],
                          vlc_rate_fraction=measured["vlc_rate_fraction"],
                          vlc_power_fraction=float(action.powers[selected_vlc].sum()/action.powers.sum()),
                          predicted_total_data_power_w=float(action.powers.sum()),
                          predicted_mean_data_power_w=float(action.powers.mean()),
                          predicted_total_electrical_power_w=float(predicted_electric.sum()),
                          predicted_vlc_link_fraction=float(selected_vlc.mean()),
                          all_pairs_qos_met=measured.get("all_pairs_qos_met"),
                          qos_metric=args.qos_metric,
                          chance_fallback=bool(reference.policy_config.get("chance_fallback", False)),
                          pilot_inclusive_gross_ee_audit=measured["link_delivered_bits"].sum()/
                          (measured["link_energy_j"].sum()+scene.n*pilot_per_vehicle_j))
            prefix = f"{method}_b{k}"
            fine = fine_by_action[common["action_sha256"]]
            for field, values in fine.items():
                raw[f"{prefix}_{field}"] = values
            raw.update({f"{prefix}_power_w": action.powers,
                        f"{prefix}_sample_time_s": t + np.arange(steps)*dt,
                        f"{prefix}_tx": np.array([e[0] for e in action.links]),
                        f"{prefix}_rx": np.array([e[1] for e in action.links]),
                        f"{prefix}_tx_port": np.array([int(action.ports['tx'][i,j]) if m == 'VLC' else -1 for i,j,m in action.links]),
                        f"{prefix}_rx_port": np.array([int(action.ports['rx'][i,j]) if m == 'VLC' else -1 for i,j,m in action.links]),
                        f"{prefix}_link_index": np.arange(len(action.links)),
                        f"{prefix}_block_index": np.array(k),
                        f"{prefix}_mode": np.array([e[2] for e in action.links]),
                        f"{prefix}_forecast_samples_bps": forecast["block_rate_samples_bps"]})
            for link_index, ((tx, rx, mode), power) in enumerate(zip(action.links, action.powers)):
                tp, rp = int(action.ports["tx"][tx, rx]), int(action.ports["rx"][tx, rx])
                selected = (tx, rx) if mode == "RF" else (tx, rx, tp, rp)
                name = mode.lower()
                values = dict(common, row_type="link", fine_prefix=prefix, link_index=link_index,
                              tx=tx, rx=rx, tx_id=str(scene.vehicle_ids[tx]), rx_id=str(scene.vehicle_ids[rx]), mode=mode,
                              tx_port=tp if mode == "VLC" else -1, rx_port=rp if mode == "VLC" else -1,
                              selected=True, active=True, power_w=power, candidate_eligible=bool(candidates[mode][tx, rx]),
                              business_allowed=bool(business[tx, rx]), predicted_rate_bps=forecast["mean_rate_bps"][link_index],
                              predicted_p05_bps=forecast["low_rate_bps"][link_index],
                              predicted_p95_bps=forecast["high_rate_bps"][link_index],
                              actual_rate_bps=measured["mean_rate_bps"][link_index],
                              delivered_bits=measured["link_delivered_bits"][link_index],
                              data_energy_j=measured["link_energy_j"][link_index], pair_pilot_energy_j=2*pilot_per_vehicle_j,
                              pair_rf_pilot_energy_j=2*duration*plan.rf_energy_per_vehicle_per_s,
                              pair_vlc_pilot_energy_j=2*duration*plan.vlc_energy_per_vehicle_per_s,
                              pair_ee_bit_per_j=measured["pair_ee_bit_per_j"][link_index],
                              pair_qos_met=measured["pair_qos_met"][link_index] if args.q_eval_bps is not None else None,
                              tx_x_m=state["center_xy_m"][tx, 0], tx_y_m=state["center_xy_m"][tx, 1],
                              rx_x_m=state["center_xy_m"][rx, 0], rx_y_m=state["center_xy_m"][rx, 1],
                              tx_heading_rad=state["heading_rad"][tx], rx_heading_rad=state["heading_rad"][rx],
                              csi_age_s=belief[name+"_age_s"][selected], csi_missing=belief[name+"_missing"][selected],
                              state_mean=json.dumps(json_value(belief[name+"_state_mean"][selected])),
                              state_variance=belief[name+"_state_var"][selected],
                              absolute_gain_mean=belief[name+"_gain_mean"][selected],
                              geometry_gain=(rg if mode == "RF" else vg)["gain"][selected], rf_los=rg["los"][tx, rx],
                              signal_unit=("W" if mode == "RF" else
                                           ("W_equiv" if cfg.vlc.sinr_model == "linear" else "A^2")))
                rows.append(json_value(values))
            block_records.append(dict(common, success=True, delivered_bits=float(measured["link_delivered_bits"].sum()),
                                      data_energy_j=float(measured["link_energy_j"].sum())))
            if method == "policy" and args.phase == "train":
                if training_forecast is not None:
                    label_action, label_forecast = policy_sample.action, training_forecast
                    label_source = "proposal_forecast"
                else:
                    label_action, label_forecast = action, forecast
                    label_source = "executed_forecast"
                if training_skip:
                    # Keep the executed label from a rejected proposal out of
                    # replay; the block is recorded instead of silently relabelled.
                    training_curve.append(dict(block=k, skipped=True, reason=training_skip,
                                               label_source="none"))
                else:
                    # Objective = each link's own rate / its own power, averaged.
                    # network EE (sum rate / sum power) is a different quantity and
                    # is recorded for reporting only, never used as the reward.
                    label_pair_ee = (np.asarray(label_forecast["mean_rate_bps"], float)
                                     / link_electrical_power_w(
                                         label_action.powers,
                                         [mode for _i, _j, mode in label_action.links], cfg))
                    predicted_mean_ee = float(label_pair_ee.mean())
                    normalized_cost = (max(0., 1-args.delta-predicted_mean_ee/reference.eta_ref)
                                       if reference.eta_ref else 0.)
                    learner.observe(features, label_action, predicted_mean_ee, cost=normalized_cost,
                                    fallback=decision.fallback, proposal=policy_sample,
                                    label_source=label_source)
                for update_index in range(args.updates_per_block):
                    training_curve.append(dict(block=k, update_in_block=update_index,
                                               reward_source="mean_of_conditional_mean_pair_ee",
                                               label_source=label_source,
                                               predicted_mean_pair_ee=None if training_skip else predicted_mean_ee,
                                               predicted_min_pair_ee=None if training_skip else float(label_pair_ee.min()),
                                               predicted_network_ee=None if training_skip else float(
                                                   label_forecast["mean_rate_bps"].sum()/label_action.powers.sum()),
                                               predicted_budget_violation=None if training_skip else normalized_cost,
                                               executed_fallback=decision.fallback,
                                               actual_mean_pair_ee_audit_only=measured["mean_pair_ee_bit_per_j"],
                                               actual_min_pair_ee_audit_only=measured["min_pair_ee_bit_per_j"],
                                               **learner.update()))
        print(f"n{scene.n} {args.split} {args.phase} block {episode_block+1}/{args.blocks}; "
              f"forecast {bank.build_time_s:.3f}s; decisions {decisions_wall_s:.3f}s", flush=True)
    fields = list(dict.fromkeys(["row_type", "tx", "rx", "mode"] + [key for row in rows for key in row]))
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_value(rows))
    raw["audit_only"] = np.array(True)
    raw["units"] = np.array("time=s;rate=bit/s;sinr=linear;RF S/I/N=W;VLC S/I/N=A^2")
    np.savez_compressed(args.out / "fine_samples.npz", **raw)
    if learner and args.phase == "train":
        provenance = dict(split=args.split, n=args.n, dataset_manifest_sha256=manifest_hash,
                          training_traffic_seed=dataset["traffic_seed"], traffic_hashes=traffic_hashes,
                          channel_seed=args.channel_seed, forecast_seed=args.forecast_seed,
                          feature_version=FEATURE_VERSION, physical_config_sha256=physical_hash,
                          source_sha256=code_hash, run_config_sha256=config_hash)
        provenance["forecast_seeds"] = sorted(set(getattr(learner, "provenance", {}).get("forecast_seeds", [])) |
                                               {args.forecast_seed+k for k in range(args.start_block, args.start_block+args.blocks)})
        learner.save(args.out / "model.pt", provenance=provenance)
    dump_json(args.out / "training_curve.json", training_curve)
    if args.power_solver == "continuous":
        dump_json(args.out / "continuous_power_audit.json", power_solver_records)
    if args.credit:
        dump_json(args.out / "credit.json", dict(
            schema_version=1, information="conditional_trajectory_forecast_only",
            calibration_status="approximate_not_certified",
            max_evals=args.credit_max_evals, focus=args.credit_focus,
            elapsed_s=credit_elapsed_s, blocks=credit_records))
    method_summary = {}
    for method in args.methods:
        records = [r for r in block_records if r["method"] == method]
        successes = [r for r in records if r["success"]]
        def stat(key):
            data = np.array([r.get(key, 0.) or 0. for r in records], float)
            return dict(mean=float(data.mean()), std=float(data.std()), min=float(data.min()), max=float(data.max()))
        bits = sum(r["delivered_bits"] for r in records)
        energy = sum(r["data_energy_j"] for r in records)
        method_summary[method] = dict(blocks=len(records), successful_blocks=len(successes),
                                      failure_rate=1-len(successes)/len(records),
                                      fallback_rate=sum(r["fallback"] for r in records)/len(records),
                                      total_delivered_bits=bits, total_data_energy_j=energy,
                                      cumulative_network_ee_bit_per_j=bits/energy if energy else None,
                                      mean_pair_ee_including_failed_as_zero=stat("mean_pair_ee_bit_per_j"),
                                      mean_pair_ee_ratio_vs_reference=stat("mean_pair_ee_ratio_vs_actual_reference"),
                                      min_pair_ee_including_failed_as_zero=stat("min_pair_ee_bit_per_j"),
                                      block_network_ee_diagnostic=stat("network_ee_bit_per_j"),
                                      # 全网功率与 VLC 使用率：性能汇报的固定两项。
                                      mean_total_data_power_w=stat("predicted_total_data_power_w"),
                                      mean_total_electrical_power_w=stat("predicted_total_electrical_power_w"),
                                      mean_vlc_link_fraction=stat("vlc_link_fraction"),
                                      mean_vlc_power_fraction=stat("vlc_power_fraction"),
                                      solve_time_s=stat("solve_time_s"),
                                      real_budget_violation_rate=sum(r.get("actual_budget_violation_vs_actual_reference", 0)>1e-6
                                                                     for r in records)/len(records))
    result = dict(schema_version=1, stage="development_short_run", trained_to_convergence=False,
                  independent_traffic=args.split != "development", config=config, config_sha256=config_hash,
                  source_sha256=code_hash, traffic=dataset, traffic_hashes=traffic_hashes,
                  dataset_manifest_sha256=manifest_hash, methods=method_summary, blocks=block_records,
                  pilot_plan=estimator.plan.as_dict(), elapsed_s=time.perf_counter()-started,
                  environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                   numpy=np.__version__, torch=torch.__version__, cuda_available=torch.cuda.is_available()),
                  updates=learner.updates if learner else 0, checkpoint_sha256=sha256(args.out/"model.pt")
                  if (args.out/"model.pt").exists() else checkpoint_hash,
                  scope=["Gross PHY data-transmit EE, pilot energy separately audited",
                         "Predicted budget protection does not guarantee the true future budget",
                         "Few forecast draws and short runs cannot establish coverage or policy ranking",
                         "100ms communication duration is not a computation deadline guarantee",
                         "Fallback-only replay has narrow action coverage; nonzero gradients do not show convergence",
                         *(["Credit deltas are conditional-trajectory forecasts, not measured truth"] if args.credit else [])])
    dump_json(args.out / "summary.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, choices=(10, 50, 100), required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--phase", choices=("train", "evaluate"), default="evaluate")
    p.add_argument("--split", choices=("development", "train", "validation", "test"), default="development")
    p.add_argument("--dataset-manifest", type=Path)
    p.add_argument("--methods", choices=("reference", "maxmin", "policy"), nargs="+", default=["reference", "maxmin", "policy"])
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--start-block", type=int, default=2)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--channel-seed", type=int, default=2026102401)
    p.add_argument("--forecast-seed", type=int, default=2026102501)
    p.add_argument("--learning-seed", type=int, default=2026102601)
    p.add_argument("--encoder", choices=("mlp", "hetero_graph", "hetero_graph_v2"))
    p.add_argument("--credit", action="store_true",
                   help="write a forecast-only counterfactual credit.json diagnostic")
    p.add_argument("--credit-max-evals", type=int, default=48,
                   help="cap on replacement evaluations per block; 0 means no cap")
    p.add_argument("--credit-focus", type=int, default=3,
                   help="number of weakest executed links whose replacements are tried first")
    p.add_argument("--estimator", choices=("two_state", "complex_ar1"), default="two_state")
    p.add_argument("--vlc-sinr", choices=("linear", "square_law"),
                   help="覆盖 cfg.vlc.sinr_model；linear 只用于复现 dink10.m")
    p.add_argument("--vlc-noise", choices=("n0_only", "physical", "physical_isi"),
                   help="覆盖 cfg.vlc.noise_model；physical_isi 会把期望信号算成噪声")
    p.add_argument("--power-solver", choices=("ascent", "continuous", "dinkelbach"), default=None,
                   help="10车默认continuous=连续二次变换/SCA；ascent=旧八档网格基线，"
                        "dinkelbach=分式规划外层（二次变换）+ SCA 内层")
    p.add_argument("--reference-search", choices=("candidates", "joint"), default="candidates",
                   help="candidates=只在启发式候选匹配里选（旧基线）；"
                        "joint=匹配+功率联合局部搜索（主算法）")
    p.add_argument("--joint-rounds", type=int, default=8,
                   help="joint 局部搜索的最大轮数（无改进即提前停止）")
    p.add_argument("--joint-max-neighbours", type=int, default=0,
                   help="每轮最多评估的邻居数（0=不限；按先动最弱链路截断）")
    p.add_argument("--q-min-bps", type=float, default=None,
                   help="predicted block-mean rate constraint; n10 default 1e6, other scales 0")
    p.add_argument("--q-eval-bps", type=float)
    p.add_argument("--qos-metric", choices=("mean", "chance"), default="mean",
                   help="QoS variable: predictive mean (default) or the 5% posterior "
                        "quantile of the same block-rate samples (per-link 95% chance constraint)")
    p.add_argument("--qos-fallback-mean", action="store_true",
                   help="chance mode only: if no candidate meets the quantile constraint, "
                        "fall back to the best mean-feasible action and flag the block")
    p.add_argument("--qos-chance-rounds", type=int, default=2,
                   help="chance repair rounds (per-link target escalation) after the search")
    p.add_argument("--qos-chance-hold-mean", action="store_true",
                   help="chance repair only: also pin links that already clear the "
                        "chance threshold at their pre-repair predicted mean (stricter "
                        "than the stated constraint; default off)")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--replay-capacity", type=int, default=128)
    p.add_argument("--updates-per-block", type=int, default=1)
    p.add_argument("--training-labels", choices=("proposal", "executed"), default="proposal",
                   help="proposal: replay the actor's proposal with its own forecast label; "
                        "executed: legacy behaviour, label the shield's executed action only")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    return p


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    options = parser().parse_args()
    run(options)
