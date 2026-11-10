"""Re-evaluate saved fair-power decisions without solving the initial allocation.

Old run directories are read-only. Decision inputs are reconstructed from causal
estimator checkpoints and the original forecast seed. Future truth is evaluated
only after improve_fair_power has returned. No learner or policy is instantiated.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parent
EXPERIMENT = CODE.parent
STORAGE = EXPERIMENT.parent
sys.path.insert(0, str(CODE))

from p3base import block, channel
from p3base.config import (FoundationConfig, RFConfig, VLCConfig, VehicleConfig,
                           EstimationConfig)
from p3base.energy import link_electrical_power_w
from p3base.estimator import ChannelEstimator
from p3base.fading import link_coherence_s
from p3base.scene import Scene
from p3learn.allocation import Action, predicted_qos_rates
from p3learn.forecast import ConditionalTrajectoryBank
from run_foundation import TruthOracle

FIXED_BLOCKS = {"development": [0, 10, 20, 30, 40],
                "validation": list(range(100, 1000, 100))}
SOURCE_RUNS = {
    "10A": ("对比算法安全运行_20261106/runs/n10_A_100s_1000blocks", None),
    "10B": ("对比算法安全运行_20261106/runs/n10_B_100s_1000blocks", None),
    "50B": ("长跑保存修复_20261105/runs/R2_n50_d010_full100s_1000blocks",
             "50车补跑_20261108/runs/n50_B_missing_100s_1000blocks"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def assert_same_samples(actual, expected, label):
    """Strict cross-version numerical equality, without claiming bitwise identity."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"{label}: shape/nonfinite mismatch")
    error = np.abs(actual - expected)
    result = dict(shape=list(actual.shape), bitwise_equal=bool(np.array_equal(actual, expected)),
                  max_abs_bps=float(error.max(initial=0.)),
                  max_relative=float((error / np.maximum(np.abs(expected), 1.)).max(initial=0.)),
                  rtol=1e-11, atol_bps=1e-6)
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-6):
        raise ValueError(f"{label}: saved samples do not reproduce: {result}")
    return result


def physical_config(value):
    return FoundationConfig(rf=RFConfig(**value["rf"]), vlc=VLCConfig(**value["vlc"]),
                            vehicle=VehicleConfig(**value["vehicle"]),
                            estimation=EstimationConfig(**value["estimation"]),
                            speed_limit_mps=value["speed_limit_mps"])


def geometry_now(context, state, cfg):
    return (channel.rf_geometry(context, state, cfg.rf),
            channel.vlc_geometry(context, context.ports(state), cfg.vlc))


@dataclass
class SavedAction:
    action: Action
    samples: np.ndarray
    actual_rates: np.ndarray
    record: dict


class SavedCase:
    def __init__(self, group):
        self.group = group
        main, state = SOURCE_RUNS[group]
        self.source = STORAGE / main
        self.state_source = STORAGE / (state or main)
        self.summary = json.loads((self.source / "summary.json").read_text(encoding="utf-8"))
        state_summary = json.loads((self.state_source / "summary.json").read_text(encoding="utf-8"))
        self.config = self.summary["config"]
        if self.config["phase"] != "evaluate" or self.summary["updates"] != 0:
            raise ValueError("only numerical evaluation runs without learning are supported")
        if "policy" in self.config["methods"] or self.config["csi"] != "estimated":
            raise ValueError("only causal estimated-CSI numerical runs are supported")
        for name in ("physical", "draws", "channel_seed", "forecast_seed", "csi", "estimator",
                     "forecast_information", "qos_metric", "q_min_bps", "qos_chance_inflation", "delta"):
            if self.config[name] != state_summary["config"][name]:
                raise ValueError(f"state source configuration differs: {name}")
        if self.summary["traffic_hashes"] != state_summary["traffic_hashes"]:
            raise ValueError("state source traffic differs")
        self.cfg = physical_config(self.config["physical"])
        directory = Path(self.summary["traffic"]["directory"].replace("\\\\", "\\"))
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for name, expected in self.summary["traffic_hashes"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"traffic input changed: {name}")
        self.scene = Scene(directory, self.cfg.vehicle)
        self.context = block.GeometryContext.from_scene(self.scene)
        self.n = self.scene.n
        self.qos = dict(q_min_bps=self.config["q_min_bps"], metric=self.config["qos_metric"],
                        inflation=self.config["qos_chance_inflation"])
        self.restart_ids = sorted(int(p.parent.name.split("_")[-1]) for p in
                                  (self.state_source / "block_checkpoints").glob("block_*/restart.pkl.gz"))
        self._causal_oracle = None
        self._causal_estimator = None
        self._causal_index = -1

    def provenance(self):
        return dict(group=self.group, source=self.source, state_source=self.state_source,
                    summary_sha256=digest(self.source / "summary.json"),
                    state_summary_sha256=digest(self.state_source / "summary.json"),
                    traffic_hashes=self.summary["traffic_hashes"], config=self.config,
                    restart_count=len(self.restart_ids))

    @staticmethod
    def checked_file(folder, name):
        commit = json.loads((folder / "commit.json").read_text(encoding="utf-8"))
        path = folder / name
        if digest(path) != commit["sha256"].get(name):
            raise ValueError(f"saved-file checksum mismatch: {path}")
        return path

    def load_runtime(self, k):
        folder = self.state_source / "block_checkpoints" / f"block_{k:06d}"
        path = self.checked_file(folder, "restart.pkl.gz")
        # These are this project's own local checkpoints, verified before unpickling.
        with gzip.open(path, "rb") as stream:
            runtime = pickle.load(stream)
        if runtime["estimator"].cfg.to_dict() != self.cfg.to_dict():
            raise ValueError("checkpoint physical configuration differs")
        runtime["oracle"].scene = self.scene
        return runtime

    def _prepare_causal(self, k):
        if k <= self._causal_index:
            raise ValueError("causal restoration must proceed in increasing block order")
        earlier = [j for j in self.restart_ids if self._causal_index < j < k]
        if earlier:
            j = max(earlier)
            runtime = self.load_runtime(j)
            self._causal_oracle, self._causal_estimator = runtime["oracle"], runtime["estimator"]
            self._causal_index = j
        elif self._causal_oracle is None:
            self._causal_oracle = TruthOracle(self.scene, self.cfg, self.config["channel_seed"])
            self._causal_estimator = ChannelEstimator(self.n, self.cfg)
        for j in range(self._causal_index + 1, k + 1):
            t = j * self.cfg.estimation.block_dt_s
            self._causal_oracle.advance_to(t)
            state = self.scene.knot(j)
            rg, _ = geometry_now(self.context, state, self.cfg)
            vel = state["speed_mps"][:, None] * np.stack(
                [np.cos(state["heading_rad"]), np.sin(state["heading_rad"])], -1)
            rel = np.linalg.norm(vel[:, None] - vel[None, :], axis=-1)
            sigma = np.where(rg["los"], self.cfg.rf.los_shadow_sigma_db, self.cfg.rf.nlos_shadow_sigma_db)
            prior = np.exp((sigma * np.log(10.) / 10.) ** 2 / 2.)
            self._causal_estimator.advance(t, self._causal_oracle, rf_prior_var=prior,
                                           rf_tau=link_coherence_s(rel, self.cfg.rf))
            self._causal_index = j
        return self._causal_estimator

    def restore_forecast(self, k):
        started = time.perf_counter()
        if k in self.restart_ids:
            # Saved after truth evaluation, but the estimator is still the causal
            # block-start posterior: no updates occur during the evaluation loop.
            estimator = self.load_runtime(k)["estimator"]
            restore_kind = "same_block_causal_estimator_checkpoint"
        else:
            estimator = self._prepare_causal(k)
            restore_kind = "sequential_observation_replay_without_allocation"
        t = k * self.cfg.estimation.block_dt_s
        if any(v > t + 1e-12 for v in estimator.state_time.values()):
            raise ValueError("future observations in restored estimator")
        state = self.scene.knot(k)
        rg, vg = geometry_now(self.context, state, self.cfg)
        belief = estimator.gain_belief(rg["gain"], vg["gain"], t,
                                      rf_candidate=rg["candidate"], vlc_candidate=vg["candidate"])
        ports = channel.select_ports(belief["vlc_gain_mean"])
        gains = {"RF": belief["rf_gain_mean"], "VLC": belief["vlc_gain_mean"]}
        bank = ConditionalTrajectoryBank(estimator, state, self.context, t,
                                        draws=self.config["draws"], seed=self.config["forecast_seed"] + k,
                                        information=self.config["forecast_information"])
        return bank, gains, ports, dict(kind=restore_kind, elapsed_s=time.perf_counter()-started,
                                       forecast_build_s=bank.build_time_s)

    def saved_action(self, k, method, ports):
        folder = self.source / "block_checkpoints" / f"block_{k:06d}"
        records = json.loads(self.checked_file(folder, "records.json").read_text(encoding="utf-8"))
        matches = [r for r in records["block_records"] if r["method"] == method]
        if len(matches) != 1:
            raise ValueError("missing or duplicate saved method record")
        record = matches[0]
        if not record["success"]:
            return None, record
        prefix = f"{method}_b{k}_"
        with np.load(self.checked_file(folder, "fine_samples.npz"), allow_pickle=False) as raw:
            tx, rx, modes = (raw[prefix + s] for s in ("tx", "rx", "mode"))
            links = [(int(i), int(j), str(m)) for i, j, m in zip(tx, rx, modes)]
            pp = {name: np.array(ports[name], copy=True) for name in ("tx", "rx")}
            for index, (i, j, mode) in enumerate(links):
                if mode == "VLC":
                    for name in ("tx", "rx"):
                        saved_port = int(raw[prefix + name + "_port"][index])
                        if pp[name][i, j] != saved_port:
                            raise ValueError("restored causal port selection differs from saved action")
            action = Action(links, raw[prefix + "power_w"].copy(), pp)
            samples = raw[prefix + "forecast_samples_bps"].copy()
            rates = raw[prefix + "rate_bps"].copy()
        block.validate_action(action.links, action.powers, action.ports, self.n, self.cfg)
        return SavedAction(action, samples, rates, record), record

    def actual_evaluate_after_decision(self, k, actions):
        """No callback to any solver; returned truth metrics are evaluation only."""
        started = time.perf_counter()
        earlier = [j for j in self.restart_ids if j < k]
        if earlier and max(earlier) == k - 1:
            oracle = self.load_runtime(k - 1)["oracle"]
        elif self._causal_index == k:
            oracle = copy.deepcopy(self._causal_oracle)
        elif k > 0:
            # A same-block estimator checkpoint does not rewind its oracle.
            # Sparse retained checkpoints (notably 10-car block 998) require
            # consuming historical observations from an earlier checkpoint.
            # Otherwise an oracle advanced from zero would accumulate a huge
            # unconsumed pilot cache and waste the available restart state.
            self._prepare_causal(k)
            oracle = copy.deepcopy(self._causal_oracle)
        else:
            # Block zero has no preceding end-of-block oracle checkpoint.
            oracle = TruthOracle(self.scene, self.cfg, self.config["channel_seed"])
        dt = self.cfg.estimation.fine_dt_s
        steps = int(round(self.cfg.estimation.block_dt_s / dt))
        t = k * self.cfg.estimation.block_dt_s
        fine = {name: {f: [] for f in ("rate_bps", "sinr", "signal", "interference", "noise")}
                for name in actions}
        for offset in range(steps):
            oracle.advance_to(t + offset * dt)
            truth = oracle.truth()
            for name, action in actions.items():
                values = block.sample_metrics(action.links, action.powers, action.ports,
                                              truth["rf"], truth["vlc"], self.cfg)
                for field, value in values.items():
                    fine[name][field].append(value)
        fine = {name: {f: np.asarray(v) for f, v in values.items()} for name, values in fine.items()}
        metrics = {name: block.summarize_block(action.links, action.powers, fine[name], dt, self.n,
                                             q_eval_bps=self.config["q_eval_bps"], cfg=self.cfg)
                   for name, action in actions.items()}
        return metrics, fine, time.perf_counter() - started


def prediction_metrics(action, forecast, case, eta_ref):
    denom = link_electrical_power_w(action.powers, [m for _, _, m in action.links], case.cfg)
    ee = np.asarray(forecast["mean_rate_bps"]) / denom
    qr = predicted_qos_rates(forecast, case.qos["metric"], case.qos["inflation"])
    return dict(mean_ee=float(ee.mean()), min_ee=float(ee.min()),
                budget=float((1-case.config["delta"])*eta_ref),
                budget_pass=bool(ee.mean() >= (1-case.config["delta"])*eta_ref-1e-7),
                quantile_min_rate=float(qr.min()),
                quantile_pass=bool(np.all(qr >= case.qos["q_min_bps"]-1e-7)))


def identical_action(left, right):
    """Exact physical action equality, ignoring ordering and unselected ports."""
    if sorted(left.links) != sorted(right.links):
        return False
    left_index = sorted(range(len(left.links)), key=lambda i: left.links[i])
    right_index = sorted(range(len(right.links)), key=lambda i: right.links[i])
    if not np.array_equal(np.asarray(left.powers)[left_index], np.asarray(right.powers)[right_index]):
        return False
    for i, j, mode in left.links:
        if mode == "VLC" and any(left.ports[p][i, j] != right.ports[p][i, j] for p in ("tx", "rx")):
            return False
    return True


def process_block(case, k, stage, method="maxmin"):
    bank, gains, ports, restore = case.restore_forecast(k)
    reference, rr = case.saved_action(k, "reference", ports)
    incumbent, ir = case.saved_action(k, method, ports)
    if reference is None or incumbent is None:
        return dict(block=k, group=case.group, status="original_failure_preserved",
                    initial_reason=rr.get("failure_reason"), old_reason=ir.get("failure_reason")), {}
    ref_forecast = bank.evaluate(reference.action)
    old_forecast = bank.evaluate(incumbent.action)
    checks = {
        "reference_prediction": assert_same_samples(ref_forecast["block_rate_samples_bps"], reference.samples, "reference"),
        f"old_{method}_prediction": assert_same_samples(old_forecast["block_rate_samples_bps"], incumbent.samples, f"old_{method}"),
    }
    eta_ref = float(rr["eta_ref"])
    ref_pred = prediction_metrics(reference.action, ref_forecast, case, eta_ref)
    if not np.isclose(ref_pred["mean_ee"], eta_ref, rtol=1e-11, atol=1e-7):
        raise ValueError("restored initial mean EE differs from saved budget reference")
    old_pred = prediction_metrics(incumbent.action, old_forecast, case, eta_ref)
    output = dict(group=case.group, block=k, method=method, status="restored", restore=restore, checks=checks,
                  delta=case.config["delta"],
                  reference_prediction=ref_pred, old_prediction=old_pred,
                  original_complete_decision_s=ir["solve_time_s"],
                  reused_initial_solve_s=rr["reference_solve_time_s"],
                  old_returned_initial_action=identical_action(incumbent.action, reference.action),
                  original_initial_mean_fallback_flag=ir.get("chance_fallback"))
    arrays = {"reference_samples_bps": reference.samples, "old_samples_bps": incumbent.samples,
              "old_power_w": incumbent.action.powers}
    if stage == "restore":
        return output, arrays

    # The solver interface has no Scene, TruthOracle, future arrays, or actual EE.
    from p3learn.guarded_refine import improve_fair_power
    started = time.perf_counter()
    new_action, new_forecast, diagnostics = improve_fair_power(
        incumbent.action, old_forecast, case.cfg, bank.evaluate,
        delta=case.config["delta"], eta_ref=eta_ref, qos=case.qos, belief_gains=gains)
    solve_s = time.perf_counter() - started
    block.validate_action(new_action.links, new_action.powers, new_action.ports, case.n, case.cfg)
    if new_action.links != incumbent.action.links or any(
            not np.array_equal(new_action.ports[x], incumbent.action.ports[x]) for x in ("tx", "rx")):
        raise ValueError("this experiment may refine powers only")
    audited_forecast = bank.evaluate(new_action)
    checks["new_prediction_self_check"] = assert_same_samples(
        audited_forecast["block_rate_samples_bps"], new_forecast["block_rate_samples_bps"], "new forecast")
    new_pred = prediction_metrics(new_action, audited_forecast, case, eta_ref)
    unchanged = bool(np.array_equal(new_action.powers, incumbent.action.powers))
    if not unchanged and (not new_pred["budget_pass"] or not new_pred["quantile_pass"]):
        raise ValueError("changed action violates original prediction constraints")
    if new_pred["min_ee"] < old_pred["min_ee"] - max(1e-6, abs(old_pred["min_ee"])*1e-11):
        raise ValueError("new candidate lost the preserved incumbent minimum EE")
    # All prediction checks and candidate selection have ended before truth access.
    measured, fine, evaluation_s = case.actual_evaluate_after_decision(
        k, {"reference": reference.action, "old": incumbent.action, "new": new_action})
    checks["reference_actual"] = assert_same_samples(fine["reference"]["rate_bps"], reference.actual_rates, "reference actual")
    checks[f"old_{method}_actual"] = assert_same_samples(fine["old"]["rate_bps"], incumbent.actual_rates, "old actual")
    output.update(status="evaluated", new_prediction=new_pred, unchanged=unchanged,
                  new_returned_initial_action=identical_action(new_action, reference.action),
                  diagnostics=diagnostics, refine_time_s=solve_s, actual_evaluation_s=evaluation_s,
                  estimated_complete_decision_s=ir["solve_time_s"]+solve_s,
                  reconstructed_time_note="historical complete incumbent decision time plus current incremental refinement; not a freshly timed full run",
                  actual=measured)
    arrays.update(new_samples_bps=audited_forecast["block_rate_samples_bps"], new_power_w=new_action.powers,
                  tx=np.array([i for i, _, _ in new_action.links]), rx=np.array([j for _, j, _ in new_action.links]),
                  mode=np.array([m for _, _, m in new_action.links]),
                  tx_port=new_action.ports["tx"], rx_port=new_action.ports["rx"])
    for name, values in fine.items():
        arrays.update({f"{name}_{field}": value for field, value in values.items()})
    return output, arrays


def commit_output(folder, record, arrays):
    temporary = folder.with_name(folder.name + ".pending")
    if folder.exists() or temporary.exists():
        raise FileExistsError(folder)
    temporary.mkdir(parents=True)
    atomic_json(temporary / "record.json", record)
    np.savez_compressed(temporary / "arrays.npz", **arrays)
    atomic_json(temporary / "commit.json", {x: digest(temporary / x) for x in ("record.json", "arrays.npz")})
    os.replace(temporary, folder)


def load_output(folder):
    hashes = json.loads((folder / "commit.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        if digest(folder / name) != expected:
            raise ValueError(f"output checkpoint changed: {folder / name}")
    return json.loads((folder / "record.json").read_text(encoding="utf-8"))


def summary(records, planned, stage):
    out = dict(planned_blocks=planned, completed_blocks=len(records), stage=stage,
               original_failures=sum(r["status"] == "original_failure_preserved" for r in records),
               restored_blocks=sum(r["status"] != "original_failure_preserved" for r in records))
    out["timing_s"] = {
        "replay_wall_total": sum(r.get("replay_wall_s", 0.) for r in records),
        "restoration_total": sum(r.get("restore", {}).get("elapsed_s", 0.) for r in records),
        "forecast_build_total": sum(r.get("restore", {}).get("forecast_build_s", 0.) for r in records),
        "incremental_refine_total": sum(r.get("refine_time_s", 0.) for r in records),
        "actual_evaluation_total": sum(r.get("actual_evaluation_s", 0.) for r in records),
    }
    measured = [r for r in records if r["status"] == "evaluated"]
    eligible = [r for r in records if r["status"] != "original_failure_preserved"]
    for key, subset in (("old_returned_initial_action", eligible),
                        ("new_returned_initial_action", measured),
                        ("original_initial_mean_fallback_flag", eligible)):
        known = [r for r in subset if key in r]
        out[key + "_checked_blocks"] = len(known)
        out[key + "_count"] = sum(bool(r[key]) for r in known) if len(known) == len(subset) else None
    if measured:
        out["retained_old_incumbent_count"] = sum(r["unchanged"] for r in measured)
        out["actual"] = {}
        for name in ("reference", "old", "new"):
            out["actual"][name] = {
                field: float(np.mean([r["actual"][name][field] for r in measured])) for field in
                ("mean_pair_ee_bit_per_j", "min_pair_ee_bit_per_j", "vlc_link_fraction", "total_electrical_power_w")}
            out["actual"][name]["all_links_service_pass_count"] = sum(
                r["actual"][name]["all_pairs_qos_met"] for r in measured)
            out["actual"][name]["planned_service_denominator"] = planned
            out["actual"][name]["successful_action_denominator"] = len(measured)
            out["actual"][name]["planned_service_pass_fraction"] = (
                out["actual"][name]["all_links_service_pass_count"] / planned)
            out["actual"][name]["violating_link_count"] = sum(
                int(np.count_nonzero(~np.asarray(r["actual"][name]["pair_qos_met"], bool))) for r in measured)
            losses = [1-r["actual"][name]["mean_pair_ee_bit_per_j"] /
                      r["actual"]["reference"]["mean_pair_ee_bit_per_j"] for r in measured]
            out["actual"][name]["loss_exceeds_delta_count"] = sum(
                loss > r["delta"] + 1e-12 for loss, r in zip(losses, measured))
            out["actual"][name]["mean_period_relative_loss"] = float(np.mean(losses))
        out["new_prediction_budget_failure_count"] = sum(not r["new_prediction"]["budget_pass"] for r in measured)
        out["new_prediction_quantile_failure_count"] = sum(not r["new_prediction"]["quantile_pass"] for r in measured)
        out["new_predicted_min_gain_count"] = sum(r["new_prediction"]["min_ee"] > r["old_prediction"]["min_ee"]+1e-6 for r in measured)
        out["new_actual_min_gain_count"] = sum(r["actual"]["new"]["min_pair_ee_bit_per_j"] > r["actual"]["old"]["min_pair_ee_bit_per_j"]+1e-6 for r in measured)
        out["new_predicted_min_nondecrease_failure_count"] = sum(
            r["new_prediction"]["min_ee"] < r["old_prediction"]["min_ee"] -
            max(1e-6, abs(r["old_prediction"]["min_ee"])*1e-11) for r in measured)
        out["actual_comparison"] = {}
        for field in ("mean_pair_ee_bit_per_j", "min_pair_ee_bit_per_j"):
            changes = np.array([r["actual"]["new"][field]-r["actual"]["old"][field] for r in measured])
            tolerance = np.array([max(1e-6, abs(r["actual"]["old"][field])*1e-11) for r in measured])
            out["actual_comparison"][field] = dict(
                wins=int(np.count_nonzero(changes > tolerance)),
                losses=int(np.count_nonzero(changes < -tolerance)),
                ties=int(np.count_nonzero(np.abs(changes) <= tolerance)))
        out["estimated_complete_decision_mean_s"] = float(np.mean([
            r["estimated_complete_decision_s"] for r in measured]))
        out["complete_decision_time_note"] = (
            "historical old complete decision time plus current incremental refinement; "
            "not a freshly measured complete new decision pipeline")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=SOURCE_RUNS, default=["50B"])
    parser.add_argument("--selection", choices=FIXED_BLOCKS, default="development")
    parser.add_argument("--blocks", nargs="+", type=int, help="explicit diagnostic subset, recorded before execution")
    parser.add_argument("--start", type=int, help="inclusive start of a continuous saved-block shard")
    parser.add_argument("--stop", type=int, help="exclusive end of a continuous saved-block shard")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--method", choices=("maxmin", "same_topo_maxmin"), default="maxmin")
    parser.add_argument("--stage", choices=("restore", "improve"), default="restore")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (args.start is None) != (args.stop is None) or (args.blocks is not None and args.start is not None):
        raise ValueError("provide both --start and --stop, without --blocks")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    from threadpoolctl import threadpool_limits
    thread_limits = threadpool_limits(limits=args.threads)
    import torch
    torch.set_num_threads(args.threads)
    selected = range(args.start, args.stop) if args.start is not None else (
        args.blocks if args.blocks is not None else FIXED_BLOCKS[args.selection])
    blocks = sorted(set(selected))
    if not blocks or min(blocks) < 0 or max(blocks) >= 1000:
        raise ValueError("blocks must be in the original 0..999 evaluation window")
    output = args.out.resolve()
    if EXPERIMENT.resolve() not in output.parents:
        raise ValueError("write results only below this new experiment directory")
    cases = [SavedCase(g) for g in args.groups]
    tracked = [Path(__file__), CODE / "p3learn/forecast.py", CODE / "run_foundation.py",
               *sorted((CODE / "p3base").glob("*.py")), CODE / "p3learn/allocation.py"]
    if args.stage == "improve":
        tracked.extend([CODE / "p3learn/guarded_refine.py", CODE / "p3learn/bottleneck_power.py",
                        CODE / "p3learn/dinkelbach_sca.py"])
    manifest = jsonable(dict(schema_version=1, stage=args.stage, method=args.method, groups=args.groups, blocks=blocks,
                            selection=("continuous_shard" if args.start is not None else
                                       args.selection if args.blocks is None else "explicit_diagnostic"),
                            threads=args.threads,
                            fixed_prespecified_selections=FIXED_BLOCKS,
                            inputs=[c.provenance() for c in cases],
                            source_hashes={str(p.relative_to(CODE)): digest(p) for p in tracked}))
    if output.exists() and any(output.iterdir()):
        if not args.resume or json.loads((output / "manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("nonempty output requires --resume with exactly matching input and code hashes")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "manifest.json", manifest)
    for case in cases:
        records = []
        for k in blocks:
            folder = output / case.group / f"block_{k:06d}"
            if folder.exists():
                records.append(load_output(folder))
                print(f"{case.group} block {k}: reused committed replay", flush=True)
                continue
            started = time.perf_counter()
            try:
                record, arrays = process_block(case, k, args.stage, args.method)
                record["replay_wall_s"] = time.perf_counter() - started
                commit_output(folder, record, arrays)
                records.append(record)
                atomic_json(output / case.group / "summary.json", summary(records, len(blocks), args.stage))
                print(f"{case.group} block {k}: {record['status']} ({record['replay_wall_s']:.3f}s)", flush=True)
            except Exception as exc:
                atomic_json(output / "failure.json", dict(group=case.group, block=k,
                            error=f"{type(exc).__name__}: {exc}", completed=len(records)))
                raise
        atomic_json(output / case.group / "summary.json", summary(records, len(blocks), args.stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
