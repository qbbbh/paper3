"""Audit allocation exports using only CSV/NPZ/JSON, without running the PHY.

The saved signal/interference/noise reconstruct SINR and mode-specific rates.
Per-link EE is ``rate / (static + p)`` with the static power taken from the run's
own recorded config, so the audited quantity is the paper objective, not ``R/p``.
Block totals are counted once, including explicit failure rows. This checks
internal reproducibility and binds the export to a source/config hash; it does
not certify causal provenance or calibration.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GROUP_FIELDS = ("method", "phase", "split", "scenario", "n", "episode", "block",
                "traffic_seed", "channel_seed", "forecast_seed", "learning_seed")
SAMPLE_FIELDS = ("rate_bps", "sinr", "signal", "interference", "noise")
RAW_FIELDS = SAMPLE_FIELDS + ("power_w", "sample_time_s", "tx", "rx", "mode", "tx_port", "rx_port",
                             "link_index", "block_index", "forecast_samples_bps")


def source_hash():
    """与 ``run_allocation.source_hash`` 同一算法：``ROOT/code`` 下全部 .py 的路径+内容。

    用来把结果绑到产生它的源码版本上；默认只记录是否一致，不因不一致而失败，
    因为旧 run 合法地出自旧源码。``--require-current-source`` 时才强制一致。
    """
    digest = hashlib.sha256()
    for path in sorted((ROOT / "code").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: missing or invalid number") from exc
    require(np.isfinite(result), f"{name}: nonfinite number")
    return result


def integer(value, name):
    result = number(value, name)
    require(result.is_integer(), f"{name}: must be integer")
    return int(result)


def boolean(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    require(value in ("True", "False", "true", "false"), f"{name}: must be a boolean")
    return value in ("True", "true")


def near(actual, expected, name, rtol=1e-9, atol=1e-6):
    try:
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: not numeric") from exc
    require(np.isfinite(actual).all() and np.isfinite(expected).all()
            and np.allclose(actual, expected, rtol=rtol, atol=atol), f"{name}: recomputed value disagrees")


def same(actual, expected, name):
    if expected is None:
        require(actual in (None, ""), f"{name}: must be empty/null")
    elif isinstance(expected, (bool, np.bool_)):
        require(boolean(actual, name) == bool(expected), f"{name}: boolean mismatch")
    elif isinstance(expected, (int, float, np.number)):
        near(number(actual, name), expected, name)
    else:
        require(str(actual) == str(expected), f"{name}: metadata mismatch")


def group_key(row):
    return tuple("" if row.get(field) is None else str(row.get(field, "")) for field in GROUP_FIELDS)


def check_hash(value, name, optional=False):
    if optional and value in (None, ""):
        return
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
            f"{name}: invalid SHA256")


def column(rows, name):
    return np.array([number(row.get(name), name) for row in rows])


def _metadata(rows, summary):
    config = summary["config"]
    computed = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    require(summary.get("config_sha256") == computed, "summary config_sha256 does not match serialized config")
    hashes = {"source_sha256": summary.get("source_sha256"), "config_sha256": computed,
              "mobility_sha256": summary["traffic_hashes"].get("mobility.npz"),
              "buildings_sha256": summary["traffic_hashes"].get("buildings.npz"),
              "dataset_manifest_sha256": summary.get("dataset_manifest_sha256")}
    for key, value in hashes.items():
        check_hash(value, key, optional=key == "dataset_manifest_sha256")
    require(integer(summary.get("schema_version"), "schema_version") == 1, "unsupported summary schema")
    require(len(config["methods"]) == len(set(config["methods"])), "duplicate configured methods")
    delta = number(config.get("delta"), "delta")
    require(0 <= delta <= 1, "delta outside [0,1]")
    if config.get("q_eval_bps") is not None:
        require(number(config["q_eval_bps"], "q_eval_bps") >= 0, "negative q_eval_bps")
    original_ids, state_references = {}, {}
    for row in rows:
        same(row.get("schema_version"), 1, "row schema_version")
        for key, value in hashes.items():
            same(row.get(key), value, key)
        for key in ("delta", "q_eval_bps", "phase", "split", "estimator", "channel_seed", "learning_seed"):
            same(row.get(key), config.get(key), key)
        same(row.get("traffic_seed"), summary["traffic"].get("traffic_seed"), "traffic_seed")
        block_index = integer(row.get("block"), "block")
        same(row.get("forecast_seed"), config["forecast_seed"] + block_index, "forecast_seed")
        require(row.get("method") in config["methods"], "unconfigured method in CSV")
        if row.get("row_type") == "link":
            for endpoint in ("tx", "rx"):
                identity = (row.get("scenario"), row.get("traffic_seed"), row.get(endpoint))
                original = row.get(endpoint + "_id")
                require(bool(original), "original vehicle ID missing")
                require(identity not in original_ids or original_ids[identity] == original,
                        "original vehicle ID changes across methods/blocks")
                original_ids[identity] = original
            state_key = group_key(row)[1:]
            values = tuple(row.get(key) for key in ("eta_ref", "actual_eta_ref", "reference_version"))
            require(state_key not in state_references or state_references[state_key] == values,
                    "same-state methods do not share the frozen reference")
            state_references[state_key] = values


def static_power(physical, modes):
    """按 run 自己记录的 config 取每链路的静态功耗（W）。

    新 run 写 ``circuit_power_w``；旧 run（2026-10-26 之前）写的是 ``bias_power_w``
    （20 W 照明偏置）。两者含义不同，**不在这里做换算**，只用各自记录的值重算，
    这样旧结果仍可审计，但等口径比较必须在同一份 config 下重跑。
    """
    rf = number(physical["rf"]["circuit_power_w"], "RF circuit power")
    block = physical["vlc"]
    if "circuit_power_w" in block:
        vlc, source = number(block["circuit_power_w"], "VLC circuit power"), "circuit_power_w"
    else:
        vlc, source = number(block["bias_power_w"], "VLC bias power (legacy)"), "bias_power_w"
    require(rf >= 0. and vlc >= 0., "negative static power")
    return np.array([rf if mode == "RF" else vlc for mode in modes]), (rf, vlc, source)


def _pilot_values(summary, n, duration):
    e, plan = summary["config"]["physical"]["estimation"], summary["pilot_plan"]
    same(plan.get("n"), n, "pilot n")
    energies = []
    for mode in ("rf", "vlc"):
        period = number(e[mode + "_pilot_period_s"], "pilot period")
        require(period > 0, "pilot period must be positive")
        slot = min(number(e[mode + "_pilot_max_slot_s"], "pilot max slot"),
                   number(e["pilot_overhead_budget"], "pilot budget") * period / n)
        power = number(e[mode + "_pilot_power_w"], "pilot power")
        require(slot >= 0 and power >= 0, "negative pilot cost")
        energy_rate = power * slot / period
        for suffix, expected in (("period_s", period), ("slot_s", slot), ("power_w", power),
                                 ("overhead", n * slot / period), ("energy_per_vehicle_per_s", energy_rate)):
            same(plan.get(mode + "_" + suffix), expected, "pilot_plan " + mode + "_" + suffix)
        energies.append(duration * energy_rate)
    return energies


def _block(rows, record, summary, raw):
    first = rows[0]
    cfg = summary["config"]
    physical = cfg["physical"]
    n = integer(first.get("n"), "n")
    require(n >= 2 and n % 2 == 0, "N must be even and at least two")
    duration = number(first.get("duration_s"), "duration_s")
    dt = number(physical["estimation"]["fine_dt_s"], "fine_dt_s")
    near(duration, .1, "100ms block duration", atol=1e-12)
    near(dt, .001, "1ms sample spacing", atol=1e-12)
    near(physical["estimation"]["block_dt_s"], duration, "physical block duration", atol=1e-12)
    index = integer(first.get("block"), "block")
    near(number(first.get("start_time_s"), "start_time_s"), index * duration, "block start", atol=1e-10)
    failed = first.get("row_type") == "block_failure"
    for row in rows:
        for key, expected in record.items():
            # These two fields are per-link in CSV and block totals in summary.
            if (key in row and key not in ("delivered_bits", "data_energy_j")
                    and not (failed and key in ("network_ee_bit_per_j",
                                                "min_pair_ee_bit_per_j",
                                                "mean_pair_ee_bit_per_j",
                                                "mean_pair_ee_ratio_vs_actual_reference"))):
                same(row[key], expected, "summary block " + key)
    require(boolean(record.get("success"), "summary success") == (not failed), "summary block success mismatch")
    rf_pilot, vlc_pilot = _pilot_values(summary, n, duration)
    block_pilot = n * (rf_pilot + vlc_pilot)
    for row in rows:
        near(number(row.get("block_pilot_energy_j"), "block_pilot_energy_j"), block_pilot, "block pilot energy")
        same(row.get("rf_pilot_overhead"), summary["pilot_plan"]["rf_overhead"], "rf pilot overhead")
        same(row.get("vlc_pilot_overhead"), summary["pilot_plan"]["vlc_overhead"], "vlc pilot overhead")
    result = dict(method=first["method"], success=not failed, fallback=boolean(first.get("fallback"), "fallback"),
                  solve_time_s=number(first.get("solve_time_s"), "solve_time_s"),
                  delivered_bits=0., data_energy_j=0., network_ee_bit_per_j=0.,
                  min_pair_ee_bit_per_j=0., mean_pair_ee_bit_per_j=0.,
                  actual_budget_violation_vs_actual_reference=0.)
    require(result["solve_time_s"] >= 0, "negative solve time")
    same(first.get("online_within_100ms"), result["solve_time_s"] <= duration, "online runtime label")
    if failed:
        require(len(rows) == 1, "failure block must have exactly one failure row")
        require(all(first.get(key) in (None, "") for key in ("tx", "rx", "mode", "action_sha256")),
                "failure row contains a fake active link")
        require(not boolean(first.get("selected"), "selected") and not boolean(first.get("active"), "active"),
                "failure row must not be selected or active")
        require(bool(first.get("failure_reason")), "failure row requires a reason")
        for key in ("delivered_bits", "data_energy_j", "network_ee_bit_per_j",
                    "min_pair_ee_bit_per_j", "mean_pair_ee_bit_per_j"):
            same(record.get(key), 0., "failed summary " + key)
        return result, None
    require(all(row.get("row_type") == "link" for row in rows), "mixed failure/link row types")
    require(len(rows) == n // 2, "successful block does not contain N/2 links")
    indices = [integer(row.get("link_index"), "link_index") for row in rows]
    require(sorted(indices) == list(range(n // 2)), "link indices must be exactly 0..N/2-1")
    rows = sorted(rows, key=lambda row: integer(row["link_index"], "link_index"))
    tx, rx = column(rows, "tx").astype(int), column(rows, "rx").astype(int)
    for row in rows:
        integer(row["tx"], "tx")
        integer(row["rx"], "rx")
    require(sorted([*tx, *rx]) == list(range(n)), "endpoint coverage is not complete exactly once")
    modes = np.array([row.get("mode") for row in rows])
    require(np.isin(modes, ("RF", "VLC")).all(), "invalid mode")
    require(len({row[endpoint + "_id"] for row in rows for endpoint in ("tx", "rx")}) == n,
            "original vehicle IDs must also cover N distinct endpoints")
    for row in rows:
        require(all(boolean(row.get(key), key) for key in ("selected", "active", "candidate_eligible", "business_allowed")),
                "selected link must be active, geometrically eligible and business allowed")
        tp, rp = integer(row.get("tx_port"), "tx_port"), integer(row.get("rx_port"), "rx_port")
        require((tp, rp) == (-1, -1) if row["mode"] == "RF" else tp in (0, 1) and rp in (0, 1),
                "invalid fixed mode/port choices")
        same(row.get("signal_unit"), "W" if row["mode"] == "RF" else "A^2", "signal unit")
    power = column(rows, "power_w")
    caps = np.array([number(physical[mode.lower()]["power_max_w"], "power cap") for mode in modes])
    require(np.all((power >= .001) & (power <= caps) & (caps <= 2)), "selected power outside common domain")
    action_values = [[int(i), int(j), str(mode), float(p), integer(row["tx_port"], "tx_port"),
                      integer(row["rx_port"], "rx_port")] for i, j, mode, p, row in zip(tx, rx, modes, power, rows)]
    action_hash = hashlib.sha256(json.dumps(action_values, separators=(",", ":")).encode()).hexdigest()
    prefix = rows[0].get("fine_prefix")
    require(bool(prefix) and all(row.get("fine_prefix") == prefix for row in rows), "inconsistent fine prefix")
    for row in rows:
        same(row.get("action_sha256"), action_hash, "fixed action hash")
    for field in RAW_FIELDS:
        require(prefix + "_" + field in raw, "missing NPZ field: " + prefix + "_" + field)
    for field, expected in (("tx", tx), ("rx", rx), ("mode", modes),
                            ("tx_port", column(rows, "tx_port").astype(int)),
                            ("rx_port", column(rows, "rx_port").astype(int)),
                            ("link_index", np.arange(n // 2))):
        require(np.array_equal(raw[prefix + "_" + field], expected), "NPZ/CSV " + field + " mismatch")
        if field != "mode":
            require(raw[prefix + "_" + field].dtype.kind in "iu", "NPZ indices/ports must use integer arrays")
    saved_block = raw[prefix + "_block_index"]
    require(saved_block.shape == () and saved_block.dtype.kind in "iu" and saved_block.item() == index,
            "NPZ block_index must be the matching scalar integer")
    require(raw[prefix + "_power_w"].shape == power.shape, "NPZ power shape mismatch")
    near(raw[prefix + "_power_w"], power, "NPZ/CSV power", atol=1e-12)
    times = raw[prefix + "_sample_time_s"]
    require(times.shape == (100,), "must save exactly 100 fine sample times")
    near(times, index * duration + np.arange(100) * dt, "fine sample time grid", atol=1e-10)
    fine = {field: np.asarray(raw[prefix + "_" + field], float) for field in SAMPLE_FIELDS}
    for field, values in fine.items():
        require(values.shape == (100, n // 2) and np.isfinite(values).all() and np.all(values >= 0),
                "invalid fine samples: " + field)
    require(np.all(fine["noise"] > 0), "noise must be positive")
    sinr = fine["signal"] / (fine["interference"] + fine["noise"])
    near(fine["sinr"], sinr, "SINR from signal/interference/noise", atol=1e-12)
    bandwidth = np.array([number(physical[m.lower()]["bandwidth_hz"], "bandwidth") for m in modes])
    gap = np.array([10. ** (number(physical[m.lower()]["gap_db"], "gap_db") / 10.) for m in modes])
    require(np.all(bandwidth > 0), "nonpositive bandwidth")
    rate = bandwidth * np.log1p(sinr / gap) / np.log(2.)
    near(fine["rate_bps"], rate, "mode-specific PHY rate")
    mean, bits, energy = rate.mean(0), rate.sum(0) * dt, power * duration
    # EE 分母是**电输入功率** static + p，不是发射功率 p。旧版这里用
    # ``mean / power``（R/p），与新口径不一致，2026-10-26 已按 run 记录的
    # 静态项重算；静态项取不到就直接失败，不做默许。
    static, static_detail = static_power(physical, modes)
    electrical = static + power
    pair_ee = mean / electrical
    network, weakest = float(bits.sum() / energy.sum()), float(pair_ee.min())
    jain = float(pair_ee.sum() ** 2 / (len(pair_ee) * np.dot(pair_ee, pair_ee))) if np.dot(pair_ee, pair_ee) else 0.
    for key, expected in (("actual_rate_bps", mean), ("delivered_bits", bits), ("data_energy_j", energy),
                          ("pair_ee_bit_per_j", pair_ee), ("network_ee_bit_per_j", network),
                          ("min_pair_ee_bit_per_j", weakest), ("jain_pair_ee", jain)):
        near(column(rows, key), expected, key)
    forecast = np.asarray(raw[prefix + "_forecast_samples_bps"], float)
    require(forecast.shape == (integer(cfg["draws"], "draws"), n // 2) and forecast.shape[0] >= 2
            and np.isfinite(forecast).all() and np.all(forecast >= 0), "invalid conditional forecast samples")
    predicted_mean = forecast.mean(0)
    q_min = number(cfg.get("q_min_bps", 0.), "q_min_bps")
    require(q_min >= 0, "negative optimization QoS")
    require(np.all(predicted_mean >= q_min*(1-1e-8)),
            "successful action violates predicted block-average minimum rate")
    near(column(rows, "predicted_rate_bps"), predicted_mean, "forecast trajectory mean")
    near(column(rows, "predicted_p05_bps"), np.quantile(forecast, .05, axis=0), "forecast p05")
    near(column(rows, "predicted_p95_bps"), np.quantile(forecast, .95, axis=0), "forecast p95")
    predicted_net = float(predicted_mean.sum() / power.sum())
    # The objective and the budget are the mean of each link's own rate over its
    # own electrical input power (static + p).  Network EE (sum rate / sum data
    # power) stays reported as a diagnostic, but it is not the objective.
    predicted_mean_pair = float(np.mean(predicted_mean / electrical))
    mean_pair = float(np.mean(pair_ee))
    eta_ref, delta = number(first.get("eta_ref"), "eta_ref"), number(cfg["delta"], "delta")
    budget = (1 - delta) * eta_ref
    require(eta_ref >= 0 and predicted_mean_pair + max(1e-6, abs(budget) * 1e-10) >= budget,
            "executed action violates predicted mean per-link EE budget")
    actual_ref = number(first.get("actual_eta_ref"), "actual_eta_ref")
    require(actual_ref >= 0, "actual reference mean per-link EE must be nonnegative")
    actual_ref_net = number(first.get("actual_reference_network_ee"),
                            "actual_reference_network_ee")
    require(actual_ref_net >= 0, "actual reference network EE must be nonnegative")
    if first["method"] == "reference":
        near(predicted_mean_pair, eta_ref, "reference predicted mean per-link EE equals eta_ref")
        near(mean_pair, actual_ref, "reference actual mean per-link EE equals actual_eta_ref")
        require(not result["fallback"], "reference cannot be its own fallback")
    real_violation = max(0., (1 - delta) * actual_ref - mean_pair)
    vlc = modes == "VLC"
    for key, expected in (("predicted_network_ee", predicted_net),
                          ("predicted_mean_pair_ee", predicted_mean_pair),
                          ("predicted_budget_violation", max(0., budget - predicted_mean_pair)),
                          ("actual_budget_violation_vs_predicted_reference", max(0., budget - mean_pair)),
                          ("actual_budget_violation_vs_actual_reference", real_violation),
                          ("mean_pair_ee_bit_per_j", mean_pair),
                          ("mean_pair_ee_ratio_vs_actual_reference",
                           mean_pair / actual_ref if actual_ref else None),
                          ("vlc_link_fraction", float(vlc.mean())),
                          ("vlc_rate_fraction", float(mean[vlc].sum() / mean.sum()) if mean.sum() else 0.),
                          ("vlc_power_fraction", float(power[vlc].sum() / power.sum())),
                          ("pilot_inclusive_gross_ee_audit", float(bits.sum() / (energy.sum() + block_pilot)))):
        near(column(rows, key), expected, key)
    for key, expected in (("predicted_total_data_power_w", float(power.sum())),
                          ("predicted_mean_data_power_w", float(power.mean())),
                          ("predicted_total_electrical_power_w", float(electrical.sum()))):
        if rows[0].get(key) not in (None, ""):
            near(column(rows, key), expected, key)
    loss = 1 - network / actual_ref_net if actual_ref_net else None
    q_eval = cfg.get("q_eval_bps")
    for li, row in enumerate(rows):
        same(row.get("actual_network_loss_fraction_vs_actual_reference"), loss, "actual network loss fraction")
        same(row.get("pair_qos_met"), None if q_eval is None else bool(mean[li] >= q_eval), "pair QoS")
        same(row.get("all_pairs_qos_met"), None if q_eval is None else bool(np.all(mean >= q_eval)), "block QoS")
    for key, expected in (("pair_rf_pilot_energy_j", 2 * rf_pilot),
                          ("pair_vlc_pilot_energy_j", 2 * vlc_pilot),
                          ("pair_pilot_energy_j", 2 * (rf_pilot + vlc_pilot))):
        near(column(rows, key), expected, key)
    near(column(rows, "pair_pilot_energy_j").sum(), block_pilot, "pair pilot allocation sums once per block")
    same(record.get("delivered_bits"), float(bits.sum()), "summary block delivered bits")
    same(record.get("data_energy_j"), float(energy.sum()), "summary block data energy")
    result.update(delivered_bits=float(bits.sum()), data_energy_j=float(energy.sum()),
                  network_ee_bit_per_j=network, min_pair_ee_bit_per_j=weakest,
                  mean_pair_ee_bit_per_j=mean_pair,
                  actual_budget_violation_vs_actual_reference=real_violation)
    return result, prefix


def _method_summary(method, records, saved):
    count = len(records)
    successful = sum(record["success"] for record in records)
    bits = sum(record["delivered_bits"] for record in records)
    energy = sum(record["data_energy_j"] for record in records)
    expected = dict(blocks=count, successful_blocks=successful, failure_rate=1 - successful / count,
                    fallback_rate=sum(record["fallback"] for record in records) / count,
                    total_delivered_bits=bits, total_data_energy_j=energy,
                    cumulative_network_ee_bit_per_j=bits / energy if energy else None,
                    real_budget_violation_rate=sum(record["actual_budget_violation_vs_actual_reference"] > 1e-6
                                                   for record in records) / count)
    for key, value in expected.items():
        same(saved.get(key), value, method + " summary " + key)
    for saved_key, record_key in (("min_pair_ee_including_failed_as_zero", "min_pair_ee_bit_per_j"),
                                  ("mean_pair_ee_including_failed_as_zero", "mean_pair_ee_bit_per_j"),
                                  ("block_network_ee_diagnostic", "network_ee_bit_per_j"),
                                  ("solve_time_s", "solve_time_s")):
        values = np.array([record[record_key] for record in records])
        for stat, value in (("mean", values.mean()), ("std", values.std()), ("min", values.min()), ("max", values.max())):
            same(saved[saved_key].get(stat), float(value), method + " " + saved_key + " " + stat)
    return expected


def audit(directory, require_current_source=False):
    """Return a structured report; every detected error makes success False."""
    directory = Path(directory)
    report = dict(directory=str(directory.resolve()), blocks=0, linkblocks=0, failed_blocks=0,
                  failedchecks=[], success=False, methods={})
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
        with (directory / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(bool(rows), "results.csv contains no block records; empty/failed export cannot pass")
        _metadata(rows, summary)
        current = source_hash()
        recorded = summary.get("source_sha256")
        report["source"] = dict(recorded=recorded, current=current, matches=recorded == current)
        if require_current_source:
            require(recorded == current, "recorded source_sha256 does not match the current source tree")
        _, (rf_static, vlc_static, vlc_key) = static_power(summary["config"]["physical"], ["RF"])
        report["static_power_w"] = dict(RF=rf_static, VLC=vlc_static, vlc_config_key=vlc_key)
        groups = defaultdict(list)
        for row in rows:
            groups[group_key(row)].append(row)
        report["blocks"] = len(groups)
        report["linkblocks"] = sum(row.get("row_type") == "link" for row in rows)
        report["failed_blocks"] = sum(row.get("row_type") == "block_failure" for row in rows)
        saved_records = summary.get("blocks", [])
        saved = {group_key(record): record for record in saved_records}
        require(len(saved) == len(saved_records), "duplicate summary block records")
        require(set(saved) == set(groups), "CSV/summary block identities or failure denominators disagree")
        cfg = summary["config"]
        require(set(summary["methods"]) == set(cfg["methods"]) == {row["method"] for row in rows},
                "CSV/config/summary methods disagree")
        for method in cfg["methods"]:
            selected = [key for key in groups if key[0] == method]
            require(len(selected) == integer(cfg["blocks"], "configured blocks"), "method block denominator differs from config")
            expected_indices = set(range(integer(cfg["start_block"], "start_block"), cfg["start_block"] + cfg["blocks"]))
            require({integer(groups[key][0]["block"], "block") for key in selected} == expected_indices,
                    "missing configured blocks, including failures")
        actual_records = defaultdict(list)
        prefixes = set()
        with np.load(directory / "fine_samples.npz", allow_pickle=False) as raw:
            require("audit_only" in raw and bool(raw["audit_only"].item()), "NPZ must be labelled audit_only")
            require("units" in raw and "RF S/I/N=W;VLC S/I/N=A^2" in str(raw["units"].item()), "NPZ units missing")
            for key, group_rows in groups.items():
                try:
                    result, prefix = _block(group_rows, saved[key], summary, raw)
                    actual_records[key[0]].append(result)
                    if prefix:
                        require(prefix not in prefixes, "fine prefix reused across distinct block identities")
                        prefixes.add(prefix)
                except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
                    report["failedchecks"].append(f"block {key}: {exc}")
            expected_keys = {prefix + "_" + field for prefix in prefixes for field in RAW_FIELDS} | {"audit_only", "units"}
            if not report["failedchecks"]:
                require(set(raw.files) == expected_keys, "NPZ contains orphaned or unindexed sample arrays")
        for method in cfg["methods"]:
            if len(actual_records[method]) != cfg["blocks"]:
                continue
            try:
                report["methods"][method] = _method_summary(method, actual_records[method], summary["methods"][method])
            except (ValueError, KeyError, TypeError) as exc:
                report["failedchecks"].append(str(exc))
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, EOFError, zipfile.BadZipFile) as exc:
        report["failedchecks"].append(str(exc))
    report["success"] = not report["failedchecks"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+", help="one or more allocation run directories")
    parser.add_argument("--require-current-source", action="store_true",
                        help="还要求 run 记录的 source_sha256 等于当前源码树（旧 run 会因此失败）")
    options = parser.parse_args(argv)
    reports = [audit(directory, require_current_source=options.require_current_source)
               for directory in options.directories]
    result = dict(blocks=sum(item["blocks"] for item in reports), linkblocks=sum(item["linkblocks"] for item in reports),
                  failedchecks=[f"{item['directory']}: {message}" for item in reports for message in item["failedchecks"]],
                  success=all(item["success"] for item in reports), runs=reports)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
