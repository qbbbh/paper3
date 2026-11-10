"""Synthetic exports exercise auditing without executing any physical simulator."""
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from audit_allocation_outputs import audit, main


def _save_csv(directory, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(directory, q_eval=None):
    directory.mkdir()
    e = dict(fine_dt_s=.001, block_dt_s=.1, pilot_overhead_budget=.2,
             rf_pilot_period_s=.001, vlc_pilot_period_s=.05,
             rf_pilot_max_slot_s=.001, vlc_pilot_max_slot_s=.001,
             rf_pilot_power_w=2., vlc_pilot_power_w=2.)
    physical = dict(estimation=e,
                    rf=dict(bandwidth_hz=22e6, gap_db=3., power_max_w=2., circuit_power_w=.1159),
                    vlc=dict(bandwidth_hz=50e6, gap_db=5.4, power_max_w=2., circuit_power_w=1.745))
    config = dict(physical=physical, methods=["maxmin"], phase="evaluate", split="development",
                  estimator="two_state", channel_seed=11, forecast_seed=21, learning_seed=31,
                  delta=.05, q_eval_bps=q_eval, draws=2, blocks=3, start_block=2)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    plan = dict(n=4)
    for mode in ("rf", "vlc"):
        period = e[mode + "_pilot_period_s"]
        slot = min(.001, .2 * period / 4)
        plan.update({mode + "_period_s": period, mode + "_slot_s": slot, mode + "_power_w": 2.,
                     mode + "_overhead": 4 * slot / period,
                     mode + "_energy_per_vehicle_per_s": 2 * slot / period})
    rows, raw, blocks = [], {}, []
    rf_pilot = .1 * plan["rf_energy_per_vehicle_per_s"]
    vlc_pilot = .1 * plan["vlc_energy_per_vehicle_per_s"]
    pilot = 4 * (rf_pilot + vlc_pilot)
    for b in (2, 3, 4):
        common = dict(schema_version=1, method="maxmin", phase="evaluate", split="development", scenario="n4", n=4,
                      episode=0, block=b, traffic_seed=5, channel_seed=11, forecast_seed=21+b, learning_seed=31,
                      source_sha256="a"*64, config_sha256=config_hash, mobility_sha256="b"*64,
                      buildings_sha256="c"*64, dataset_manifest_sha256=None, estimator="two_state",
                      start_time_s=.1*b, duration_s=.1, delta=.05, q_eval_bps=q_eval, fallback=b == 3,
                      coverage_failure=b == 4, failure_reason="coverage_failure" if b == 4 else "selected",
                      solve_time_s=.2+b*.01, online_within_100ms=False,
                      rf_pilot_overhead=plan["rf_overhead"], vlc_pilot_overhead=plan["vlc_overhead"],
                      block_pilot_energy_j=pilot, action_sha256="", reference_version="fixed-v1")
        if b == 4:
            rows.append(dict(common, row_type="block_failure", selected=False, active=False))
            blocks.append(dict(common, success=False, delivered_bits=0., data_energy_j=0.,
                               network_ee_bit_per_j=0., min_pair_ee_bit_per_j=0.,
                               mean_pair_ee_bit_per_j=0.,
                               mean_pair_ee_ratio_vs_actual_reference=0.))
            continue
        power = np.array([.02, .3]) if b == 2 else np.array([.04, .2])
        tx, rx, mode = np.array([0, 2]), np.array([1, 3]), np.array(["RF", "VLC"])
        interference = np.tile([.2, .3], (100, 1))
        noise = np.ones((100, 2))
        signal = np.stack([.3 + np.arange(100)*.0001, np.full(100, .1)], -1) * (b - 1)
        sinr = signal / (interference + noise)
        bandwidth = np.array([22e6, 50e6])
        gap = 10. ** (np.array([3., 5.4]) / 10.)
        rate = bandwidth * np.log1p(sinr/gap) / np.log(2.)
        mean, bits, energy = rate.mean(0), rate.sum(0)*.001, power*.1
        # EE 分母是电输入功率 static + p（导出文件的口径），不是发射功率。
        static = np.where(mode == "RF", .1159, 1.745)
        pair = mean/(static + power)
        net, weakest = bits.sum()/energy.sum(), pair.min()
        jain = pair.sum()**2/(2*np.dot(pair, pair))
        forecast = np.stack([mean*.8, mean*1.2])
        action = [[0, 1, "RF", float(power[0]), -1, -1], [2, 3, "VLC", float(power[1]), 1, 0]]
        ahash = hashlib.sha256(json.dumps(action, separators=(",", ":")).encode()).hexdigest()
        mean_pair = float(pair.mean())
        common.update(action_sha256=ahash, eta_ref=mean_pair, actual_eta_ref=mean_pair,
                      actual_reference_network_ee=net,
                      predicted_network_ee=net, predicted_mean_pair_ee=mean_pair,
                      predicted_budget_violation=0.,
                      actual_budget_violation_vs_predicted_reference=0.,
                      actual_budget_violation_vs_actual_reference=0.,
                      actual_network_loss_fraction_vs_actual_reference=0.,
                      mean_pair_ee_ratio_vs_actual_reference=1.,
                      network_ee_bit_per_j=net, min_pair_ee_bit_per_j=weakest,
                      mean_pair_ee_bit_per_j=mean_pair, jain_pair_ee=jain,
                      vlc_link_fraction=.5, vlc_rate_fraction=mean[1]/mean.sum(),
                      vlc_power_fraction=power[1]/power.sum(),
                      predicted_total_data_power_w=float(power.sum()),
                      predicted_mean_data_power_w=float(power.mean()),
                      predicted_total_electrical_power_w=float((static + power).sum()),
                      all_pairs_qos_met=None if q_eval is None else bool(np.all(mean >= q_eval)),
                      pilot_inclusive_gross_ee_audit=bits.sum()/(energy.sum()+pilot))
        prefix = f"maxmin_b{b}"
        for key, value in dict(signal=signal, interference=interference, noise=noise, sinr=sinr, rate_bps=rate,
                               power_w=power, sample_time_s=.1*b+np.arange(100)*.001,
                               tx=tx, rx=rx, mode=mode, tx_port=np.array([-1, 1]), rx_port=np.array([-1, 0]),
                               link_index=np.arange(2), block_index=np.array(b), forecast_samples_bps=forecast).items():
            raw[prefix + "_" + key] = value
        for li in range(2):
            rows.append(dict(common, row_type="link", fine_prefix=prefix, link_index=li,
                             tx=int(tx[li]), rx=int(rx[li]), tx_id=f"v{tx[li]}", rx_id=f"v{rx[li]}", mode=mode[li],
                             tx_port=-1 if li == 0 else 1, rx_port=-1 if li == 0 else 0,
                             selected=True, active=True, candidate_eligible=True, business_allowed=True,
                             power_w=power[li], predicted_rate_bps=mean[li],
                             predicted_p05_bps=np.quantile(forecast, .05, axis=0)[li],
                             predicted_p95_bps=np.quantile(forecast, .95, axis=0)[li], actual_rate_bps=mean[li],
                             delivered_bits=bits[li], data_energy_j=energy[li], pair_ee_bit_per_j=pair[li],
                             pair_qos_met=None if q_eval is None else bool(mean[li] >= q_eval),
                             pair_pilot_energy_j=2*(rf_pilot+vlc_pilot), pair_rf_pilot_energy_j=2*rf_pilot,
                             pair_vlc_pilot_energy_j=2*vlc_pilot, signal_unit="W" if li == 0 else "A^2"))
        blocks.append(dict(common, success=True, delivered_bits=float(bits.sum()), data_energy_j=float(energy.sum())))

    def stat(key):
        values = np.array([record[key] for record in blocks], float)
        return dict(mean=float(values.mean()), std=float(values.std()), min=float(values.min()), max=float(values.max()))

    bits_total = sum(record["delivered_bits"] for record in blocks)
    energy_total = sum(record["data_energy_j"] for record in blocks)
    method = dict(blocks=3, successful_blocks=2, failure_rate=1/3, fallback_rate=1/3,
                  total_delivered_bits=bits_total, total_data_energy_j=energy_total,
                  cumulative_network_ee_bit_per_j=bits_total/energy_total,
                  min_pair_ee_including_failed_as_zero=stat("min_pair_ee_bit_per_j"),
                  mean_pair_ee_including_failed_as_zero=stat("mean_pair_ee_bit_per_j"),
                  mean_pair_ee_ratio_vs_reference=stat("mean_pair_ee_ratio_vs_actual_reference"),
                  block_network_ee_diagnostic=stat("network_ee_bit_per_j"), solve_time_s=stat("solve_time_s"),
                  real_budget_violation_rate=0.)
    summary = dict(schema_version=1, config=config, config_sha256=config_hash, source_sha256="a"*64,
                   traffic=dict(traffic_seed=5), traffic_hashes={"mobility.npz": "b"*64, "buildings.npz": "c"*64},
                   dataset_manifest_sha256=None, pilot_plan=plan, blocks=blocks, methods=dict(maxmin=method))
    raw.update(audit_only=np.array(True), units=np.array("time=s;rate=bit/s;sinr=linear;RF S/I/N=W;VLC S/I/N=A^2"))
    _save_csv(directory, rows)
    np.savez_compressed(directory / "fine_samples.npz", **raw)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, raw, summary


def test_audit_reconstructs_mixed_modes_and_keeps_failures_in_denominator(tmp_path):
    directory = tmp_path / "valid"
    _, _, summary = fixture(directory)
    result = audit(directory)
    assert result["success"], result["failedchecks"]
    assert result["blocks"] == 3 and result["linkblocks"] == 4 and result["failed_blocks"] == 1
    assert result["methods"]["maxmin"]["failure_rate"] == pytest.approx(1/3)
    expected = summary["methods"]["maxmin"]["cumulative_network_ee_bit_per_j"]
    assert result["methods"]["maxmin"]["cumulative_network_ee_bit_per_j"] == pytest.approx(expected)
    assert expected != pytest.approx(summary["methods"]["maxmin"]["block_network_ee_diagnostic"]["mean"])


@pytest.mark.parametrize("field,change", [("rx", 0), ("tx_port", 0), ("network_ee_bit_per_j", 42),
                                         ("pair_pilot_energy_j", 1), ("q_eval_bps", 0),
                                         ("predicted_rate_bps", 1), ("mobility_sha256", "d"*64)])
def test_csv_tampering_fails(tmp_path, field, change):
    directory = tmp_path / "changed"
    rows, _, _ = fixture(directory)
    # Port tamper must touch the VLC link; other changes can touch either mode.
    rows[1 if field == "tx_port" else 0][field] = change
    _save_csv(directory, rows)
    assert not audit(directory)["success"]


@pytest.mark.parametrize("field", ["sinr", "rate_bps", "sample_time_s", "forecast_samples_bps", "mode",
                                   "tx_port", "rx_port", "link_index", "block_index"])
def test_npz_tampering_fails(tmp_path, field):
    directory = tmp_path / "changed"
    _, raw, _ = fixture(directory)
    key = "maxmin_b2_" + field
    if field == "mode":
        raw[key][0] = "VLC"
    elif field == "sample_time_s":
        raw[key] = raw[key][:-1]
    else:
        raw[key].flat[0] += 1e5
    np.savez_compressed(directory / "fine_samples.npz", **raw)
    assert not audit(directory)["success"]


def test_summary_cannot_double_count_link_network_fields_or_drop_failed_block(tmp_path):
    directory = tmp_path / "changed"
    _, _, summary = fixture(directory)
    summary["methods"]["maxmin"]["total_delivered_bits"] *= 2
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert not audit(directory)["success"]
    summary["blocks"].pop()
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert not audit(directory)["success"]


def test_executed_prediction_budget_is_enforced_independently_of_recorded_zero(tmp_path):
    directory = tmp_path / "budget"
    rows, _, summary = fixture(directory)
    for row in rows[:2]:
        row["eta_ref"] *= 2
    summary["blocks"][0]["eta_ref"] *= 2
    _save_csv(directory, rows)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = audit(directory)
    assert not result["success"]
    assert any("executed action violates predicted" in message for message in result["failedchecks"])


def test_mode_bandwidth_and_gap_come_from_hashed_physical_config(tmp_path):
    directory = tmp_path / "bandwidth"
    rows, _, summary = fixture(directory)
    summary["config"]["physical"]["vlc"]["bandwidth_hz"] *= 2
    summary["config"]["physical"]["vlc"]["gap_db"] += 1
    changed = hashlib.sha256(json.dumps(summary["config"], sort_keys=True).encode()).hexdigest()
    summary["config_sha256"] = changed
    for row in rows:
        row["config_sha256"] = changed
    for record in summary["blocks"]:
        record["config_sha256"] = changed
    _save_csv(directory, rows)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = audit(directory)
    assert not result["success"]
    assert any("mode-specific PHY rate" in message for message in result["failedchecks"])


def test_qos_is_evaluation_only_and_explicit_threshold_is_recomputed(tmp_path):
    directory = tmp_path / "qos"
    fixture(directory, q_eval=1e6)
    result = audit(directory)
    assert result["success"], result["failedchecks"]


def test_empty_export_rejected_and_multi_directory_cli_exit_status(tmp_path, capsys):
    first, second = tmp_path / "first", tmp_path / "second"
    fixture(first)
    fixture(second)
    assert main([str(first), str(second)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["blocks"] == 6 and result["linkblocks"] == 8 and result["success"]
    (second / "results.csv").write_text("row_type\n", encoding="utf-8")
    assert not audit(second)["success"]
    assert main([str(first), str(second)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["success"] and result["failedchecks"]
