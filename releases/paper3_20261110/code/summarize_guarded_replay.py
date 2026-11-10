"""Read-only audit and aggregation of committed fixed-action power replays.

This module imports no allocator and starts no optimization or simulation. Default
inputs are the four 50B shards and separate 10A/10B runs. Incomplete inputs fail
closed unless --allow-partial is explicitly supplied; partial output is named
partial_summary.json and cannot replace the completed full report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
ZAPPONE = "zappone_mmee_fp_fixed"
METHODS = ("reference", "old", "new", "zappone")
LABELS = dict(reference="初始分配", old="旧主方法", new="增加功率改进后的主方法", zappone="Zappone 对照")
DEFAULT_RUNS = [ROOT / "runs" / f"full_50B_s{i}" for i in range(4)] + [
    ROOT / "runs/full_10A", ROOT / "runs/full_10B"]
INDEPENDENT_AUDIT = ROOT / "evidence/改进求解器独立审计_20261110.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def close(actual, expected, label, rtol=1e-10, atol=1e-6):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=rtol, atol=atol), f"numeric mismatch: {label}")


def distribution(values):
    a = np.asarray(values, float)
    require(np.isfinite(a).all(), "nonfinite summary input")
    if not len(a):
        return dict(n=0, mean=None, median=None, p05=None, p95=None, min=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                min=float(a.min()), max=float(a.max()))


def current_environment():
    """Current host metadata is not retroactive proof of historical hardware."""
    out = dict(host=platform.node(), platform=platform.platform(),
               python=sys.version, executable=sys.executable, numpy=np.__version__,
               note="Read at aggregation time on this host; old summaries did not record CPU or RAM.")
    if sys.platform == "win32":
        command = ("$taskCpu=Get-CimInstance Win32_Processor; "
                   "$taskMachine=Get-CimInstance Win32_ComputerSystem; "
                   "[pscustomobject]@{cpu=($taskCpu.Name -join ', '); "
                   "cores=($taskCpu.NumberOfCores | Measure-Object -Sum).Sum; "
                   "logical_processors=($taskCpu.NumberOfLogicalProcessors | Measure-Object -Sum).Sum; "
                   "memory_bytes=$taskMachine.TotalPhysicalMemory} | ConvertTo-Json -Compress")
        try:
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                  capture_output=True, text=True, timeout=15, check=True)
            out["hardware"] = json.loads(proc.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            out["hardware_query_error"] = str(exc)
    return out


def result_comparison(left, right):
    """Paired comparisons; every reported relative change names its denominator."""
    ids = sorted(set(left) & set(right))
    out = dict(common_success_count=len(ids), common_success_blocks=ids)
    for key in ("mean_ee", "min_ee"):
        a = np.array([left[k][key] for k in ids])
        b = np.array([right[k][key] for k in ids])
        tolerance = np.maximum(1e-6, abs(b) * 1e-11)
        d = a - b
        positive = b > 0
        out[key] = dict(left=distribution(a), right=distribution(b),
                        absolute_change=distribution(d),
                        period_relative_change=distribution(a[positive] / b[positive] - 1),
                        relative_change_denominator_positive_count=int(positive.sum()),
                        ratio_of_means_change=float(a.mean()/b.mean()-1) if len(b) and b.mean() > 0 else None,
                        wins=int(np.sum(d > tolerance)), losses=int(np.sum(d < -tolerance)),
                        ties=int(np.sum(abs(d) <= tolerance)),
                        loss_blocks=[k for k, loss in zip(ids, d < -tolerance) if loss])
    return out


def provenance(run_roots, allow_partial):
    expected, locations, inputs, source_hashes = {}, {}, {}, None
    manifests, threads = [], set()
    for root in run_roots:
        mpath = root / "manifest.json"
        m = read_json(mpath)
        require(m["stage"] == "improve" and m["method"] == "maxmin", f"wrong stage/method: {root}")
        require(m["schema_version"] == 1, f"unknown manifest schema: {root}")
        if source_hashes is None:
            source_hashes = m["source_hashes"]
        require(m["source_hashes"] == source_hashes, f"source versions differ between shards: {root}")
        threads.add(m["threads"])
        require(len(m["blocks"]) == len(set(m["blocks"])), f"duplicate manifest blocks: {root}")
        require(all(0 <= k < 1000 for k in m["blocks"]), f"invalid block IDs: {root}")
        manifests.append(dict(path=str(mpath), sha256=digest(mpath), groups=m["groups"],
                              blocks=m["blocks"], threads=m["threads"]))
        for item in m["inputs"]:
            group = item["group"]
            require(group in m["groups"], f"input outside groups: {root}")
            if group in inputs:
                require(item == inputs[group], f"input provenance differs within {group}")
            inputs[group] = item
        for group in m["groups"]:
            require(group in inputs, f"missing input provenance for {group}")
            expected.setdefault(group, set())
            locations.setdefault(group, {})
            duplicate = expected[group] & set(m["blocks"])
            require(not duplicate, f"overlapping shards {group}: {sorted(duplicate)}")
            expected[group].update(m["blocks"])
            folder = root / group
            if not folder.exists():
                continue
            for child in folder.glob("block_*"):
                if child.name.endswith(".pending"):
                    require(allow_partial, f"uncommitted block directory: {child}")
                    continue
                k = int(child.name.split("_")[-1])
                require(k in m["blocks"], f"block outside manifest: {child}")
                require(k not in locations[group], f"duplicate committed block: {child}")
                require((child / "commit.json").is_file(), f"missing commit: {child}")
                locations[group][k] = child
    require(set(expected) == {"10A", "10B", "50B"}, "all three groups must be declared")
    for group, ids in expected.items():
        require(ids == set(range(1000)), f"planned union is not 0..999 for {group}")
        require(allow_partial or set(locations[group]) == ids,
                f"incomplete {group}: {len(locations[group])}/1000 committed")
    require(len(threads) == 1, "new replay thread settings differ between runs")
    for name, expected_hash in source_hashes.items():
        path = CODE / name.replace("\\", "/")
        require(digest(path) == expected_hash, f"live frozen source differs: {path}")
    for group, item in inputs.items():
        for prefix in ("", "state_"):
            source = Path(item[prefix + "source"])
            require(digest(source / "summary.json") == item[prefix + "summary_sha256"],
                    f"historical summary changed: {source}")
        s = read_json(Path(item["source"]) / "summary.json")
        require(s["config"] == item["config"] and s["traffic_hashes"] == item["traffic_hashes"],
                f"manifest/source disagreement: {group}")
        traffic = Path(s["traffic"]["directory"].replace("\\\\", "\\"))
        for name, expected_hash in item["traffic_hashes"].items():
            require(digest(traffic / name) == expected_hash, f"traffic input changed: {group}/{name}")
    return locations, inputs, dict(manifests=manifests, source_hashes=source_hashes,
                                   replay_threads=next(iter(threads)),
                                   all_3000_committed=sum(map(len, locations.values())) == 3000)


def load_historical(source, method_names):
    summary = read_json(source / "summary.json")
    records = {}
    for r in summary["blocks"]:
        if r["method"] not in method_names:
            continue
        key = (r["method"], int(r["block"]))
        require(key not in records, f"duplicate source record: {source}/{key}")
        records[key] = r
    links = defaultdict(list)
    numeric = ("power_w", "actual_rate_bps", "predicted_rate_bps", "predicted_p05_bps", "pair_ee_bit_per_j")
    with (source / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["row_type"] != "link" or row["method"] not in method_names:
                continue
            key = row["method"], int(row["block"])
            require(row["active"] == "True" and row["selected"] == "True", f"inactive link row: {key}")
            link = {name: float(row[name]) for name in numeric}
            link.update({name: int(row[name]) for name in ("link_index", "tx", "rx", "tx_port", "rx_port")})
            link["mode"] = row["mode"]
            links[key].append(link)
    for key, rows in links.items():
        rows.sort(key=lambda r: r["link_index"])
        require([r["link_index"] for r in rows] == list(range(len(rows))), f"nonunique link indices: {key}")
        require(key in records and records[key]["success"], f"links without successful record: {key}")
    return summary, records, links, dict(source=str(source), summary_sha256=digest(source / "summary.json"),
                                        results_csv_sha256=digest(source / "results.csv"))


def metrics(powers, modes, rates, predicted_mean, low, cfg, eta_ref):
    p, modes, rates = np.asarray(powers, float), np.asarray(modes), np.asarray(rates, float)
    rf = modes == "RF"
    require(np.all(rf | (modes == "VLC")), "unknown mode")
    phy = cfg["physical"]
    pc = np.where(rf, phy["rf"]["circuit_power_w"], phy["vlc"]["circuit_power_w"])
    lo = np.where(rf, phy["rf"]["power_min_w"], phy["vlc"]["power_min_w"])
    hi = np.where(rf, phy["rf"]["power_max_w"], phy["vlc"]["power_max_w"])
    require(np.isfinite(p).all() and np.all(p >= lo-1e-9) and np.all(p <= hi+1e-9), "power outside saved limits")
    ee = rates / (p + pc)
    mean = np.asarray(predicted_mean, float)
    calibrated = np.maximum(mean, 1e-9) * (np.maximum(low, 1e-9) / np.maximum(mean, 1e-9)) ** cfg["qos_chance_inflation"]
    require(cfg["qos_metric"] == "chance", "this report requires chance-service settings")
    pred_ee = mean / (p + pc)
    return dict(mean_ee=float(ee.mean()), min_ee=float(ee.min()), link_ee=ee,
                mean_rate=float(rates.mean()), service_all=bool(np.all(rates >= cfg["q_eval_bps"])),
                violating_links=int(np.count_nonzero(rates < cfg["q_eval_bps"])),
                powers=p, modes=modes, vlc_fraction=float(np.mean(~rf)),
                predicted_mean_ee=float(pred_ee.mean()), predicted_min_ee=float(pred_ee.min()),
                predicted_budget_pass=bool(pred_ee.mean() >= (1-cfg["delta"])*eta_ref-1e-7),
                final_quantile_pass=bool(np.all(calibrated >= cfg["q_min_bps"]-1e-7)),
                quantile_min_rate=float(calibrated.min()))


def original_metrics(record, rows, cfg):
    result = metrics([r["power_w"] for r in rows], [r["mode"] for r in rows],
                     [r["actual_rate_bps"] for r in rows], [r["predicted_rate_bps"] for r in rows],
                     [r["predicted_p05_bps"] for r in rows], cfg, record["eta_ref"])
    close(result["link_ee"], [r["pair_ee_bit_per_j"] for r in rows], "historical link EE")
    for dest, src in (("mean_ee", "mean_pair_ee_bit_per_j"), ("min_ee", "min_pair_ee_bit_per_j"),
                      ("vlc_fraction", "vlc_link_fraction"), ("predicted_mean_ee", "predicted_mean_pair_ee")):
        close(result[dest], record[src], f"historical {src}")
    require(result["service_all"] == record["all_pairs_qos_met"], "historical service flag mismatch")
    return result


def audit_replay(folder, record, original, rows, cfg):
    hashes = read_json(folder / "commit.json")
    require(set(hashes) == {"record.json", "arrays.npz"}, f"unexpected commit entries: {folder}")
    for name, expected_hash in hashes.items():
        require(digest(folder / name) == expected_hash, f"commit hash mismatch: {folder / name}")
    if record["status"] == "original_failure_preserved":
        require(not original["reference"]["success"] or not original["old"]["success"], "unexplained preserved failure")
        with np.load(folder / "arrays.npz", allow_pickle=False) as a:
            require(not a.files, "preserved failure unexpectedly has arrays")
        return None
    require(record["status"] == "evaluated" and record["method"] == "maxmin", f"unexpected replay status: {folder}")
    for key in ("old_returned_initial_action", "new_returned_initial_action", "original_initial_mean_fallback_flag"):
        require(isinstance(record.get(key), bool), f"missing/untyped audit flag {key}: {folder}")
    require(record["original_initial_mean_fallback_flag"] == original["old"]["chance_fallback"], "initial fallback flag changed")
    with np.load(folder / "arrays.npz", allow_pickle=False) as a:
        oldrows, refrows = rows["old"], rows["reference"]
        for key in ("tx", "rx", "mode"):
            require(np.array_equal(a[key], [r[key] for r in oldrows]), f"saved topology differs: {folder}/{key}")
        for row in oldrows:
            if row["mode"] == "VLC":
                for port in ("tx_port", "rx_port"):
                    require(int(a[port][row["tx"],row["rx"]]) == row[port], "selected VLC port changed")
        close(a["old_power_w"], [r["power_w"] for r in oldrows], "old action powers", atol=1e-12)
        for name in ("reference", "old", "new"):
            p = a["new_power_w"] if name == "new" else [r["power_w"] for r in rows[name]]
            mode = a["mode"] if name == "new" else [r["mode"] for r in rows[name]]
            samples, fine = a[f"{name}_samples_bps"], a[f"{name}_rate_bps"]
            require(samples.shape == (cfg["draws"], len(p)) and fine.shape == (100, len(p)), "unexpected saved sample shape")
            result = metrics(p, mode, fine.mean(0), samples.mean(0), np.quantile(samples, .05, axis=0),
                             cfg, original["reference"]["eta_ref"])
            actual, pred = record["actual"][name], record[f"{name}_prediction"]
            for dest, src in (("mean_ee", "mean_pair_ee_bit_per_j"), ("min_ee", "min_pair_ee_bit_per_j")):
                close(result[dest], actual[src], f"replay actual {name}/{dest}")
            close(result["link_ee"], actual["pair_ee_bit_per_j"], f"replay actual links {name}")
            close(fine.mean(0), actual["mean_rate_bps"], f"replay actual rates {name}")
            close(np.asarray(p) * actual["duration_s"], actual["link_energy_j"], "TX input energy")
            for dest, src in (("predicted_mean_ee", "mean_ee"), ("predicted_min_ee", "min_ee"), ("quantile_min_rate", "quantile_min_rate")):
                close(result[dest], pred[src], f"replay predicted {name}/{dest}")
            require(result["service_all"] == actual["all_pairs_qos_met"], "replay service flag mismatch")
            require(result["violating_links"] == np.count_nonzero(~np.asarray(actual["pair_qos_met"], bool)), "replay violating-link mismatch")
            require(result["predicted_budget_pass"] == pred["budget_pass"] and result["final_quantile_pass"] == pred["quantile_pass"], "replay prediction flags mismatch")
            if name != "new":
                close(fine.mean(0), [r["actual_rate_bps"] for r in rows[name]], "replay/historical actual")
                close(samples.mean(0), [r["predicted_rate_bps"] for r in rows[name]], "replay/historical prediction")
                close(np.quantile(samples, .05, axis=0), [r["predicted_p05_bps"] for r in rows[name]], "replay/historical p05")
            else:
                new_result = result
        require(record["unchanged"] == np.array_equal(a["old_power_w"], a["new_power_w"]), "retained-incumbent flag mismatch")
        require(record["diagnostics"]["retained_incumbent"] == record["unchanged"], "diagnostic retained flag mismatch")
        def action_signature(name):
            if name == "new":
                return sorted((int(i), int(j), str(m), float(p), int(a["tx_port"][i,j]) if m == "VLC" else -1,
                               int(a["rx_port"][i,j]) if m == "VLC" else -1)
                              for i,j,m,p in zip(a["tx"],a["rx"],a["mode"],a["new_power_w"]))
            return sorted((r["tx"],r["rx"],r["mode"],r["power_w"],r["tx_port"] if r["mode"] == "VLC" else -1,
                           r["rx_port"] if r["mode"] == "VLC" else -1) for r in rows[name])
        for name in ("old", "new"):
            require(record[f"{name}_returned_initial_action"] == (action_signature(name) == action_signature("reference")), "returned-initial flag mismatch")
        if not record["unchanged"]:
            require(new_result["predicted_budget_pass"] and new_result["final_quantile_pass"], "changed action violates prediction constraints")
        oldmin = record["old_prediction"]["min_ee"]
        require(new_result["predicted_min_ee"] >= oldmin-max(1e-6, abs(oldmin)*1e-11), "predicted minimum EE decreased")
    close(record["original_complete_decision_s"], original["old"]["solve_time_s"], "old timing")
    close(record["estimated_complete_decision_s"], record["original_complete_decision_s"]+record["refine_time_s"], "estimated complete timing")
    return new_result


def power_statistics(values, cfg):
    modes = np.concatenate([r["modes"] for r in values]) if values else np.array([], str)
    powers = np.concatenate([r["powers"] for r in values]) if values else np.array([], float)
    out = dict(active_link_instances=len(powers), vlc_link_instances=int(np.sum(modes == "VLC")),
               vlc_link_fraction=float(np.mean(modes == "VLC")) if len(powers) else None)
    for mode in ("RF", "VLC"):
        p = powers[modes == mode]
        spec = cfg["physical"][mode.lower()]
        eff = spec["pa_efficiency" if mode == "RF" else "eo_efficiency"]
        out[mode] = dict(electrical_input_W=distribution(p),
                         output_W=distribution(p * eff), output_mW=distribution(p * eff * 1000),
                         output_kind="RF radiated power" if mode == "RF" else "VLC optical power",
                         conversion_efficiency=eff, circuit_power_W=spec["circuit_power_w"],
                         lower_bound_W=spec["power_min_w"], upper_bound_W=spec["power_max_w"],
                         lower_bound_count=int(np.sum(np.isclose(p, spec["power_min_w"], rtol=1e-7, atol=1e-10))),
                         upper_bound_count=int(np.sum(np.isclose(p, spec["power_max_w"], rtol=1e-7, atol=1e-10))),
                         bound_tolerance="numpy isclose rtol=1e-7, atol=1e-10 W")
    return out


def search_error_statistics(records):
    """An optional-candidate error is distinct from no executable block action."""
    event_counts, period_counts = Counter(), Counter()
    details = []
    retained, changed = 0, 0
    for record in records:
        diagnostic = record.get("diagnostics")
        require(isinstance(diagnostic, dict) and isinstance(diagnostic.get("errors"), list),
                f"missing search-error list in block {record['block']}; do not assume zero")
        errors = diagnostic["errors"]
        keys = []
        for error in errors:
            require(isinstance(error, dict) and isinstance(error.get("stage"), str)
                    and isinstance(error.get("error"), str) and error["error"],
                    f"unknown search-error format in block {record['block']}")
            keys.append((error["stage"], error["error"]))
        event_counts.update(keys)
        period_counts.update(set(keys))
        if errors:
            details.append(dict(block=record["block"], errors=errors,
                                retained_old_incumbent=record["unchanged"]))
            retained += int(record["unchanged"])
            changed += int(not record["unchanged"])
    return dict(checked_evaluated_blocks=len(records), error_period_count=len(details),
                error_event_count=sum(event_counts.values()),
                error_blocks=[d["block"] for d in details],
                retained_old_with_error_count=retained, changed_action_with_error_count=changed,
                reasons=[dict(stage=k[0], error=k[1], event_count=event_counts[k],
                              period_count=period_counts[k]) for k in sorted(event_counts)],
                details=details,
                classification="Additional candidate search errors with an executable returned action; not whole-period solve failures.")


def summarize_method(values, failures, reference, cfg, completed_count, planned=1000):
    rows = list(values.values())
    passes = sum(r["service_all"] for r in rows)
    common = sorted(set(values) & set(reference))
    loss = [1-values[k]["mean_ee"]/reference[k]["mean_ee"] for k in common]
    out = dict(success_count=len(rows), success_blocks=sorted(values), failure_count=len(failures),
               failure_blocks=sorted(failures), failure_reasons=dict(Counter(failures.values())),
               completed_blocks=completed_count, planned_blocks=planned, pending_blocks=planned-completed_count,
               mean_link_EE_bit_per_J=distribution([r["mean_ee"] for r in rows]),
               per_period_min_link_EE_bit_per_J=distribution([r["min_ee"] for r in rows]),
               mean_link_rate_bps=distribution([r["mean_rate"] for r in rows]),
               all_links_service_pass_count=passes, planned_service_rate=passes/planned,
               completed_service_rate=passes/completed_count if completed_count else None,
               successful_action_service_rate=passes/len(rows) if rows else None,
               violating_link_count=sum(r["violating_links"] for r in rows),
               actual_loss_vs_initial=dict(common_success_count=len(common),
                                           period_relative_mean_EE_loss=distribution(loss),
                                           exceeds_delta_count=sum(x > cfg["delta"]+1e-12 for x in loss),
                                           exceeds_delta_blocks=[k for k,x in zip(common,loss) if x > cfg["delta"]+1e-12]),
               predicted_budget_below_threshold_count=sum(not r["predicted_budget_pass"] for r in rows),
               final_quantile_fail_count=sum(not r["final_quantile_pass"] for r in rows),
               final_quantile_fail_blocks=[k for k,r in values.items() if not r["final_quantile_pass"]],
               power=power_statistics(rows,cfg), relative_to_initial=result_comparison(values,reference))
    return out


def summarize_group(group, locations, item):
    cfg = item["config"]
    source, state_source = Path(item["source"]), Path(item["state_source"])
    same = source == state_source
    s, records, links, source_audit = load_historical(source, {"reference", "maxmin", ZAPPONE} if same else {"reference", "maxmin"})
    if same:
        zs, zr, zl, zaudit = s, records, links, source_audit
    else:
        zs, zr, zl, zaudit = load_historical(state_source, {"reference", ZAPPONE})
        require(zs["traffic_hashes"] == s["traffic_hashes"], "baseline traffic differs")
        for key in ("physical", "channel_seed", "forecast_seed", "draws", "qos_metric", "q_min_bps", "q_eval_bps", "qos_chance_inflation", "delta"):
            require(zs["config"][key] == cfg[key], f"baseline settings differ: {key}")
        # Same-period baseline reference must be the exact saved physical action.
        for k in range(1000):
            a, b = records["reference",k], zr["reference",k]
            require(a["success"] == b["success"], "baseline reference success differs")
            if a["success"]:
                require(a["action_sha256"] == b["action_sha256"], "baseline reference action differs")
                close(a["eta_ref"],b["eta_ref"],"baseline reference predicted EE")
                close(a["mean_pair_ee_bit_per_j"],b["mean_pair_ee_bit_per_j"],"baseline reference actual EE")
    methods = {name:{} for name in METHODS}
    failures = {name:{} for name in METHODS}
    replay, times = [], defaultdict(list)
    for k, folder in sorted(locations.items()):
        raw = read_json(folder/"record.json")
        require(raw["group"] == group and raw["block"] == k, f"record identity mismatch: {folder}")
        original = {"reference": records["reference",k], "old": records["maxmin",k]}
        blockrows = {"reference": links.get(("reference",k),[]), "old": links.get(("maxmin",k),[])}
        for name, rr, rows in (("reference",original["reference"],blockrows["reference"]),
                               ("old",original["old"],blockrows["old"]),
                               ("zappone",zr[ZAPPONE,k],zl.get((ZAPPONE,k),[]))):
            times[name+"_historical_complete_all_attempts_s"].append(rr["solve_time_s"])
            if rr["success"]:
                require(len(rows) == int(rr["n"])//2, "wrong link count")
                methods[name][k] = original_metrics(rr,rows,cfg)
                times[name+"_historical_complete_success_s"].append(rr["solve_time_s"])
            else:
                failures[name][k] = rr.get("failure_reason") or "unspecified saved failure"
        new = audit_replay(folder,raw,original,blockrows,cfg)
        if new is None:
            failures["new"][k] = raw.get("old_reason") or raw.get("initial_reason") or "original_failure_preserved"
            continue
        methods["new"][k] = new
        replay.append(raw)
        for key in ("refine_time_s","estimated_complete_decision_s","original_complete_decision_s",
                    "reused_initial_solve_s","actual_evaluation_s","replay_wall_s"):
            times[key].append(raw[key])
        times["historical_forecast_build_s"].append(original["old"]["forecast_build_time_s"])
        residual = original["old"]["solve_time_s"]-original["old"]["forecast_build_time_s"]-original["reference"]["reference_solve_time_s"]
        require(residual >= -1e-6, "negative historical fairness timing remainder")
        times["historical_fair_stage_residual_s"].append(max(0.,residual))
    out = dict(completed=len(locations), planned=1000, missing_blocks=sorted(set(range(1000))-set(locations)),
               settings=dict(config=cfg, historical_environment=s.get("environment"),
                             baseline_environment=zs.get("environment")),
               historical_provenance=[source_audit] if same else [source_audit,zaudit],
               methods={name:summarize_method(methods[name],failures[name],methods["reference"],cfg,len(locations)) for name in METHODS},
               comparisons={name:result_comparison(methods[a],methods[b]) for name,a,b in (
                   ("new_vs_old","new","old"),("new_vs_zappone","new","zappone"),("old_vs_zappone","old","zappone"))},
               timing_seconds={key:distribution(v) for key,v in times.items()})
    out["flow_counts"] = dict(evaluated_count=len(replay),
        initial_mean_fallback_count=sum(r["original_initial_mean_fallback_flag"] for r in replay),
        old_returned_initial_action_count=sum(r["old_returned_initial_action"] for r in replay),
        new_returned_initial_action_count=sum(r["new_returned_initial_action"] for r in replay),
        retained_old_incumbent_count=sum(r["unchanged"] for r in replay),
        changed_power_action_count=sum(not r["unchanged"] for r in replay),
        old_final_quantile_failure_count=sum(not r["old_prediction"]["quantile_pass"] for r in replay),
        new_final_quantile_failure_count=sum(not r["new_prediction"]["quantile_pass"] for r in replay),
        initial_mean_fallback_and_old_final_quantile_pass_count=sum(r["original_initial_mean_fallback_flag"] and r["old_prediction"]["quantile_pass"] for r in replay),
        initial_mean_fallback_and_new_final_quantile_pass_count=sum(r["original_initial_mean_fallback_flag"] and r["new_prediction"]["quantile_pass"] for r in replay),
        new_returned_initial_and_final_quantile_fail_count=sum(r["new_returned_initial_action"] and not r["new_prediction"]["quantile_pass"] for r in replay),
        predicted_min_EE_decrease_count=0)
    out["new_search_errors"] = search_error_statistics(replay)
    out["methods"]["zappone"]["prediction_budget_is_binding_constraint"] = False
    out["timing_note"] = "新全程为历史旧全程加本次新增量的估算；历史/新增测量时环境与并发不同。未重新测量端到端新流程；恢复、真实信道评估和保存开销不计入决策新增量。"
    return out


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def ee_display(value):
    return fmt(value / 1e6 if value is not None else None)


def make_report(result):
    complete = result["complete"]
    text = ["# 保存动作功率改进：全量汇总草稿" if complete else "# 未完成：保存动作功率改进部分核查", "",
            f"生成时间：{result['generated_beijing']}（北京时间）。状态：{'3000 个计划周期均已提交并通过完整性核查' if complete else '实验尚未全部完成；以下仅为已提交周期，不作最终结论'}。", "",
            "平均链路 EE 先在每个成功周期内对活动链路取算术平均，再跨成功周期平均；最低链路 EE 先取每周期最弱链路，再跨成功周期平均，单位 Mbit/J。方法间比较严格采用同周期共同成功集合。", "",
            "服务率的计划分母为每组 1000 周期；求解失败单列并不记为成功服务。实际服务按执行后周期平均速率达到固定 1 Mbit/s 检查。预测预算检查与真实平均 EE 损失分别报告；Zappone 的预算检查只是同阈值诊断，其算法没有该预算约束。", "",
            "功率表中的 p 是发送支路电输入功率；RF 辐射功率为 0.1p，VLC 光功率为 0.25p。EE 分母分别为 p+0.1159 W 和 p+1.745 W，不重复除以转换效率。功率统计的分母是该模式的活动链路次数。", ""]
    for group,g in result["groups"].items():
        text += [f"## {group}（已提交 {g['completed']}/1000）", "",
                 "| 方法 | 成功/失败 | 平均 EE | 每周期最低 EE 均值 | 全链路实际达标/计划周期 | 违约链路次数 | VLC 使用率 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name,m in g["methods"].items():
            text.append(f"| {LABELS[name]} | {m['success_count']}/{m['failure_count']} | {ee_display(m['mean_link_EE_bit_per_J']['mean'])} | {ee_display(m['per_period_min_link_EE_bit_per_J']['mean'])} | {m['all_links_service_pass_count']}/1000 | {m['violating_link_count']} | {fmt(100*m['power']['vlc_link_fraction']) if m['power']['vlc_link_fraction'] is not None else '—'}% |")
        text += ["", "| 方法 | 相对初始的平均 EE 损失：逐周期比例均值 | 真实损失超过 10% | 预测均值低于预算阈值 | 最终校准下分位不通过 |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for name,m in g["methods"].items():
            loss=m["actual_loss_vs_initial"]
            text.append(f"| {LABELS[name]} | {fmt(100*loss['period_relative_mean_EE_loss']['mean']) if loss['period_relative_mean_EE_loss']['mean'] is not None else '—'}%（n={loss['common_success_count']}） | {loss['exceeds_delta_count']}/{loss['common_success_count']} | {m['predicted_budget_below_threshold_count']}/{m['success_count']} | {m['final_quantile_fail_count']}/{m['success_count']} |")
        text += ["", "| 方法/模式 | 电输入均值 W | 中位数 W | P95 W | 最大 W | 下界/上界次数 | 输出功率均值 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name,m in g["methods"].items():
            for mode in ("RF","VLC"):
                p=m["power"][mode]; d=p["electrical_input_W"]
                emitted=fmt(p["output_mW" if mode == "RF" else "output_W"]["mean"])+( " mW 辐射" if mode == "RF" else " W 光功率")
                text.append(f"| {LABELS[name]}/{mode}（n={d['n']}） | {fmt(d['mean'],6)} | {fmt(d['median'],6)} | {fmt(d['p95'],6)} | {fmt(d['max'],6)} | {p['lower_bound_count']}/{p['upper_bound_count']} | {emitted} |")
        text += ["", "| 同周期比较（左/右） | 共同成功 n | 平均 EE：左/右 Mbit/J | 最低 EE：左/右 Mbit/J | 最低 EE 胜/负/平 |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for name,c in g["comparisons"].items():
            mn,mi=c["mean_ee"],c["min_ee"]
            pair=lambda d: f"{fmt(d['left']['mean']/1e6)}/{fmt(d['right']['mean']/1e6)}" if d['left']['n'] else '—'
            text.append(f"| {name} | {c['common_success_count']} | {pair(mn)} | {pair(mi)} | {mi['wins']}/{mi['losses']}/{mi['ties']} |")
        f=g["flow_counts"]
        text += ["",f"流程核对（n={f['evaluated_count']}）：初始阶段均值后备 {f['initial_mean_fallback_count']}；旧/新返回初始动作 {f['old_returned_initial_action_count']}/{f['new_returned_initial_action_count']}；本次保留旧主方法动作 {f['retained_old_incumbent_count']}，改变功率 {f['changed_power_action_count']}。保留旧动作与返回初始动作是不同事件。", "",
                 f"初始均值后备后，旧/新最终仍通过下分位检查的周期分别为 {f['initial_mean_fallback_and_old_final_quantile_pass_count']}/{f['initial_mean_fallback_and_new_final_quantile_pass_count']}；新动作返回初始且最终下分位不通过 {f['new_returned_initial_and_final_quantile_fail_count']}。", "",
                 ]
        errors=g["new_search_errors"]
        reason_text="；".join(f"{r['stage']}: {r['error']}（{r['period_count']} 周期/{r['event_count']} 条）" for r in errors["reasons"]) or "未记录异常"
        text += [f"新增搜索异常：{errors['error_period_count']}/{errors['checked_evaluated_blocks']} 周期，共 {errors['error_event_count']} 条；其中保留旧动作 {errors['retained_old_with_error_count']} 周期、返回改进动作 {errors['changed_action_with_error_count']} 周期。这些周期仍有可执行动作，不并入整周期求解失败数。原因：{reason_text}。", "",
                 "| 时间项 | 有效周期 n | 平均 s | 中位 s | P95 s | 最大 s |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        timing_labels={"original_complete_decision_s":"历史旧主方法全程（成功周期）", "refine_time_s":"本次功率改进新增量", "estimated_complete_decision_s":"旧全程+新增量（估算）", "historical_forecast_build_s":"历史条件预测", "reused_initial_solve_s":"历史初始分配", "historical_fair_stage_residual_s":"历史公平阶段（全程扣除预测与初始）", "zappone_historical_complete_success_s":"历史 Zappone 全程（成功周期）"}
        for key,label in timing_labels.items():
            d=g["timing_seconds"].get(key,distribution([]))
            text.append(f"| {label} | {d['n']} | {fmt(d['mean'])} | {fmt(d['median'])} | {fmt(d['p95'])} | {fmt(d['max'])} |")
        text += ["",g["timing_note"], "",f"设置：每周期 {g['settings']['config']['physical']['estimation']['block_dt_s']} s；预测样本数 {g['settings']['config']['draws']}；旧记录线程数 {g['settings']['config']['threads']}；本次每个进程 {result['audit']['replay_threads']} 线程。历史软件环境保存在 JSON 的 settings.historical_environment；原记录未保存 CPU 型号与内存，不能把本次硬件查询追溯为历史硬件证据。求解等待未加入通信轨迹。", ""]
        drops=g["comparisons"]["new_vs_old"]["min_ee"]["loss_blocks"]
        text += [f"新增动作的真实最低 EE 下降周期共 {len(drops)} 个；完整编号保存在 JSON 的 comparisons.new_vs_old.min_ee.loss_blocks。预测最低 EE 不下降已逐周期核验，这不等同于真实执行最低 EE 必然不下降。", ""]
    hardware=result["aggregation_environment"].get("hardware")
    if hardware:
        text += [f"本次汇总时核对的本机硬件：{hardware['cpu']}，{hardware['cores']} 核 / {hardware['logical_processors']} 逻辑处理器，内存 {hardware['memory_bytes']/1024**3:.2f} GiB。", ""]
    text += ["## 审计与解释范围", "", "逐块核对 record.json/arrays.npz 的提交校验值、分片间源代码版本、输入摘要与轨迹哈希、计划周期并集及重复周期；逐链路重算 EE、校准分位检查、功率单位与服务结果，并与历史记录及新保存数组相互核对。", "",
             f"模型、条件预测器与基线版本保持原值的逐文件比对，以及新代码允许变更的范围，见[独立审计报告第2节](<{INDEPENDENT_AUDIT.as_posix()}>); 冻结求解源码的摘要见该报告第7节。配套 JSON 同时保存本次读取该审计报告的 SHA-256。", "",
             "这里评估的是保存的旧主方法动作上追加一次功率改进的效果；配对、模式、端口和预测信息保持原记录。既不重新求解初始分配，也不将事后真实信道评价用于选择动作。所有比例、失败周期、比较集合、功率分布和时间分布均见配套 JSON。"]
    return "\n".join(text)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs="+",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--out",type=Path,default=ROOT/"analysis")
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args(argv)
    out=args.out.resolve()
    require(ROOT.resolve() in out.parents, "write summary only below this experiment directory")
    locations,inputs,audit=provenance(args.runs,args.allow_partial)
    audit["supporting_independent_audit"] = dict(path=str(INDEPENDENT_AUDIT),sha256=digest(INDEPENDENT_AUDIT),
                                                sections=["2. 模型与原流程未被改动", "7. 本次审计的求解源码标识"])
    generated=datetime.now(timezone.utc)
    result=dict(schema_version=1,generated_utc=generated.isoformat(),
                generated_beijing=generated.astimezone(timezone(timedelta(hours=8))).isoformat(),
                complete=audit["all_3000_committed"],audit=audit,
                aggregation_environment=current_environment(), groups={})
    for group in ("10A","10B","50B"):
        result["groups"][group]=summarize_group(group,locations[group],inputs[group])
        print(f"Audited {group}: {len(locations[group])}/1000 committed",flush=True)
    out.mkdir(parents=True,exist_ok=True)
    stem="full" if result["complete"] else "partial"
    (out/f"{stem}_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (out/f"{stem}_summary_中文报告草稿.md").write_text(make_report(result),encoding="utf-8")
    print(f"Saved {stem} summary to {out}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
