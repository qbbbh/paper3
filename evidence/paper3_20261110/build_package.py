"""Export published numerical evidence; never runs simulation or optimization.

This local build helper needs the original local files. The independent
verify_package.py checker only needs the published evidence package.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from collections import Counter
from urllib.parse import unquote

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
SRC = Path(r"E:\博士\课题论文\第3篇-修改版\代码\公平功率求解改进_20261110")
sys.path.insert(0, str(SRC / "code"))
from summarize_guarded_replay import load_historical

GROUPS = ("10A", "10B", "50B")
SAMPLE_BLOCKS = list(range(100, 1000, 100))
METHOD_NAMES = dict(reference="reference", maxmin="main_old", same_topo_maxmin="same_topo_old",
                    no_refine="no_refine", aboagye_matching_qt="aboagye_matching_qt",
                    zappone_mmee_fp_fixed="zappone_mmee_fp_fixed")
manifest = dict(schema_version=1, purpose="Full-period audit export; no new simulation/optimization",
                source_root_local_only=str(SRC), sources=[], transformations=[], sample_blocks=SAMPLE_BLOCKS,
                excluded="Full restart states, full runs, third-party PDFs/PNGs remain local and are not included")
source_ids = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source(path, published=None, transform="none"):
    path = Path(path)
    key = str(path)
    if key not in source_ids:
        sid = "S%05d" % len(source_ids)
        source_ids[key] = sid
        row = dict(id=sid, local_path=key, exists=path.exists(), included=False)
        if path.is_file(): row.update(bytes=path.stat().st_size, sha256=sha(path))
        manifest["sources"].append(row)
    row = manifest["sources"][int(source_ids[key][1:])]
    if published:
        row.update(included=True, published_path=Path(published).relative_to(OUT).as_posix(),
                   published_sha256=sha(published), transformation=transform)
    return row["id"]


copy_map = {}
for relative in ("analysis/10车与50车_功率改进全量核查_20261110.md", "analysis/full_summary.json",
                 "analysis/same_topo/同拓扑消融_补算复用与全量结论.md",
                 "analysis/same_topo/same_topo_full_summary.json",
                 "analysis/same_topo/same_topo_period_metrics.csv",
                 "analysis/same_topo/same_topo_reuse_provenance.json",
                 "analysis/same_topo/same_topo_execution_and_reuse_plan.json",
                 "analysis/same_topo/same_topo_output_manifest.json"):
    copy_map[SRC/relative] = OUT/"reports"/Path(relative).name
for p in (SRC/"evidence").iterdir():
    if p.suffix.lower() in {".md", ".json", ".py"}: copy_map[p] = OUT/"audits"/p.name


def published_target(path):
    if path in copy_map: return copy_map[path]
    try: return REPO/"releases/paper3_20261110/code"/path.relative_to(SRC/"code")
    except ValueError: return None


def rewrite_markdown(text, dest):
    def replace(match):
        label, raw = match.group(1), unquote(match.group(2).strip("<>"))
        if not re.match(r"^[A-Za-z]:[/\\]", raw): return match.group(0)
        raw_path = re.sub(r":\d+(?:-\d+)?$", "", raw)
        p = Path(raw_path)
        target = published_target(p)
        if target:
            rel = Path(os.path.relpath(target, dest.parent)).as_posix()
            return "["+label+"](<"+rel+">)"
        sid = source(p)
        return label+"（本地来源 "+sid+"，见 SOURCE_MANIFEST.json；未打包）"
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, text)


def copy_reports():
    for src, dest in copy_map.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".md":
            body = rewrite_markdown(src.read_text(encoding="utf-8"), dest)
            body += "\n\n---\n发布说明：本文由本地原报告复制，仅将可映射的本地链接改为仓库相对链接；未打包链接注明本地来源。来源原文与发布文本的 SHA-256 分别记录于 SOURCE_MANIFEST.json。\n"
            dest.write_text(body, encoding="utf-8")
            source(src, dest, "Markdown link rewriting plus publication provenance note; numerical text unchanged")
        else:
            shutil.copyfile(src, dest); source(src, dest)


def primary(group, k):
    return SRC/"runs"/(f"full_50B_s{k//250}" if group == "50B" else f"full_{group}")/group/f"block_{k:06d}"


def committed(folder):
    hashes = read(folder/"commit.json")
    for name, expected in hashes.items():
        assert sha(folder/name) == expected, folder/name
        source(folder/name)
    source(folder/"commit.json")
    return read(folder/"record.json")


LINK_FIELDS = ["group", "block", "method", "link_index", "tx", "rx", "mode", "tx_port", "rx_port",
 "power_electrical_input_w", "emitted_rf_or_optical_power_w", "circuit_power_w", "power_min_w", "power_max_w",
 "at_power_lower_bound", "at_power_upper_bound", "forecast_draws", "predicted_mean_rate_bps", "predicted_p05_raw_bps",
 "predicted_p05_calibrated_bps", "actual_mean_rate_bps", "predicted_link_ee_bit_per_j", "actual_link_ee_bit_per_j",
 "actual_service_pass", "final_quantile_link_pass", "source_id"]

PERIOD_FIELDS = ["group", "block", "method", "success", "failure_reason", "source_kind", "source_id", "links",
 "actual_mean_link_ee_bit_per_j", "actual_min_link_ee_bit_per_j", "predicted_mean_link_ee_bit_per_j",
 "predicted_min_link_ee_bit_per_j", "actual_mean_link_rate_bps", "actual_violating_links", "all_links_actual_service_pass",
 "actual_reference_mean_link_ee_bit_per_j", "actual_mean_ee_loss_vs_reference", "actual_loss_over_delta",
 "predicted_reference_mean_link_ee_bit_per_j", "delta", "predicted_mean_budget_check_pass", "budget_constraint_applies",
 "final_quantile_pass", "final_quantile_violating_links", "initial_mean_fallback", "returned_initial_action",
 "retained_previous_incumbent", "vlc_fraction", "total_electrical_input_power_w", "total_input_plus_circuit_power_w",
 "mean_rf_input_power_w", "mean_vlc_input_power_w", "mean_rf_radiated_power_w", "mean_vlc_optical_power_w",
 "power_lower_bound_fraction", "power_upper_bound_fraction", "historical_complete_decision_s", "incremental_refine_s",
 "estimated_complete_decision_s", "incremental_time_origin", "historical_forecast_build_s", "historical_reference_solve_s"]


def selected_links(group, k, method, raw, cfg, sid):
    result = []
    for x in raw:
        phy = cfg["physical"][x["mode"].lower()]
        p, mean, low, actual = (float(x[n]) for n in ("power_w", "predicted_rate_bps", "predicted_p05_bps", "actual_rate_bps"))
        calibrated = max(mean, 1e-9)*(max(low, 1e-9)/max(mean, 1e-9))**cfg["qos_chance_inflation"]
        pc = phy["circuit_power_w"]
        result.append(dict(group=group, block=k, method=method, **{n:x[n] for n in ("link_index","tx","rx","mode","tx_port","rx_port")},
            power_electrical_input_w=p, emitted_rf_or_optical_power_w=p*phy["pa_efficiency" if x["mode"]=="RF" else "eo_efficiency"],
            circuit_power_w=pc, power_min_w=phy["power_min_w"], power_max_w=phy["power_max_w"],
            at_power_lower_bound=bool(np.isclose(p,phy["power_min_w"],rtol=1e-8,atol=1e-10)),
            at_power_upper_bound=bool(np.isclose(p,phy["power_max_w"],rtol=1e-8,atol=1e-10)),
            forecast_draws=cfg["draws"], predicted_mean_rate_bps=mean, predicted_p05_raw_bps=low,
            predicted_p05_calibrated_bps=calibrated, actual_mean_rate_bps=actual,
            predicted_link_ee_bit_per_j=mean/(p+pc), actual_link_ee_bit_per_j=actual/(p+pc),
            actual_service_pass=actual>=cfg["q_eval_bps"], final_quantile_link_pass=calibrated>=cfg["q_min_bps"]-1e-7, source_id=sid))
    return result


def period(group,k,method,rec,links,cfg,sid,kind,actual_ref,eta_ref,**extra):
    row = dict(group=group,block=k,method=method,success=bool(rec["success"]),failure_reason=rec.get("failure_reason"),
        source_kind=kind,source_id=sid,links=len(links),all_links_actual_service_pass=False,
        delta=cfg["delta"],budget_constraint_applies=rec.get("budget_constraint_applies",method in {"main_old","same_topo_old","main_new","same_topo_new"}),
        initial_mean_fallback=rec.get("chance_fallback"), historical_complete_decision_s=rec.get("solve_time_s"),
        historical_forecast_build_s=rec.get("forecast_build_time_s"), historical_reference_solve_s=rec.get("reference_solve_time_s"))
    if links:
        assert rec["success"]
        a=np.array([x["actual_link_ee_bit_per_j"] for x in links]); pred=np.array([x["predicted_link_ee_bit_per_j"] for x in links])
        mean=float(a.mean());loss=1-mean/actual_ref if actual_ref is not None and actual_ref>0 else None
        power=np.array([x["power_electrical_input_w"] for x in links]); rf=np.array([x["mode"]=="RF" for x in links])
        row.update(actual_mean_link_ee_bit_per_j=mean,actual_min_link_ee_bit_per_j=float(a.min()),
           predicted_mean_link_ee_bit_per_j=float(pred.mean()),predicted_min_link_ee_bit_per_j=float(pred.min()),
           actual_mean_link_rate_bps=float(np.mean([x["actual_mean_rate_bps"] for x in links])),
           actual_violating_links=sum(not x["actual_service_pass"] for x in links),
           all_links_actual_service_pass=all(x["actual_service_pass"] for x in links),
           actual_reference_mean_link_ee_bit_per_j=actual_ref,actual_mean_ee_loss_vs_reference=loss,
           actual_loss_over_delta=loss>cfg["delta"] if loss is not None else None,predicted_reference_mean_link_ee_bit_per_j=eta_ref,
           predicted_mean_budget_check_pass=bool(pred.mean()>=(1-cfg["delta"])*eta_ref-1e-7) if eta_ref is not None else None,
           final_quantile_pass=all(x["final_quantile_link_pass"] for x in links),
           final_quantile_violating_links=sum(not x["final_quantile_link_pass"] for x in links),
           vlc_fraction=float(np.mean(~rf)),total_electrical_input_power_w=float(power.sum()),
           total_input_plus_circuit_power_w=float(sum(x["power_electrical_input_w"]+x["circuit_power_w"] for x in links)),
           mean_rf_input_power_w=float(power[rf].mean()) if rf.any() else None,
           mean_vlc_input_power_w=float(power[~rf].mean()) if (~rf).any() else None,
           mean_rf_radiated_power_w=float(power[rf].mean())*cfg["physical"]["rf"]["pa_efficiency"] if rf.any() else None,
           mean_vlc_optical_power_w=float(power[~rf].mean())*cfg["physical"]["vlc"]["eo_efficiency"] if (~rf).any() else None,
           power_lower_bound_fraction=float(np.mean([x["at_power_lower_bound"] for x in links])),
           power_upper_bound_fraction=float(np.mean([x["at_power_upper_bound"] for x in links])))
    row.update(extra)
    return row


def main():
    for d in ("reports","audits","tables","samples","manifests"): (OUT/d).mkdir(parents=True, exist_ok=True)
    copy_reports()
    full=read(SRC/"analysis/full_summary.json")
    traces=read(SRC/"analysis/same_topo/same_topo_reuse_provenance.json")["blocks"]
    trace={(r["group"],r["block"]):r for r in traces}
    all_periods=[]; sample_index=[]; counts={};configs={}; checks=[]
    for group in GROUPS:
        cfg=full["groups"][group]["settings"]["config"];configs[group]=cfg
        records={};oldlinks={};record_sources={}
        for provenance in full["groups"][group]["historical_provenance"]:
            src=Path(provenance["source"])
            assert sha(src/"summary.json")==provenance["summary_sha256"]
            assert sha(src/"results.csv")==provenance["results_csv_sha256"]
            ss=source(src/"summary.json");ls=source(src/"results.csv")
            summary,rs,lks,_=load_historical(src,set(METHOD_NAMES))
            for key,r in rs.items():
                if key in records:
                    assert r["success"]==records[key]["success"]
                    if r["success"]:
                        for key2 in ("mean_pair_ee_bit_per_j","predicted_mean_pair_ee"):
                            assert np.isclose(r[key2],records[key][key2],rtol=1e-11)
                    continue
                records[key]=r;oldlinks[key]=lks[key];record_sources[key]=(ss,ls)
        table=OUT/"tables"/f"{group}_links.csv.gz"
        preview=[]; nlinks=0
        with gzip.open(table,"wt",encoding="utf-8",newline="") as fout, gzip.open(OUT/"tables"/f"{group}_replay_records.jsonl.gz","wt",encoding="utf-8") as log:
            writer=csv.DictWriter(fout,fieldnames=LINK_FIELDS);writer.writeheader()
            for k in range(1000):
                pref=records["reference",k]
                actual_ref=pref.get("mean_pair_ee_bit_per_j") if pref["success"] else None
                eta_ref=pref.get("eta_ref") if pref["success"] else None
                pf=primary(group,k);mr=committed(pf);tr=trace[group,k]
                sf=Path(tr["expected_source_folder"])
                sr=mr if sf==pf else committed(sf)
                for oldmethod,label in METHOD_NAMES.items():
                    key=oldmethod,k;rec=records[key];ss,ls=record_sources[key]
                    links=selected_links(group,k,label,oldlinks[key],cfg,ls)
                    writer.writerows(links);nlinks+=len(links)
                    if k==100: preview.extend(links)
                    extra={}
                    if oldmethod=="maxmin": extra["returned_initial_action"]=mr.get("old_returned_initial_action")
                    if oldmethod=="same_topo_maxmin": extra["returned_initial_action"]=tr.get("old_returned_initial_action")
                    all_periods.append(period(group,k,label,rec,links,cfg,ss,"historical_saved",actual_ref,eta_ref,**extra))
                for family,folder,rr in (("main",pf,mr),("same_topo",sf,sr)):
                    label=family+"_new";sid=source(folder/"arrays.npz")
                    historical=records["maxmin" if family=="main" else "same_topo_maxmin",k]
                    origin=("separate_main_optimization" if rr["status"]=="evaluated" else "preserved_original_failure") if family=="main" else tr["kind"]
                    logical=dict(group=group,block=k,logical_method=label,input_origin=origin,
                                 original_record_source_id=source(folder/"record.json"),record=rr)
                    if family=="same_topo":logical["same_topology_metadata"]=tr
                    log.write(json.dumps(logical,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
                    extra=dict(initial_mean_fallback=historical.get("chance_fallback"),
                               returned_initial_action=rr.get("new_returned_initial_action") if family=="main" else tr.get("new_returned_initial_action"),
                               retained_previous_incumbent=rr.get("unchanged"),
                               incremental_refine_s=rr.get("refine_time_s",0),
                               estimated_complete_decision_s=historical["solve_time_s"]+rr.get("refine_time_s",0),
                               incremental_time_origin="measured_for_main" if family=="main" else tr.get("increment_time_origin","preserved_original_failure"))
                    links=[]
                    with np.load(folder/"arrays.npz",allow_pickle=False) as a:
                        if rr["status"]=="evaluated":
                            means=a["new_samples_bps"].mean(0);low=np.quantile(a["new_samples_bps"],.05,axis=0);actual=a["new_rate_bps"].mean(0)
                            raw=[dict(link_index=i,tx=int(tx),rx=int(rx),mode=str(mode),power_w=float(p),
                                 tx_port=int(a["tx_port"][tx,rx]) if mode=="VLC" else 0,
                                 rx_port=int(a["rx_port"][tx,rx]) if mode=="VLC" else 0,
                                 predicted_rate_bps=float(means[i]),predicted_p05_bps=float(low[i]),actual_rate_bps=float(actual[i]))
                                 for i,(tx,rx,mode,p) in enumerate(zip(a["tx"],a["rx"],a["mode"],a["new_power_w"]))]
                            links=selected_links(group,k,label,raw,cfg,sid)
                            assert np.allclose([x["actual_link_ee_bit_per_j"] for x in links],rr["actual"]["new"]["pair_ee_bit_per_j"],rtol=1e-11)
                        else:assert rr["status"]=="original_failure_preserved" and not a.files
                    outrow=period(group,k,label,historical,links,cfg,source(folder/"record.json"),origin,actual_ref,eta_ref,**extra)
                    if links:
                        assert outrow["final_quantile_pass"]==rr["new_prediction"]["quantile_pass"]
                        assert outrow["predicted_mean_budget_check_pass"]==rr["new_prediction"]["budget_pass"]
                        assert outrow["all_links_actual_service_pass"]==rr["actual"]["new"]["all_pairs_qos_met"]
                    writer.writerows(links);nlinks+=len(links)
                    if k==100: preview.extend(links)
                    all_periods.append(outrow)
                    if k in SAMPLE_BLOCKS:
                        dest=OUT/"samples"/group/family/f"block_{k:06d}"
                        dest.mkdir(parents=True,exist_ok=True)
                        for name in ("arrays.npz","record.json","commit.json"):
                            shutil.copyfile(folder/name,dest/name);source(folder/name,dest/name)
                        sample_index.append(dict(group=group,block=k,family=family,input_origin=origin,
                            path=(dest/"arrays.npz").relative_to(OUT).as_posix(),source_id=sid,
                            note="The reference action topology/power is in tables/*_links.csv.gz method=reference; arrays tx/rx/mode apply to old/new only"))
        with (OUT/"tables"/f"{group}_links_preview.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=LINK_FIELDS);w.writeheader();w.writerows(preview)
        group_periods=[r for r in all_periods if r["group"]==group]
        with (OUT/"tables"/f"{group}_periods.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=PERIOD_FIELDS);w.writeheader();w.writerows(group_periods)
        counts[group]=dict(period_rows=len(group_periods),link_rows=nlinks,methods={})
        for label in sorted(set(r["method"] for r in group_periods)):
            rows=[r for r in group_periods if r["method"]==label]; good=[r for r in rows if r["success"]]
            assert len(rows)==1000 and {r["block"] for r in rows}==set(range(1000))
            counts[group]["methods"][label]=dict(planned=1000,success=len(good),failure=1000-len(good),
                failure_blocks=[r["block"] for r in rows if not r["success"]],service_pass_count=sum(r["all_links_actual_service_pass"] for r in rows),
                mean_link_ee_bit_per_j=float(np.mean([r["actual_mean_link_ee_bit_per_j"] for r in good])),
                per_period_min_link_ee_bit_per_j=float(np.mean([r["actual_min_link_ee_bit_per_j"] for r in good])))
        for label,summarylabel in (("reference","reference"),("main_old","old"),("main_new","new"),("zappone_mmee_fp_fixed","zappone")):
            a=counts[group]["methods"][label]; b=full["groups"][group]["methods"][summarylabel]
            assert a["success"]==b["success_count"] and a["service_pass_count"]==b["all_links_service_pass_count"]
        print(group,counts[group]["period_rows"],nlinks,flush=True)
    for run in sorted({primary(g,k).parents[1] for g in GROUPS for k in (0,250,500,750)} | {SRC/"runs"/f"same_topo_{g}_diff" for g in GROUPS}):
        for name in ("manifest.json",):
            dest=OUT/"manifests"/(run.name+"_"+name);shutil.copyfile(run/name,dest);source(run/name,dest)
    sample_bytes=sum((OUT/r["path"]).stat().st_size for r in sample_index)
    assert sample_bytes<10_000_000, sample_bytes
    write(OUT/"samples/INDEX.json",dict(selection=SAMPLE_BLOCKS,selection_rule="Fixed validation set declared before optimization; all 9 blocks in all 3 groups and both logical methods",raw_npz_bytes=sample_bytes,items=sample_index))
    write(OUT/"tables/GROUP_CONFIGS.json",configs)
    write(OUT/"tables/EXPORT_COUNTS.json",counts)
    write(OUT/"SOURCE_MANIFEST.json",manifest)
    write(OUT/"tables/SCHEMA.json",dict(period_fields=PERIOD_FIELDS,link_fields=LINK_FIELDS,
       missing="CSV empty cells and JSON null are unavailable/not-applicable, never zero-imputed failure EE",
       ids="block is zero-based 0..999; tx/rx/link_index are zero-based; each group has 1000 planned 0.1-second periods",
       units="Rates bit/s, powers W, EE bit/J, time s. Divide EE by 1e6 for Mbit/J.",
       method_labels={**{v:k for k,v in METHOD_NAMES.items()},"main_new":"guarded refinement of original maxmin","same_topo_new":"guarded refinement of original same_topo_maxmin; explicit same-input reuse"},
       service="all_links_actual_service_pass requires success and every link period mean actual rate >= q_eval_bps",
       budget="predicted_mean_budget_check_pass checks (1-delta)*predicted reference mean link EE; applies only when budget_constraint_applies=true. Actual loss uses actual same-period reference mean EE.",
       calibration="p05_calibrated = max(mean,1e-9)*(max(raw_p05,1e-9)/max(mean,1e-9))**inflation; sample p05 uses numpy.quantile(q=.05, default linear)",
       time="estimated_complete_decision_s is historical_complete_decision_s + measured/reused incremental_refine_s; not a measured new end-to-end run",
       npz="All NPZ files are unmodified original committed selected-action arrays. Samples do not contain the full conditional channel bank needed for arbitrary new-power forecasts."))
    print('NPZ bytes',sample_bytes,'sources',len(manifest['sources']),flush=True)


if __name__=="__main__":main()
