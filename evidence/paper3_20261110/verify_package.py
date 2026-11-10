"""Read-only portable checks of the published package; requires only NumPy.

No source data outside this directory, allocator, model checkpoint, or simulator
is loaded. This verifies stored actions and arithmetic, not global optimality or
the conditional channel model used to produce candidate distributions.
"""
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path, compressed=False):
    h=hashlib.sha256()
    opener=gzip.open if compressed else open
    with opener(path,"rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    return h.hexdigest()


def close(a,b,label):
    if not np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-10,atol=1e-7):
        raise AssertionError(label)


def yes(value):
    assert value in ("True","False"),value
    return value=="True"


def main():
    config=read(ROOT/"tables/GROUP_CONFIGS.json")
    expected=read(ROOT/"tables/EXPORT_COUNTS.json")
    report=read(ROOT/"reports/full_summary.json")
    same_report=read(ROOT/"reports/same_topo_full_summary.json")
    result=dict(status="passed",groups={},sample_actions_checked=0,baseline_gzip_files_checked=0)
    package_manifest=ROOT/"PACKAGE_MANIFEST.json"
    if package_manifest.exists():
        for row in read(package_manifest)["files"]:
            assert digest(ROOT/row["path"])==row["sha256"],row["path"]
        result["package_files_checked"]=len(read(package_manifest)["files"])
    partition_path=ROOT/"tables/by_method/INDEX.json"
    if partition_path.exists():
        partitions=read(partition_path);assert partitions["file_count"]==24
        checked=0
        for item in partitions["files"]:
            original=ROOT/item["source_path"];split=ROOT/item["path"]
            assert digest(original)==item["source_sha256"] and digest(split)==item["sha256"]
            with original.open(encoding="utf-8",newline="") as f:
                reader=csv.reader(f);header=next(reader);mi=header.index("method")
                expected_rows=[row for row in reader if row[mi]==item["method"]]
            with split.open(encoding="utf-8",newline="") as f:
                reader=csv.reader(f);assert next(reader)==header;actual_rows=list(reader)
            assert actual_rows==expected_rows and len(actual_rows)==1000
            checked+=len(actual_rows)
        assert checked==24000
        result["browser_partition_rows_checked"]=checked
    for row in read(ROOT/"SOURCE_MANIFEST.json")["sources"]:
        if row.get("included"):
            assert digest(ROOT/row["published_path"])==row["published_sha256"],row["published_path"]
    for run in read(ROOT/"baseline_records/INDEX.json")["runs"]:
        for row in run["files"]:
            p=ROOT/row["path"]
            assert digest(p)==row["compressed_sha256"] and digest(p,True)==row["uncompressed_sha256"]
            result["baseline_gzip_files_checked"]+=1
    periods={};sample_links={}
    for group,cfg in config.items():
        with (ROOT/"tables"/(group+"_periods.csv")).open(encoding="utf-8",newline="") as f:
            rows=list(csv.DictReader(f))
        lookup={(int(r["block"]),r["method"]):r for r in rows}
        assert len(rows)==len(lookup)==8000
        for method in {r["method"] for r in rows}:
            assert {int(r["block"]) for r in rows if r["method"]==method}==set(range(1000))
        nlinks=0;found=set()
        with gzip.open(ROOT/"tables"/(group+"_links.csv.gz"),"rt",encoding="utf-8",newline="") as f:
            reader=csv.DictReader(f)
            for key,iterator in itertools.groupby(reader,lambda r:(int(r["block"]),r["method"])):
                links=list(iterator);r=lookup[key];assert key not in found;found.add(key)
                assert yes(r["success"]);assert len(links)==int(r["links"])
                assert [int(x["link_index"]) for x in links]==list(range(len(links)))
                assert sorted([int(x[y]) for x in links for y in ("tx","rx")])==list(range(2*len(links)))
                p=np.array([float(x["power_electrical_input_w"]) for x in links])
                pc=np.array([cfg["physical"][x["mode"].lower()]["circuit_power_w"] for x in links])
                actual=np.array([float(x["actual_mean_rate_bps"]) for x in links]);mean=np.array([float(x["predicted_mean_rate_bps"]) for x in links])
                low=np.array([float(x["predicted_p05_raw_bps"]) for x in links])
                calibrated=np.maximum(mean,1e-9)*(np.maximum(low,1e-9)/np.maximum(mean,1e-9))**cfg["qos_chance_inflation"]
                ee=actual/(p+pc);pe=mean/(p+pc)
                close(ee,[x["actual_link_ee_bit_per_j"] for x in links],"link EE")
                close(pe,[x["predicted_link_ee_bit_per_j"] for x in links],"prediction EE")
                close(calibrated,[x["predicted_p05_calibrated_bps"] for x in links],"calibration")
                close(ee.mean(),r["actual_mean_link_ee_bit_per_j"],"period mean")
                close(ee.min(),r["actual_min_link_ee_bit_per_j"],"period minimum")
                close(pe.mean(),r["predicted_mean_link_ee_bit_per_j"],"predicted mean")
                close(pe.min(),r["predicted_min_link_ee_bit_per_j"],"predicted minimum")
                close(p.sum(),r["total_electrical_input_power_w"],"input power")
                close((p+pc).sum(),r["total_input_plus_circuit_power_w"],"branch power")
                assert bool(np.all(actual>=cfg["q_eval_bps"]))==yes(r["all_links_actual_service_pass"])
                assert int(np.count_nonzero(actual<cfg["q_eval_bps"]))==int(r["actual_violating_links"])
                assert bool(np.all(calibrated>=cfg["q_min_bps"]-1e-7))==yes(r["final_quantile_pass"])
                if r["actual_reference_mean_link_ee_bit_per_j"]:
                    loss=1-ee.mean()/float(r["actual_reference_mean_link_ee_bit_per_j"])
                    close(loss,r["actual_mean_ee_loss_vs_reference"],"actual loss")
                    assert bool(loss>cfg["delta"])==yes(r["actual_loss_over_delta"])
                else:assert not r["actual_mean_ee_loss_vs_reference"] and not r["actual_loss_over_delta"]
                if r["predicted_reference_mean_link_ee_bit_per_j"]:
                    assert bool(pe.mean()>=(1-cfg["delta"])*float(r["predicted_reference_mean_link_ee_bit_per_j"])-1e-7)==yes(r["predicted_mean_budget_check_pass"])
                else:assert not r["predicted_mean_budget_check_pass"]
                for x in links:
                    phy=cfg["physical"][x["mode"].lower()];power=float(x["power_electrical_input_w"])
                    assert phy["power_min_w"]-1e-9<=power<=phy["power_max_w"]+1e-9
                    close(power*phy["pa_efficiency" if x["mode"]=="RF" else "eo_efficiency"],x["emitted_rf_or_optical_power_w"],"power conversion")
                if key[0] in range(100,1000,100):sample_links[(group,)+key]=links
                nlinks+=len(links)
        assert nlinks==expected[group]["link_rows"]
        for key,r in lookup.items():
            assert yes(r["success"])==(key in found)
            if not yes(r["success"]):
                assert r["actual_mean_link_ee_bit_per_j"]==r["actual_min_link_ee_bit_per_j"]==""
                assert not yes(r["all_links_actual_service_pass"])
        for method,exp in expected[group]["methods"].items():
            subset=[r for r in rows if r["method"]==method]; good=[r for r in subset if yes(r["success"])]
            assert len(good)==exp["success"]
            assert sum(yes(r["all_links_actual_service_pass"]) for r in subset)==exp["service_pass_count"]
            close(np.mean([float(r["actual_mean_link_ee_bit_per_j"]) for r in good]),exp["mean_link_ee_bit_per_j"],"aggregate mean")
            close(np.mean([float(r["actual_min_link_ee_bit_per_j"]) for r in good]),exp["per_period_min_link_ee_bit_per_j"],"aggregate minimum")
            report_labels={"reference":"reference","main_old":"old","main_new":"new","zappone_mmee_fp_fixed":"zappone"}
            if method in report_labels:
                prior=report["groups"][group]["methods"][report_labels[method]]
            elif method in ("same_topo_old","same_topo_new"):
                prior=same_report["groups"][group]["methods"]["old_same_topo" if method=="same_topo_old" else "new_same_topo"]
            else:prior=None
            if prior is not None:
                close(exp["mean_link_ee_bit_per_j"],prior["mean_link_EE_bit_per_J"]["mean"],"published report mean")
                close(exp["per_period_min_link_ee_bit_per_j"],prior["per_period_min_link_EE_bit_per_J"]["mean"],"published report minimum")
                assert sum(yes(r["final_quantile_pass"])==False for r in good)==prior["final_quantile_fail_count"]
                assert sum(yes(r["predicted_mean_budget_check_pass"])==False for r in good)==prior["predicted_budget_below_threshold_count"]
        result["groups"][group]=dict(period_rows=len(rows),link_rows=nlinks,methods=len(expected[group]["methods"]))
        periods[group]=lookup
    samples=read(ROOT/"samples/INDEX.json")
    assert samples["selection"]==list(range(100,1000,100)) and len(samples["items"])==54
    assert samples["raw_npz_bytes"]<10_000_000
    for item in samples["items"]:
        group,k,family=item["group"],item["block"],item["family"];cfg=config[group]
        folder=(ROOT/item["path"]).parent;commit=read(folder/"commit.json");rec=read(folder/"record.json")
        for filename,h in commit.items():assert digest(folder/filename)==h
        with np.load(folder/"arrays.npz",allow_pickle=False) as a:
            for prefix,method in (("reference","reference"),("old",family+"_old"),("new",family+"_new")):
                links=sample_links[group,k,method]
                p=np.array([float(x["power_electrical_input_w"]) for x in links]);modes=[x["mode"] for x in links]
                pc=np.array([cfg["physical"][m.lower()]["circuit_power_w"] for m in modes])
                bw=np.array([cfg["physical"][m.lower()]["bandwidth_hz"] for m in modes])
                gap=np.array([10**(cfg["physical"][m.lower()]["gap_db"]/10) for m in modes])
                sinr=a[prefix+"_signal"]/(a[prefix+"_interference"]+a[prefix+"_noise"])
                close(sinr,a[prefix+"_sinr"],"raw SINR")
                rate=bw*np.log1p(sinr/gap)/np.log(2.)
                close(rate,a[prefix+"_rate_bps"],"raw rate formula")
                close(rate.mean(0),[x["actual_mean_rate_bps"] for x in links],"sample versus full table")
                smp=a[prefix+"_samples_bps"];assert smp.shape==(cfg["draws"],len(links))
                close(smp.mean(0),[x["predicted_mean_rate_bps"] for x in links],"sample prediction mean")
                close(np.quantile(smp,.05,axis=0),[x["predicted_p05_raw_bps"] for x in links],"sample p05")
                result["sample_actions_checked"]+=1
    result["limitations"]="Verifies saved selected-action arithmetic, consistency, completeness and provenance; does not prove global optimality, generalization, or arbitrary-action forecast reconstruction."
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
