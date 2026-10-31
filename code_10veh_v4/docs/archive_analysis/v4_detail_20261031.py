"""v4 批量的细项：QoS 满足率、贴下界链路、最弱链路功率，以及与 v3 的机制对照。"""
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code" / "runs"
FLOOR = {"RF": 0.01, "VLC": 0.1}


def load(name):
    return list(csv.DictReader(io.open(ROOT / name / "results.csv", encoding="utf-8-sig")))


def stats(name, method):
    rows = [r for r in load(name) if r["method"] == method]
    if not rows:
        return None
    links = [r for r in rows]
    qos = sum(r["pair_qos_met"] == "True" for r in links) / len(links)
    per_mode = defaultdict(lambda: [0, 0])
    floor_qos = defaultdict(lambda: [0, 0])
    for r in links:
        m = r["mode"]
        per_mode[m][0] += 1
        at = abs(float(r["power_w"]) - FLOOR[m]) <= 1e-9
        if at:
            floor_qos[m][0] += 1
            floor_qos[m][1] += r["pair_qos_met"] == "True"
    # 每块的最低 EE 链路的功率与模式
    by_block = defaultdict(list)
    for r in links:
        by_block[r["block"]].append(r)
    weak = []
    for b, rs in by_block.items():
        w = min(rs, key=lambda r: float(r["pair_ee_bit_per_j"]))
        weak.append((w["mode"], float(w["power_w"])))
    return dict(qos=qos, per_mode=dict(per_mode), floor=dict(floor_qos), weak=weak)


def show(name):
    print(f"\n## {name}")
    for method in ("reference", "maxmin"):
        s = stats(name, method)
        if not s:
            continue
        rf_n = s["per_mode"].get("RF", [0, 0])[0]
        vlc_n = s["per_mode"].get("VLC", [0, 0])[0]
        fr = s["floor"].get("RF", [0, 0])
        fv = s["floor"].get("VLC", [0, 0])
        w = s["weak"]
        pw = sorted(p for _m, p in w)
        vlc_at_weak = sum(m == "VLC" for m, _p in w)
        print(f"  {method:9s} qos_ok={s['qos']:.3f} "
              f"RFfloor={fr[0]}/{rf_n}({fr[1]} qos) VLCfloor={fv[0]}/{vlc_n}({fv[1]} qos) "
              f"weak_link_power p50={pw[len(pw)//2]:.3f} p90={pw[int(.9*(len(pw)-1))]:.3f} "
              f"max={pw[-1]:.2f} vlc_weak={vlc_at_weak}/{len(w)}")


if __name__ == "__main__":
    for name in ("e5_B_mean_d010_20261031", "e5_D_chance_d010_20261031",
                 "e6_B_d010_20261031", "e6_A_chance_d100_20261031",
                 "e5_B_mean_d010_20261028", "e5_D_chance_d010_20261028"):
        show(name)
