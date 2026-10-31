"""对比 ideal CSI 与估计 CSI 的 reference：逐块差异定位。

只读 results.csv / summary.json。
"""
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code" / "runs"
RUNS = {
    "est": "e5_D_chance_d010_20261031b",
    "ideal": "e7_ideal_ref_20261031b",
    "random": "e7_random_chance_d010_20261031b",
}
FLOOR = {"RF": 0.01, "VLC": 0.1}
STATIC = {"RF": 0.1159, "VLC": 1.745}


def rows(name, method="reference"):
    out = []
    with io.open(ROOT / name / "results.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["row_type"] == "link" and r["method"] == method:
                out.append(r)
    return out


def per_block(link_rows):
    blocks = defaultdict(list)
    for r in link_rows:
        blocks[int(r["block"])].append(r)
    stats = {}
    for b, ls in blocks.items():
        ee = [float(r["pair_ee_bit_per_j"]) for r in ls]
        p = [float(r["power_w"]) for r in ls]
        stats[b] = dict(
            n=len(ls),
            mean_ee=sum(ee) / len(ee),
            min_ee=min(ee),
            pelec=sum(STATIC[r["mode"]] + float(r["power_w"]) for r in ls),
            pdata=sum(p),
            vlc=sum(r["mode"] == "VLC" for r in ls),
            at_floor=sum(abs(float(r["power_w"]) - FLOOR[r["mode"]]) <= 1e-9 * 1 for r in ls),
            qos=sum(r["pair_qos_met"] == "True" for r in ls),
        )
    return stats


def main():
    data = {}
    for k, name in RUNS.items():
        method = "random" if k == "random" else "reference"
        data[k] = per_block(rows(name, method))

    # 种子与配对一致性
    raw = rows(RUNS["est"])
    raw_i = rows(RUNS["ideal"])
    print("channel_seed est/ideal:", raw[0]["channel_seed"], raw_i[0]["channel_seed"])
    print("csi est/ideal:", raw[0]["csi"], raw_i[0]["csi"])
    pair_e = {(r["block"], r["tx"], r["rx"]): r["mode"] for r in raw}
    pair_i = {(r["block"], r["tx"], r["rx"]): r["mode"] for r in raw_i}
    same_pair = sum(pair_e.get(k) == pair_i.get(k) for k in set(pair_e) | set(pair_i))
    print(f"配对/模式相同 {same_pair}/{len(set(pair_e) | set(pair_i))} 个链路块")

    print(f"\n{'blk':>4s} {'est_mean':>9s} {'ideal_mean':>10s} {'Δ%':>7s} "
          f"{'est_min':>9s} {'ideal_min':>9s} {'est_P':>6s} {'ideal_P':>7s} "
          f"{'estfloor':>8s} {'idfloor':>7s} {'estqos':>6s} {'idqos':>5s}")
    dmean = []
    for b in sorted(data["est"]):
        e, i = data["est"][b], data["ideal"][b]
        rel = 100 * (i["mean_ee"] - e["mean_ee"]) / e["mean_ee"]
        dmean.append(rel)
        print(f"{b:4d} {e['mean_ee']/1e6:9.1f} {i['mean_ee']/1e6:10.1f} {rel:7.1f} "
              f"{e['min_ee']/1e6:9.1f} {i['min_ee']/1e6:9.1f} {e['pelec']:6.2f} {i['pelec']:7.2f} "
              f"{e['at_floor']:8d} {i['at_floor']:7d} {e['qos']:6d} {i['qos']:5d}")
    better = sum(1 for r in dmean if r < 0)
    print(f"\nideal 更差的块 {better}/{len(dmean)}；平均差 {sum(dmean)/len(dmean):+.2f}%")
    for k in ("est", "ideal", "random"):
        if k not in data:
            continue
        bs = data[k]
        m = sum(v["mean_ee"] for v in bs.values()) / len(bs)
        mn = sum(v["min_ee"] for v in bs.values()) / len(bs)
        p = sum(v["pelec"] for v in bs.values()) / len(bs)
        fl = sum(v["at_floor"] for v in bs.values()) / sum(v["n"] for v in bs.values())
        q = sum(v["qos"] for v in bs.values()) / sum(v["n"] for v in bs.values())
        print(f"{k:7s} 块均 meanEE={m/1e6:8.1f} minEE={mn/1e6:8.1f} "
              f"Pelec={p:6.2f} 贴下界={100*fl:5.1f}% QoS满足={100*q:5.1f}%")


if __name__ == "__main__":
    main()
