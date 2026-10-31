"""复算两件事，不依赖 summary 里的现成标量：

1. 含固定电路功耗 P_c 的网络 EE：ΣR / Σ(mode 的 P_c + p)，逐块求和后再汇总。
   同时给出只含可调功率的分母（summary 的 total_data_energy_j 口径）作为对照，
   两者不能用同一个符号讲。
2. 每块 reference 与 maxmin 的选中链路集合是否相同（tx, rx, mode），
   以及 VLC 使用率、动作是否逐块一致。

用法:
    python -B -I verify_network_ee_and_matching_20261028.py <runs_root> <tag> [...]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

PC = {"RF": 0.1159, "VLC": 1.745}


def load(path):
    """按 (block, method) 收集选中链路的 (tx, rx, mode, power, rate, duration)。"""
    per = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["row_type"] != "link" or r["selected"] != "True":
                continue
            per[(int(r["block"]), r["method"])].append(dict(
                tx=r["tx_id"], rx=r["rx_id"], mode=r["mode"],
                p=float(r["power_w"]), rate=float(r["actual_rate_bps"]),
                T=float(r["duration_s"]), port=(r["tx_port"], r["rx_port"])))
    return per


def network_ee(per, method):
    bits = 0.0
    e_bill = 0.0
    e_data = 0.0
    for (b, m), links in per.items():
        if m != method:
            continue
        for x in links:
            bits += x["rate"] * x["T"]
            e_data += x["p"] * x["T"]
            e_bill += (PC[x["mode"]] + x["p"]) * x["T"]
    return bits, e_bill, e_data


def main():
    root = Path(sys.argv[1]).resolve()
    for tag in sys.argv[2:]:
        d = root / tag
        per = load(d / "results.csv")
        print(f"===== {tag}")
        for method in ("reference", "maxmin"):
            if (2, method) not in per:
                continue
            bits, e_bill, e_data = network_ee(per, method)
            print(f"  {method:9s} bits={bits/1e9:7.3f} Gbit  "
                  f"E含Pc={e_bill:7.3f} J  网络EE(含Pc)={bits/e_bill/1e6:7.2f} Mbit/J  "
                  f"E仅数据={e_data:6.3f} J  网络EE(仅数据)={bits/e_data/1e6:8.2f} Mbit/J")

        if (2, "maxmin") not in per:
            continue
        # 逐块匹配对比
        changed = []
        vlc_d = []
        for b in sorted({k[0] for k in per}):
            ref = per.get((b, "reference"))
            mm = per.get((b, "maxmin"))
            if not ref or not mm:
                continue
            sr = {(x["tx"], x["rx"], x["mode"]) for x in ref}
            sm = {(x["tx"], x["rx"], x["mode"]) for x in mm}
            if sr != sm:
                changed.append(dict(block=b, only_ref=len(sr - sm), only_mm=len(sm - sr),
                                    n_ref=len(sr), n_mm=len(sm)))
            vr = sum(1 for x in ref if x["mode"] == "VLC") / len(ref)
            vm = sum(1 for x in mm if x["mode"] == "VLC") / len(mm)
            vlc_d.append((b, vr, vm))
        tot_r = sum(len(per[(b, "reference")]) for b in sorted({k[0] for k in per})
                    if (b, "reference") in per)
        tot_m = sum(len(per[(b, "maxmin")]) for b in sorted({k[0] for k in per})
                    if (b, "maxmin") in per)
        vr_all = sum(1 for k, v in per.items() if k[1] == "reference"
                     for x in v if x["mode"] == "VLC") / tot_r
        vm_all = sum(1 for k, v in per.items() if k[1] == "maxmin"
                     for x in v if x["mode"] == "VLC") / tot_m
        print(f"  匹配逐块一致: {len(vlc_d) - len(changed)}/{len(vlc_d)} 块；"
              f"改动的块: {[c['block'] for c in changed]}")
        for c in changed[:12]:
            print(f"    block {c['block']}: ref {c['n_ref']} 条 / mm {c['n_mm']} 条，"
                  f"仅 ref {c['only_ref']}、仅 mm {c['only_mm']}")
        # 逐块 VLC 使用率的差
        diffs = [(b, vr, vm) for b, vr, vm in vlc_d if abs(vr - vm) > 1e-12]
        print(f"  VLC 链路份额：reference 汇总 {vr_all*100:.2f}%（{tot_r} 条），"
              f"maxmin 汇总 {vm_all*100:.2f}%（{tot_m} 条）；"
              f"逐块份额不同的块数 {len(diffs)}")
        for b, vr, vm in diffs[:8]:
            print(f"    block {b}: ref {vr*100:5.1f}% -> mm {vm*100:5.1f}%")
        print()


if __name__ == "__main__":
    main()
