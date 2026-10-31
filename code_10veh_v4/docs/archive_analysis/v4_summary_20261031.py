"""汇总 v4 批量（Tag 20261031）的 15 个 run。

口径提醒：
* 均 EE / min EE 都是逐链路 EE（r_k/(P_c+p_k)）的算术平均，单位 bit/J；
* "贴下界"指 RF 贴 0.01 W、VLC 贴 0.1 W（各自模式下界），并单独统计这些链路
  是否满足 1 Mbit/s 的 QoS，因为"贴下界"本身不是成绩，可能在违约；
* 主算法 = maxmin + refine。
"""
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code" / "runs"
TAG = "20261031"
FLOOR_RTOL = 1e-9


def load(name):
    d = json.load(io.open(ROOT / name / "summary.json", encoding="utf-8"))
    rows = list(csv.DictReader(io.open(ROOT / name / "results.csv", encoding="utf-8-sig")))
    return d, rows


def floor_stats(rows):
    """按模式统计贴下界的链路占比，以及这些链路是否满足 QoS。"""
    out = defaultdict(lambda: dict(links=0, floor=0, floor_qos_ok=0, qos_met=0))
    for r in rows:
        mode = r["mode"]
        rec = out[mode]
        rec["links"] += 1
        p = float(r["power_w"])
        floor = 0.01 if mode == "RF" else 0.1
        at = abs(p - floor) <= FLOOR_RTOL * max(1.0, floor)
        ok = r["pair_qos_met"] == "True"
        rec["qos_met"] += ok
        if at:
            rec["floor"] += 1
            rec["floor_qos_ok"] += ok
    return out


def main():
    groups = sorted(p.name for p in ROOT.iterdir()
                    if p.is_dir() and p.name.endswith("_" + TAG) and (p / "summary.json").exists())
    print(f"{'group':34s} {'method':10s} {'fb':>5s} {'meanEE':>9s} {'ratio':>6s} "
          f"{'minEE':>8s} {'Pdata':>6s} {'Pelec':>6s} {'vlc%':>5s} {'rfFloor':>8s} "
          f"{'vlcFloor':>8s} {'floorQoS':>9s}")
    for g in groups:
        d, rows = load(g)
        per = defaultdict(list)
        for r in rows:
            if r["row_type"] != "link":
                continue
            per[r["method"]].append(r)
        for method in ("reference", "random", "maxmin"):
            if method not in d["methods"]:
                continue
            s = d["methods"][method]
            fs = floor_stats(per.get(method, []))
            rf, vlc = fs.get("RF", {}), fs.get("VLC", {})
            floor_qos = (rf.get("floor_qos_ok", 0) + vlc.get("floor_qos_ok", 0),
                         rf.get("floor", 0) + vlc.get("floor", 0))
            print(f"{g:34s} {method:10s} {s.get('fallback_rate', 0):5.2f} "
                  f"{s['mean_pair_ee_including_failed_as_zero']['mean']/1e6:9.1f} "
                  f"{s['mean_pair_ee_ratio_vs_reference']['mean']:6.3f} "
                  f"{s['min_pair_ee_including_failed_as_zero']['mean']/1e6:8.1f} "
                  f"{s['mean_total_data_power_w']['mean']:6.2f} "
                  f"{s['mean_total_electrical_power_w']['mean']:6.2f} "
                  f"{100*s['mean_vlc_link_fraction']['mean']:5.1f} "
                  f"{100*rf.get('floor',0)/max(1,rf.get('links',1)):7.1f}% "
                  f"{100*vlc.get('floor',0)/max(1,vlc.get('links',1)):7.1f}% "
                  f"{floor_qos[0]:4d}/{floor_qos[1]:<4d}")

    print("\n=== 逐 run 细项：QoS 全满足率 / 真实预算违约率 / 求解耗时 ===")
    for g in groups:
        d, _ = load(g)
        for method, s in d["methods"].items():
            print(f"{g:34s} {method:10s} qos_ok={s.get('all_pairs_qos_met_rate', float('nan')):.2f} "
                  f"budget_viol={s.get('real_budget_violation_rate', float('nan')):.2f} "
                  f"solve={s.get('solve_time_s', {}).get('mean', float('nan')):.1f}s")


if __name__ == "__main__":
    main()
