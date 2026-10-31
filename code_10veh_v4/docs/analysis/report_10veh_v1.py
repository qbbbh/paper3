"""按批次打印 10 车评估的统一统计行。只读 runs，不写任何文件。

用法:
    python -B -I report_10veh_v1.py <runs_root> <tag>:<method> [<tag>:<method> ...]

method 取 reference / maxmin。每行含：均逐链路 EE、块内最低 EE 的 100 块均值、
全网平均电功率、VLC 使用率、1 mW 贴地链路数与其中违速率数、真值违约链路、
回退块、adopt 块、worst20 命中、adopt 块的均 EE 相对同块 reference 的中位/最差损失。
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

Q = 1e6
FLOOR = 1.001e-3


def load(path):
    per = defaultdict(lambda: defaultdict(list))
    reason = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["row_type"] != "link" or r["selected"] != "True":
                continue
            per[int(r["block"])][r["method"]].append(r)
            reason[int(r["block"])] = r["decision_reason"]
    return per, reason


def stats(per, reason, method):
    ks = sorted(per)
    if method not in per[ks[0]]:
        return None
    ee = [sum(float(x["pair_ee_bit_per_j"]) for x in per[k][method]) / len(per[k][method])
          for k in ks]
    mn = [min(float(x["pair_ee_bit_per_j"]) for x in per[k][method]) for k in ks]
    links = [x for k in ks for x in per[k][method]]
    floor = [x for x in links if float(x["power_w"]) <= FLOOR]
    floor_viol = sum(1 for x in floor if float(x["actual_rate_bps"]) < Q)
    viol = sum(1 for x in links if float(x["actual_rate_bps"]) < Q)
    ad = [k for k in ks if reason.get(k) == "max_min_candidate_selected"
          and "reference" in per[k]]
    refmn = {k: min(float(x["pair_ee_bit_per_j"]) for x in per[k]["reference"]) for k in ks}
    worst = set(sorted(ks, key=lambda k: refmn[k])[:20])
    loss = []
    for k in ad:
        ref = per[k]["reference"]
        r = sum(float(x["pair_ee_bit_per_j"]) for x in ref) / len(ref)
        cur = sum(float(x["pair_ee_bit_per_j"]) for x in per[k][method]) / len(per[k][method])
        loss.append(cur / r - 1)
    return dict(blocks=len(ks), ee=sum(ee) / len(ee), mn=sum(mn) / len(mn),
                floor=len(floor), floor_share=len(floor) / len(links), floor_viol=floor_viol,
                viol=viol, adopted=len(ad), worst_hit=sum(1 for k in ad if k in worst),
                loss_med=(sorted(loss)[len(loss) // 2] if loss else 0.),
                loss_min=(min(loss) if loss else 0.))


def main():
    root = Path(sys.argv[1]).resolve()
    for spec in sys.argv[2:]:
        tag, method = spec.rsplit(":", 1)
        d = root / tag
        if not (d / "summary.json").exists():
            print(f"{tag:34s} {method:9s} 未完成")
            continue
        per, reason = load(d / "results.csv")
        s = json.load((d / "summary.json").open(encoding="utf-8"))
        m = s["methods"].get(method)
        if m is None:
            print(f"{tag:34s} {method:9s} summary 无该 method")
            continue
        st = stats(per, reason, method)
        fb = m.get("fallback_rate")
        print(f"{tag:34s} {method:9s} EE={st['ee']/1e6:7.2f} min={st['mn']/1e6:6.2f} "
              f"Pel={m['mean_total_electrical_power_w']['mean']:.3f} "
              f"vlc={m['mean_vlc_link_fraction']['mean']*100:5.1f}% "
              f"floor={st['floor']:3d}({st['floor_share']*100:4.1f}%,{st['floor_viol']}违) "
              f"违约={st['viol']:3d} "
              f"fb={'  n/a' if fb is None else f'{fb:.2f}'} adopt={st['adopted']:3d} "
              f"worst={st['worst_hit']:2d}/20 "
              f"loss中位={st['loss_med']*100:+.2f}% 最差={st['loss_min']*100:+.1f}% "
              f"solve={m['solve_time_s']['mean']:.1f}s")


if __name__ == "__main__":
    main()
