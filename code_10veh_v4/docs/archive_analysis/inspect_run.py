"""打印一个 run 的指标（CSV 只有 link 行，块级量在每行上重复；失败/预算状态见 summary.json）。

用法：python inspect_run.py <run 目录> [link|block]
"""
import csv
import io
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "block"
rows = list(csv.DictReader(io.open(root / "results.csv", encoding="utf-8-sig")))
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
seen = {}
for r in rows:
    seen.setdefault((int(r["block"]), r["method"]), []).append(r)


def f(r, key, digits=4):
    try:
        return f"{float(r[key]):.{digits}g}"
    except (KeyError, TypeError, ValueError):
        return "-"


print(f"blocks={summary.get('config',{}).get('blocks')} "
      f"draws={summary['config']['draws']} methods={list(summary['methods'])}")
for method, m in summary["methods"].items():
    print(f"  {method:11s} fail={m['failure_rate']:.2f} fb={m['fallback_rate']:.2f} "
          f"meanEE={m['mean_pair_ee_including_failed_as_zero']['mean']/1e6:.1f}M "
          f"minEE={m['min_pair_ee_including_failed_as_zero']['mean']/1e6:.1f}M "
          f"P_el={m['mean_total_electrical_power_w']['mean']:.3f}W "
          f"vlcL={m['mean_vlc_link_fraction']['mean']:.3f} "
          f"vlcP={m['mean_vlc_power_fraction']['mean']:.3f} "
          f"solve={m['solve_time_s']['mean']:.2f}s "
          f"budget_viol={m['real_budget_violation_rate']:.2f}")
for (block, method), links in sorted(seen.items()):
    r = links[0]
    notes = [r["decision_reason"]]
    for key in ("failure_reason", "fallback_reason", "chance_fallback"):
        if r.get(key):
            notes.append(f"{key}={r[key]}")
    print(f"{method:11s} b{block:<3d} predEE={f(r,'predicted_mean_pair_ee')} "
          f"actEE={f(r,'mean_pair_ee_bit_per_j')} actMin={f(r,'min_pair_ee_bit_per_j')} "
          f"netEE={f(r,'network_ee_bit_per_j')} P_el={f(r,'predicted_total_electrical_power_w')} "
          f"vlcL={f(r,'vlc_link_fraction')} vlcP={f(r,'vlc_power_fraction')} "
          f"qos_ok={r['all_pairs_qos_met']} qv={f(r,'qos_violation_links','g')} "
          f"floor={sum(float(x['power_w']) <= {'RF': 0.01, 'VLC': 0.1}[x['mode']] * (1 + 1e-9) for x in links)}/{len(links)} "
          f"| {' '.join(n for n in notes if n)}")
    if mode == "link":
        for x in sorted(links, key=lambda z: int(z["link_index"])):
            print(f"    L{x['link_index']} {x['tx']}->{x['rx']} {x['mode']:3s} "
                  f"p={f(x,'power_w')} rate={f(x,'actual_rate_bps')} "
                  f"pred={f(x,'predicted_rate_bps')} p05={f(x,'predicted_p05_bps')} "
                  f"qos={x['pair_qos_met']}")
