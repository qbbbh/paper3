"""按块配对比对不同 run / 方法，并给出功率下界占比与这些链路的速率达标情况。

用法：
    python compare_final.py <run1> [<run2> ...]

- 块级指标取 summary.json 的 blocks 记录（含失败块）；CSV 只有 link 行，
  失败块没有 link 行，所以下界占比只能对成功块统计（分母如实写出）。
- 失败块在"含失败"口径下按 0 计（与 summary 的 including_failed_as_zero 一致）。
- 下界口径是 v4：RF 0.01 W、VLC 0.1 W（不是旧的 1 mW）。
- "贴下界且欠速率"= 功率在各自下界上、但实际速率低于 q_eval——贴下界本身不违规。
- 结果同时写到 stdout 与相邻的 .txt（UTF-8），避免控制台编码把中文弄乱。
"""
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

SEED = "2026102401"
FLOOR = {"RF": 0.01, "VLC": 0.1}


def load(run):
    root = Path(run)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    blocks = {}
    for r in summary["blocks"]:
        if str(r["channel_seed"]) != SEED:
            continue
        blocks[(int(r["block"]), r["method"])] = r
    links = defaultdict(list)
    for r in csv.DictReader(io.open(root / "results.csv", encoding="utf-8-sig")):
        if r["channel_seed"] != SEED or r["row_type"] != "link":
            continue
        links[(int(r["block"]), r["method"])].append(r)
    return root.name, blocks, links, summary


def num(r, key, default=None):
    try:
        v = r.get(key)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def floor_stats(rows, q_eval):
    at_floor, at_floor_short, zero = 0, 0, 0
    for x in rows:
        p = num(x, "power_w")
        rate = num(x, "actual_rate_bps", 0.0)
        if rate < 1e-9:
            zero += 1
        cap = FLOOR.get(x["mode"])
        if p is not None and cap is not None and p <= cap * (1 + 1e-9):
            at_floor += 1
            if q_eval is not None and rate < q_eval * (1 - 1e-9):
                at_floor_short += 1
    return at_floor, at_floor_short, zero


def main():
    runs = [load(p) for p in sys.argv[1:]]
    out = []

    def say(text=""):
        print(text)
        out.append(text)

    common = None
    for _n, blocks, _l, _s in runs:
        own = {b for (b, _m) in blocks}
        common = own if common is None else common & own
    common = sorted(common)
    say(f"配对块数 {len(common)}（{common[0]}..{common[-1]}）")

    table = {}
    for name, blocks, links, summary in runs:
        for method in summary["methods"]:
            recs = [blocks[(b, method)] for b in common if (b, method) in blocks]
            if not recs:
                continue
            ok = [r for r in recs if r.get("success")]
            share_blocks = [r for r in ok if (len(links[(int(r['block']), method)]) > 0)]
            floors = [floor_stats(links[(int(r["block"]), method)], num(r, "q_eval_bps"))
                      for r in share_blocks]
            n_links = sum(len(links[(int(r["block"]), method)]) for r in share_blocks)
            at_floor = sum(f[0] for f in floors)
            short = sum(f[1] for f in floors)
            zero = sum(f[2] for f in floors) / len(share_blocks) if share_blocks else 0.0
            mean = lambda key: sum(num(r, key, 0.0) for r in recs) / len(recs)
            table[f"{name}::{method}"] = dict(
                blocks=recs, links=links, n=len(recs), ok=len(ok),
                mean_ee=mean("mean_pair_ee_bit_per_j") / 1e6,
                min_ee=mean("min_pair_ee_bit_per_j") / 1e6,
                net_ee=mean("network_ee_bit_per_j") / 1e6,
                p_el=mean("predicted_total_electrical_power_w"),
                vlc_link=mean("vlc_link_fraction"),
                vlc_power=mean("vlc_power_fraction"),
                qos_ok=sum(1 for r in ok if r.get("all_pairs_qos_met")) / len(recs),
                floor_share=at_floor / n_links if n_links else 0.0,
                floor_short=short / len(share_blocks) if share_blocks else 0.0,
                zero_rate=zero,
                mean_links=n_links / len(share_blocks) if share_blocks else 0.0,
            )

    say(f"{'run::method':46s} {'均EE(M)':>9s} {'minEE(M)':>9s} {'netEE(M)':>9s} "
        f"{'P_el(W)':>8s} {'vlcL':>6s} {'vlcP':>6s} {'成功%':>6s} {'QoS%':>6s} "
        f"{'下界%':>6s} {'贴界欠速率':>10s} {'零速率':>6s}")
    for tag, d in table.items():
        say(f"{tag:46s} {d['mean_ee']:9.1f} {d['min_ee']:9.1f} {d['net_ee']:9.1f} "
            f"{d['p_el']:8.3f} {d['vlc_link']:6.3f} {d['vlc_power']:6.3f} "
            f"{100*d['ok']/d['n']:6.1f} {100*d['qos_ok']:6.1f} "
            f"{100*d['floor_share']:6.1f} {d['floor_short']:10.2f} {d['zero_rate']:6.2f}")
    say("注：均/min/net EE 与功率是按块平均、失败块按 0 计；下界% 与贴界欠速率的分母只有成功块。")

    say()
    say("逐块配对（相对同 run 的 reference 实际均 EE，单位 %）")
    for name, blocks, links, summary in runs:
        ref = {b: num(blocks[(b, "reference")], "mean_pair_ee_bit_per_j")
               for b in common if (b, "reference") in blocks}
        for method in summary["methods"]:
            if method == "reference":
                continue
            pairs = [(num(blocks[(b, method)], "mean_pair_ee_bit_per_j"), ref[b])
                     for b in common if (b, method) in blocks and ref.get(b)]
            if not pairs:
                continue
            deltas = [(a / r - 1) * 100 for a, r in pairs]
            wins = sum(d > 0 for d in deltas)
            say(f"  {name}::{method:10s} 配对 {len(deltas):3d} 块  "
                f"均值 {sum(deltas)/len(deltas):+6.2f}%  "
                f"中位 {sorted(deltas)[len(deltas)//2]:+6.2f}%  胜 {wins}/{len(deltas)}")

    mains = [t for t in table if t.endswith("::maxmin")]
    lits = [t for t in table if t.endswith("::literature")]
    if mains and lits:
        say()
        say("主算法 vs 文献（逐块配对，同 channel seed）")
        for a in mains:
            for b in lits:
                da, db = table[a], table[b]
                mine = {int(r["block"]): num(r, "mean_pair_ee_bit_per_j")
                        for r in da["blocks"] if r.get("success")}
                lit = {int(r["block"]): num(r, "mean_pair_ee_bit_per_j")
                       for r in db["blocks"] if r.get("success")}
                pairs = [x for x in mine if x in lit and lit[x]]
                ratios = [mine[x] / lit[x] for x in pairs]
                lit_fail = len(db["blocks"]) - len(lit)
                mine_fail = len(da["blocks"]) - len(mine)
                say(f"  {a} / {b}")
                say(f"    两边都成功的配对 {len(pairs)} 块；均 EE 比 均值 "
                    f"{100*(sum(ratios)/len(ratios)-1):+.2f}%，"
                    f"中位 {100*(sorted(ratios)[len(ratios)//2]-1):+.2f}%，"
                    f"胜 {sum(r > 1 for r in ratios)}/{len(ratios)}")
                say(f"    失败块 主算法 {mine_fail} / 文献 {lit_fail}（含失败按 0 计的均 EE："
                    f"主 {da['mean_ee']:.1f} M vs 文献 {db['mean_ee']:.1f} M，"
                    f"{100*(da['mean_ee']/db['mean_ee']-1):+.1f}%）")
                say(f"    min EE（含失败按 0）：主 {da['min_ee']:.1f} M vs 文献 {db['min_ee']:.1f} M，"
                    f"{100*(da['min_ee']/db['min_ee']-1):+.1f}%")
                say(f"    电功率：主 {da['p_el']:.3f} W vs 文献 {db['p_el']:.3f} W；"
                    f"VLC 链路占比 {100*da['vlc_link']:.1f}% vs {100*db['vlc_link']:.1f}%；"
                    f"贴下界 {100*da['floor_share']:.1f}% vs {100*db['floor_share']:.1f}%")

    target = Path(sys.argv[1]).parent / f"compare_{'_'.join(Path(p).name for p in sys.argv[1:])}.txt"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n[written] {target}")


if __name__ == "__main__":
    main()
