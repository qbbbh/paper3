"""逐块对比两个方法的动作：链路、模式、功率、端口、速率。

用法：python diff_block.py <run> <block> <methodA> <methodB>
"""
import csv
import io
import sys
from pathlib import Path

run, block, ma = sys.argv[1], int(sys.argv[2]), sys.argv[3]
mb = sys.argv[4] if len(sys.argv) > 4 else None
rows = [r for r in csv.DictReader(io.open(Path(run) / "results.csv", encoding="utf-8-sig"))
        if int(r["block"]) == block and r["row_type"] == "link"]
for method in [m for m in (ma, mb) if m]:
    mine = sorted([r for r in rows if r["method"] == method], key=lambda x: int(x["link_index"]))
    print(f"== {method} ({len(mine)} links)")
    for r in mine:
        print(f"  L{r['link_index']} {r['tx']}->{r['rx']} {r['mode']:3s} "
              f"p={float(r['power_w']):.5f} port={r['tx_port']}/{r['rx_port']} "
              f"pred={float(r['predicted_rate_bps'])/1e6:.1f}M "
              f"act={float(r['actual_rate_bps'])/1e6:.1f}M "
              f"ee={float(r['pair_ee_bit_per_j'])/1e6:.1f}M")
