import csv,sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1])
name="e4_B_mean_maxmin_d010_20261027"
per=defaultdict(dict)
with (root/name/"results.csv").open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        if r["row_type"]!="link" or r["selected"]!="True" or r["block"]!="60": continue
        per[r["method"]][r["link_index"]]=r
for li in sorted(per["reference"],key=int):
    a,b=per["reference"][li],per["maxmin"][li]
    print(f"L{li} {a['mode']:3s} p={float(a['power_w']):.3f}->{float(b['power_w']):.3f} "
          f"pred={float(a['predicted_rate_bps'])/1e6:6.2f}->{float(b['predicted_rate_bps'])/1e6:6.2f}M "
          f"p05={float(a['predicted_p05_bps'])/1e6:6.2f}->{float(b['predicted_p05_bps'])/1e6:6.2f}M "
          f"act={float(a['actual_rate_bps'])/1e6:6.2f}->{float(b['actual_rate_bps'])/1e6:6.2f}M "
          f"EE={float(a['pair_ee_bit_per_j'])/1e6:6.2f}->{float(b['pair_ee_bit_per_j'])/1e6:6.2f}M")
print("reasons", per["maxmin"][li]["decision_reason"], "loss", per["maxmin"][li]["actual_mean_pair_loss_fraction_vs_actual_reference"])
