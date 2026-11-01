"""逐条校验 README.md 的结论是否真的由 data/ 里的数据支持。

不依赖 v5 源码树、不重跑仿真；只读本目录下的 JSON/CSV。

用法::

    D:\\anaconda\\envs\\pymarl\\python.exe -B -I verify_claims.py

输出每条 claim 的实测值、期望值、PASS/FAIL。任何 FAIL 都说明 README 与数据不符。
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

RESULTS = []


def check(claim, expected, actual, ok):
    RESULTS.append((claim, expected, actual, ok))
    flag = "PASS" if ok else "**FAIL**"
    print(f"[{flag}] {claim:10s} expected={expected}  actual={actual}")


def close(claim, expected, actual, tol, fmt="{:.4f}"):
    ok = actual is not None and abs(actual - expected) <= tol
    check(claim, fmt.format(expected), "None" if actual is None else fmt.format(actual), ok)


def load(name):
    with open(DATA / name, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


calib100 = load("calib_100s_999blocks.json")
calib10 = load("calib_gate_weak_10s_v5C.json")
routeb = load("routeB_summary.json")

with open(DATA / "routeB_blocks.csv", "r", encoding="utf-8-sig", newline="") as handle:
    blocks = list(csv.DictReader(handle))

summary = routeb["summary"]
records = [r for r in routeb["blocks"] if r.get("mismatch") is None and "error" not in r]

# ---------------------------------------------------------------- 主线一
print("\n=== 主线一：分位数校准 ===")
overall = calib100["parts"]["calib"]["overall"]
close("C1 跌破率", 0.1027, overall["violation_rate"], 5e-4)
check("C1 n", "9990", str(overall["n"]), overall["n"] == 9990)

alpha100 = calib100["parts"]["alpha"]
close("C2 α*(100s)", 1.3003, alpha100.get("alpha_star"), 5e-3, "{:.4f}")
trace135 = [t for t in alpha100["trace"] if abs(t["alpha"] - 1.35) < 1e-9]
close("C2 α=1.35 跌破率", 0.0458, trace135[0]["violation_rate"] if trace135 else None, 5e-4)

alpha10 = calib10["parts"]["alpha"]
close("C2b α*(10s)", 1.2314, alpha10.get("alpha_star"), 5e-3, "{:.4f}")
check("C3 α*∈[1.23,1.30]",
      "True",
      str(1.23 <= alpha10["alpha_star"] and alpha100["alpha_star"] <= 1.301),
      1.23 <= alpha10["alpha_star"] and alpha100["alpha_star"] <= 1.301)

by_age = calib100["parts"]["calib"]["by_csi_age"]
by_mode = calib100["parts"]["calib"]["by_mode"]
check("C4 年龄分层反直觉",
      "age<5ms > 5-50ms",
      f"{by_age['age<5ms']['violation_rate']:.4f} vs {by_age['5-50ms']['violation_rate']:.4f}",
      by_age["age<5ms"]["violation_rate"] > by_age["5-50ms"]["violation_rate"])
close("C4 RF 违约率", 0.1073, by_mode["RF"]["violation_rate"], 5e-4)
close("C4 VLC 违约率", 0.0900, by_mode["VLC"]["violation_rate"], 5e-4)

# ---------------------------------------------------------------- 主线二
print("\n=== 主线二：峰值钉住 / 反事实 ===")
holds = [r["premise"]["premise_holds"] for r in records]
n_holds = sum(1 for h in holds if h)
check("C5 premise_holds==0", "0", str(n_holds), n_holds == 0)

over = sum(1 for r in records if r["premise"]["over_powered"])
check("C5 p_cur>p_svc==95", "95", str(over), over == 95)

ratio_cur_ee = [r["premise"]["p_cur_w"] / max(r["premise"]["p_ee_w"], 1e-12) for r in records]
ratio_cur_svc = [r["premise"]["p_cur_w"] / max(r["premise"]["p_svc_w"], 1e-12) for r in records]
ratio_ee_svc = [r["premise"]["p_ee_w"] / max(r["premise"]["p_svc_w"], 1e-12) for r in records]
close("C6 p_cur/p_ee 中位", 0.98, st.median(ratio_cur_ee), 0.02, "{:.4f}")
close("C6 p_cur/p_svc 中位", 8.26, st.median(ratio_cur_svc), 0.05, "{:.3f}")
close("C6 p_ee/p_svc 中位", 10.96, st.median(ratio_ee_svc), 0.05, "{:.3f}")

by_stage = summary["by_stage"]
for key in ("A", "B", "C", "S2"):
    if key not in by_stage:
        check(f"C7-C9 族 {key}", "present", "MISSING", False)

a_dmin = by_stage["A"]["d_min_median"]
check("C7 A Δmin≈0", "|x|<=0.005", f"{a_dmin:.6f}", abs(a_dmin) <= 5e-3)
a_identical = sum(1 for r in records for c in r["candidates"]
                  if c["stage"].startswith("A_") and abs(c["d_min_vs_exec_pct"]) < 1e-6)
check("C7 A 逐位相同区块", "15", str(a_identical), a_identical == 15)
close("C8 B Δmin", -97.71, by_stage["B"]["d_min_median"], 0.02, "{:.2f}")
check("C8 B n/better", "102/6", f"{by_stage['B']['n']}/{by_stage['B']['better']}",
      by_stage["B"]["n"] == 102 and by_stage["B"]["better"] == 6)
close("C9 C Δmin", -27.62, by_stage["C"]["d_min_median"], 0.02, "{:.2f}")
close("C9 S2 Δmin", -45.34, by_stage["S2"]["d_min_median"], 0.02, "{:.2f}")

# ---------------------------------------------------------------- 主线三
print("\n=== 主线三：排序有效性 / 守卫价值 ===")
close("C10 τ 均值", 0.3576, summary.get("kendall_tau_mean"), 5e-4)
taus = [r["kendall_tau"] for r in records if r["kendall_tau"] is not None]
check("C10 τ n", "66", str(len(taus)), len(taus) == 66)
pos = sum(1 for t in taus if t > 0)
check("C10 τ 正相关块", "45/66", f"{pos}/{len(taus)}", pos == 45)

legit_gain = []
for r in records:
    base = r["executed"]["min_ee"]
    better = [c for c in r["candidates"]
              if c["truth_min_ee"] > base + 1e-9
              and c.get("truth_qos_ok")
              and c.get("truth_budget_ok_true_ref")]
    if better:
        best = max(better, key=lambda c: c["truth_min_ee"])
        legit_gain.append(100.0 * (best["truth_min_ee"] - base) / base)
check("C11 合法改进块数", "29", str(len(legit_gain)), len(legit_gain) == 29)
if legit_gain:
    close("C11 中位增益", 0.01, st.median(legit_gain), 0.01, "{:.3f}")
    check("C11 >10% 块", "1", str(sum(1 for x in legit_gain if x > 10)),
          sum(1 for x in legit_gain if x > 10) == 1)
    check("C11 >50% 块", "1", str(sum(1 for x in legit_gain if x > 50)),
          sum(1 for x in legit_gain if x > 50) == 1)

# C16：真值 QoS 从来不是约束
tri = summary.get("truth_three_conditions") or {}
check("C16 真值更优候选总数", "78", str(tri.get("total")), tri.get("total") == 78)
check("C16 真值QoS失败数", "0", str(tri.get("better_qos_fail")),
      tri.get("better_qos_fail") == 0)
check("C16 三条件全过(候选)", "30", str(tri.get("better_qos_ok_budget_true")),
      tri.get("better_qos_ok_budget_true") == 30)

# C15：Block 23 的唯一显著漏拒
blk23 = [r for r in records if r["block"] == 23]
if blk23:
    r23 = blk23[0]
    cand = [c for c in r23["candidates"] if c["stage"].startswith("B_")]
    if cand:
        c = cand[0]
        close("C15 B23 增益", 334.30, c["d_min_vs_exec_pct"], 0.05, "{:.2f}")
        close("C15 B23 Δmean", -7.86, c["d_mean_vs_exec_pct"], 0.05, "{:.2f}")
        check("C15 B23 真值QoS", "True", str(c.get("truth_qos_ok")), c.get("truth_qos_ok") is True)
        check("C15 B23 真值参考预算", "True", str(c.get("truth_budget_ok_true_ref")),
              c.get("truth_budget_ok_true_ref") is True)
        close("C15 B23 预测违约占 η_ref", 0.0033,
              c["pred_budget_violation"] / max(r23["eta_ref_pred"], 1e-12), 5e-4, "{:.4f}")
    else:
        check("C15 B23 候选存在", "B_swap_rx_1", "MISSING", False)
else:
    check("C15 Block 23 存在", "present", "MISSING", False)

force = summary["force_accept"]
check("C12 真值更优块", "35", str(force["wins_on_truth_min"]), force["wins_on_truth_min"] == 35)
close("C12 Δmean 中位", -1.96, force["d_mean_median_pct"], 0.02, "{:.2f}")
close("C12 真值预算内比例", 0.47, force["truth_budget_ok_frac"], 0.005, "{:.3f}")

# ---------------------------------------------------------------- C14 bootstrap
print("\n=== C14 block bootstrap ===")
boot = load("bootstrap_100s.json")
b100 = boot["alpha"]["1"]["block_bootstrap"]
close("C14 未校准点估计", 0.1027, b100["point"], 5e-4)
close("C14 CI 下界", 0.0949, b100["ci_low"], 5e-4)
close("C14 CI 上界", 0.1111, b100["ci_high"], 5e-4)
check("C14 P(rate>5%)", "1.000", f"{b100['probability_above_nominal']:.3f}",
      b100["probability_above_nominal"] > 0.99)
b135 = boot["alpha"]["1.35"]["block_bootstrap"]
close("C14 α=1.35 点估计", 0.0458, b135["point"], 5e-4)

# ---------------------------------------------------------------- 自检
print("\n=== 自检 S1/S2 ===")
check("S1 复现通过块", "100", str(summary["blocks_verified"]), summary["blocks_verified"] == 100)
check("S1 排除块", "0", str(summary["blocks_excluded"]), summary["blocks_excluded"] == 0)
dev = summary.get("truth_ee_max_rel_dev")
check("S2 真值口径偏差", "0.0", str(dev), dev is not None and dev < 1e-9)

# blocks.csv 与 summary.json 必须一致（防止两个文件来自不同次运行）
csv_premise = sum(1 for row in blocks if row.get("premise_holds") == "True")
check("CSV/summary 前提一致", "0 vs 0", f"{csv_premise} vs {n_holds}", csv_premise == n_holds)

# ---------------------------------------------------------------- 汇总
failed = [r for r in RESULTS if not r[3]]
print("\n" + "=" * 68)
print(f"共 {len(RESULTS)} 条检查，PASS {len(RESULTS) - len(failed)}，FAIL {len(failed)}")
for claim, expected, actual, _ok in failed:
    print(f"  FAIL {claim}: expected {expected}, actual {actual}")
sys.exit(1 if failed else 0)
