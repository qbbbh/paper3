"""三条主线的一体化诊断：纯 CSV/JSON，不重跑仿真、不重建真值、不改 code/。

为什么单独放一个目录：本脚本不位于任何被审计的 ``code/`` 树内，因此
**不会改变 source_sha256**，已归档的 23 个 run 与 v5 的 run 都不受影响。

用法（在 第3篇-修改版 目录下）::

    D:\\anaconda\\envs\\pymarl\\python.exe -B -I diagnose_three_lines.py ^
        --run "代码\\开发_10车机制定型_v5_20261101\\code\\runs\\v5_C_ref_sca_s_v5a" ^
        --run "代码\\开发_10车机制定型_v4_20261027\\code\\runs\\e5_D_chance_d010_20261031h"

    # 只跑主线一（不需要 weak_link.json）
    ... --part calib

    # 只跑 H1/H2/H3 排查（最紧急的一条）
    ... --part gate

产出：终端报告 + ``out/report_<tag>.json``。

对应文档：论文/Paper3_三条主线的可证命题与实验设计_20261101.md
  --part calib 检验命题 1 与推论 1.1（覆盖率 + α 反解）
  --part gate  排查 §5.2 的 H1/H2/H3
  --part weak  检验命题 2/3/4 的候选侧统计（需要 weak_link.json）
  --part all   全跑（默认）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# 项目用 ``-B -I`` 调用，而 -I 会忽略 PYTHONIOENCODING；不显式改 stdout，
# Windows 控制台的 gbk 编码会在打印中文路径/字段时抛 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def read_link_rows(results_csv):
    """读 results.csv 的 link 行。只保留本脚本用到的列，缺列时跳过。"""
    wanted = (
        "row_type", "method", "block", "mode", "tx_id", "rx_id",
        "power_w", "active",
        "predicted_rate_bps", "predicted_p05_bps", "predicted_p95_bps",
        "actual_rate_bps", "pair_ee_bit_per_j", "pair_qos_met",
        "csi_age_s", "csi_missing", "q_eval_bps",
        "all_pairs_qos_met", "qos_metric", "chance_fallback",
        "fallback", "min_pair_ee_bit_per_j", "mean_pair_ee_bit_per_j",
    )
    rows = []
    # v5 的 results.csv 带 UTF-8 BOM：用 utf-8-sig 读，否则首列名会变成
    # "\ufeffrow_type"，被误判为缺列。
    with open(results_csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in wanted if c not in (reader.fieldnames or [])]
        for raw in reader:
            if (raw.get("row_type") or "") != "link":
                continue
            row = {}
            for key in wanted:
                row[key] = raw.get(key, "")
            rows.append(row)
    return rows, missing


def as_float(value, default=float("nan")):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def as_bool(value):
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def parse_links(value):
    """``qos_violation_links`` 可能是 [] / [0,3] / 字符串。返回列表。"""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text.replace("True", "true").replace("False", "false"))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return [text]


# --------------------------------------------------------------------------- #
# 主线一：分位数校准（命题 1 与推论 1.1）
# --------------------------------------------------------------------------- #
def _inflate(mean, low, alpha):
    """代码 ``inflate_low`` 的等价实现：Mean * (low/mean) ** alpha。"""
    floor = 1e-9
    m = max(mean, floor)
    ratio = max(low, floor) / m
    try:
        return m * ratio ** float(alpha)
    except (OverflowError, ValueError):
        return 0.0


def coverage_table(rows, epsilon=0.05, q_default=1.0e6):
    """按名义 ε 分位统计实际被跌破的频率，并分层。"""
    buckets = defaultdict(list)
    by_mode = defaultdict(list)
    by_age = defaultdict(list)
    n_used = 0
    n_skipped = 0
    for row in rows:
        p05 = as_float(row["predicted_p05_bps"])
        mean = as_float(row["predicted_rate_bps"])
        actual = as_float(row["actual_rate_bps"])
        if not (math.isfinite(p05) and math.isfinite(mean) and math.isfinite(actual)):
            n_skipped += 1
            continue
        if p05 <= 0.0 or mean <= 0.0 or mean <= p05:
            n_skipped += 1
            continue
        hit = 1.0 if actual < p05 else 0.0          # 跌破名义 p05 = 一次"下尾事件"
        n_used += 1
        by_mode[row["mode"]].append(hit)
        age = as_float(row["csi_age_s"])
        if math.isfinite(age):
            if age < 5e-3:
                label = "age<5ms"
            elif age < 5e-2:
                label = "5-50ms"
            elif age < 1e-1:
                label = "50-100ms"
            else:
                label = ">=100ms"
        else:
            label = "age_unknown"
        by_age[label].append(hit)
        buckets["all"].append(hit)

    def summarise(values):
        if not values:
            return None
        return {
            "n": len(values),
            "violation_rate": sum(values) / len(values),
            "nominal": epsilon,
            "over_eps": sum(values) / len(values) - epsilon,
        }

    return {
        "overall": summarise(buckets["all"]),
        "by_mode": {k: summarise(v) for k, v in sorted(by_mode.items())},
        "by_csi_age": {k: summarise(v) for k, v in sorted(by_age.items())},
        "n_used": n_used,
        "n_skipped": n_skipped,
    }


def solve_alpha(rows, epsilon=0.05, lo=1.0, hi=4.0, tol=1e-3):
    """推论 1.1：二分求最小的 α 使下尾事件率 ≤ ε。

    α 增大 ⇒ R~(α) 单调下降 ⇒ 事件率单调下降，故二分合法。
    """
    usable = []
    for row in rows:
        p05 = as_float(row["predicted_p05_bps"])
        mean = as_float(row["predicted_rate_bps"])
        actual = as_float(row["actual_rate_bps"])
        if not (math.isfinite(p05) and math.isfinite(mean) and math.isfinite(actual)):
            continue
        if p05 <= 0.0 or mean <= 0.0 or mean <= p05:
            continue
        usable.append((mean, p05, actual))
    if not usable:
        return {"status": "no_usable_rows"}

    def rate(alpha):
        hits = 0
        for mean, p05, actual in usable:
            if actual < _inflate(mean, p05, alpha):
                hits += 1
        return hits / len(usable)

    trace = []
    for alpha in (lo, 1.35, 1.6, 2.0, 2.5, 3.0, hi):
        trace.append({"alpha": alpha, "violation_rate": rate(alpha)})

    a, b = lo, hi
    if rate(lo) <= epsilon:
        return {"status": "alpha_at_or_below_lo", "alpha_star": lo, "trace": trace}
    if rate(hi) > epsilon:
        return {"status": "alpha_above_hi", "alpha_star": None, "trace": trace}
    while b - a > tol:
        mid = 0.5 * (a + b)
        if rate(mid) <= epsilon:
            b = mid
        else:
            a = mid
    return {"status": "ok", "alpha_star": b, "violation_rate_at_star": rate(b),
            "trace": trace, "n_used": len(usable)}


# --------------------------------------------------------------------------- #
# H1 / H2 / H3：为什么每条候选都报 QoS 违约
# --------------------------------------------------------------------------- #
def gate_diagnosis(rows):
    """按 block 汇总：基准解自己是否已经违约。

    H2（门不可满足）的判据：**基准解本身**在所有块上都违约。
    若如此，任何"继承基准匹配"的候选必然违约，整层修复门按构造即不可满足。
    """
    per_block = {}
    for row in rows:
        method = row["method"]
        block = row["block"]
        key = (method, block)
        entry = per_block.setdefault(key, {
            "method": method, "block": block,
            "links": 0, "qos_fail_links": 0,
            "all_pairs_qos_met": None, "qos_metric": row.get("qos_metric", ""),
            "chance_fallback": as_bool(row.get("chance_fallback")),
            "fallback": as_bool(row.get("fallback")),
        })
        entry["links"] += 1
        if as_bool(row["pair_qos_met"]) is False:
            entry["qos_fail_links"] += 1
        flag = as_bool(row["all_pairs_qos_met"])
        if flag is not None:
            entry["all_pairs_qos_met"] = flag

    out = {}
    for method in sorted({k[0] for k in per_block}):
        blocks = [v for k, v in per_block.items() if k[0] == method]
        n = len(blocks)
        all_met = [b for b in blocks if b["all_pairs_qos_met"] is True]
        all_fail = [b for b in blocks if b["all_pairs_qos_met"] is False]
        links = sum(b["links"] for b in blocks)
        fails = sum(b["qos_fail_links"] for b in blocks)
        out[method] = {
            "blocks": n,
            "blocks_all_pairs_met": len(all_met),
            "blocks_all_pairs_failed": len(all_fail),
            "link_rows": links,
            "link_rows_failing_qos": fails,
            "link_fail_rate": (fails / links) if links else None,
            "chance_fallback_blocks": sum(1 for b in blocks if b["chance_fallback"]),
            "fallback_blocks": sum(1 for b in blocks if b["fallback"]),
        }
    return out


# --------------------------------------------------------------------------- #
# 主线二 / 三：weak_link.json 的候选侧统计
# --------------------------------------------------------------------------- #
def weak_link_stats(path, delta_hint=None):
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    blocks = data.get("blocks") or []
    cand = []
    adopted_any = 0
    blocks_with_submission = 0
    s1 = {"blocks": 0, "ratio_sum": 0.0, "empty_ranking": 0, "ratio": []}
    for entry in blocks:
        repairs = entry.get("repair") or []
        submitted = 0
        for item in repairs:
            stage = item.get("stage", "?")
            if stage.startswith("s1_probe"):
                s1["blocks"] += 1
                used = as_float(item.get("used_w"))
                service = as_float(item.get("service_w"))
                ratio = as_float(item.get("ratio"))
                if math.isfinite(used) and math.isfinite(service):
                    s1["ratio"].append(service / used if used > 0 else float("nan"))
                if math.isfinite(ratio):
                    s1["ratio_sum"] += ratio
                if not (item.get("ranking") or []):
                    s1["empty_ranking"] += 1
                continue
            submitted += 1
            qos_links = parse_links(item.get("qos_violation_links"))
            cand.append({
                "block": entry.get("block"),
                "stage": stage,
                "accepted": bool(item.get("accepted")),
                "min_pair_ee": as_float(item.get("min_pair_ee")),
                "budget_violation": as_float(item.get("budget_violation")),
                "qos_violation_count": len(qos_links),
                "budget_ok": (as_float(item.get("budget_violation"), float("inf")) == 0.0),
                "qos_ok": len(qos_links) == 0,
            })
            if item.get("accepted"):
                adopted_any += 1
        if submitted:
            blocks_with_submission += 1

    # 按 stage 归类（B_swap_rx_* / C_mode_* 合并）
    def family(stage):
        if stage.startswith("A_"):
            return "A_raise_power"
        if stage.startswith("B_"):
            return "B_swap_rx"
        if stage.startswith("C_"):
            return "C_mode"
        return stage

    families = {}
    for item in cand:
        b = family(item["stage"])
        agg = families.setdefault(b, {"submitted": 0, "accepted": 0,
                                      "budget_ok": 0, "qos_ok": 0, "both_ok": 0,
                                      "min_ee": []})
        agg["submitted"] += 1
        agg["accepted"] += int(item["accepted"])
        agg["budget_ok"] += int(item["budget_ok"])
        agg["qos_ok"] += int(item["qos_ok"])
        agg["both_ok"] += int(item["budget_ok"] and item["qos_ok"])
        if math.isfinite(item["min_pair_ee"]):
            agg["min_ee"].append(item["min_pair_ee"])

    for agg in families.values():
        values = sorted(agg.pop("min_ee"))
        agg["min_ee_median"] = statistics.median(values) if values else None
        agg["min_ee_max"] = values[-1] if values else None

    ratios = sorted(r for r in s1["ratio"] if math.isfinite(r))
    s1.pop("ratio")
    s1["ratio_median"] = statistics.median(ratios) if ratios else None
    s1["ratio_min"] = ratios[0] if ratios else None
    s1["ratio_max"] = ratios[-1] if ratios else None

    return {
        "path": path,
        "blocks": len(blocks),
        "blocks_with_submission": blocks_with_submission,
        "candidates_submitted": len(cand),
        "candidates_accepted": adopted_any,
        "by_family": families,
        "s1_probe": s1,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_one(run_dir, parts, epsilon, q_eval):
    results_csv = os.path.join(run_dir, "results.csv")
    weak_json = os.path.join(run_dir, "weak_link.json")
    tag = os.path.basename(os.path.normpath(run_dir))
    report = {"run": run_dir, "tag": tag, "parts": {}}
    if not os.path.isfile(results_csv):
        report["error"] = f"missing results.csv under {run_dir}"
        return report

    rows, missing = read_link_rows(results_csv)
    report["link_rows"] = len(rows)
    report["missing_columns"] = missing

    if parts in ("calib", "all"):
        report["parts"]["calib"] = coverage_table(rows, epsilon=epsilon)
        report["parts"]["alpha"] = solve_alpha(rows, epsilon=epsilon)
    if parts in ("gate", "all"):
        report["parts"]["gate"] = gate_diagnosis(rows)
    if parts in ("weak", "all"):
        if os.path.isfile(weak_json):
            report["parts"]["weak"] = weak_link_stats(weak_json)
        else:
            report["parts"]["weak"] = {"status": "no weak_link.json"}
    return report


def print_report(report):
    print("=" * 78)
    print("run:", report["run"])
    if "error" in report:
        print("  ERROR:", report["error"])
        return
    print("  link rows:", report["link_rows"])
    if report["missing_columns"]:
        print("  缺少列（旧 run 可能没有）:", ", ".join(report["missing_columns"]))

    parts = report["parts"]
    if "calib" in parts:
        print("\n-- 主线一：分位数校准（名义 ε 分位被跌破的频率）--")
        overall = parts["calib"]["overall"]
        if overall:
            print(f"  总体: n={overall['n']}  违约率={overall['violation_rate']:.4f}"
                  f"  名义={overall['nominal']:.3f}  超出={overall['over_eps']:+.4f}")
        for label, stat in (parts["calib"]["by_csi_age"] or {}).items():
            if stat:
                print(f"  CSI 年龄 {label:12s} n={stat['n']:5d} 违约率={stat['violation_rate']:.4f}")
        for label, stat in (parts["calib"]["by_mode"] or {}).items():
            if stat:
                print(f"  模式 {label:4s} n={stat['n']:5d} 违约率={stat['violation_rate']:.4f}")
        if parts["calib"]["n_skipped"]:
            print(f"  （跳过 {parts['calib']['n_skipped']} 行：p05/mean 缺失或 mean<=p05）")

    if "alpha" in parts:
        alpha = parts["alpha"]
        print("\n-- 推论 1.1：α 反解 --")
        print(f"  状态: {alpha.get('status')}   α* = {alpha.get('alpha_star')}")
        for item in alpha.get("trace", []):
            print(f"    α={item['alpha']:.2f} 违约率={item['violation_rate']:.4f}")

    if "gate" in parts:
        print("\n-- H1/H2/H3 排查：基准解自己是否已违约 --")
        for method, stat in parts["gate"].items():
            print(f"  {method:12s} 块={stat['blocks']:4d}"
                  f" 全块达标={stat['blocks_all_pairs_met']:4d}"
                  f" 全块失败={stat['blocks_all_pairs_failed']:4d}"
                  f" 链路违约率={stat['link_fail_rate']}"
                  f" chance_fallback={stat['chance_fallback_blocks']:3d}"
                  f" fallback={stat['fallback_blocks']:3d}")

    weak = parts.get("weak")
    if weak and weak.get("blocks"):
        print("\n-- 主线二/三：候选采纳与门 --")
        print(f"  块={weak['blocks']} 提交候选={weak['candidates_submitted']}"
              f" 采纳={weak['candidates_accepted']}")
        for name, agg in weak["by_family"].items():
            med = agg["min_ee_median"]
            print(f"  {name:14s} 提交={agg['submitted']:4d} 采纳={agg['accepted']:3d}"
                  f" 预算通过={agg['budget_ok']:4d} QoS通过={agg['qos_ok']:4d}"
                  f" 双通过={agg['both_ok']:3d}"
                  f" 预测minEE中位={med if med is None else round(med/1e6, 2)} M")
        probe = weak["s1_probe"]
        print(f"  S1: 块={probe['blocks']} 服务/实际功率比中位={probe['ratio_median']}"
              f" 空排序块={probe['empty_ranking']}")
        if probe["ratio_median"] is not None:
            print(f"      → 可回收功率份额中位 {1-probe['ratio_median']:.1%}")
    elif weak:
        print("\n-- 主线二/三 --", weak.get("status"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[],
                        help="run 目录（可重复给多个）")
    parser.add_argument("--part", choices=("calib", "gate", "weak", "all"), default="all")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="名义下尾概率，默认 0.05")
    parser.add_argument("--q-eval", type=float, default=1.0e6,
                        help="事后 QoS 门限，默认 1e6 bit/s")
    parser.add_argument("--out", default="out",
                        help="JSON 输出目录")
    args = parser.parse_args(argv)

    if not args.run:
        parser.error("至少给一个 --run")

    os.makedirs(args.out, exist_ok=True)
    reports = []
    for run_dir in args.run:
        report = run_one(run_dir, args.part, args.epsilon, args.q_eval)
        print_report(report)
        reports.append(report)
        slug = report["tag"].replace(os.sep, "_")
        with open(os.path.join(args.out, f"report_{slug}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print("\n写出:", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
