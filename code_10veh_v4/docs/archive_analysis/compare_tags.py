"""比对两个 Tag 的同一组 run：验证同种子同物理是否复现，并列出对比算法。

用法：python -I compare_tags.py 20261031 20261031b
只读，不改任何 run 目录。
"""
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code" / "runs"

FIELDS = [
    ("mean_pair_ee_including_failed_as_zero", 1e6),
    ("mean_pair_ee_ratio_vs_reference", 1.0),
    ("min_pair_ee_including_failed_as_zero", 1e6),
    ("mean_total_data_power_w", 1.0),
    ("mean_total_electrical_power_w", 1.0),
    ("mean_vlc_link_fraction", 100.0),
]


def load(cls, tag):
    d = json.load(io.open(ROOT / f"{cls}_{tag}" / "summary.json", encoding="utf-8"))
    return d


def val(d, method, key, scale):
    s = d["methods"][method]
    if key == "mean_vlc_link_fraction":
        return s[key]["mean"] * scale
    if key in ("mean_pair_ee_including_failed_as_zero",
               "min_pair_ee_including_failed_as_zero"):
        return s[key]["mean"] / scale
    if key == "mean_pair_ee_ratio_vs_reference":
        return s[key]["mean"]
    return s[key]["mean"]


def main():
    a_tag, b_tag = sys.argv[1], sys.argv[2]
    a_dir = {p.name[:-len("_" + a_tag)] for p in ROOT.iterdir()
             if p.is_dir() and p.name.endswith("_" + a_tag)}
    b_dir = {p.name[:-len("_" + b_tag)] for p in ROOT.iterdir()
             if p.is_dir() and p.name.endswith("_" + b_tag)}
    common = sorted(a_dir & b_dir)
    print(f"tagA={a_tag} tagB={b_tag} 共同组 {len(common)}："
          f"仅A {sorted(a_dir - b_dir)} 仅B {sorted(b_dir - a_dir)}\n")

    worst = defaultdict(float)
    print(f"{'group':26s} {'method':10s} {'field':38s} {'A':>10s} {'B':>10s} {'Δ%':>8s}")
    for cls in common:
        try:
            A, B = load(cls, a_tag), load(cls, b_tag)
        except FileNotFoundError:
            print(f"{cls}: 缺 summary.json，跳过")
            continue
        for method in ("reference", "maxmin", "random"):
            if method not in A["methods"] or method not in B["methods"]:
                continue
            for key, scale in FIELDS:
                va, vb = val(A, method, key, scale), val(B, method, key, scale)
                if va == 0:
                    rel = 0.0 if vb == 0 else float("inf")
                else:
                    rel = 100.0 * (vb - va) / abs(va)
                if abs(rel) > abs(worst[(cls, method)]):
                    worst[(cls, method)] = rel
                flag = "  <-- 不一致" if abs(rel) > 1e-6 else ""
                print(f"{cls:26s} {method:10s} {key:38s} {va:10.4f} {vb:10.4f} {rel:8.4f}{flag}")
        extra = []
        for method in ("reference", "maxmin", "random"):
            if method in B["methods"]:
                fa = A["methods"].get(method, {}).get("fallback_rate", float("nan"))
                fb = B["methods"][method].get("fallback_rate", float("nan"))
                extra.append(f"{method} fallback {fa:.3f}->{fb:.3f}")
        print("  " + " | ".join(extra))
    print("\n=== 每组最大相对差 ===")
    for (cls, method), r in sorted(worst.items()):
        print(f"{cls:26s} {method:10s} max|Δ| = {r:.6f}%")

    # 对比算法组（只在 B 里）
    only_b = sorted(b_dir - a_dir)
    if only_b:
        print("\n=== 仅 B 有的组（对比算法） ===")
        for cls in only_b:
            try:
                d = load(cls, b_tag)
            except FileNotFoundError:
                print(f"\n{cls}: 还没有 summary.json（可能正在跑），跳过")
                continue
            print(f"\n{cls}: methods={list(d['methods'])}")
            for method, s in d["methods"].items():
                print(f"  {method:10s} fallback={s.get('fallback_rate', float('nan')):.3f} "
                      f"meanEE={s['mean_pair_ee_including_failed_as_zero']['mean']/1e6:9.1f} "
                      f"ratio={s['mean_pair_ee_ratio_vs_reference']['mean']:.4f} "
                      f"minEE={s['min_pair_ee_including_failed_as_zero']['mean']/1e6:8.1f} "
                      f"Pelec={s['mean_total_electrical_power_w']['mean']:6.2f} "
                      f"vlc%={100*s['mean_vlc_link_fraction']['mean']:5.1f} "
                      f"qos_ok={s.get('all_pairs_qos_met_rate', float('nan')):.2f} "
                      f"budget_viol={s.get('real_budget_violation_rate', float('nan')):.2f}")


if __name__ == "__main__":
    main()
