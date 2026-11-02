"""Merge one design-seed and one independent validation-seed Guard-R audit."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def arm_rows(payload, arm):
    return [
        r[arm] for r in payload.get("records", [])
        if r.get("reconstruction_mismatch") is None and not r.get("error") and r.get(arm)
    ]


def summarize_pair(design, validation):
    seeds = (int(design["meta"]["channel_seed"]), int(validation["meta"]["channel_seed"]))
    if seeds[0] == seeds[1]:
        raise ValueError("design and validation channel seeds must be different")
    if design["meta"].get("role") != "design":
        raise ValueError("first input must have role=design")
    if validation["meta"].get("role") != "validation":
        raise ValueError("second input must have role=validation")
    if design["meta"].get("source_tree_sha256") != validation["meta"].get("source_tree_sha256"):
        raise ValueError("source-tree hashes differ; do not pool runs from different source snapshots")
    if design["meta"].get("delta") != validation["meta"].get("delta"):
        raise ValueError("delta differs between design and validation")
    if design["meta"].get("q_min_bps") != validation["meta"].get("q_min_bps"):
        raise ValueError("QoS threshold differs between design and validation")

    arms = (
        "legacy_guard_r_candidate",
        "legacy_same_topology_control",
        "strict_feasible_first_candidate",
        "strict_same_topology_control",
    )
    out = {"channel_seeds": {"design": seeds[0], "validation": seeds[1]}, "arms": {}}
    for arm in arms:
        rows = arm_rows(design, arm) + arm_rows(validation, arm)
        tr = [r["truth"] for r in rows if r.get("truth")]
        gains = [float(x["min_ee_gain_pct_vs_original"]) for x in tr]
        out["arms"][arm] = {
            "n": len(rows),
            "truth_all_gates_pass": sum(bool(x["all_true_gates_ok"]) for x in tr),
            "legitimate_improvement_count": sum(bool(x["legitimate_improvement"]) for x in tr),
            "truth_min_gain_positive_count": sum(g > 0 for g in gains),
            "truth_min_gain_gt_1pct_count": sum(g > 1 for g in gains),
            "truth_min_gain_gt_10pct_count": sum(g > 10 for g in gains),
            "truth_min_gain_pct_median": float(np.median(gains)) if gains else None,
            "truth_min_gain_pct_p10": float(np.percentile(gains, 10)) if gains else None,
            "truth_min_gain_pct_p90": float(np.percentile(gains, 90)) if gains else None,
        }

    pairs = []
    for payload in (design, validation):
        for r in payload.get("records", []):
            if r.get("reconstruction_mismatch") is not None or r.get("error"):
                continue
            c = r.get("strict_feasible_first_candidate")
            s = r.get("strict_same_topology_control")
            if not c or not s or not c.get("truth") or not s.get("truth"):
                continue
            ct, st = c["truth"], s["truth"]
            pairs.append({
                "role": payload["meta"]["role"], "seed": payload["meta"]["channel_seed"],
                "block": r["block"], "stage": r["stage"],
                "candidate_minus_control_gain_pct": (
                    float(ct["min_ee_gain_pct_vs_original"])
                    - float(st["min_ee_gain_pct_vs_original"])
                ),
                "candidate_all_true_gates_ok": bool(ct["all_true_gates_ok"]),
                "control_all_true_gates_ok": bool(st["all_true_gates_ok"]),
                "candidate_legitimate_improvement": bool(ct["legitimate_improvement"]),
            })
    diffs = [x["candidate_minus_control_gain_pct"] for x in pairs]
    out["strict_feasible_first_vs_equal_budget_control"] = {
        "n_pairs": len(pairs),
        "candidate_better_count": sum(x > 0 for x in diffs),
        "candidate_not_worse_count": sum(x >= 0 for x in diffs),
        "median_gain_difference_pct": float(np.median(diffs)) if diffs else None,
        "both_true_feasible_pairs": sum(
            x["candidate_all_true_gates_ok"] and x["control_all_true_gates_ok"] for x in pairs
        ),
        "candidate_legitimate_improvements": sum(x["candidate_legitimate_improvement"] for x in pairs),
        "details": pairs,
    }
    out["design_summary"] = design.get("summary", {})
    out["validation_summary"] = validation.get("summary", {})
    out["claim_limit"] = (
        "Two-seed replication is evidence about these sampled seeds, not a global guarantee. "
        "A local multistart failure is not an infeasibility proof."
    )
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    d, v = load(args.design), load(args.validation)
    merged = summarize_pair(d, v)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "combined_summary.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (out / "paired_candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        rows = merged["strict_feasible_first_vs_equal_budget_control"]["details"]
        cols = list(rows[0].keys()) if rows else [
            "role", "seed", "block", "stage", "candidate_minus_control_gain_pct",
            "candidate_all_true_gates_ok", "control_all_true_gates_ok",
            "candidate_legitimate_improvement",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({
        "seeds": merged["channel_seeds"],
        "arms": merged["arms"],
        "strict_pair_result": {
            k: v for k, v in merged["strict_feasible_first_vs_equal_budget_control"].items()
            if k != "details"
        },
        "out": str(out.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
