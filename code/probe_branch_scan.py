"""Run the three learning branches over several training seeds and aggregate.

Branches (identical block, BC stage, replay, batch, round budget and seed):

* ``original``              -- unchanged objective.
* ``no_continuous_entropy`` -- ``alpha_c = 0``; power is still sampled.
* ``fixed_power``           -- ``alpha_c = 0`` and the actor's power channel is
  pinned to the 1 mW floor.  Diagnostic only: it removes power drift from the
  comparison, it does **not** make discrete exploration easier.

Each (branch, seed) pair writes its own fresh directory under ``--out`` and is
executed as a subprocess, so the runs cannot share optimiser or RNG state.

The aggregate answers one question per branch: does the *mean per-link EE* of
the policy's own proposal rise above the cloned policy, and is that rise robust
across seeds?  One frozen block can still only show whether the mechanism is
alive; it is not a cross-state or convergence result.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROBE = HERE / "probe_bc_feedback.py"
BRANCHES = ("original", "no_continuous_entropy", "fixed_power")


def run_one(branch, seed, args):
    out = args.out / f"{branch}_seed{seed}"
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"run directory must be fresh: {out}")
    cmd = [sys.executable, "-B", "-I", str(PROBE),
           "--out", str(out), "--branch", branch,
           "--learning-seed", str(seed),
           "--n", str(args.n), "--block", str(args.block), "--draws", str(args.draws),
           "--delta", str(args.delta), "--rounds", str(args.rounds),
           "--proposals-per-round", str(args.proposals_per_round),
           "--bc-steps", str(args.bc_steps), "--warmup-max", str(args.warmup_max),
           "--coverage", str(args.coverage), "--coverage-clone", str(args.coverage_clone),
           "--selection-k", str(args.selection_k),
           "--selection-repeats", str(args.selection_repeats),
           "--channel-seed", str(args.channel_seed),
           "--forecast-seed", str(args.forecast_seed),
           "--encoder", args.encoder, "--threads", str(args.threads)]
    if args.truth:
        cmd.append("--truth")
    if args.save_checkpoints:
        cmd.append("--save-checkpoints")
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    seconds = time.perf_counter() - started
    if proc.returncode != 0:
        return dict(branch=branch, seed=seed, ok=False, seconds=seconds,
                    returncode=proc.returncode, stderr=proc.stderr[-4000:],
                    stdout_tail=proc.stdout[-2000:])
    probe = json.loads((out / "probe.json").read_text(encoding="utf-8"))
    return dict(branch=branch, seed=seed, ok=True, seconds=seconds,
                path=str(out / "probe.json"), probe=probe)


def flatten(run):
    """Scalar summary of one probe.json; every number is traceable to it."""
    probe = run["probe"]
    ref = probe["reference"]["metrics"]
    bc = probe["arms"]["bc_only"]["evaluation"]
    masac = probe["arms"]["bc_masac"]["evaluation"]
    sel = probe["selection"]["bc_masac"]
    by = sel["by_selector"]
    row = dict(
        branch=run["branch"], seed=run["seed"], ok=True, seconds=run["seconds"],
        trusted=probe["critic_warmup"]["trusted"],
        warmup_steps=probe["critic_warmup"]["steps"],
        reference_mean_ee=ref["mean_ee"],
        reference_min_pair_ee=ref["min_ee"],
        bc_det_mean_ee=bc["deterministic"]["metrics"]["mean_ee"],
        bc_det_cost=bc["deterministic"]["metrics"]["cost"],
        bc_det_structural=bc["deterministic"]["structural_match_with_reference"],
        bc_stoch_mean=bc["stochastic"]["mean_ee"],
        bc_stoch_cost=bc["stochastic"]["mean_budget_cost"],
        masac_det_mean_ee=masac["deterministic"]["metrics"]["mean_ee"],
        masac_det_cost=masac["deterministic"]["metrics"]["cost"],
        masac_det_structural=masac["deterministic"]["structural_match_with_reference"],
        masac_det_floor_control=masac["deterministic"]["floor_power_control"]["mean_ee"],
        masac_stoch_mean=masac["stochastic"]["mean_ee"],
        masac_stoch_median=masac["stochastic"]["median_ee"],
        masac_stoch_cost=masac["stochastic"]["mean_budget_cost"],
        masac_stoch_acceptance=masac["stochastic"]["acceptance_rate"],
        masac_power_mean=masac["stochastic"]["power"]["mean"],
        masac_power_floor_fraction=masac["stochastic"]["power"]["near_floor_fraction"],
        proposals_best_mean_ee=probe["exploration"]["proposals"]["best_mean_ee"],
        neighbours_improving=probe["neighbours"]["improved"],
        neighbours_best_ratio=probe["neighbours"]["best_improvement_ratio"],
        critic_mae=probe["critic_warmup"]["history"][-1]["mae"] if
        probe["critic_warmup"]["history"] else None,
        selection_critic_mean_ee=by["critic"]["predicted_mean_ee"].get("mean"),
        selection_training_label_mean_ee=by["training_label"]["predicted_mean_ee"].get("mean"),
        selection_feasible_mean_ee=by["feasible_forecast"]["predicted_mean_ee"].get("mean"),
        selection_critic_regret=by["critic"]["regret_predicted"].get("mean"),
        selection_critic_fallback=by["critic"]["fallback_rate"],
        selection_critic_equals_feasible=sel["agreements"][
            "critic_equals_feasible_forecast"])
    if "truth" in masac["deterministic"]:
        row["masac_det_truth_mean_ee"] = masac["deterministic"]["truth"]["mean_ee"]
        row["bc_det_truth_mean_ee"] = bc["deterministic"]["truth"]["mean_ee"]
        row["reference_truth_mean_ee"] = probe["reference"]["truth"]["mean_ee"]
    return row


def aggregate(rows):
    """Per-branch mean/min/max across seeds; no pooling across branches."""
    out = {}
    for branch in BRANCHES:
        items = [r for r in rows if r.get("branch") == branch and r.get("ok")]
        if not items:
            out[branch] = dict(runs=0)
            continue
        metrics = {}
        for key in items[0]:
            if key in ("branch", "ok", "seconds"):
                continue
            values = [r.get(key) for r in items]
            numbers = [float(v) for v in values if isinstance(v, (int, float))]
            if not numbers:
                continue
            metrics[key] = dict(mean=float(sum(numbers) / len(numbers)),
                                min=float(min(numbers)), max=float(max(numbers)),
                                values=numbers)
        out[branch] = dict(runs=len(items), metrics=metrics)
    return out


def main(args):
    jobs = [(branch, seed) for branch in args.branches for seed in args.seeds]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        runs = list(pool.map(lambda job: run_one(job[0], job[1], args), jobs))
    rows = [flatten(r) if r["ok"] else dict(branch=r["branch"], seed=r["seed"],
                                            ok=False, returncode=r["returncode"],
                                            stderr=r["stderr"],
                                            stdout_tail=r["stdout_tail"])
            for r in runs]
    result = dict(
        scope="three-branch single-state seed scan; not convergence, not a "
              "cross-state result",
        n=args.n, block=args.block, delta=args.delta, draws=args.draws,
        rounds=args.rounds, proposals_per_round=args.proposals_per_round,
        bc_steps=args.bc_steps, warmup_max=args.warmup_max,
        seeds=list(args.seeds), branches=list(args.branches),
        seeds_channel=args.channel_seed, seeds_forecast=args.forecast_seed,
        total_seconds=time.perf_counter() - started,
        runs=rows, aggregate=aggregate(rows),
        notes=[
            "each (branch, seed) writes its own fresh directory; no state is shared",
            "fixed_power is a diagnostic branch: it removes power drift only and "
            "leaves the discrete sampling probabilities unchanged",
            "a truth column appears only when --truth is passed and is validation "
            "only",
        ])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scan.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(50, 100), default=50)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--proposals-per-round", type=int, default=8)
    parser.add_argument("--bc-steps", type=int, default=600)
    parser.add_argument("--warmup-max", type=int, default=1600)
    parser.add_argument("--coverage", type=int, default=200)
    parser.add_argument("--coverage-clone", type=int, default=60)
    parser.add_argument("--selection-k", type=int, default=16)
    parser.add_argument("--selection-repeats", type=int, default=4)
    parser.add_argument("--encoder", choices=("mlp", "hetero_graph",
                                              "hetero_graph_v2"), default="mlp")
    parser.add_argument("--channel-seed", type=int, default=2026102401)
    parser.add_argument("--forecast-seed", type=int, default=2026102500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument("--branches", nargs="+", choices=BRANCHES,
                        default=list(BRANCHES))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--truth", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
