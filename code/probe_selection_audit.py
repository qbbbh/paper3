"""Audit the k-sample selection rule with already-trained checkpoints.

The previous ``selection`` block compared the critic's pick with
``argmax(predicted minimum per-link EE)`` over **all** k candidates, infeasible
ones included.  The critic is fitted on ``mean_ee/reward_scale - cost``, so that
comparison used two different objectives and a mismatch said nothing about the
critic's ranking.  The objective here is the mean of each link's own rate over
its own power; minimum per-link EE is a diagnostic only.  This script re-runs
the comparison with consistent references on the *same* candidate sets:

1. ``critic``          -- argmax of the ensemble Q.
2. ``training_label``  -- argmax of the critic's own fitted target.
3. ``feasible_forecast`` -- argmax predicted mean per-link EE among
   budget-feasible candidates only.  This is the rule that matches the research
   question.
4. ``forecast_ignoring_budget`` -- the old column, kept as an explicit
   contrast.

Every rule sees the same k candidates, the same frozen forecast bank and the
same budget shield, so the only difference is the ranking.  The script also
reports, per arm: pairwise concordance between Q and its training target, the
regret of each pick against the feasible predicted best, the truth of the
executed action, and the evaluation/time cost of each rule.

Truth is read only after the candidate set and every pick are fixed; it never
drives a decision.  No training happens here.  One frozen block cannot rank
algorithms and cannot establish cross-state generalisation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from p3learn.masac import HybridMASAC
from probe_bc_feedback import selection_policy
from probe_common import (action_metrics, build_frozen_block, jsonable,
                          power_summary, truth_metrics, truth_window)


def deterministic_row(learner, frozen, delta, window):
    """Deterministic policy action: forecast and executed truth, side by side."""
    action = learner.select_action(frozen.features, deterministic=True).action
    eta = frozen.reference.eta_ref
    forecast = action_metrics(frozen.bank, action, eta, delta)
    row = dict(
        forecast={k: v for k, v in forecast.items() if k != "powers"},
        power=power_summary(forecast["powers"]),
        links=[(int(tx), int(rx), mode) for tx, rx, mode in action.links])
    if window is not None:
        truth = truth_metrics(frozen, window, action, eta, delta)
        row["truth"] = {k: v for k, v in truth.items() if k != "powers"}
    return row


def audit_arm(name, learner, frozen, delta, k, repeats, window):
    started = time.perf_counter()
    row = dict(
        name=name,
        updates=int(learner.updates),
        alpha_continuous=float(learner.config.alpha_continuous),
        deterministic=deterministic_row(learner, frozen, delta, window),
        selection=selection_policy(learner, frozen, delta, k, repeats,
                                   window=window))
    row["seconds"] = time.perf_counter() - started
    return row


def main(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("audit output must be fresh")
    torch.set_num_threads(args.threads)
    frozen = build_frozen_block(args.n, args.block, args.draws,
                                args.channel_seed, args.forecast_seed)
    eta = frozen.reference.eta_ref
    reference_forecast = action_metrics(frozen.bank, frozen.reference.action,
                                        eta, args.delta)
    window = truth_window(frozen)
    reference_truth = truth_metrics(frozen, window, frozen.reference.action,
                                    eta, args.delta)

    arms = {}
    for name, path in args.checkpoint:
        learner = HybridMASAC.load(path)
        arms[name] = audit_arm(name, learner, frozen, args.delta,
                               args.k, args.repeats, window)

    result = dict(
        scope="selection-rule audit on one frozen block with existing checkpoints; "
              "no training, no algorithm ranking, no cross-state claim",
        n=args.n, block=args.block, delta=args.delta, draws=args.draws,
        seeds=dict(channel=args.channel_seed, forecast=args.forecast_seed),
        k=args.k, repeats=args.repeats,
        reference=dict(
            forecast={k_: v for k_, v in reference_forecast.items() if k_ != "powers"},
            truth={k_: v for k_, v in reference_truth.items() if k_ != "powers"},
            eta_ref=eta),
        arms=arms,
        notes=[
            "'training_label' is the exact quantity the critic was fitted on; "
            "'feasible_forecast' is the study's actual target.",
            "'forecast_ignoring_budget' reproduces the old oracle column and is not "
            "a like-for-like reference for the critic.",
            "regret = best feasible predicted mean per-link EE - that of the pick; "
            "negative values are possible for the old rule because it may pick an "
            "infeasible candidate with a higher raw EE.",
            "truth columns are validation only and never enter a decision.",
            "a plain local search with the same evaluation budget is NOT included "
            "yet; until it is, 'learning vs search' cannot be attributed.",
        ])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(50, 100), default=50)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--channel-seed", type=int, default=2026102401)
    parser.add_argument("--forecast-seed", type=int, default=2026102500)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint", action="append", nargs=2,
                        metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
