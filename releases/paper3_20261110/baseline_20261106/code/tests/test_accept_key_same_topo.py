"""G_c accept_key ties and same_topo_maxmin topology freeze."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from p3base.config import FoundationConfig
from p3learn.allocation import (
    Action, _accepts_better, budget_margin_gc, same_topo_maxmin_allocate,
    pure_ee_reference, protect_budget,
)
from run_allocation import apply_scale_defaults, parser


def test_budget_margin_gc_matches_protect_formula():
    assert abs(budget_margin_gc(90.0, 100.0, 0.1) - 0.0) < 1e-12
    assert budget_margin_gc(95.0, 100.0, 0.1) > 0
    assert budget_margin_gc(80.0, 100.0, 0.1) < 0


def test_accept_key_min_ee_ignores_margin_tiebreak():
    assert _accepts_better(candidate_min=2.0, best_min=1.0,
                           candidate_gc=-10.0, best_gc=10.0, accept_key="min_ee")
    assert not _accepts_better(candidate_min=1.0, best_min=1.0,
                               candidate_gc=100.0, best_gc=0.0, accept_key="min_ee")


def test_accept_key_margin_breaks_near_ties():
    assert _accepts_better(candidate_min=1.0, best_min=1.0,
                           candidate_gc=5.0, best_gc=1.0,
                           accept_key="min_ee_then_margin")
    assert not _accepts_better(candidate_min=1.0, best_min=1.0,
                               candidate_gc=1.0, best_gc=5.0,
                               accept_key="min_ee_then_margin")
    # Clear primary win still wins even with worse margin.
    assert _accepts_better(candidate_min=2.0, best_min=1.0,
                           candidate_gc=-1.0, best_gc=100.0,
                           accept_key="min_ee_then_margin")


def domain(n=4):
    business = ~np.eye(n, dtype=bool)
    candidates = {mode: business.copy() for mode in ("RF", "VLC")}
    ports = {name: np.zeros((n, n), dtype=int) for name in ("tx", "rx")}
    return FoundationConfig(), candidates, business, ports


def prediction(mean):
    mean = np.asarray(mean, float)
    samples = np.stack([mean, mean])
    return dict(mean_rate_bps=mean, block_rate_samples_bps=samples,
                low_rate_bps=np.quantile(samples, .05, axis=0),
                high_rate_bps=np.quantile(samples, .95, axis=0))


def test_same_topo_does_not_change_matching():
    cfg, candidates, business, ports = domain(4)
    gains = {"RF": np.full((4, 4), 1e-8), "VLC": np.full((4, 4), 1e-6)}

    def evaluator(action):
        # Higher EE at moderate power; keeps QoS trivial with large rates.
        p = np.asarray(action.powers, float)
        rates = 5e6 * np.ones(len(p))
        return prediction(rates)

    reference = pure_ee_reference(
        4, cfg, candidates, business, ports, evaluator,
        belief_gains=gains, power_solver="ascent", search="candidates",
        q_min_bps=0.)
    assert reference.success
    result = same_topo_maxmin_allocate(
        4, cfg, candidates, business, ports, evaluator, delta=0.1,
        reference=reference, qos=None, power_refine=True, refine_maxiter=20,
        refine_restarts=0, belief_gains=gains, accept_key="min_ee")
    assert result.success
    assert list(result.action.links) == list(reference.action.links)


def test_n50_scale_defaults_fill_when_flags_absent():
    args = parser().parse_args([
        "--n", "50", "--delta", "0.1", "--out", "unused_out",
        "--methods", "reference",
    ])
    apply_scale_defaults(args, argv=["--n", "50", "--delta", "0.1",
                                     "--out", "unused_out", "--methods", "reference"])
    assert args.power_solver == "continuous"
    assert args.q_min_bps == 1e6
    assert args.joint_rounds == 4
    assert args.joint_max_neighbours == 64
    assert args.draws == 16
    assert args.maxmin_refine_maxiter == 120
    assert args.maxmin_refine_restarts == 1
    assert args.accept_key == "min_ee_then_margin"
    assert args.reference_search == "joint"
    assert args.vlc_sinr == "square_law"


def test_n50_explicit_flags_override_scale_defaults():
    args = parser().parse_args([
        "--n", "50", "--delta", "0.1", "--out", "unused_out",
        "--methods", "reference",
        "--joint-rounds", "8", "--joint-max-neighbours", "0",
        "--draws", "32", "--accept-key", "min_ee",
        "--maxmin-refine-maxiter", "240", "--maxmin-refine-restarts", "2",
    ])
    apply_scale_defaults(args, argv=[
        "--n", "50", "--delta", "0.1", "--out", "unused_out", "--methods", "reference",
        "--joint-rounds", "8", "--joint-max-neighbours", "0", "--draws", "32",
        "--accept-key", "min_ee", "--maxmin-refine-maxiter", "240",
        "--maxmin-refine-restarts", "2",
    ])
    assert args.joint_rounds == 8
    assert args.joint_max_neighbours == 0
    assert args.draws == 32
    assert args.accept_key == "min_ee"
    assert args.maxmin_refine_maxiter == 240
    assert args.maxmin_refine_restarts == 2


def test_cli_accepts_same_topo_method_choice():
    args = parser().parse_args([
        "--n", "10", "--delta", "0.1", "--out", "x",
        "--methods", "reference", "same_topo_maxmin", "maxmin",
    ])
    assert args.methods == ["reference", "same_topo_maxmin", "maxmin"]
