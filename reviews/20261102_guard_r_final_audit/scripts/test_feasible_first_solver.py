from __future__ import annotations

import unittest
import numpy as np

from feasible_first_solver import SolverConfig, solve_feasible_first


class FeasibleFirstSolverTests(unittest.TestCase):
    def setUp(self):
        self.lo = np.array([0.02, 0.02], dtype=float)
        self.hi = np.array([0.8, 0.8], dtype=float)

    @staticmethod
    def evaluator_factory(q_min):
        def evaluate(p):
            p = np.asarray(p, dtype=float)
            rates = np.array([
                2.0e6 * p[0] / (0.20 + p[0]) / (1.0 + 0.12 * p[1]),
                1.8e6 * p[1] / (0.18 + p[1]) / (1.0 + 0.10 * p[0]),
            ])
            ee = rates / (0.12 + p)
            return {"ee": ee, "qos_rates": rates}
        return evaluate

    def test_finds_qos_and_budget_feasible_point_then_maxmin(self):
        cfg = SolverConfig(
            delta=0.10, q_min_bps=0.70e6, qos_metric="chance",
            qos_inflation=1.35, max_evals=600, max_starts=4,
            maxiter_qos=60, maxiter_mean=80, maxiter_maxmin=100,
        )
        result = solve_feasible_first(
            self.evaluator_factory(cfg.q_min_bps),
            self.lo, self.hi,
            starts=[
                np.array([0.05, 0.05]),
                np.array([0.20, 0.20]),
                np.sqrt(self.lo * self.hi),
                self.hi,
            ],
            eta_ref=2.2e6,
            config=cfg,
        )
        self.assertTrue(result["qos_feasible_found"], result)
        self.assertTrue(result["budget_feasible_found"], result)
        self.assertIsNotNone(result["selected"], result)
        self.assertTrue(result["selected"]["qos_ok"], result)
        self.assertTrue(result["selected"]["budget_ok"], result)
        p = np.asarray(result["selected"]["powers"])
        self.assertTrue(np.all(p >= self.lo - 1e-12))
        self.assertTrue(np.all(p <= self.hi + 1e-12))
        self.assertLessEqual(result["evaluation_calls"], cfg.max_evals)

    def test_budget_not_found_is_not_labeled_infeasible(self):
        cfg = SolverConfig(
            delta=0.10, q_min_bps=0.50e6, max_evals=400, max_starts=3,
            maxiter_qos=40, maxiter_mean=50, maxiter_maxmin=60,
        )
        result = solve_feasible_first(
            self.evaluator_factory(cfg.q_min_bps),
            self.lo, self.hi,
            starts=[np.array([0.2, 0.2]), np.sqrt(self.lo * self.hi)],
            eta_ref=1.0e10,
            config=cfg,
        )
        self.assertTrue(result["qos_feasible_found"], result)
        self.assertFalse(result["budget_feasible_found"], result)
        self.assertEqual(result["status"], "qos_feasible_budget_point_not_found")
        self.assertIn("not an infeasibility proof", result["solver_claim"])

    def test_qos_point_not_found_is_reported_separately(self):
        cfg = SolverConfig(
            delta=0.10, q_min_bps=1.0e12, max_evals=300, max_starts=3,
            maxiter_qos=35, maxiter_mean=40, maxiter_maxmin=40,
        )
        result = solve_feasible_first(
            self.evaluator_factory(cfg.q_min_bps),
            self.lo, self.hi,
            starts=[np.array([0.2, 0.2]), np.sqrt(self.lo * self.hi)],
            eta_ref=2.2e6,
            config=cfg,
        )
        self.assertFalse(result["qos_feasible_found"], result)
        self.assertFalse(result["budget_feasible_found"], result)
        self.assertEqual(result["status"], "qos_feasible_point_not_found")

    def test_invalid_power_bounds_rejected(self):
        with self.assertRaises(ValueError):
            solve_feasible_first(
                self.evaluator_factory(0.0),
                np.array([0.0, 0.1]), self.hi,
                starts=[np.array([0.1, 0.1])], eta_ref=1.0,
                config=SolverConfig(q_min_bps=0.0),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
