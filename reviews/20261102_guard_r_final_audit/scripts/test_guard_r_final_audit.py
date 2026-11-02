from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

import guard_r_final_audit as audit


class Phase0FilterTests(unittest.TestCase):
    def test_prefilter_uses_only_preregistered_b_c_rule(self):
        payload = {
            "blocks": [
                {"block": 2, "exec_pred_min": 100.0, "candidates": [
                    {"stage": "B_switch_rx", "pred_min_own": 112.0,
                     "qos_ok_pred": True, "margin_fixed": -1.0},
                    {"stage": "C_switch_mode", "pred_min_own": 111.0,
                     "qos_ok_pred": True, "margin_fixed": -1.0},
                    {"stage": "A_raise_power", "pred_min_own": 150.0,
                     "qos_ok_pred": True, "margin_fixed": -1.0},
                    {"stage": "B_bad_qos", "pred_min_own": 150.0,
                     "qos_ok_pred": False, "margin_fixed": -1.0},
                    {"stage": "C_no_budget", "pred_min_own": 150.0,
                     "qos_ok_pred": True, "margin_fixed": 0.0},
                ]},
                {"block": 3, "error": "reference failed"},
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "phase0.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            wanted, info = audit._load_phase0(path, 0.10)
        self.assertEqual(wanted, {2: ["B_switch_rx", "C_switch_mode"]})
        self.assertEqual(info["selected_candidates"], 2)
        self.assertEqual(info["bc_candidates"], 4)

    def test_power_mapping_stays_inside_mode_bounds(self):
        ref = SimpleNamespace(links=[(0, 1, "RF")], powers=np.array([1.5]))
        cfg = SimpleNamespace(
            rf=SimpleNamespace(power_min_w=.01, power_max_w=2.0),
            vlc=SimpleNamespace(power_min_w=.1, power_max_w=3.0),
        )
        got = audit._map_reference_powers(ref, [(0, 2, "VLC")], cfg)
        self.assertEqual(got.shape, (1,))
        self.assertGreaterEqual(got[0], .1)
        self.assertLessEqual(got[0], 3.0)

    def test_action_change_detects_topology_and_continuous_power(self):
        a = SimpleNamespace(links=[(0, 1, "RF")], powers=np.array([.5]))
        same = SimpleNamespace(links=[(0, 1, "RF")], powers=np.array([.5 + 1e-10]))
        changed_power = SimpleNamespace(links=[(0, 1, "RF")], powers=np.array([.6]))
        changed_link = SimpleNamespace(links=[(0, 2, "RF")], powers=np.array([.5]))
        self.assertFalse(audit._action_changed(a, same))
        self.assertTrue(audit._action_changed(a, changed_power))
        self.assertTrue(audit._action_changed(a, changed_link))


    def test_summary_and_csv_allow_solver_arm_without_selected_action(self):
        record = {
            "block": 2, "stage": "B_x", "reconstruction_mismatch": None,
            "strict_feasible_first_candidate": {
                "solver": {"status": "qos_feasible_point_not_found"},
                "truth": None, "calls": 12,
            },
            "strict_same_topology_control": {
                "solver": {"status": "maxmin_local_solution_found"},
                "truth": {
                    "min_ee": 11.0, "min_ee_gain_pct_vs_original": 1.0,
                    "all_true_gates_ok": True, "qos_ok": True,
                    "budget_ok_vs_pred_ref": True, "budget_ok_vs_true_ref": True,
                    "legitimate_improvement": True,
                },
                "predicted": {"qos_ok": True, "budget_ok_vs_pred_ref": True},
                "calls": 20,
            },
        }
        summary = audit._summarise([record], 0.1)
        self.assertEqual(summary["arms"]["strict_feasible_first_candidate"]["n"], 1)
        self.assertEqual(summary["arms"]["strict_feasible_first_candidate"]["truth_all_gates_pass"], 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candidate.csv"
            audit._write_csv(path, [record])
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
