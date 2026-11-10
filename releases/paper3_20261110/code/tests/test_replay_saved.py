"""Audit failures must stop replay rather than produce a new performance row."""
import json

import numpy as np
import pytest

from replay_saved import assert_same_samples, commit_output, load_output, summary, identical_action
from p3learn.allocation import Action


def test_saved_sample_mismatch_is_rejected():
    original = np.array([[1e6, 2e6], [3e6, 4e6]])
    changed = original.copy()
    changed[1, 0] += 1.
    with pytest.raises(ValueError, match="do not reproduce"):
        assert_same_samples(changed, original, "saved initial action")


def test_committed_output_detects_tampering(tmp_path):
    folder = tmp_path / "block_000010"
    commit_output(folder, {"block": 10, "status": "restored"}, {"rates": np.ones((2, 2))})
    assert load_output(folder)["block"] == 10
    (folder / "record.json").write_text(json.dumps({"block": 11}), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint changed"):
        load_output(folder)


def test_original_failure_is_not_counted_as_restored_success():
    result = summary([
        {"status": "original_failure_preserved", "replay_wall_s": 0.5},
        {"status": "restored", "restore": {"elapsed_s": 1., "forecast_build_s": 0.2},
         "replay_wall_s": 1.5},
    ], 2, "restore")
    assert result["original_failures"] == 1
    assert result["restored_blocks"] == 1
    assert result["timing_s"]["replay_wall_total"] == 2.


def test_initial_action_comparison_uses_only_selected_physical_values():
    ports = {"tx": np.zeros((4, 4), int), "rx": np.zeros((4, 4), int)}
    left = Action([(0, 1, "RF"), (2, 3, "VLC")], np.array([.02, .4]), ports)
    reordered_ports = {name: value.copy() for name, value in ports.items()}
    reordered_ports["tx"][0, 3] = 1  # unused choice does not alter the action
    reordered_ports["rx"][0, 1] = 1  # RF has no selected optical port
    right = Action(list(reversed(left.links)), left.powers[::-1].copy(), reordered_ports)
    assert identical_action(left, right)
    reordered_ports["tx"][2, 3] = 1
    assert not identical_action(left, right)
    reordered_ports["tx"][2, 3] = 0
    right.powers[0] = np.nextafter(right.powers[0], np.inf)
    assert not identical_action(left, right)
