"""Causal feature interface and fixed-port axis selection, without model training."""
import copy
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from p3learn.features import EDGE_DIM, NODE_DIM, build_features


def feature_inputs():
    n = 4
    state = dict(center_xy_m=np.array([[0., 0.], [10., 1.], [20., 0.], [30., 1.]]),
                 heading_rad=np.array([0., .2, .5, 1.]), speed_mps=np.arange(n, dtype=float))
    allowed = ~np.eye(n, dtype=bool)
    index = np.arange(n * n).reshape(n, n)
    rf_gain = (1 + index) * 1e-11
    optical = np.ones((n, n, 2, 2)) * np.array([[1., 2.], [11., 12.]]) * 1e-6
    age_rf = np.full((n, n), np.inf)
    age_rf[0, 1] = .003
    age_vlc = np.full((n, n, 2, 2), np.inf)
    age_vlc[..., 0, 0] = .05
    age_vlc[..., 0, 1] = .1
    age_vlc[..., 1, 0] = .2
    age_vlc[..., 1, 1] = .3
    belief = dict(rf_candidate=allowed.copy(), vlc_candidate=allowed.copy(),
                  rf_gain_mean=rf_gain, rf_gain_low=rf_gain * .5, rf_gain_high=rf_gain * 2,
                  rf_state_mean=np.ones((n, n), complex), rf_state_var=np.full((n, n), .2),
                  rf_age_s=age_rf, vlc_gain_mean=optical, vlc_gain_low=optical * .8,
                  vlc_gain_high=optical * 1.3, vlc_state_mean=np.zeros((n, n, 2, 2)),
                  vlc_state_var=np.full((n, n, 2, 2), .1), vlc_age_s=age_vlc)
    geom_rf = dict(gain=rf_gain * .9, los=allowed.copy())
    geom_vlc = dict(gain=optical * .9)
    ii, jj = np.indices((n, n))
    ports = dict(tx=ii % 2, rx=jj % 2)
    business = allowed.copy()
    business[0, 1] = False
    return state, belief, geom_rf, geom_vlc, ports, business


def test_feature_shapes_finite_missing_ages_and_business_intersection():
    args = feature_inputs()
    before = copy.deepcopy(args)
    features = build_features(*args)
    assert features.node_features.shape == (4, NODE_DIM)
    assert features.edge_features.shape == (4, 4, 2, EDGE_DIM)
    assert features.mode_mask.shape == (4, 4, 2)
    assert features.port_choices.shape == (4, 4, 2, 2)
    assert np.isfinite(features.node_features).all()
    assert np.isfinite(features.edge_features).all()
    assert not features.mode_mask[0, 1].any()
    assert not features.mode_mask[np.arange(4), np.arange(4)].any()
    assert features.edge_features[1, 2, 0, 3] == 1  # explicit missing RF CSI
    assert features.edge_features[0, 1, 0, 3] == 0
    for current, original in zip(args, before):
        if isinstance(current, dict):
            for key in current:
                np.testing.assert_array_equal(current[key], original[key])
        else:
            np.testing.assert_array_equal(current, original)


def test_vlc_gain_and_age_features_follow_transmitter_receiver_port_order():
    state, belief, rf, vlc, ports, business = feature_inputs()
    features = build_features(state, belief, rf, vlc, ports, business)
    for i, j in ((1, 2), (2, 1), (3, 1)):
        tp, rp = ports["tx"][i, j], ports["rx"][i, j]
        gain = belief["vlc_gain_mean"][i, j, tp, rp]
        assert features.edge_features[i, j, 1, 0] == pytest.approx(np.log10(gain) / 15)
        assert features.edge_features[i, j, 1, 2] == pytest.approx(belief["vlc_age_s"][i, j, tp, rp])
    np.testing.assert_array_equal(features.port_choices[:, :, 1, 0], ports["tx"])
    np.testing.assert_array_equal(features.port_choices[:, :, 1, 1], ports["rx"])
    assert np.all(features.port_choices[:, :, 0] == -1)


def incident(mask, node):
    """Directed edges incident to ``node``, each counted once per endpoint."""
    return ([(node, j) for j in range(len(mask)) if mask[node, j]]
            + [(j, node) for j in range(len(mask)) if mask[j, node]])


def test_node_weakest_edge_summaries_follow_incident_legal_edges():
    args = feature_inputs()
    features = build_features(*args)
    edge, mask = features.edge_features, features.mode_mask
    for node in range(4):
        for mode in (0, 1):
            values = [edge[i, j, mode, 0] for i, j in incident(mask[..., mode], node)]
            expected = min(values) if values else -2.
            assert features.node_features[node, 8 + mode] == pytest.approx(expected, abs=1e-5)
        variances = [edge[i, j, 0, 1] for i, j in incident(mask[..., 0], node)]
        expected = float(np.mean(variances)) if variances else 0.
        assert features.node_features[node, 10] == pytest.approx(expected, abs=1e-5)
        stale = [(edge[i, j, mode, 2] >= 1.) or (edge[i, j, mode, 3] > 0)
                 for mode in (0, 1) for i, j in incident(mask[..., mode], node)]
        expected = float(np.mean(stale)) if stale else 0.
        assert features.node_features[node, 11] == pytest.approx(expected, abs=1e-5)


def test_node_summaries_have_documented_empty_sentinels():
    state, belief, rf, vlc, ports, business = feature_inputs()
    blocked = dict(belief, rf_candidate=np.zeros_like(belief['rf_candidate']),
                   vlc_candidate=np.zeros_like(belief['vlc_candidate']))
    features = build_features(state, blocked, rf, vlc, ports, business)
    assert not features.mode_mask.any()
    assert np.isfinite(features.node_features).all()
    np.testing.assert_allclose(features.node_features[:, 8:10], -2., atol=1e-6)
    np.testing.assert_allclose(features.node_features[:, 10:12], 0., atol=1e-6)


def test_features_ignore_unconsumed_future_diagnostics_and_have_no_scene_argument():
    args = feature_inputs()
    first = build_features(*args)
    args[0]["future_positions_audit_only"] = np.full((100, 4, 2), 1e30)
    args[1]["future_truth_audit_only"] = np.full((4, 4), np.nan)
    second = build_features(*args)
    np.testing.assert_array_equal(first.node_features, second.node_features)
    np.testing.assert_array_equal(first.edge_features, second.edge_features)
    parameters = inspect.signature(build_features).parameters
    assert "scene" not in parameters and "oracle" not in parameters
