import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from p3base.config import FoundationConfig
from p3base.estimator import ChannelEstimator
from p3base.block import GeometryContext, predict_block
from p3learn.forecast import ConditionalTrajectoryBank
from p3learn.allocation import Action


def fixture():
    cfg = FoundationConfig()
    ctx = GeometryContext(4, cfg.vehicle, np.empty((0, 6)),
                          np.full(4, 4.5), np.full(4, 1.8), np.full(4, 1.5))
    state = dict(center_xy_m=np.array([[0., 0.], [12., 0.], [25., 0.], [37., 0.]]),
                 heading_rad=np.zeros(4), speed_mps=np.array([1., 2., 1., 0.]))
    ports = dict(tx=np.zeros((4, 4), int), rx=np.ones((4, 4), int))
    return ChannelEstimator(4, cfg), state, ctx, Action([(0, 1, "RF"), (2, 3, "VLC")], np.array([.02, .3]), ports)


def test_bank_equals_original_predictor_and_does_not_mutate_posterior():
    est, state, ctx, action = fixture()
    before = pickle.dumps(est)
    expected = predict_block(est, state, ctx, action.links, action.powers, action.ports, 0., draws=4, seed=77)
    bank = ConditionalTrajectoryBank(est, state, ctx, 0., draws=4, seed=77)
    actual = bank(action)
    np.testing.assert_array_equal(expected["block_rate_samples_bps"], actual["block_rate_samples_bps"])
    assert before == pickle.dumps(est)
    assert not hasattr(bank, "scene") and not hasattr(bank, "oracle")
    assert not bank.rf.flags.writeable
    action.powers[:] = .1
    assert not np.array_equal(actual["mean_rate_bps"], bank(action)["mean_rate_bps"])


def test_future_rejected_and_point_explicit():
    est, state, ctx, action = fixture()
    bank = ConditionalTrajectoryBank(est, state, ctx, 0., draws=3, seed=1, information="point")
    assert bank(action)["information"] == "point"
    est.state_time["rf"] = .1
    with pytest.raises(ValueError):
        ConditionalTrajectoryBank(est, state, ctx, 0., draws=3, seed=1)
