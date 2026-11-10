"""Independently difference the constraint actually submitted to SLSQP.

Unequal powers, asymmetric interference and mixed modes expose both an axis
transpose error and the omitted derivative of the variable EE denominator.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from p3base.config import FoundationConfig
from p3learn import bottleneck_power as bp
from p3learn.dinkelbach_sca import _Physics


def _mixed_physics():
    rng = np.random.default_rng(472)
    n = 8
    cfg = FoundationConfig()
    gains = {
        "RF": rng.uniform(0.3, 3.0, (n, n)) * 1e-8,
        "VLC": rng.uniform(0.3, 3.0, (n, n)) * 1e-5,
    }
    ports = {"tx": np.zeros((n, n), int), "rx": np.zeros((n, n), int)}
    links = [(0, 1, "RF"), (2, 3, "RF"), (4, 5, "VLC"), (6, 7, "VLC")]
    return _Physics(links, gains, ports, cfg)


def _centered_jacobian(fun, point):
    result = []
    for index in range(point.size):
        step = 1e-6 * max(abs(float(point[index])), 1.0)
        above, below = point.copy(), point.copy()
        above[index] += step
        below[index] -= step
        result.append((fun(above) - fun(below)) / (2.0 * step))
    return np.asarray(result).T


@pytest.mark.parametrize("fixed_denominator", [False, True])
@pytest.mark.parametrize("with_qos", [False, True])
def test_sca_constraint_jacobian_matches_finite_difference(
    monkeypatch, fixed_denominator, with_qos
):
    phys = _mixed_physics()
    powers = np.array([0.05, 0.4, 0.3, 2.0])
    threshold = 0.9 * float(bp.link_ee(phys, powers).min())
    denominator = phys.static + powers if fixed_denominator else None
    q = np.array([2e5, 3e5, 4e5, 5e5]) if with_qos else None
    checked = []

    def inspect_subproblem(fun, x0, *, jac, method, bounds, constraints, options):
        constraint = constraints[0]
        # Move away from the tangent point while remaining in the same smooth
        # branch; asymmetric derivatives cannot accidentally cancel there.
        point = x0.copy()
        point[:phys.k] *= np.array([1.01, 0.99, 1.005, 0.995])
        supplied = constraint["jac"](point)
        numerical = _centered_jacobian(constraint["fun"], point)
        np.testing.assert_allclose(supplied, numerical, rtol=3e-5, atol=2e-5)
        checked.append(True)
        return SimpleNamespace(x=x0)

    monkeypatch.setattr(bp, "minimize", inspect_subproblem)
    bp._sca_step(phys, powers, None if fixed_denominator else threshold,
                 q, trust=2.0, denominator=denominator)
    assert checked


def test_rate_bound_jacobian_orientation_is_parameter_then_link():
    phys = _mixed_physics()
    powers = np.array([0.05, 0.4, 0.3, 2.0])
    bound = bp._linear_rates(phys, powers)
    rate, jac = bound(powers)
    numerical = _centered_jacobian(lambda p: bound(p)[0], powers)
    np.testing.assert_allclose(jac.T, numerical, rtol=3e-5, atol=2e-3)
    assert not np.allclose(jac, jac.T), "Fixture must expose an orientation error"
