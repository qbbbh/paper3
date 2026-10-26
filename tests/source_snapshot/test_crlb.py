import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from crlb_report import validate_bounds, required_pilot_energy, local_db_crlb_sd
from p3base.config import FoundationConfig


def test_matched_object_mse():
    r=validate_bounds(repetitions=30000)
    assert abs(r['observation_ratio']-1)<.03
    assert max(abs(x['mse']/x['bayes_mse']-1) for x in r['linear_gaussian'])<.04


def test_pilot_energy_inverse():
    cfg=FoundationConfig(); g=1e-11
    energy=required_pilot_energy(cfg,g,.1)
    np.testing.assert_allclose(local_db_crlb_sd(cfg,g,energy/2e-5,2e-5),.1)
    np.testing.assert_allclose(required_pilot_energy(cfg,g,.05),energy*4)
