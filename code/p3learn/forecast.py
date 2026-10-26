"""Action-independent causal trajectory cache, shared by every method in a block.

This implements the same sampling/integration as p3base.block.predict_block.
It caches predicted gains, never future observed gains or Scene/TruthOracle objects.
"""
import copy
import time
import numpy as np

from p3base import block, channel
from p3base.fading import link_coherence_s


class ConditionalTrajectoryBank:
    def __init__(self, estimator, state_now, geometry, t, *, draws, seed,
                 information="trajectory"):
        started = time.perf_counter()
        if information not in ("trajectory", "point"):
            raise ValueError("information must be trajectory or point")
        if draws < 2 or any(v > t + 1e-12 for v in estimator.state_time.values()):
            raise ValueError("requires >=2 draws and a causal posterior")
        self.cfg, self.n = estimator.cfg, geometry.n
        self.draws, self.seed, self.information = int(draws), int(seed), information
        self.dt = self.cfg.estimation.fine_dt_s
        self.steps = int(round(self.cfg.estimation.block_dt_s / self.dt))
        if not np.isclose(self.steps * self.dt, self.cfg.estimation.block_dt_s):
            raise ValueError("block length must be a multiple of fine sample interval")
        state = {key: np.array(value, copy=True) for key, value in state_now.items()}
        heading = state["heading_rad"]
        velocity = state["speed_mps"][:, None] * np.stack([np.cos(heading), np.sin(heading)], -1)
        relative_speed = np.linalg.norm(velocity[:, None] - velocity[None, :], axis=-1)
        tau = link_coherence_s(relative_speed, self.cfg.rf)
        rng = np.random.default_rng(seed)

        def cn(shape):
            return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.)

        rf, vf = copy.deepcopy(estimator.rf), copy.deepcopy(estimator.vlc)
        estimator._predict_inplace(rf, estimator.state_time["rf"], t,
                                   estimator._rf_doppler, estimator._rf_tau)
        estimator._predict_inplace(vf, estimator.state_time["vlc"], t, None, None)
        two = rf.state == "two_state"
        if (not two and rf.state != "complex_ar1") or vf.state != "log_intensity":
            raise ValueError("unsupported state model")
        # Point ablation plugs conditional mean gains into the nonlinear PHY.
        # It is not the expected rate and is explicitly labelled as an ablation.
        count = self.draws if information == "trajectory" else 1
        shape = (self.draws, self.n, self.n)
        if two:
            a = rf.a + np.sqrt(np.maximum(rf.P_aa, 0)) * cn(shape)
            slow = rf.L + np.sqrt(np.maximum(rf.P_LL, 0)) * rng.standard_normal(shape)
        else:
            a = rf.mean + np.sqrt(np.maximum(rf.var, 0)) * cn(shape)
            slow = np.zeros(shape)
        z = vf.mean + np.sqrt(np.maximum(vf.var, 0)) * rng.standard_normal((self.draws, self.n, self.n, 2, 2))
        self.rf = np.empty((self.steps, count, self.n, self.n), dtype=np.float64)
        self.vlc = np.empty((self.steps, count, self.n, self.n, 2, 2), dtype=np.float64)
        rho = np.exp(-self.dt / tau)
        rs = np.exp(-self.dt * relative_speed / self.cfg.rf.shadow_decorrelation_m)
        rv = np.exp(-self.dt / self.cfg.vlc.fading_coherence_s)
        zv = self.cfg.vlc.log_stationary_sd ** 2
        zm = -0.5 * zv
        for k in range(self.steps):
            current = dict(state, center_xy_m=state["center_xy_m"] + velocity * (k * self.dt))
            rg = channel.rf_geometry(geometry, current, self.cfg.rf)
            vg = channel.vlc_geometry(geometry, geometry.ports(current), self.cfg.vlc)
            if information == "point":
                mu, variance = rf.moments() if two else (rf.mean, rf.var)
                self.rf[k, 0] = rg["gain"] * (np.abs(mu) ** 2 + variance)
                self.vlc[k, 0] = vg["gain"] * np.exp(vf.mean + 0.5 * vf.var)
                rf.predict(self.dt, estimator._rf_doppler, tau)
                vf.predict(self.dt)
            else:
                self.rf[k] = rg["gain"] * np.exp(slow) * np.abs(a) ** 2
                self.vlc[k] = vg["gain"] * np.exp(z)
                if two:
                    sigma = np.where(rg["los"], self.cfg.rf.los_shadow_sigma_db,
                                     self.cfg.rf.nlos_shadow_sigma_db) * np.log(10.) / 10.
                    slow = rs * slow + np.sqrt(1 - rs ** 2) * sigma * rng.standard_normal(shape)
                    stationary = 1.
                else:
                    stationary = rf.stationary_var
                a = rho * a + np.sqrt((1 - rho ** 2) * stationary) * cn(shape)
                z = zm + rv * (z - zm) + np.sqrt(zv * (1 - rv ** 2)) * rng.standard_normal(z.shape)
        self.rf.flags.writeable = self.vlc.flags.writeable = False
        self.build_time_s = time.perf_counter() - started
        self.bytes = self.rf.nbytes + self.vlc.nbytes
        self.evaluator_calls = 0

    def evaluate(self, action):
        block.validate_action(action.links, action.powers, action.ports, self.n, self.cfg)
        self.evaluator_calls += 1
        total = np.zeros((self.rf.shape[1], len(action.links)))
        # Small time chunks bound temporary cross-interference arrays at N=100.
        for start in range(0, self.steps, 10):
            values = block.sample_metrics(action.links, action.powers, action.ports,
                                         self.rf[start:start+10], self.vlc[start:start+10], self.cfg)
            for rates in values["rate_bps"]:
                total += rates
        samples = total / self.steps
        return dict(mean_rate_bps=samples.mean(0), low_rate_bps=np.quantile(samples, .05, axis=0),
                    high_rate_bps=np.quantile(samples, .95, axis=0), block_rate_samples_bps=samples,
                    draws=int(samples.shape[0]), seed=self.seed, information=self.information,
                    calibration_status="approximate_not_certified")

    __call__ = evaluate
