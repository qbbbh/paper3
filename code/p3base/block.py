"""100 ms fixed-action integration and causal Monte Carlo prediction.

RF input is POWER gain (never square it); VLC input is optical gain.
Prediction is an approximate conditional model, not a reliability certificate.
Only static map/dimensions and the current state enter geometry prediction.
"""
from dataclasses import dataclass
import copy
import numpy as np
from . import channel
from .scene import Scene
from .fading import link_coherence_s
from .energy import link_electrical_power_w
from .noise import vlc_noise_base_w2, vlc_noise_linear_w, vlc_shot_coeff


@dataclass(frozen=True)
class GeometryContext:
    n: int
    vehicle: object
    buildings: np.ndarray
    length_m: np.ndarray
    width_m: np.ndarray
    height_m: np.ndarray

    @classmethod
    def from_scene(cls, scene):
        return cls(scene.n, scene.vehicle, scene.buildings.copy(),
                   scene.length_m.copy(), scene.width_m.copy(), scene.height_m.copy())

    def ports(self, state):
        return Scene.ports(self, state=state)


def validate_action(links, powers, ports, n, cfg, candidates=None, business=None):
    """Require full endpoint coverage. Candidate/business masks are block-start only."""
    if n % 2 or len(links) != n // 2 or len(powers) != len(links):
        raise ValueError('完整配对必须有 N/2 条边和相同数量的功率')
    p = np.asarray(powers, float)
    seen = set()
    for (i, j, mode), power in zip(links, p):
        if mode not in ('RF', 'VLC') or not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)):
            raise ValueError('模式或索引非法')
        if i == j or i in seen or j in seen or not (0 <= i < n and 0 <= j < n):
            raise ValueError('端点冲突或越界')
        seen.update((i, j))
        cap = cfg.rf.power_max_w if mode == 'RF' else cfg.vlc.power_max_w
        if not np.isfinite(power) or not 1e-3 <= power <= cap:
            raise ValueError('活动功率必须在 [0.001, cap] W')
        if candidates is not None and not candidates[mode][i, j]:
            raise ValueError('块初不具备几何候选资格')
        if business is not None and not business[i, j]:
            raise ValueError('不具备业务接收资格')
        if mode == 'VLC':
            for key in ('tx', 'rx'):
                value = ports[key][i, j]
                if value not in (0, 1):
                    raise ValueError('VLC端口必须为0或1')
    return True


def sample_metrics(links, powers, ports, rf_gain, vlc_gain, cfg, include_isi=None):
    """Vectorized instantaneous physics; gains may have leading trajectory axis.

    Returns (..., K). RF S/I/N in W.  Under the default ``linear`` VLC model,
    VLC S/I/N are in the same electrical-equivalent domain as ``h*p`` so that
    ``rate = B log2(1+SINR/gap)`` matches dink10.m.  Under ``square_law`` they
    stay in A^2 for reproduction.
    This function intentionally does not reselect ports or remove blocked links.
    """
    i = np.array([e[0] for e in links], int)
    j = np.array([e[1] for e in links], int)
    mode = np.array([e[2] for e in links])
    p = np.asarray(powers, float)
    k = len(links)
    rf = np.asarray(rf_gain)[..., i[:, None], j[None, :]]
    tp, rp = ports['tx'][i, j], ports['rx'][i, j]
    vlc = np.asarray(vlc_gain)[..., i[:, None], j[None, :], tp[:, None], rp[None, :]]
    same = (mode[:, None] == mode[None, :])
    rf_terms = rf * p[:, None] * same
    optical = cfg.vlc.eo_efficiency * vlc * p[:, None] * same
    if cfg.vlc.sinr_model == "linear":
        vlc_terms = optical
    else:
        vlc_terms = (cfg.vlc.responsivity * optical) ** 2
    terms = np.where(mode[None, :] == 'RF', rf_terms, vlc_terms)
    signal = np.diagonal(terms, axis1=-2, axis2=-1).copy()
    offdiag = ~np.eye(k, dtype=bool)
    interference = (terms * offdiag).sum(axis=-2)
    optical_sum = optical.sum(axis=-2)
    # 噪声只由 ``cfg.vlc.noise_model`` 决定；``include_isi`` 仅为旧签名保留。
    vn = vlc_noise_linear_w(optical_sum, cfg.vlc)
    noise = np.where(mode == 'RF', cfg.rf.noise_w, vn)
    sinr = signal / (interference + noise)
    bw = np.where(mode == 'RF', cfg.rf.bandwidth_hz, cfg.vlc.bandwidth_hz)
    gap = np.where(mode == 'RF', cfg.rf.gap, cfg.vlc.gap)
    rate = bw * np.log1p(sinr / gap) / np.log(2.)
    return dict(rate_bps=rate, sinr=sinr, signal=signal, interference=interference, noise=noise)


def summarize_block(links, powers, fine, dt_s, n, q_eval_bps=None, cfg=None):
    """Exactly one fixed-duration block; no history/catch-up and no double counting.

    With ``cfg`` the per-link EE is ``rate / (static_power + p)``, i.e. the
    electrical input power -- the objective.  Without it the denominator is the
    transmit power alone, which is kept only for legacy callers/tests.
    """
    rates = np.asarray(fine['rate_bps'], float)
    if rates.ndim != 2 or not len(rates) or rates.shape[1] != len(links):
        raise ValueError('细样本应为 (time, link)')
    if not np.isfinite(rates).all() or np.any(rates < 0) or dt_s <= 0:
        raise ValueError('无效速率或采样步长')
    duration = rates.shape[0] * dt_s
    p = np.asarray(powers, float)
    if p.shape != (len(links),) or not np.isfinite(p).all() or np.any(p < .001):
        raise ValueError('选中链路的持续功率必须有限且至少1 mW')
    ids = [endpoint for i, j, _ in links for endpoint in (i, j)]
    if sorted(ids) != list(range(n)):
        raise ValueError('未完整覆盖的块须单独记录失败，不能按完整配置汇总')
    avg = rates.mean(axis=0)
    energy = duration * p
    modes = [mode for _i, _j, mode in links]
    if cfg is None:
        denom = p
    else:
        denom = link_electrical_power_w(p, modes, cfg)
    eta = avg / denom
    vehicle = np.zeros(n)
    for (i, j, _), r in zip(links, avg):
        vehicle[i] = vehicle[j] = r  # participation diagnostic, not twice network throughput
    vlc = np.array([m == 'VLC' for _, _, m in links])
    out = dict(duration_s=duration, mean_rate_bps=avg, link_energy_j=energy,
               link_delivered_bits=avg * duration, pair_ee_bit_per_j=eta,
               pair_ee_tx_only_bit_per_j=avg / p,
               network_ee_bit_per_j=float(avg.sum() / p.sum()),
               # Per-link EE, then average.  This is the paper's definition and
               # the objective; network EE (sum rate / sum power) is a different
               # quantity and is reported separately.
               mean_pair_ee_bit_per_j=float(eta.mean()),
               min_pair_ee_bit_per_j=float(eta.min()),
               mean_pair_ee_tx_only_bit_per_j=float(np.mean(avg / p)),
               jain_pair_ee=float(eta.sum() ** 2 / (len(eta) * np.dot(eta, eta))) if np.dot(eta, eta) else 0.,
               participation_rate_bps=vehicle,
               vlc_link_fraction=float(vlc.mean()),
               mean_data_power_w=float(p.mean()),
               total_data_power_w=float(p.sum()),
               total_electrical_power_w=float(denom.sum()),
               vlc_rate_fraction=float(avg[vlc].sum() / avg.sum()) if avg.sum() else 0.)
    if q_eval_bps is not None:
        q = np.broadcast_to(np.asarray(q_eval_bps, float), avg.shape)
        if not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError('q_eval必须有限非负')
        out.update(pair_qos_met=avg >= q, all_pairs_qos_met=bool(np.all(avg >= q)))
    return out


def predict_block(estimator, state_now, geometry, links, powers, ports, t,
                  draws=64, seed=1, include_isi=True):
    """Approximate conditional trajectories; no future Scene/TruthOracle argument.

    Current velocity/heading extrapolate geometry. RF two-state uses independent
    posterior factors (the filter's explicit approximation); no bounds guarantee.
    Samples retain temporal correlation and same-mode interference. Endpoints and
    ports stay fixed, including after predicted blockage.
    """
    cfg = estimator.cfg
    if any(value > t + 1e-12 for value in estimator.state_time.values()):
        raise ValueError('不能用未来后验追溯预测旧块')
    if draws < 2:
        raise ValueError('至少2条预测轨迹')
    n, dt, duration = geometry.n, cfg.estimation.fine_dt_s, cfg.estimation.block_dt_s
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration):
        raise ValueError('块长必须为细步长的整数倍')
    validate_action(links, powers, ports, n, cfg)
    st = {key: np.array(value, copy=True) for key, value in state_now.items()}
    h = st['heading_rad']
    vel = st['speed_mps'][:, None] * np.stack([np.cos(h), np.sin(h)], axis=-1)
    rel = np.linalg.norm(vel[:, None, :] - vel[None, :, :], axis=-1)
    tau = link_coherence_s(rel, cfg.rf)
    rng = np.random.default_rng(seed)
    def cn(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.)
    shape = (draws, n, n)
    rf = copy.deepcopy(estimator.rf)
    vf = copy.deepcopy(estimator.vlc)
    estimator._predict_inplace(rf, estimator.state_time['rf'], t, estimator._rf_doppler, estimator._rf_tau)
    estimator._predict_inplace(vf, estimator.state_time['vlc'], t, None, None)
    two = rf.state == 'two_state'
    if not two and rf.state != 'complex_ar1':
        raise ValueError('块预测仅支持two_state或complex_ar1 RF配置')
    if vf.state != 'log_intensity':
        raise ValueError('块预测要求VLC log_intensity')
    if two:
        a = rf.a + np.sqrt(np.maximum(rf.P_aa, 0)) * cn(shape)
        L = rf.L + np.sqrt(np.maximum(rf.P_LL, 0)) * rng.standard_normal(shape)
    else:
        a = rf.mean + np.sqrt(np.maximum(rf.var, 0)) * cn(shape)
        L = np.zeros(shape)
    z = vf.mean + np.sqrt(np.maximum(vf.var, 0)) * rng.standard_normal((draws, n, n, 2, 2))
    total = np.zeros((draws, len(links)))
    rho = np.exp(-dt / tau)
    rs = np.exp(-dt * rel / cfg.rf.shadow_decorrelation_m)
    rv = np.exp(-dt / cfg.vlc.fading_coherence_s)
    zv = cfg.vlc.log_stationary_sd ** 2
    zm = -0.5 * zv
    for k in range(steps):
        # Left-endpoint integration on [t,t+T); same convention as truth evaluation.
        state = dict(st, center_xy_m=st['center_xy_m'] + vel * (k * dt))
        rg = channel.rf_geometry(geometry, state, cfg.rf)
        vg = channel.vlc_geometry(geometry, geometry.ports(state), cfg.vlc)
        gains_rf = rg['gain'] * np.exp(L) * np.abs(a) ** 2
        gains_vlc = vg['gain'] * np.exp(z)
        total += sample_metrics(links, powers, ports, gains_rf, gains_vlc, cfg, include_isi)['rate_bps']
        if two:
            sigma = np.where(rg['los'], cfg.rf.los_shadow_sigma_db, cfg.rf.nlos_shadow_sigma_db) * np.log(10.) / 10.
            L = rs * L + np.sqrt(1 - rs ** 2) * sigma * rng.standard_normal(shape)
            stationary = 1.
        else:
            stationary = rf.stationary_var
        a = rho * a + np.sqrt((1 - rho ** 2) * stationary) * cn(shape)
        z = zm + rv * (z - zm) + np.sqrt(zv * (1 - rv ** 2)) * rng.standard_normal(z.shape)
    samples = total / steps
    return dict(mean_rate_bps=samples.mean(axis=0), low_rate_bps=np.quantile(samples, .05, axis=0),
                high_rate_bps=np.quantile(samples, .95, axis=0), block_rate_samples_bps=samples,
                draws=draws, seed=seed, geometry_model='constant_velocity_heading',
                calibration_status='approximate_not_certified')
