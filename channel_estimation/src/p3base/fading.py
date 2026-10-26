"""真值衰落过程（仅用于物理评价与估计误差统计，决策器不得读取）。

按 (n,n) 与 (n,n,2,2) 矢量化，10/50/100 车共用同一实现；状态随块推进，
不做 N^2 条链路 × 10^5 个细时间点的全量存储。

RF：g_true = g_geom * 10^(X/10) * |a|^2
    a 为零均值单位方差复高斯 AR(1)，|a|^2 服从指数分布（Rayleigh 功率衰落）
    X ~ N(0, sigma_SF^2)，sigma_SF 由 LoS/NLoS 决定（3/4 dB）
VLC：h_true = h_geom * exp(z)
    z 为对数域 AR(1)，平稳均值 -sd^2/2、平稳方差 sd^2

阴影 X 的空间相关性用"相对速度/去相关距离"近似成时间 AR(1)：同一路段的共享散射体
没有建模，链路之间相互独立。这是近似，必须在实验记录中写明。
"""
import numpy as np

from .config import RFConfig, VLCConfig


SPEED_OF_LIGHT = 299_792_458.0


def _cn(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def link_coherence_s(rel_speed_mps, rf_cfg: RFConfig):
    """按相对速度推相干时间 T_c = 0.423 * lambda / v_rel。

    真值场与卡尔曼滤波器必须调用同一个函数，否则状态转移系数不一致。
    """
    lam = SPEED_OF_LIGHT / (rf_cfg.frequency_ghz * 1e9)
    v = np.asarray(rel_speed_mps, float)
    tau = rf_cfg.doppler_coherence_coeff * lam / np.maximum(v, 1e-6)
    return np.clip(tau, rf_cfg.fading_coherence_min_s, rf_cfg.fading_coherence_max_s)


class FadingField:
    def __init__(self, n, rf_cfg: RFConfig, vlc_cfg: VLCConfig, seed, n_ports=2):
        self.n = int(n)
        self.rf_cfg, self.vlc_cfg = rf_cfg, vlc_cfg
        self.seed = int(seed)
        generators = [np.random.default_rng(s)
                      for s in np.random.SeedSequence(self.seed).spawn(3)]
        self._rf_rng, self._sh_rng, self._vlc_rng = generators
        self.rf_amp = _cn(self._rf_rng, (self.n, self.n))
        # 阴影的平稳初值需要逐链路的 LoS/NLoS 标准差，构造时还不知道，故延迟到首次 advance。
        self.rf_shadow_db = np.zeros((self.n, self.n))
        self._shadow_initialized = False
        self.vlc_sd = float(vlc_cfg.log_stationary_sd)
        self.vlc_log = (np.full((self.n, self.n, n_ports, n_ports), -0.5 * self.vlc_sd ** 2)
                        + self._vlc_rng.normal(0.0, self.vlc_sd,
                                               (self.n, self.n, n_ports, n_ports)))
        self.time_s = 0.0

    def advance(self, dt, los=None, rel_speed=None):
        """推进 dt 秒。

        los: (n,n) 布尔，决定阴影标准差；None 时全部按 LoS 处理。
        rel_speed: (n,n) 相对速度 m/s；None 时用 0（阴影不随时间变化）。
        """
        dt = float(dt)
        if dt <= 0:
            return
        self.time_s += dt
        if rel_speed is None:
            rho = np.exp(-dt / self.rf_cfg.fading_coherence_s)
        else:
            rho = np.exp(-dt / link_coherence_s(rel_speed, self.rf_cfg))
        self.rf_amp = rho * self.rf_amp + np.sqrt(1.0 - rho ** 2) * _cn(
            self._rf_rng, (self.n, self.n))

        if los is None:
            sigma = np.full((self.n, self.n), self.rf_cfg.los_shadow_sigma_db)
        else:
            sigma = np.where(np.asarray(los, bool),
                             self.rf_cfg.los_shadow_sigma_db,
                             self.rf_cfg.nlos_shadow_sigma_db)
        if not self._shadow_initialized:
            # 平稳初值：X ~ N(0, sigma^2)，避免从 0 开始的长时间暂态被当成物理现象。
            self.rf_shadow_db = sigma * self._sh_rng.standard_normal((self.n, self.n))
            self._shadow_initialized = True
        if rel_speed is None:
            rho_s = 1.0
        else:
            d_corr = max(self.rf_cfg.shadow_decorrelation_m, 1e-9)
            rho_s = np.exp(-dt * np.asarray(rel_speed, float) / d_corr)
        self.rf_shadow_db = (rho_s * self.rf_shadow_db
                             + np.sqrt(np.maximum(1.0 - rho_s ** 2, 0.0)) * sigma
                             * self._sh_rng.standard_normal((self.n, self.n)))

        rho_v = np.exp(-dt / self.vlc_cfg.fading_coherence_s)
        mean = -0.5 * self.vlc_sd ** 2
        self.vlc_log = (mean + rho_v * (self.vlc_log - mean)
                        + np.sqrt(self.vlc_sd ** 2 * (1.0 - rho_v ** 2))
                        * self._vlc_rng.standard_normal(self.vlc_log.shape))

    def rf_gain(self, geometry_gain):
        return np.asarray(geometry_gain, float) * 10.0 ** (self.rf_shadow_db / 10.0) \
            * np.abs(self.rf_amp) ** 2

    def vlc_gain(self, geometry_gain):
        return np.asarray(geometry_gain, float) * np.exp(self.vlc_log)

    def snapshot(self):
        return dict(rf_amp=self.rf_amp.copy(), rf_shadow_db=self.rf_shadow_db.copy(),
                    vlc_log=self.vlc_log.copy(), time_s=self.time_s)
