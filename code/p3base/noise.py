"""接收端噪声。VLC 噪声按论文 (8)-(11) 式拆分。

sigma^2_VLC = sigma^2_shot + sigma^2_thermal + sigma^2_ISI
  sigma^2_shot    = 2 q R P_r B + 2 q I_bg I2 B
  sigma^2_thermal = (8 pi k T / G) eta S I2 B^2
                    + (16 pi^2 k T Gamma / g_m) eta^2 S^2 I3 B^3
  sigma^2_ISI     = (R P_r)^2

旧 DJ 管线只算了 thermal + 背景散粒，没有 ISI 项。本层默认按照论文计入 ISI，
并把 include_isi 显式暴露出来，避免两种口径被悄悄混用。
"""
import numpy as np

from .config import RFConfig, VLCConfig


def rf_noise_w(cfg: RFConfig) -> float:
    return cfg.noise_w


def vlc_noise_base_w2(cfg: VLCConfig) -> float:
    """不含信号项的热噪声 + 背景散粒噪声（即旧管线的 VLC_NOISE_BASE）。"""
    cap = cfg.capacitance_per_area_f_m2 * cfg.area_m2
    b = cfg.bandwidth_hz
    return (2.0 * cfg.electron_charge_c * cfg.background_current_a * cfg.c2 * b
            + 8.0 * np.pi * cfg.boltzmann_j_k * cfg.temperature_k / cfg.voltage_gain
            * cap * cfg.c2 * b * b
            + 16.0 * np.pi ** 2 * cfg.boltzmann_j_k * cfg.temperature_k
            * cfg.noise_factor / cfg.transconductance_s * cap * cap * cfg.c3 * b ** 3)


def vlc_shot_coeff(cfg: VLCConfig) -> float:
    """乘以 R*P_optical 得到信号相关散粒噪声方差。"""
    return 2.0 * cfg.electron_charge_c * cfg.bandwidth_hz


def vlc_noise_linear_w(optical_power_w, cfg: VLCConfig, model=None):
    """线性 SINR 口径下的接收噪声功率（数值上与 h·p 同量纲）。

    ``optical_power_w`` 是到达光功率之和（期望 + 干扰）。三种口径：

    ``n0_only``      ``N0_VLC * B``，严格照 dink10.m，= 5.0e-14 W。
    ``physical``     热噪声 + 背景散粒 + 信号散粒，不含 ISI，= 6.0e-14 W。
    ``physical_isi`` 再加 ``(R P_opt)^2``；该式等于把期望信号本身算成噪声，
                     会让 SINR <= 1，仅保留用于复现旧结果。

    ``physical`` 与 ``n0_only`` 只差 20%，速率差异 < 1%：本层把 A^2 与 W 混用，
    两者在数量级上本来就一致（5.8e-14 vs 5.0e-14），不改变 SINR 的量级。
    """
    model = cfg.noise_model if model is None else model
    p = np.asarray(optical_power_w, float)
    if model == "n0_only":
        return cfg.n0_w_per_hz * cfg.bandwidth_hz + 0.0 * p
    total = vlc_noise_base_w2(cfg) + vlc_shot_coeff(cfg) * cfg.responsivity * p
    if model == "physical_isi":
        total = total + (cfg.responsivity * p) ** 2
    return total


def vlc_noise_var(optical_power_w, cfg: VLCConfig, include_isi=True):
    """给定到达光功率（含干扰的合成功率和），返回噪声方差（A^2）。"""
    p = np.asarray(optical_power_w, float)
    total = vlc_noise_base_w2(cfg) + vlc_shot_coeff(cfg) * cfg.responsivity * p
    if include_isi:
        total = total + (cfg.responsivity * p) ** 2
    return total


def pilot_gain_variance(noise_power_w, bandwidth_hz, pilot_power_w, integration_s,
                        geometry_gain):
    """导频观测的等效方差：y = h + n，Var(n) = N0 / (P T g)。

    匹配滤波输出 SNR = E_s/N0，其中 E_s = P T g 为接收符号能量，
    N0 = 带内噪声功率 / 带宽 为单边功率谱密度。旧写法直接用带内噪声功率
    （W）当 N0，方差被放大 B 倍；本函数显式除以带宽。

    geometry_gain 为确定性几何增益；g=0 的链路不可观测，返回 inf。
    """
    g = np.asarray(geometry_gain, float)
    n0 = float(noise_power_w) / float(bandwidth_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = n0 / (pilot_power_w * integration_s * g)
    return np.where(g > 0, var, np.inf)


def vlc_pilot_log_variance(optical_gain, pilot_power_w, integration_s, cfg: VLCConfig,
                           floor_sd):
    """VLC 导频的对数域观测方差。

    物理项：SNR = (R P_r)^2 / sigma^2 * (B T)，对数域方差 ~ 1/SNR。
    导频只发已知符号，不存在数据 ISI，故此处不含 (R P_r)^2 项。
    本配置下物理项远低于下限，实际由 floor_sd 控制（标定/对准误差）。
    """
    g = np.asarray(optical_gain, float)
    p_r = g * float(pilot_power_w)
    sigma2 = vlc_noise_var(p_r, cfg, include_isi=False)
    signal = cfg.responsivity * p_r
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = signal ** 2 / sigma2 * cfg.bandwidth_hz * float(integration_s)
        var = np.maximum(1.0 / np.maximum(snr, 1e-300), float(floor_sd) ** 2)
    return np.where(g > 0, var, np.inf)


def observation_var_vlc(log_sd, gain):
    """固定对数观测方差（不看功率时使用）。不可见链路返回 inf。"""
    g = np.asarray(gain, float)
    return np.where(g > 0, float(log_sd) ** 2, np.inf)
