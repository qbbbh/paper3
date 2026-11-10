"""卡尔曼滤波 + MMSE 信道估计（按 paper3 的式 (6)(7)(12) 实现）。

论文递推：
  预测  g^- = A_s g^+ ,  P^- = A_s P^+ A_s^H + Q_p                (6)
  更新  K   = P^- H^H (H P^- H^H + Q_m)^-1
        g^+ = g^- + K (y - H g^-) ,  P^+ = (I - K H) P^-           (7)
  MMSE  g_MMSE = sigma_h^2 / (sigma_h^2 + sigma_VLC^2) * g^+      (12)
  A_s = exp(-j 2 pi v dt / lambda)                                 (6)

与论文原文的三处差异，必须在使用记录里写明：
1. 论文把观测矩阵取成 H = g^VLC（状态本身），这使观测方程不是线性的。本实现取
   H = 1，即导频先除以已知的 R*P_t 还原成归一化增益再观测；等价地可理解为
   H = R*P_t 的标量观测。
2. 论文用 |A_s| = 1 的纯相位旋转。纯相位无法跟踪车辆接近/远离造成的幅度变化，
   因此额外提供 complex_ar1（含衰减）与 log_intensity（对数光强）两种状态模型。
   论文对应的是 paper_doppler，另外两种是块协议下的工程扩展，不是论文原式。
3. 式 (12) 的 MMSE 收缩是在卡尔曼更新之后再乘一个因子。若 sigma_VLC^2 与 Q_m 是同一个
   观测噪声，这一步等于把噪声扣了两次，估计会偏低。默认保留以便复现论文，可用
   mmse=False 关闭。
"""
import numpy as np
from scipy.stats import ncx2, norm


class KalmanMMSE:
    """标量状态卡尔曼 + MMSE 收缩；mean/var 可以是任意形状的数组。"""

    def __init__(self, state="paper_doppler", *, tau_s=0.5, process_var=None,
                 stationary_mean=0.0, stationary_var=1.0):
        if state not in ("paper_doppler", "complex_ar1", "log_intensity"):
            raise ValueError(f"未知状态模型 {state}")
        self.state = state
        self.tau_s = float(tau_s)
        self.process_var = process_var
        self.stationary_mean = float(stationary_mean)
        self.stationary_var = float(stationary_var)
        self.mean = None
        self.var = None

    # ---- 初始化 --------------------------------------------------------
    def reset(self, shape, mean=None, var=None):
        if self.state == "log_intensity":
            self.mean = np.full(shape, self.stationary_mean if mean is None else mean, float)
            self.var = np.full(shape, self.stationary_var if var is None else var, float)
        else:
            self.mean = np.zeros(shape, complex) if mean is None else np.asarray(mean, complex).copy()
            self.var = np.full(shape, self.stationary_var if var is None else var, float)
        return self

    def _q(self, dt, tau=None):
        if self.process_var is not None:
            return float(self.process_var)
        if self.state == "paper_doppler":
            return 0.0
        rho = np.exp(-dt / (self.tau_s if tau is None else tau))
        return (1.0 - rho ** 2) * self.stationary_var

    # ---- 式 (6) 预测 ---------------------------------------------------
    def propagate(self, mean, var, dt, doppler_hz=0.0, tau=None):
        """式 (6) 的状态传播，**不修改自身状态**，返回 (mean, var) 副本。

        估计器需要在"保留在测量时刻的后验"的同时对外给出查询时刻的预测，
        因此把递推抽成纯函数；predict 只是它的原地版本，两者必须完全一致。
        """
        dt = float(dt)
        if dt <= 0:
            return mean, var
        if self.state == "log_intensity":
            rho = np.exp(-dt / self.tau_s)
            return (self.stationary_mean + rho * (mean - self.stationary_mean),
                    rho ** 2 * var + (1.0 - rho ** 2) * self.stationary_var)
        if doppler_hz is None:
            doppler_hz = 0.0
        phase = np.exp(-2j * np.pi * np.asarray(doppler_hz, float) * dt)
        if self.state == "complex_ar1":
            # 与 fading.py 完全一致：rho 是复幅度的相关时间常数，
            # a' = rho a + sqrt(1-rho^2) w，故均值乘 rho、方差乘 rho^2。
            rho = np.exp(-dt / (self.tau_s if tau is None else tau))
            return (rho * phase * mean,
                    rho ** 2 * var + (1.0 - rho ** 2) * self.stationary_var)
        # 论文原式：|A_s| = 1，只有 Q_p 提供不确定性增长
        return phase * mean, var + self._q(dt, tau)

    def predict(self, dt, doppler_hz=0.0, tau=None):
        self.mean, self.var = self.propagate(self.mean, self.var, dt, doppler_hz, tau)
        return self

    # ---- 式 (7) 更新 ---------------------------------------------------
    def update(self, y, obs_var, valid=None):
        """H = 1；obs_var 为观测噪声方差，inf 表示该链路不可观测。"""
        if self.mean is None:
            raise RuntimeError("先调用 reset")
        y = np.asarray(y, complex if self.state != "log_intensity" else float)
        r = np.asarray(obs_var, float)
        mask = np.isfinite(r) & np.isfinite(y)
        if valid is not None:
            mask &= np.asarray(valid, bool)
        r_safe = np.where(mask, r, np.inf)
        gain = np.where(mask, self.var / (self.var + r_safe), 0.0)
        innovation = np.where(mask, y - self.mean, 0.0)
        self.mean = self.mean + gain * innovation
        self.var = (1.0 - gain) * self.var
        return self

    def set_prior_var(self, var, reset_state=False):
        """设置平稳方差（可逐链路），用于把阴影衰落并入复合状态。"""
        self.stationary_var = np.asarray(var, float)
        if reset_state or self.var is None:
            self.var = self.stationary_var.copy()
        return self

    # ---- 式 (12) MMSE 收缩 --------------------------------------------
    @staticmethod
    def mmse_factor(signal_var, noise_var):
        """sigma_h^2 / (sigma_h^2 + sigma_v^2)；缺测（inf）返回 1，即不收缩。"""
        s = np.maximum(np.asarray(signal_var, float), 0.0)
        n = np.asarray(noise_var, float)
        denom = np.where(np.isfinite(n), s + np.maximum(n, 0.0), np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.isfinite(denom) & (denom > 0), s / denom, 1.0)

    @staticmethod
    def mmse(gain_estimate, signal_var, noise_var):
        """sigma_h^2 / (sigma_h^2 + sigma_v^2) 的标量收缩。"""
        s = np.maximum(np.asarray(signal_var, float), 0.0)
        n = np.maximum(np.asarray(noise_var, float), 0.0)
        denom = np.where(s + n > 0, s + n, 1.0)
        return gain_estimate * (s / denom)

    # ---- 先验方差（论文的 sigma_h^2）-----------------------------------
    @staticmethod
    def prior_signal_var(mean_gain, var_gain, log_state=False):
        """把状态后验翻译成增益先验方差，供式 (12) 使用。"""
        if log_state:
            m = np.asarray(mean_gain, float)
            v = np.asarray(var_gain, float)
            return np.exp(2 * m + v) * np.expm1(v)
        m = np.asarray(mean_gain, complex)
        v = np.asarray(var_gain, float)
        return v * v + 2 * v * np.abs(m) ** 2


def paper_process_var(observations):
    """论文对 Q_p 的估计：sigma_p^2 = 1/(N-1) sum (y_{k+1}-y_k)^2。

    observations 第一维是时间；NaN 表示该时刻无观测，跳过相应差分。
    注意：若 y 本身含观测噪声，这个量约等于 2(sigma_p^2 + sigma_m^2)，直接当 Q_p 会偏大。
    """
    y = np.asarray(observations)
    diff = np.diff(y, axis=0)
    ok = np.isfinite(diff)
    power = np.where(ok, np.abs(diff) ** 2, 0.0)
    count = ok.sum(axis=0)
    total = power.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def corrected_process_var(observations, obs_var):
    """把观测差分的均方拆掉测量噪声：sigma_p^2 = (D^2 - 2 sigma_m^2)/2。"""
    d2 = paper_process_var(observations)
    noise = float(np.nanmean(np.asarray(obs_var, float)))
    return np.maximum((d2 - 2.0 * noise) / 2.0, 0.0)


class PredictiveBelief:
    """把状态后验翻译成块开始时刻的增益点估计与单侧区间。"""

    def __init__(self, tail_probability=0.05):
        self.alpha = float(tail_probability)

    def rf(self, mean_amp, var_amp, geometry_gain):
        """RF 功率增益 |a|^2 的均值与 alpha 分位（非中心卡方）。"""
        m = np.asarray(mean_amp, complex)
        v = np.asarray(var_amp, float)
        g = np.asarray(geometry_gain, float)
        mean = g * (np.abs(m) ** 2 + v)
        nc = np.where(v > 0, 2 * np.abs(m) ** 2 / np.maximum(v, 1e-30), 0.0)
        scale = g * v / 2.0
        low = scale * ncx2.ppf(self.alpha, 2, nc)
        high = scale * ncx2.ppf(1.0 - self.alpha, 2, nc)
        # v=0 是退化分布：分位必须等于确定值，不能返回 0。
        degenerate = v <= 0.0
        low = np.where(degenerate, mean, low)
        high = np.where(degenerate, mean, high)
        return dict(mean=mean, low=low, high=high, available=True)

    def vlc(self, mean_log, var_log, geometry_gain):
        """VLC 光增益 exp(z) 的均值与 alpha 分位（对数正态）。"""
        m = np.asarray(mean_log, float)
        v = np.asarray(var_log, float)
        g = np.asarray(geometry_gain, float)
        mean = g * np.exp(m + v / 2.0)
        q = norm.ppf(1.0 - self.alpha)
        low = g * np.exp(m - q * np.sqrt(np.maximum(v, 0.0)))
        high = g * np.exp(m + q * np.sqrt(np.maximum(v, 0.0)))
        return dict(mean=mean, low=low, high=high, available=True)
