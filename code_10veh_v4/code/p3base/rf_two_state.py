"""RF 快慢分离的两状态估计：慢阴影 + 快小尺度。

真值（fading.FadingField）本来就是两个过程相乘：

    h = sqrt(10**(X/10)) * a = s * a,
    a  零均值单位方差复高斯 AR(1)，相关常数 T_c = 0.423*lambda/v_rel；
    X  阴影 dB，AR(1)，相关常数 D / v_rel（D = shadow_decorrelation_m）。

单状态 complex_ar1 把二者压成一个复幅度、只用一个短相关时间，于是每隔几毫秒
就把均值收缩到 0、方差涨到复合平稳方差——阴影信息（本可延续几百毫秒）被一起丢掉。
本模块保留两状态结构，但**不**宣称能把误差降到某个理论值：快衰落在一个 T_c 之后
仍然不可预测，可降的只是阴影那一部分。

观测方程（每个导频、每条链路，pilot-normalized gain）：

    y = s * a + n,            n ~ CN(0, R)

对 (a, X) 的联合观测是非线性的，这里按"强度通道 + 幅度通道"做两次标量卡尔曼更新，
并显式写出被忽略的相关项：

1. 强度通道（定阴影）：z = ln|y|^2 = L + ln|a|^2 + e,  L = ln(s^2) = (ln10/5) * (X/2)
   对 Rayleigh 衰落，ln|a|^2 的均值 -gamma、方差 pi^2/6 是精确已知的常数，
   因此 z + gamma 是 L 的无偏观测量，观测方差 pi^2/6 + (R/|y|^2)^2。
   这一步不把 X 当成无噪可读量：单个 |y|^2 的 dB 散布约 5.57 dB，比 3/4 dB 的阴影
   标准差大得多，必须靠多次导频由卡尔曼增益累积。

2. 幅度通道（定 a）：a_obs = y / s，方差 R/s^2 加上 s 自身不确定性带来的
   |a_obs|^2 (exp(P_LL) - 1)。

两步之间的 a-L 交叉相关被丢弃（近似），因此这不是联合最优估计器；它的作用是
把"阴影可预测、快衰落不可预测"这一结构显式保留下来，便于误差归因。

对外接口与 KalmanMMSE 对齐：predict / propagate / update / reset / mean / var /
state，另外提供 clone() 供"查询时刻预测副本"使用。
"""
import numpy as np

from .fading import SPEED_OF_LIGHT

EULER_GAMMA = 0.5772156649015329
LOG_EXP_VAR = np.pi ** 2 / 6.0          # Var(ln|a|^2)，a ~ CN(0,1)
LN10_OVER_10 = np.log(10.0) / 10.0      # L = ln(s^2) = ln(10^(X/10)) = (ln10/10) * X


class TwoStateRF:
    state = "two_state"

    def __init__(self, rf_cfg, est_cfg=None):
        self.cfg = rf_cfg
        self.est = est_cfg
        self.a = None                    # 复合快幅度（归一化到阴影之外）
        self.P_aa = None
        self.L = None                    # 阴影的对数功率 ln(s^2)
        self.P_LL = None
        self.sigma_L2 = None             # 阴影平稳方差（对数功率域），逐链路
        self._pending_prior = None

    # ---- 平稳方差 / 初始化 ---------------------------------------------
    @property
    def stationary_var(self):
        v = np.exp(self.sigma_L2) if self.sigma_L2 is not None else 1.0
        return v

    @stationary_var.setter
    def stationary_var(self, prior):
        p = np.asarray(prior, float)
        # E[10^(X/10)] = exp(k^2 sigma_X^2 / 2) => sigma_L^2 = k^2 sigma_X^2 = 2 ln(prior)
        self.sigma_L2 = np.maximum(2.0 * np.log(np.maximum(p, 1e-12)), 0.0)

    def reset(self, shape, mean=None, var=None):
        if var is not None:
            self.stationary_var = var
        elif self.sigma_L2 is None or np.shape(self.sigma_L2) != tuple(shape):
            self.sigma_L2 = np.full(shape, self._default_sigma_L2())
        self.a = np.zeros(shape, complex)
        self.P_aa = np.ones(shape, float)
        self.L = np.zeros(shape, float)
        self.P_LL = np.array(self.sigma_L2, float, copy=True)
        return self

    def _default_sigma_L2(self):
        sigma_db = float(getattr(self.cfg, "los_shadow_sigma_db", 3.0))
        return (np.log(10.0) / 10.0) ** 2 * sigma_db ** 2

    # ---- 预测 -----------------------------------------------------------
    def _rhos(self, dt, tau):
        """快、慢两个 AR 系数。慢系数与真值同一公式 exp(-dt*v_rel/D)。"""
        if tau is None:
            rho_f = np.exp(-dt / self.cfg.fading_coherence_s)
            rho_s = np.ones_like(np.asarray(rho_f, float)) if np.ndim(rho_f) else 1.0
            return rho_f, rho_s
        tau = np.asarray(tau, float)
        rho_f = np.exp(-dt / np.maximum(tau, 1e-9))
        lam = SPEED_OF_LIGHT / (self.cfg.frequency_ghz * 1e9)
        v_rel = self.cfg.doppler_coherence_coeff * lam / np.maximum(tau, 1e-9)
        rho_s = np.exp(-dt * v_rel / max(self.cfg.shadow_decorrelation_m, 1e-9))
        return rho_f, rho_s

    def propagate(self, a, P_aa, L, P_LL, dt, doppler_hz=0.0, tau=None):
        """式 (6) 的两状态版本，纯函数。"""
        dt = float(dt)
        if dt <= 0:
            return a, P_aa, L, P_LL
        rho_f, rho_s = self._rhos(dt, tau)
        if doppler_hz is None:           # 与 KalmanMMSE.propagate 同一约定
            doppler_hz = 0.0
        phase = np.exp(-2j * np.pi * np.asarray(doppler_hz, float) * dt)
        a = rho_f * phase * a
        P_aa = rho_f ** 2 * P_aa + (1.0 - rho_f ** 2)
        L = rho_s * L
        P_LL = rho_s ** 2 * P_LL + (1.0 - rho_s ** 2) * self.sigma_L2
        return a, P_aa, L, P_LL

    def predict(self, dt, doppler_hz=0.0, tau=None):
        self.a, self.P_aa, self.L, self.P_LL = self.propagate(
            self.a, self.P_aa, self.L, self.P_LL, dt, doppler_hz, tau)
        return self

    def clone(self):
        out = object.__new__(TwoStateRF)
        out.__dict__ = dict(self.__dict__)
        for k in ("a", "P_aa", "L", "P_LL", "sigma_L2"):
            v = getattr(self, k)
            setattr(out, k, None if v is None else np.array(v, copy=True))
        return out

    # ---- 更新 -----------------------------------------------------------
    def update(self, y, obs_var, valid=None):
        y = np.asarray(y, complex)
        R = np.asarray(obs_var, float)
        mask = np.isfinite(R) & np.isfinite(y)
        if valid is not None:
            mask &= np.asarray(valid, bool)
        power = np.abs(y) ** 2
        mask &= power > 0.0
        if not mask.any():
            return self

        # 1) 强度通道：定阴影 L = ln(s^2)
        z = np.where(mask, np.log(np.where(power > 0, power, 1.0)) + EULER_GAMMA, 0.0)
        # 只有 mask 内的链路参与；先把 inf/0 换成安全值，避免无效链路溢出成 inf*0=nan。
        R_safe = np.where(mask, R, 1.0)
        p_safe = np.where(mask & (power > 0), power, 1.0)
        var_z = LOG_EXP_VAR + np.where(mask, (R_safe / p_safe) ** 2, 0.0)
        k_l = np.where(mask, self.P_LL / (self.P_LL + var_z), 0.0)
        self.L = np.where(mask, self.L + k_l * (z - self.L), self.L)
        self.P_LL = np.where(mask, (1.0 - k_l) * self.P_LL, self.P_LL)

        # 2) 幅度通道：定尺度 a_obs = y / s
        s = np.exp(0.5 * self.L)
        a_obs = np.where(mask, y / np.where(s > 0, s, 1.0), 0.0)
        R_a = np.where(mask, R, np.inf)
        var_a = np.where(mask,
                         R_a / np.maximum(s ** 2, 1e-300)
                         + np.abs(a_obs) ** 2 * np.expm1(np.maximum(self.P_LL, 0.0)),
                         np.inf)
        k_a = np.where(mask, self.P_aa / (self.P_aa + var_a), 0.0)
        self.a = np.where(mask, self.a + k_a * (a_obs - self.a), self.a)
        self.P_aa = np.where(mask, (1.0 - k_a) * self.P_aa, self.P_aa)
        return self

    # ---- 对外一阶矩（复合复幅度 h = s*a）--------------------------------
    def moments(self):
        """返回 (E[h], Var[h])，供 PredictiveBelief 使用。"""
        e_s = np.exp(0.5 * self.L + self.P_LL / 8.0)
        e_s2 = np.exp(self.L + 0.5 * self.P_LL)
        mean = e_s * self.a
        e_h2 = e_s2 * (np.abs(self.a) ** 2 + self.P_aa)
        var = np.maximum(e_h2 - np.abs(mean) ** 2, 0.0)
        return mean, var

    @property
    def mean(self):
        return self.moments()[0]

    @property
    def var(self):
        return self.moments()[1]
