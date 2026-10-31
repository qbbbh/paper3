"""块级因果信道估计：把导频、到达时延、遮挡与卡尔曼-MMSE 串起来。

因果性规则（不可放宽）：
- 时刻 p 发出的导频，最早在 p + delay 才可用；
- 块开始时刻 t 的动作只能使用 p + delay <= t 的观测；
- 块内到达的观测只更新后台估计，供下一块使用，不得追溯修改本块动作。

时间一致性规则（2026-10-23 修正）：
- 滤波器**自身状态**永远停在"最后一次被应用观测的时刻"，绝不预测到查询时刻后停留；
- 查询时刻 t 的信念由该后验**预测一个副本**得到，副本用完即弃；
- 内部传播按不超过 `fine_dt_s` 的子步长进行，因此外部以 1 ms 还是 100 ms 的粒度调用，
  只要观测集合相同，结果就相同。

旧实现把状态预测到查询时刻并把 filt_time 设成 t；当导频周期小于信息时延时，
下一个待处理导频的时刻早于 filt_time，predict 的负 dt 被静默跳过，随后又把同一段
老化重算一遍，造成"同一批观测、不同调用粒度得到不同结果"。

观测提供者 provider 必须实现：
    provider.rf(pilot_time)  -> (y_complex (n,n), obs_var (n,n), valid (n,n))
    provider.vlc(pilot_time) -> (y_log (n,n,2,2), obs_var (n,n,2,2), valid (n,n,2,2))
"""
import numpy as np

from .kalman_mmse import KalmanMMSE, PredictiveBelief
from .pilots import PilotPlan
from .rf_two_state import TwoStateRF


class ChannelEstimator:
    def __init__(self, n, cfg, n_ports=2):
        self.n = int(n)
        self.cfg = cfg
        self.n_ports = int(n_ports)
        self.plan = PilotPlan.for_n(self.n, cfg)
        e = cfg.estimation
        if e.rf_state == "two_state":
            self.rf = TwoStateRF(cfg.rf, e)
        else:
            self.rf = KalmanMMSE(e.rf_state, tau_s=cfg.rf.fading_coherence_s,
                                 process_var=e.rf_process_var or None,
                                 stationary_var=1.0)
        self.vlc = KalmanMMSE(e.vlc_state, tau_s=cfg.vlc.fading_coherence_s,
                              process_var=e.vlc_process_var or None,
                              stationary_mean=-0.5 * cfg.vlc.log_stationary_sd ** 2,
                              stationary_var=cfg.vlc.log_stationary_sd ** 2)
        self.rf.reset((self.n, self.n))
        self.vlc.reset((self.n, self.n, self.n_ports, self.n_ports))
        # 后验所在的时刻（最后一次被应用观测的时刻），不是查询时刻。
        self.state_time = {"rf": 0.0, "vlc": 0.0}
        # 导频时刻用整数索引 k*period 生成，绝不累加：累加误差会让 k=8001 之后的
        # 请求落在缓存键之外（KeyError）或与真值场对不上，从而静默漏掉一次观测。
        self.pilot_index = {"rf": 0, "vlc": 0}
        self.next_pilot = {"rf": 0.0, "vlc": 0.0}
        self.rf_last_used = np.full((self.n, self.n), -np.inf)
        self.vlc_last_used = np.full((self.n, self.n, self.n_ports, self.n_ports), -np.inf)
        self.belief_builder = PredictiveBelief(e.tail_probability)
        self._mmse = bool(e.mmse_stage)
        self._rf_prior_set = False
        # 内部传播的子步长上限：保证调用粒度不影响数值结果。
        self.max_step_s = float(getattr(e, "fine_dt_s", 1e-3))
        self._rf_doppler = 0.0
        self._rf_tau = None

    @property
    def mmse_stage(self):
        return self._mmse

    # ---- 推进与更新 ----------------------------------------------------
    def advance(self, t, provider, doppler_hz=None, rf_prior_var=None, rf_tau=None):
        """推进到时刻 t，返回时刻 t 的因果信念。

        rf_prior_var: (n,n) 平稳方差，把阴影衰落并入复合复幅度状态。
        rf_tau: (n,n) 逐链路相干时间，由相对速度给出；必须与真值场同一公式。
        VLC 的对数状态平稳方差已由配置给出，不需要逐链路。
        """
        t = float(t)
        if rf_prior_var is not None:
            prior = np.asarray(rf_prior_var, float)
            # 平稳方差必须先于 reset 写入：否则首次调用走后验初始化分支时它仍是标量 1.0，
            # 而之后的调用才被设成 prior，等于让"调用次数"改变了过程噪声（旧缺陷）。
            self.rf.stationary_var = prior
            if not self._rf_prior_set:
                self.rf.reset(prior.shape, var=prior)
                self._rf_prior_set = True
        self._rf_doppler = 0.0 if doppler_hz is None else doppler_hz
        self._rf_tau = rf_tau
        self._consume(self.rf, "rf", t, provider, self.plan.rf_period_s,
                      self.plan.rf_delay_s, doppler_hz, rf_tau)
        self._consume(self.vlc, "vlc", t, provider, self.plan.vlc_period_s,
                      self.plan.vlc_delay_s, None, None)
        return self.belief(t)

    def _consume(self, filt, mode, t, provider, period, delay, doppler, tau):
        """按时间顺序消费所有已到达的导频；后验停在最后一次更新的时刻。"""
        while self.pilot_index[mode] * period + delay <= t + 1e-12:
            p = self.pilot_index[mode] * period
            # 观测 y 描述的是导频时刻 p 的信道。dt = p - state_time >= 0 恒成立，
            # 因为后验从不越过尚未消费的导频时刻。
            self._predict_inplace(filt, self.state_time[mode], p, doppler, tau)
            self.state_time[mode] = p
            y, obs_var, valid = getattr(provider, mode)(p)
            filt.update(y, obs_var, valid)
            mask = np.asarray(valid, bool) & np.isfinite(obs_var)
            if self._mmse and filt.state not in ("log_intensity", "two_state"):
                # 只在确实有观测时收缩；缺测（inf）必须保持上一时刻的均值。
                filt.mean = filt.mean * np.where(
                    mask, KalmanMMSE.mmse_factor(filt.stationary_var, obs_var), 1.0)
            if mode == "rf":
                self.rf_last_used = np.where(mask, p, self.rf_last_used)
            else:
                self.vlc_last_used = np.where(mask, p, self.vlc_last_used)
            self.pilot_index[mode] += 1
            self.next_pilot[mode] = self.pilot_index[mode] * period

    def _predict_inplace(self, filt, t0, t1, doppler, tau):
        while t1 - t0 > 1e-12:
            dt = min(self.max_step_s, t1 - t0)
            filt.predict(dt, doppler, tau)
            t0 += dt

    def _predict_copy(self, filt, t0, t1, doppler, tau):
        if filt.state == "two_state":
            # 两状态的后验是 (a, L) 两个量与它们的方差，不能只拿复合一阶矩传播。
            clone = filt.clone()
            while t1 - t0 > 1e-12:
                dt = min(self.max_step_s, t1 - t0)
                clone.predict(dt, doppler, tau)
                t0 += dt
            return clone.moments()
        mean, var = filt.mean, filt.var
        while t1 - t0 > 1e-12:
            dt = min(self.max_step_s, t1 - t0)
            mean, var = filt.propagate(mean, var, dt, doppler, tau)
            t0 += dt
        return mean, var

    # ---- 信念 ----------------------------------------------------------
    def belief(self, t=None):
        if t is None:
            t = max(self.state_time.values())
        # 对外给的是"后验在当前时刻的预测副本"，滤波器自身状态不被搬动。
        rf_mean, rf_var = self._predict_copy(self.rf, self.state_time["rf"], t,
                                             self._rf_doppler, self._rf_tau)
        vlc_mean, vlc_var = self._predict_copy(self.vlc, self.state_time["vlc"], t, None, None)
        rf = self.belief_builder.rf(rf_mean, rf_var, 1.0)
        vlc = self.belief_builder.vlc(vlc_mean, vlc_var, 1.0)
        rf_age = np.where(np.isfinite(self.rf_last_used), t - self.rf_last_used, np.inf)
        vlc_age = np.where(np.isfinite(self.vlc_last_used), t - self.vlc_last_used, np.inf)
        return dict(
            time_s=t,
            rf_state_mean=rf_mean.copy(), rf_state_var=np.asarray(rf_var, float).copy(),
            rf_gain_mean=rf["mean"], rf_gain_low=rf["low"], rf_gain_high=rf["high"],
            rf_age_s=rf_age, rf_missing=~np.isfinite(rf_age),
            vlc_state_mean=vlc_mean.copy(), vlc_state_var=np.asarray(vlc_var, float).copy(),
            vlc_gain_mean=vlc["mean"], vlc_gain_low=vlc["low"], vlc_gain_high=vlc["high"],
            vlc_age_s=vlc_age, vlc_missing=~np.isfinite(vlc_age),
            rf_state=self.rf.state, vlc_state=self.vlc.state,
            pilot_plan=self.plan.as_dict(),
        )

    # ---- 与几何增益合成 ------------------------------------------------
    def gain_belief(self, geometry_rf, geometry_vlc, t=None,
                    rf_candidate=None, vlc_candidate=None):
        """把估计出的归一化状态乘上确定性几何增益，得到块级信道信念。

        候选资格必须由调用方按距离/可见性显式传入；用 gain>0 反推会丢掉
        100 m / 45 m 范围限制，把交叉干扰增益误当作候选边。
        """
        b = self.belief(t)
        b["rf_gain_mean"] = np.asarray(geometry_rf, float) * b["rf_gain_mean"]
        b["rf_gain_low"] = np.asarray(geometry_rf, float) * b["rf_gain_low"]
        b["rf_gain_high"] = np.asarray(geometry_rf, float) * b["rf_gain_high"]
        b["vlc_gain_mean"] = np.asarray(geometry_vlc, float) * b["vlc_gain_mean"]
        b["vlc_gain_low"] = np.asarray(geometry_vlc, float) * b["vlc_gain_low"]
        b["vlc_gain_high"] = np.asarray(geometry_vlc, float) * b["vlc_gain_high"]
        if rf_candidate is None:
            rf_candidate = np.asarray(geometry_rf, float) > 0
        if vlc_candidate is None:
            vlc_candidate = np.asarray(geometry_vlc, float).max(axis=(-1, -2)) > 0
        b["rf_candidate"] = np.asarray(rf_candidate, bool)
        b["vlc_candidate"] = np.asarray(vlc_candidate, bool)
        return b


def relative_doppler_hz(prev_xy, xy, dt, frequency_hz):
    """径向多普勒频移估计，供 paper_doppler / complex_ar1 使用。

    prev_xy、xy 形状 (n,2)。返回 (n,n)，只对 i!=j 有意义。
    """
    v = (np.asarray(xy, float) - np.asarray(prev_xy, float)) / max(float(dt), 1e-12)
    delta = np.asarray(xy, float)[:, None, :] - np.asarray(xy, float)[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    unit = delta / np.maximum(dist, 1e-9)[..., None]
    radial = (v[:, None, :] - v[None, :, :]) * unit           # 相对速度沿视线分量
    return radial.sum(-1) / 299_792_458.0 * float(frequency_hz)
