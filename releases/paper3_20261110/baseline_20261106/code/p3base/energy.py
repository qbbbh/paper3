"""逐链路能耗分母：静态功耗 + 电输入功率。

``p`` 是**电输入功率**，与 dink10.m 的 VLC 20–30 W / RF 1–30 W 同一口径，
所以分母直接相加，不再除以电光效率：

    P_elec(RF)  = rf.circuit_power_w  + p
    P_elec(VLC) = vlc.circuit_power_w + p

两边的静态项都是**通信有源前端的固定电耗**，不是照明：

- VLC 1.745 W = DAC 1.400 W + TA 0.345 W（Mayahi 2022 Table 5.2，路灯设备模型，
  非车载实测；TA 已含在内，不再重复计）；
- RF 0.1159 W：Wang et al., Scientific Reports 14:9341 (2024), Table 2（P_c = 115.9 mW），
  该文的仿真取值；原文另设功放效率，不是车载实测。

旧版 VLC 用 20 W（整灯照明电输入）,把"灯本来就要付的电费"整体算给通信；
本课题只算通信功能引入的发射端能耗，所以 2026-10-26 改为有源前端口径。

EE 的唯一定义是 ``rate_k / P_elec_k``，再对 k 取算术平均。
"""
import numpy as np


def static_power_w(mode, cfg) -> float:
    """单个模式的静态功耗（W）。"""
    return cfg.rf.circuit_power_w if str(mode).upper() == "RF" else cfg.vlc.circuit_power_w


def link_electrical_power_w(powers, modes, cfg):
    """每条链路的电输入功率（W），与 powers 同形。

    ``modes`` 是与 powers 等长的模式序列（"RF"/"VLC"）。
    """
    p = np.asarray(powers, float)
    flat_modes = list(modes)
    if p.ndim != 1 or len(flat_modes) != len(p):
        raise ValueError("modes 必须与 powers 等长")
    is_rf = np.array([str(mode).upper() == "RF" for mode in flat_modes])
    # 动作 p 是电输入功率，功放效率已经在信号项里，分母直接相加。
    return np.where(is_rf, cfg.rf.circuit_power_w, cfg.vlc.circuit_power_w) + p


def link_electrical_energy_j(powers, modes, cfg, duration_s):
    return float(duration_s) * link_electrical_power_w(powers, modes, cfg)


def total_electrical_power_w(powers, modes, cfg) -> float:
    return float(link_electrical_power_w(powers, modes, cfg).sum())
