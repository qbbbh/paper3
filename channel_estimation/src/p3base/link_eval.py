"""链路级 SINR / 速率 / 能耗。同模干扰全量计入，异模互不干扰。

RF :  S = p g_ij ,  I = sum_{b != a, same mode} p_b g_{sender_b, j} ,  N = N_RF

VLC（默认 ``linear``，与 dink10.m 一致）
    :  S = p h_ij,tp,rp ,  I = sum_{b != a} p_b h_{u_b -> j, tx_b, rx_a}
       N 由 ``noise.vlc_noise_linear_w`` 按 ``cfg.vlc.noise_model`` 给出

VLC（``square_law``，旧口径，仅用于复现）
    :  S = (R p h)^2 ,  I = sum (R p h_cross)^2 ,  N = 基础 + 散粒 [+ ISI]

速率 R = B log2(1 + SINR / GAP)，GAP 为各模路的实现余量。

EE 的分母是**电输入功率** ``P0_mode + p``（见 ``p3base/energy.py``），
不是发射功率本身；p 与 dink10.m 的 20–30 W / 1–30 W 同一口径。
"""
import math

import numpy as np

from .energy import link_electrical_power_w
from .noise import vlc_noise_base_w2, vlc_noise_linear_w, vlc_shot_coeff


def evaluate(links, powers, ports, rf_gain, vlc_gain, cfg, include_isi=None):
    """links: [(i, j, "RF"/"VLC")]，powers 与之一一对应。

    VLC 噪声由 ``cfg.vlc.noise_model`` 唯一决定；``include_isi`` 仅为旧调用签名
    保留，不再生效（传 True 也不再加 ISI）。要复现含 ISI 的旧结果，
    请把 ``vlc.noise_model`` 设为 ``physical_isi``。
    """
    links = [(int(i), int(j), str(m).upper()) for i, j, m in links]
    p = np.asarray(powers, float)
    if len(p) != len(links):
        raise ValueError("每条链路一个功率")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("功率必须有限非负")
    n = np.asarray(rf_gain).shape[0]
    k = len(links)
    rates = np.zeros(k)
    sinr = np.zeros(k)
    signal = np.zeros(k)
    interference = np.zeros(k)
    noise = np.zeros(k)
    occupied = set()
    for a, (i, j, mode) in enumerate(links):
        if mode not in ("RF", "VLC"):
            raise ValueError(f"未知模式 {mode}")
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError("车辆编号非法")
        if i in occupied or j in occupied:
            raise ValueError("端点冲突")
        occupied.update((i, j))
        cap = cfg.rf.power_max_w if mode == "RF" else cfg.vlc.power_max_w
        if p[a] > cap + 1e-10:
            raise ValueError(f"{mode} 功率超上限")

    linear = cfg.vlc.sinr_model == "linear"
    for a, (i, j, mode) in enumerate(links):
        if mode == "RF":
            s = p[a] * rf_gain[i, j]
            inter = 0.0
            for b, (u, v, m2) in enumerate(links):
                if b != a and m2 == mode:
                    inter += p[b] * rf_gain[u, j]
            nz = cfg.rf.noise_w
            bw, gap = cfg.rf.bandwidth_hz, cfg.rf.gap
        elif mode == "VLC":
            rho = cfg.vlc.responsivity
            eta = cfg.vlc.eo_efficiency
            tp, rp = int(ports["tx"][i, j]), int(ports["rx"][i, j])
            h_des = vlc_gain[i, j, tp, rp]
            optical = eta * p[a] * h_des
            if linear:
                # dink10.m 的线性形式，但光功率仍要先经过电->光效率 eta。
                s = eta * p[a] * h_des
                inter = 0.0
                for b, (u, v, m2) in enumerate(links):
                    if b != a and m2 == mode:
                        h_cross = vlc_gain[u, j, int(ports["tx"][u, v]), rp]
                        inter += eta * p[b] * h_cross
                        optical += eta * p[b] * h_cross
                nz = vlc_noise_linear_w(optical, cfg.vlc)
            else:
                s = (rho * eta * p[a] * h_des) ** 2
                inter = 0.0
                for b, (u, v, m2) in enumerate(links):
                    if b != a and m2 == mode:
                        h_cross = vlc_gain[u, j, int(ports["tx"][u, v]), rp]
                        inter += (rho * eta * p[b] * h_cross) ** 2
                        optical += eta * p[b] * h_cross
                # 噪声只由 ``cfg.vlc.noise_model`` 决定（含不含 ISI），与
                # ``block.sample_metrics`` 和 ``p3learn.allocation._rates_and_ee``
                # 走同一个函数，避免三处口径漂移。
                nz = vlc_noise_linear_w(optical, cfg.vlc)
            bw, gap = cfg.vlc.bandwidth_hz, cfg.vlc.gap
        else:
            raise ValueError(f"未知模式 {mode}")
        signal[a], interference[a], noise[a] = s, inter, nz
        sinr[a] = s / (nz + inter)
        rates[a] = bw * math.log2(1.0 + sinr[a] / gap)
    return dict(link_rates_mbps=rates / 1e6, link_rates_bps=rates,
                sinr=sinr, signal=signal, interference=interference, noise=noise,
                energy_j=float(cfg.estimation.block_dt_s * p.sum()),
                powers=p.copy(), links=links)


def network_metrics(result, n, block_dt_s, demand_mbps=None, cfg=None):
    """网络速率、最弱通信对、总能量与网络 EE。

    total_mbit 用"链路速率(Mbit/s) × 块时长(s)"直接累加，与旧管线的块级口径一致；
    网络 EE = 总传输量 / 总发送能量，不是逐块 EE 的平均。

    传入 ``cfg`` 时，``pair_ee`` 用**电输入功率**作分母（静态项 + p），并额外给出
    ``*_tx_only`` 的发射功率版本供对比。``network_ee_mbit_per_j`` 保持"总速率/总发射
    功率"的旧口径，``network_ee_electrical_mbit_per_j`` 才是电网口径。
    """
    rates = np.asarray(result["link_rates_mbps"], float)
    links = result["links"]
    if len(rates) == 0 or np.any(np.asarray(result['powers']) <= 0):
        raise ValueError('空配置或零功率须记为失败，不能计算有效通信对EE')
    powers = np.asarray(result['powers'], float)
    modes = [mode for _i, _j, mode in links]
    served = np.zeros(n)
    for (i, j, _m), r in zip(links, rates):
        served[i] += r
        served[j] += r  # endpoint participation only; network throughput counted once below
    total_mbit = float(rates.sum() * block_dt_s)
    energy = float(powers.sum() * block_dt_s)
    out = dict(total_mbit=total_mbit, energy_j=energy,
               network_ee_mbit_per_j=total_mbit / energy if energy > 0 else np.nan,
               min_link_rate_mbps=float(rates.min()) if len(rates) else np.nan,
               median_link_rate_mbps=float(np.median(rates)) if len(rates) else np.nan,
               served_vehicle_rate_mbps=served,
               pair_ee_tx_only_mbit_per_j=rates / powers,
               min_pair_ee_tx_only_mbit_per_j=float(np.min(rates / powers)) if len(rates) else np.nan)
    if cfg is not None:
        electric = link_electrical_power_w(powers, modes, cfg)
        out["pair_ee_mbit_per_j"] = rates / electric
        out["min_pair_ee_mbit_per_j"] = float(np.min(rates / electric))
        out["mean_pair_ee_mbit_per_j"] = float(np.mean(rates / electric))
        out["total_electrical_power_w"] = float(electric.sum())
        out["mean_pair_ee_tx_only_mbit_per_j"] = float(np.mean(rates / powers))
        out["network_ee_electrical_mbit_per_j"] = (
            total_mbit / float(electric.sum() * block_dt_s) if electric.sum() > 0 else np.nan)
        out["electrical_energy_j"] = float(electric.sum() * block_dt_s)
    if demand_mbps is not None:
        out["demand_satisfied"] = rates >= demand_mbps
    return out
