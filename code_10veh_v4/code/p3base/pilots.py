"""名义导频能量预算。slot=min(max_slot, budget*period/N)。
采用周期批量观测抽象，不实现各车时隙偏移或MAC数据资源复用。
RF 1 ms时单车slot为20/4/2 us（N=10/50/100）；功率固定2 W。
RF与VLC导频发送能量分别记录，link_eval毛速率不自动扣除导频。
规模影响积分能量，但精度还受链路几何、老化、噪声和估计方法影响。
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class PilotPlan:
    n: int
    rf_period_s: float
    rf_slot_s: float
    rf_delay_s: float
    rf_power_w: float
    rf_overhead: float
    rf_energy_per_vehicle_per_s: float
    vlc_period_s: float
    vlc_slot_s: float
    vlc_delay_s: float
    vlc_power_w: float
    vlc_overhead: float
    vlc_energy_per_vehicle_per_s: float

    @classmethod
    def for_n(cls, n, cfg):
        e = cfg.estimation
        n = int(n)
        if n <= 0:
            raise ValueError("车辆数必须为正")
        budget = float(e.pilot_overhead_budget)
        rf_slot = min(e.rf_pilot_max_slot_s, budget * e.rf_pilot_period_s / n)
        vlc_slot = min(e.vlc_pilot_max_slot_s, budget * e.vlc_pilot_period_s / n)
        return cls(
            n=n,
            rf_period_s=e.rf_pilot_period_s, rf_slot_s=rf_slot,
            rf_delay_s=e.rf_pilot_delay_s, rf_power_w=e.rf_pilot_power_w,
            rf_overhead=n * rf_slot / e.rf_pilot_period_s,
            rf_energy_per_vehicle_per_s=e.rf_pilot_power_w * rf_slot / e.rf_pilot_period_s,
            vlc_period_s=e.vlc_pilot_period_s, vlc_slot_s=vlc_slot,
            vlc_delay_s=e.vlc_pilot_delay_s, vlc_power_w=e.vlc_pilot_power_w,
            vlc_overhead=n * vlc_slot / e.vlc_pilot_period_s,
            vlc_energy_per_vehicle_per_s=e.vlc_pilot_power_w * vlc_slot / e.vlc_pilot_period_s,
        )

    def as_dict(self):
        return asdict(self)

    def observable_pairs(self):
        """每个周期内能被谁观测到：第 i 辆车发导频，其余 n-1 辆在各自波束内接收。

        这里只返回规模，几何可见性由信道层的遮挡结果决定。
        """
        return self.n * (self.n - 1)


def pilot_times(start_s, end_s, period_s):
    """[start_s, end_s) 内的导频时刻，含 start_s 本身。"""
    if period_s <= 0:
        raise ValueError("导频周期必须为正")
    k0 = int(np.ceil(start_s / period_s - 1e-12))
    times = []
    k = k0
    while k * period_s < end_s - 1e-12:
        times.append(k * period_s)
        k += 1
    return times
