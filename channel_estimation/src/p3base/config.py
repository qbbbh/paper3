"""基础层参数。所有参数与车辆数无关，10/50/100 车共用同一套配置。

射频路损采用 3GPP TR 37.885 城市 V2V 模型（见 p3base/channel.py 的注释），
载频单位 GHz、三维距离单位 m。旧 40 m 场景的 47.86+27.5log10(d) 单斜率模型
仅作为可选的复现开关保留，不作为新场景默认值。
"""
from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class VehicleConfig:
    length_m: float = 4.5
    width_m: float = 1.8
    height_m: float = 1.5
    port_height_m: float = 1.0          # LED/PD 离地高度，与旧场景诊断口径一致
    port_longitudinal_frac: float = 0.5  # 端口位于车长两端的保险杠中心


@dataclass(frozen=True)
class RFConfig:
    candidate_range_m: float = 100.0
    bandwidth_hz: float = 22e6
    gap_db: float = 3.0
    power_max_w: float = 2.0          # 数据链路功率上限，与 1 mW–2 W 的当前设定一致
    frequency_ghz: float = 5.89
    noise_reference_dbm: float = -95.0
    noise_reference_bandwidth_hz: float = 10e6
    # "tr37885" 为 3GPP 模型；"legacy_47p86" 只用于复现旧 40 m 场景
    pathloss_model: str = "tr37885"
    los_shadow_sigma_db: float = 3.0
    nlos_shadow_sigma_db: float = 4.0
    shadow_decorrelation_m: float = 10.0
    # 小尺度相干时间按多普勒推：T_c = doppler_coherence_coeff * lambda / v_rel。
    # 不能用全链路同一个固定值，否则等速行驶与对向交会得到相同的衰落速率。
    doppler_coherence_coeff: float = 0.423      # Clarke 模型的 0.423/f_d
    fading_coherence_s: float = 0.020           # 未知相对速度时的回退值
    fading_coherence_min_s: float = 1e-3
    fading_coherence_max_s: float = 0.5
    legacy_path_loss_1m_db: float = 47.86
    legacy_path_loss_exponent: float = 2.75
    # 每条 RF 链路固定的收发电路功耗（电输入，W）。EE 的分母是
    # ``circuit_power_w + p``：p 是电输入功率，与 dink10.m 的 1--30 W 同一口径。
    #
    # 出处：Wang, Liu, Dai & He, "Joint cooperative caching and power control for
    # UAV-assisted internet of vehicles", Scientific Reports 14:9341 (2024),
    # DOI 10.1038/s41598-024-59823-9, Table 2: P_c = 115.9 mW。这是该文的仿真取值，
    # 原文另设功放效率，采用它不代表本模型的完整能耗与原文相同，也不能写成
    # 车载硬件实测。旧版 0.3 W 是测试假设，已废弃。
    circuit_power_w: float = 0.1159

    @property
    def noise_w(self) -> float:
        return (10.0 ** ((self.noise_reference_dbm - 30.0) / 10.0)
                * self.bandwidth_hz / self.noise_reference_bandwidth_hz)

    @property
    def gap(self) -> float:
        return 10.0 ** (self.gap_db / 10.0)


@dataclass(frozen=True)
class VLCConfig:
    candidate_range_m: float = 45.0
    bandwidth_hz: float = 50e6
    gap_db: float = 5.4
    power_max_w: float = 2.0          # 与 RF 同一数据功率域：1 mW–2 W
    area_m2: float = 1.5e-4
    responsivity: float = 0.45
    background_current_a: float = 0.0051
    fov_half_deg: float = 60.0
    zeta_rad: float = 0.1585
    epsilon: float = 0.0175
    temperature_k: float = 298.0
    voltage_gain: float = 10.0
    capacitance_per_area_f_m2: float = 1.12e-6
    noise_factor: float = 1.5
    transconductance_s: float = 0.03
    c2: float = 0.562
    c3: float = 0.0868
    electron_charge_c: float = 1.602e-19
    boltzmann_j_k: float = 1.380649e-23
    min_port_distance_m: float = 1.0
    log_stationary_sd: float = 0.25   # 对数域乘性衰落的平稳标准差
    fading_coherence_s: float = 0.5

    # ---- 功率维与 SINR 口径（2026-10-25，2026-10-26 改静态项） ----
    # p 是电输入功率，EE 分母 = circuit_power_w + p。
    #
    # 1.745 W = 发射端**通信有源前端**固定电耗：DAC 1400 mW + TA 345 mW。
    # 出处：Mayahi 2022 博士论文 Table 5.2（PDF p69，
    # https://pepite-depot.univ-lille.fr/LIBRE/EDMADIS/2022/2022ULILB046.pdf），
    # 与 DJ 线 ``physics/energy.py`` 的 ``PC_VLC=1.745`` 同源。
    # **该表是路灯基础设施 VLC 设备模型，不是车载发送模块实测**，所以只能作为
    # 有出处的工程参考参数，不能写成"车载 VLC 电路功耗实测为 1.745 W"。
    #
    # 旧版 20 W 是整灯照明电输入（dink10.m 的 20--30 W 下限），含义完全不同：
    # 按它计，灯本来就要付的电费被整体算给通信。本课题研究的是"通信功能引入的
    # 发射端能耗"，车辆原本就需要的照明不随 RF/VLC 模式选择计入通信，故改用 1.745 W。
    # TA 只在这里出现一次（DAC 与 TA 已合并在 1.745 W 内），不要重复计入。
    circuit_power_w: float = 1.745
    # 电->光效率，只用于把电功率换成到达光功率去算散粒噪声；SINR 里的 h 已把
    # 电输入映射到接收端电域，不再重复乘。
    eo_efficiency: float = 0.25
    n0_w_per_hz: float = 1e-21
    # "linear"      dink10.m:  S = h*p,  I = sum h_cross*p      （默认）
    # "square_law"  旧实现:    S = (R p h)^2, I = sum (R p h)^2  （仅用于复现）
    sinr_model: str = "linear"
    # "n0_only"     严格照原文：N = N0_VLC*B
    # "physical"    热噪声 + 背景散粒 + 信号散粒（不含 ISI，默认）
    # "physical_isi" 再加 (R*P_opt)^2，即上一版会把 SINR 钉死在 1 的口径
    noise_model: str = "physical"

    def __post_init__(self):
        if self.sinr_model not in ("linear", "square_law"):
            raise ValueError(f"未知 VLC SINR 口径 {self.sinr_model}")
        if self.noise_model not in ("n0_only", "physical", "physical_isi"):
            raise ValueError(f"未知 VLC 噪声口径 {self.noise_model}")
        if not (0.0 < self.eo_efficiency <= 1.0):
            raise ValueError("eo_efficiency 必须在 (0, 1]")
        if self.circuit_power_w < 0.0 or self.n0_w_per_hz <= 0.0:
            raise ValueError("circuit_power_w 非负且 n0_w_per_hz 为正")

    @property
    def bias_power_w(self) -> float:
        """旧字段名（20 W 照明偏置时期），现在等价于 ``circuit_power_w``。

        只用于读旧 run 的 summary / 旧调用；新代码一律用 ``circuit_power_w``。
        """
        return self.circuit_power_w

    @property
    def gap(self) -> float:
        """实现余量（线性），与 RFConfig.gap 同名同义。"""
        return 10.0 ** (self.gap_db / 10.0)

    @property
    def noise_w(self) -> float:
        # 历史字段：旧代码把它当作 gap 用（数值上确实等于 gap）。新代码请用
        # ``gap``；VLC 噪声请用 ``p3base.noise.vlc_noise_linear_w``。
        return 10.0 ** (self.gap_db / 10.0)

    @property
    def aperture_diameter_m(self) -> float:
        import math
        return math.sqrt(4.0 * self.area_m2 / math.pi)


@dataclass(frozen=True)
class EstimationConfig:
    """导频与估计器配置。

    rf_state / vlc_state 选择状态演化模型：
      "paper_doppler"  论文式 A_s = exp(-j2πvΔt/λ)，模值恒为 1
      "complex_ar1"    复幅度 AR(1)，含多普勒相位，模值随机游走
      "log_intensity"  对数光强实状态 AR(1)（VLC 专用）
      "two_state"      仅 RF：慢阴影 L=ln(s^2) 与快复幅度 a 分开滤波（见 rf_two_state.py）
    论文对应的默认值是 paper_doppler；100 ms 块协议推荐 complex_ar1/log_intensity，
    因为纯相位旋转无法跟踪车辆接近/远离造成的幅度变化。
    """
    fine_dt_s: float = 1e-3
    block_dt_s: float = 0.1

    rf_pilot_period_s: float = 0.001
    rf_pilot_delay_s: float = 0.002
    rf_pilot_power_w: float = 2.0      # 导频功率取数据功率上限
    rf_pilot_max_slot_s: float = 1e-3
    # 只决定观测噪声的实现种子，不构成 L 次平均；能量已由 slot 时长给出。
    rf_pilot_combined_symbols: int = 16
    rf_state: str = "two_state"
    rf_process_var: float = 0.0        # 0 表示由 AR(1) 时间常数自动给出
    # 导频空口开销上限：slot = min(max_slot, budget * period / n)。
    # 固定预算只统一名义观测资源；实际精度还取决于几何、衰落、时延和估计器。
    pilot_overhead_budget: float = 0.20

    vlc_pilot_period_s: float = 0.050
    vlc_pilot_delay_s: float = 0.005
    vlc_pilot_power_w: float = 2.0
    vlc_pilot_max_slot_s: float = 1e-3
    # 对数域观测噪声下限（标准差）。物理噪声在 1 ms 导频下远低于此值，
    # 该下限代表标定误差/对准误差/模型误差，本实现将其作为独立高斯观测噪声，不能视为不可消除的误差界。
    vlc_log_obs_sd: float = 0.12
    vlc_observable_gain_min: float = 1e-8
    vlc_state: str = "log_intensity"
    vlc_process_var: float = 0.0

    # 论文式 (12) 的标量收缩。默认关闭：本实现的 (7) 已是完整 Kalman 后验，
    # 再用同一观测噪声收缩一次等于把噪声扣两遍，且缺测时会把均值乘成 0。
    # 复现论文时用 --mmse 打开。
    mmse_stage: bool = False
    tail_probability: float = 0.05

    def __post_init__(self):
        allowed = {"rf_state": ("paper_doppler", "complex_ar1", "two_state"),
                   "vlc_state": ("paper_doppler", "complex_ar1", "log_intensity")}
        for name, ok in allowed.items():
            if getattr(self, name) not in ok:
                raise ValueError(f"{name} 取值非法")
        if self.vlc_state == "log_intensity" and self.rf_state == "log_intensity":
            raise ValueError("RF 状态不能使用 log_intensity")
        if not 0.0 < self.block_dt_s:
            raise ValueError("block_dt_s 必须为正")


@dataclass(frozen=True)
class FoundationConfig:
    rf: RFConfig = RFConfig()
    vlc: VLCConfig = VLCConfig()
    vehicle: VehicleConfig = VehicleConfig()
    estimation: EstimationConfig = EstimationConfig()
    speed_limit_mps: float = 8.3333333333

    def to_dict(self):
        return {
            "rf": asdict(self.rf),
            "vlc": asdict(self.vlc),
            "vehicle": asdict(self.vehicle),
            "estimation": asdict(self.estimation),
            "speed_limit_mps": self.speed_limit_mps,
        }

    def with_(self, **parts):
        """返回替换了若干子配置的新配置，例如 cfg.with_(rf=replace(...))."""
        return replace(self, **parts)
