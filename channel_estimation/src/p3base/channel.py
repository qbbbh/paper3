"""基础信道：RF 路损与 VLC 光增益、几何候选、建筑/车辆遮挡。

RF 路损（3GPP TR 37.885 城市 V2V，d_3D 单位 m、fc 单位 GHz）：
    LoS : PL = 38.77 + 16.7 log10(d_3D) + 18.2 log10(fc)
    NLoS: PL = 36.85 + 30   log10(d_3D) + 18.9 log10(fc)
阴影衰落标准差 LoS 3 dB、NLoS 4 dB（见 p3base/fading.py），小尺度为 Rayleigh。
LoS/NLoS 由建筑是否遮挡三维直线决定，与 3GPP 的 V2V 定义一致；车辆不把 RF 置零。

VLC 光增益（沿用旧场景文档的口径，仅把端口数、遮挡改为与车辆数无关）：
    h = chi * [D_R / (zeta * d_port)]^2 * cos(theta)^(2/epsilon)
chi 是 0/1 门控：发射角 <90°、入射角 <= 60° 半视场、无建筑或第三辆车遮挡。
注意 D_R 由 PD 面积推出，不再使用原稿 7 mm 直径的口径。
"""
import numpy as np

from .config import RFConfig, VLCConfig
from .geometry import blocked_by_vehicles, blocked_segments, emission_incidence

PORT_COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))   # (发射端口, 接收端口)
_CHUNK = 512


def rf_path_loss_db(d3d_m, fc_ghz, los, cfg: RFConfig):
    """3GPP TR 37.885 路损；d < 1 m 按 1 m 计（公式适用范围之外）。"""
    d = np.maximum(np.asarray(d3d_m, float), 1.0)
    fc = float(fc_ghz)
    if cfg.pathloss_model == "tr37885":
        los_arr = np.asarray(los, bool)
        los_db = 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(fc)
        nlos_db = 36.85 + 30.0 * np.log10(d) + 18.9 * np.log10(fc)
        return np.where(los_arr, los_db, nlos_db)
    if cfg.pathloss_model == "legacy_47p86":
        return cfg.legacy_path_loss_1m_db + 10.0 * cfg.legacy_path_loss_exponent * np.log10(d)
    raise ValueError(f"未知路径损耗模型 {cfg.pathloss_model}")


def rf_geometry(scene, state, cfg: RFConfig):
    """确定性 RF 几何增益（不含阴影和小尺度衰落）。

    candidate 只表示距离可达；交叉干扰项仍保留非候选链路的增益。
    """
    c = np.asarray(state["center_xy_m"], float)
    n = len(c)
    z = np.full((n, 1), scene.vehicle.port_height_m)
    p = np.concatenate([c, z], axis=1)
    delta = p[:, None, :] - p[None, :, :]
    d3d = np.linalg.norm(delta, axis=-1)
    los = ~blocked_segments(p[:, None, :], p[None, :, :], scene.buildings)
    loss = rf_path_loss_db(d3d, cfg.frequency_ghz, los, cfg)
    gain = 10.0 ** (-loss / 10.0)
    eye = np.eye(n, dtype=bool)
    gain[eye] = 0.0
    candidate = (d3d <= cfg.candidate_range_m) & ~eye
    return dict(d3d_m=d3d, los=los, gain=gain, candidate=candidate)


def _pair_segments(ports):
    """所有有向车对 i->j 与 4 种端口组合的线段，形状 (P,4,3)。"""
    pos = ports["pos"]
    outward = ports["outward"]
    n = pos.shape[0]
    ii, jj = np.nonzero(~np.eye(n, dtype=bool))
    start = np.stack([pos[ii, tp] for tp, _ in PORT_COMBOS], axis=1)
    end = np.stack([pos[jj, rp] for _, rp in PORT_COMBOS], axis=1)
    dir_tx = np.stack([outward[ii, tp] for tp, _ in PORT_COMBOS], axis=1)
    dir_rx = np.stack([outward[jj, rp] for _, rp in PORT_COMBOS], axis=1)
    return ii, jj, start, end, dir_tx, dir_rx


def vlc_geometry(scene, ports, cfg: VLCConfig, strict=False):
    """VLC 端口级光增益，形状 (n,n,2,2)。

    返回 gain 与诊断计数：too_close（端口距离 < 1 m）、angle_gated、blocked_building、
    blocked_vehicle。gate 失败或遮挡处 gain 置 0，不裁剪成小正数。
    """
    ii, jj, start, end, dir_tx, dir_rx = _pair_segments(ports)
    p = len(ii)
    n = ports["pos"].shape[0]
    d_port, cos_tx, cos_rx = emission_incidence(start[..., :2], dir_tx, end[..., :2], dir_rx)

    too_close = d_port < cfg.min_port_distance_m
    if strict and too_close.any():
        raise ValueError(f"存在端口距离 < {cfg.min_port_distance_m} m 的端口组合")
    fov_cos = np.cos(np.deg2rad(cfg.fov_half_deg))
    forward = cos_tx > 0.0
    in_fov = cos_rx >= fov_cos
    gated = forward & in_fov & ~too_close

    d_safe = np.maximum(d_port, cfg.min_port_distance_m)
    aperture = cfg.aperture_diameter_m
    base = (aperture / (cfg.zeta_rad * d_safe)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        h = base * np.power(np.where(forward, cos_tx, 0.0), 2.0 / cfg.epsilon)
    h = np.where(gated, h, 0.0)

    blocked_b = blocked_segments(start, end, scene.buildings) if len(scene.buildings) else \
        np.zeros((p, len(PORT_COMBOS)), bool)
    blocked_v = np.zeros((p, len(PORT_COMBOS)), bool)
    centres = np.asarray(ports["center_xy_m"], float)
    headings = np.asarray(ports["heading_rad"], float)
    if n > 2:
        # Exact pruning: vehicle blockage cannot change a ray already eliminated
        # by angular/range gate or a building. Diagnostics also mask these rays.
        rows, combos = np.nonzero(gated & ~blocked_b)
        for lo in range(0, len(rows), _CHUNK):
            rr, cc = rows[lo:lo + _CHUNK], combos[lo:lo + _CHUNK]
            hit = blocked_by_vehicles(
                start[rr, cc], end[rr, cc], centres, headings,
                scene.length_m, scene.width_m, scene.height_m)
            own = (np.arange(n)[None, :] == ii[rr, None]) | \
                  (np.arange(n)[None, :] == jj[rr, None])
            blocked_v[rr, cc] = (hit & ~own).any(axis=-1)

    visible = gated & ~blocked_b & ~blocked_v
    h = np.where(visible, h, 0.0)

    gain = np.zeros((n, n, 2, 2))
    gain[ii, jj] = h.reshape(p, 2, 2)
    d_port_map = np.zeros((n, n, 2, 2))
    d_port_map[ii, jj] = d_port.reshape(p, 2, 2)

    centre = np.asarray(state_center(ports), float)
    d2 = np.linalg.norm(centre[:, None, :] - centre[None, :, :], axis=-1)
    eye = np.eye(n, dtype=bool)
    in_range = (d2 <= cfg.candidate_range_m) & ~eye
    has_port = gain.reshape(n, n, 4).max(axis=-1) > 0.0
    candidate = in_range & has_port

    return dict(gain=gain, candidate=candidate, d_port_m=d_port_map,
                diagnostics=dict(too_close=int(too_close.sum()),
                                 angle_gated=int((~gated).sum()),
                                 blocked_building=int((gated & blocked_b).sum()),
                                 blocked_vehicle=int((gated & ~blocked_b & blocked_v).sum()),
                                 visible=int(visible.sum())))


def state_center(ports):
    """端口位置的中点即车辆中心（前后端口关于中心对称）。"""
    return ports["pos"][:, :, :2].mean(axis=1)


def _scatter(ii, jj, values, n):
    out = np.zeros((n, n, 4))
    out[ii, jj] = values.reshape(len(ii), 4)
    return out


def select_ports(vlc_gain):
    """每条有向边选光增益最大的收发端口组合，返回 tx/rx 索引图与对应增益。

    端口组合顺序为 PORT_COMBOS = ((0,0),(0,1),(1,0),(1,1))，故 index = 2*tx + rx。
    不可见边仍会返回某个索引，其增益为 0，调用方用 candidate 过滤。
    """
    n = vlc_gain.shape[0]
    flat = vlc_gain.reshape(n, n, len(PORT_COMBOS))
    idx = flat.argmax(axis=-1)
    tx, rx = np.divmod(idx, 2)
    return dict(tx=tx, rx=rx, gain=flat.max(axis=-1))


def candidate_edges(candidate_rf, candidate_vlc):
    """可行动的有向边（发送车, 接收车, 模式），顺序稳定便于复现。"""
    edges = []
    for i, j in np.argwhere(candidate_rf):
        edges.append((int(i), int(j), "RF"))
    for i, j in np.argwhere(candidate_vlc):
        edges.append((int(i), int(j), "VLC"))
    return edges


def port_gain_for(gain, i, j, tp, rp):
    return gain[i, j, tp, rp]
