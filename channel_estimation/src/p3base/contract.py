"""基础层契约自检。任何一项失败都应视为接口错误，而不是"数值不理想"。"""
import numpy as np


class ContractError(AssertionError):
    pass


def check_scene(scene, n_expected=None):
    if n_expected is not None and scene.n != n_expected:
        raise ContractError(f"场景车辆数 {scene.n} 与请求 {n_expected} 不一致")
    if scene.n < 2:
        raise ContractError("至少需要两辆车")
    if not np.all(np.isfinite(scene.center_xy)):
        raise ContractError("轨迹含 NaN/Inf")
    if not np.allclose(np.diff(scene.time_s), scene.frame_dt, atol=1e-6):
        raise ContractError("mobility 时间步长不均匀")
    return True


def check_rf(rf):
    g = np.asarray(rf["gain"], float)
    if np.any(~np.isfinite(g)):
        raise ContractError("RF 增益含非有限值")
    if np.any(g < 0):
        raise ContractError("RF 增益为负")
    if np.any(np.diag(g) != 0):
        raise ContractError("自连边 RF 增益必须为 0")
    if np.any(rf["candidate"] & (g <= 0)):
        raise ContractError("RF 候选边增益必须为正")
    return True


def check_vlc(vlc):
    h = np.asarray(vlc["gain"], float)
    if np.any(~np.isfinite(h)):
        raise ContractError("VLC 增益含非有限值")
    if np.any(h < 0):
        raise ContractError("VLC 增益为负")
    if np.any(np.diagonal(h, axis1=0, axis2=1) != 0):
        raise ContractError("自连边 VLC 增益必须为 0")
    has_port = h.reshape(h.shape[0], h.shape[1], -1).max(axis=-1) > 0
    if np.any(vlc["candidate"] & ~has_port):
        raise ContractError("VLC 候选边必须至少有一个端口组合可见")
    return True


def check_belief(belief):
    for key in ("rf_gain_mean", "rf_gain_low", "rf_gain_high",
                "vlc_gain_mean", "vlc_gain_low", "vlc_gain_high"):
        arr = np.asarray(belief[key], float)
        if np.any(~np.isfinite(arr)):
            raise ContractError(f"{key} 含 NaN/Inf")
        if np.any(arr < 0):
            raise ContractError(f"{key} 为负")
    if np.any(belief["rf_gain_low"] > belief["rf_gain_mean"] + 1e-15):
        raise ContractError("RF 下分位高于均值")
    if np.any(belief["rf_gain_mean"] > belief["rf_gain_high"] + 1e-15):
        raise ContractError("RF 均值高于上分位")
    age = np.asarray(belief["rf_age_s"], float)
    if np.any(age[~belief["rf_missing"]] < -1e-9):
        raise ContractError("CSI 年龄为负")
    return True


def check_block_plan(links, powers, n, cfg):
    """完整配对、模式合法、选中边功率在 [1 mW, P_MAX]。"""
    if n % 2 or len(links) != n // 2 or len(powers) != len(links):
        raise ContractError("必须完整配对，且功率数量与链路数量相同")
    seen = set()
    for (i, j, mode), p in zip(links, powers):
        if mode not in ("RF", "VLC"):
            raise ContractError("未知模式")
        if i == j or i in seen or j in seen:
            raise ContractError("端点重复或自连")
        if not (0 <= i < n and 0 <= j < n):
            raise ContractError("车辆编号越界")
        seen.update((i, j))
        cap = cfg.rf.power_max_w if mode == "RF" else cfg.vlc.power_max_w
        if not (1e-3 - 1e-12 <= p <= cap + 1e-12):
            raise ContractError(f"{mode} 选中边功率 {p} 不在 [1e-3, {cap}]")
    return True


def check_pilot_plan(plan):
    for mode in ("rf", "vlc"):
        period = getattr(plan, f"{mode}_period_s")
        slot = getattr(plan, f"{mode}_slot_s")
        delay = getattr(plan, f"{mode}_delay_s")
        if not (period > 0 and slot > 0 and delay >= 0):
            raise ContractError(f"{mode} 导频参数非正")
        if slot > period:
            raise ContractError(f"{mode} 时隙长于周期")
        if plan.n * slot > period + 1e-12:
            raise ContractError(f"{mode} 导频开销超过 100%")
    return True
