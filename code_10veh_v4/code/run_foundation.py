"""基础层入口：10/50/100 车场景下的信道统计与信道估计精度诊断。

用法（需要用户明确同意后才运行）：
    python -B -I run_foundation.py --n 10
    python -B -I run_foundation.py --n 50 --duration 100
    python -B -I run_foundation.py --n 100 --duration 100 --out results/n100

本入口只做两件事：验证基础层接口，统计候选/遮挡/导频开销与估计误差。
它不求解资源分配问题，也不产生任何 EE 结论。
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

import networkx as nx
import numpy as np

# `python -I` 不会把脚本目录放进 sys.path，这里显式加入（本目录是我们自己的代码）。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3base import channel, contract, noise
from p3base.config import FoundationConfig
from p3base.estimator import ChannelEstimator
from p3base.fading import FadingField, link_coherence_s
from p3base.pilots import PilotPlan
from p3base.scene import Scene, find_scene_root

CODE_ROOT = Path(__file__).resolve().parents[1]


class TruthOracle:
    """真值场 + 导频观测缓存，同时充当估计器的观测提供者。

    advance_to(target) 先按时间顺序生成 (当前时刻, target] 内的所有导频观测：
    每个导频使用**它自己时刻**的几何（不是块末几何），并使用它自己时刻的衰落状态；
    再推进到 target 并设置块几何。估计器只能拿到缓存，不可能读到未来观测。
    已被消费的观测立即从缓存移除，只保留最近 4 次供作图，避免 100 车 100 s 时
    缓存增长到 GB 级。
    """

    def __init__(self, scene, cfg, seed):
        self.scene, self.cfg = scene, cfg
        self.seed = int(seed)
        self.plan = PilotPlan.for_n(scene.n, cfg)
        self.fading = FadingField(scene.n, cfg.rf, cfg.vlc, seed)
        self.t = 0.0
        self._grid_t = 0.0
        self._grid_k = 0
        self.truth_dt = float(cfg.estimation.fine_dt_s)
        # 阴影 LoS/NLoS 判定的固定时钟。它必须与导频周期无关：旧实现直接用
        # "上一次 _set_geometry 留下的 los"，而几何只在导频/查询时刻刷新，
        # 于是换导频周期就换了阴影 σ，真值随之改变。
        self.los_dt = 0.01
        self._los_k = 0
        self._los = None
        self.state = None
        self.ports = None
        self.rf_geom = None
        self.vlc_geom = None
        self.cache = {"rf": {}, "vlc": {}}
        self.recent = {"rf": {}, "vlc": {}}
        # 诊断用：跟踪一条链路的逐导频真值/观测，供作图。估计器永远读不到这里。
        self.track = None
        self.trace = {"rf": [], "vlc": []}
        self._next = {"rf": 0.0, "vlc": 0.0}
        self._next_i = {"rf": 0, "vlc": 0}
        self._set_geometry(0.0)
        self._los = self.rf_geom["los"]

    # ---- 几何 ----------------------------------------------------------
    def _set_geometry(self, t):
        state = self.scene.at(t)
        ports = self.scene.ports(state)
        rf = channel.rf_geometry(self.scene, state, self.cfg.rf)
        vlc = channel.vlc_geometry(self.scene, ports, self.cfg.vlc)
        contract.check_rf(rf)
        contract.check_vlc(vlc)
        self.state, self.ports, self.rf_geom, self.vlc_geom = state, ports, rf, vlc

    def rf_prior_var(self):
        """复合 RF 状态的平稳方差 E[10^(X/10)]，X 为阴影 dB 值。"""
        sigma = np.where(self.rf_geom["los"], self.cfg.rf.los_shadow_sigma_db,
                         self.cfg.rf.nlos_shadow_sigma_db)
        return np.exp((sigma * np.log(10.0) / 10.0) ** 2 / 2.0)

    # ---- 时间推进 ------------------------------------------------------
    def advance_to(self, target):
        """推进真值到 target。

        真值只在**固定网格**（`truth_dt`，默认 1 ms）上推进，网格由整数步索引决定，
        与导频周期、调用次数、查询粒度完全无关。导频在自己所在的网格点采样。
        这样 20 ms 与 1 ms 导频看到的是同一条物理轨迹，可以互相比较。

        旧实现按导频事件切分推进区间，每次切分都重新抽随机数，改变导频周期就改变了
        随机数消耗与分段，等于换了另一条信道实现。
        """
        target = float(target)
        if target < self.t - 1e-12:
            raise ValueError("真值时间不能倒退")
        events = {}
        for mode, period in (("rf", self.plan.rf_period_s), ("vlc", self.plan.vlc_period_s)):
            # 用整数索引算 p = k*period，不做累加：累加 10^4 次后与网格点会差 1e-12，
            # 事件键就与网格键对不上，那一次导频被静默丢弃（缓存里出现空洞）。
            k = self._next_i[mode]
            while k * period <= target + 1e-12:
                events.setdefault(round(k * period, 9), []).append(mode)
                k += 1
            self._next_i[mode] = k
            self._next[mode] = k * period

        if self._grid_k == 0 and 0.0 in events:
            # 0 时刻的导频落在网格起点，网格循环从第 1 步开始，需单独采样。
            self._set_geometry(0.0)
            for mode in events.pop(0.0):
                self._sample_pilot(mode, 0.0)

        step = self.truth_dt
        while (self._grid_k + 1) * step <= target + 1e-12:
            self._grid_k += 1
            g = self._grid_k * step
            while (self._los_k + 1) * self.los_dt <= g + 1e-12:
                self._los_k += 1
                # 只按固定时钟刷新遮挡判定，与是否有导频无关。
                self._los = channel.rf_geometry(
                    self.scene, self.scene.at(g), self.cfg.rf)["los"]
            self._advance(g)            # 用上一网格点的几何推进（因果）
            key = round(g, 9)
            if key in events:
                self._set_geometry(g)   # 导频用它自己时刻的几何
                for mode in events[key]:
                    self._sample_pilot(mode, g)
        self.t = target
        self._set_geometry(target)

    def _advance(self, t):
        """把衰落从上一个**网格点**推进到 t（t 必为网格点）。"""
        dt = t - self._grid_t
        if dt <= 0:
            return
        vel = self._velocity(self.scene.at(0.5 * (self._grid_t + t)))
        rel = np.linalg.norm(vel[:, None, :] - vel[None, :, :], axis=-1)
        # los 来自固定时钟（见 advance_to），不是上一次几何刷新的残留。
        self.fading.advance(dt, los=self._los, rel_speed=rel)
        self._grid_t = t

    @staticmethod
    def _velocity(state):
        speed = np.asarray(state["speed_mps"], float)
        heading = np.asarray(state["heading_rad"], float)
        return np.stack([speed * np.cos(heading), speed * np.sin(heading)], axis=-1)

    def rel_speed(self, t):
        """时刻 t 的二维相对速度大小（估计器可用，因为位置/速度是已知观测量）。"""
        vel = self._velocity(self.scene.at(t))
        return np.linalg.norm(vel[:, None, :] - vel[None, :, :], axis=-1)

    # ---- 导频观测 ------------------------------------------------------
    def pilot_observation_var(self, mode):
        """当前几何下的导频观测方差（诊断用，不参与估计）。"""
        if mode == "rf":
            return noise.pilot_gain_variance(
                self.cfg.rf.noise_w, self.cfg.rf.bandwidth_hz,
                self.cfg.estimation.rf_pilot_power_w, self.plan.rf_slot_s,
                self.rf_geom["gain"])
        return noise.vlc_pilot_log_variance(
            self.vlc_geom["gain"], self.cfg.estimation.vlc_pilot_power_w,
            self.plan.vlc_slot_s, self.cfg.vlc, self.cfg.estimation.vlc_log_obs_sd)

    def _sample_pilot(self, mode, p):
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(round(p * 1e6)), 7 if mode == "rf" else 11]))
        if mode == "rf":
            g = self.rf_geom["gain"]
            valid = g > 0
            var = self.pilot_observation_var("rf")
            # 复幅度状态含阴影：h = 10^(X/20) a，于是 |h|^2 = 10^(X/10)|a|^2，
            # 与 fading.rf_gain 的功率口径严格一致。
            amp = self.fading.rf_amp * 10.0 ** (self.fading.rf_shadow_db / 20.0)
            noise_c = ((rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape))
                       / np.sqrt(2.0))
            y = amp + np.sqrt(np.where(valid, var, 0.0)) * noise_c
            # 第 4 项是该时刻的真值，只给作图/误差核对用，估计器拿不到（_serve 只返回前三项）。
            self.cache["rf"][round(p, 9)] = (y, np.where(valid, var, np.inf), valid, amp)
            if self.track is not None:
                i, j = self.track
                self.trace["rf"].append((float(p), float(g[i, j]), complex(y[i, j]),
                                         complex(amp[i, j]), bool(valid[i, j])))
        else:
            h = self.vlc_geom["gain"]
            valid = h >= self.cfg.estimation.vlc_observable_gain_min
            var = self.pilot_observation_var("vlc")
            y = self.fading.vlc_log + rng.standard_normal(h.shape) * np.sqrt(
                np.where(valid, var, 0.0))
            self.cache["vlc"][round(p, 9)] = (
                np.where(valid, y, np.nan), np.where(valid, var, np.inf), valid,
                self.fading.vlc_log.copy())
            if self.track is not None:
                i, j = self.track
                self.trace["vlc"].append(
                    (float(p), h[i, j].reshape(4).copy(), np.asarray(y[i, j]).reshape(4).copy(),
                     self.fading.vlc_log[i, j].reshape(4).copy(),
                     np.asarray(valid[i, j]).reshape(4).copy()))

    def _serve(self, mode, p):
        key = round(float(p), 9)
        item = self.cache[mode].pop(key)
        self.recent[mode][key] = item
        while len(self.recent[mode]) > 8:
            self.recent[mode].pop(next(iter(self.recent[mode])))
        return item[:3]

    def last_observation(self, mode, p):
        """(y, obs_var, valid, truth_at_pilot)，供作图与误差核对。"""
        return self.recent[mode].get(round(float(p), 9))

    # ---- 估计器接口 ----------------------------------------------------
    def rf(self, p):
        return self._serve("rf", p)

    def vlc(self, p):
        return self._serve("vlc", p)

    def truth(self):
        return dict(rf=self.fading.rf_gain(self.rf_geom["gain"]),
                    rf_amp=self.fading.rf_amp * 10.0 ** (self.fading.rf_shadow_db / 20.0),
                    vlc=self.fading.vlc_gain(self.vlc_geom["gain"]),
                    vlc_log=self.fading.vlc_log)


def summarise(name, values, out):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out[name] = dict(count=0)
        return
    out[name] = dict(count=int(arr.size), mean=float(arr.mean()),
                     p05=float(np.percentile(arr, 5)), p50=float(np.percentile(arr, 50)),
                     p95=float(np.percentile(arr, 95)))


def matching_cover(candidate):
    """把有向候选边投影成无向端点图，求最大基数匹配覆盖比例。

    旧实现用"左右两份车辆集合"的二部图匹配，允许同一辆车既发送又接收，
    会高估覆盖（两个独立三角形报 6，合法上限是 4）。块协议要求每车恰好一次，
    所以必须用普通图匹配，2*匹配数 才是被覆盖的车辆数。
    """
    n = int(candidate.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    eye = np.eye(n, dtype=bool)
    for i, j in np.argwhere(np.asarray(candidate, bool) & ~eye):
        graph.add_edge(int(i), int(j))
    matched = nx.max_weight_matching(graph, maxcardinality=True)
    return 2.0 * len(matched) / n


def main():
    ap = argparse.ArgumentParser(description="P3 基础层诊断入口")
    ap.add_argument("--n", type=int, required=True, choices=(10, 50, 100))
    ap.add_argument("--duration", type=float, default=100.0, help="统计时长 / s")
    ap.add_argument("--blocks", type=int, default=None, help="覆盖 duration 的块数")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="只做估计预热、不计入统计的秒数")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mmse", action="store_true",
                    help="打开论文式 (12) 的 MMSE 收缩（默认关闭，见 README）")
    ap.add_argument("--rf-period", type=float, default=None, help="RF 导频周期 / s")
    ap.add_argument("--vlc-period", type=float, default=None, help="VLC 导频周期 / s")
    ap.add_argument("--rf-state", default=None,
                    choices=("paper_doppler", "complex_ar1", "two_state"),
                    help="RF 状态模型；two_state 为慢阴影+快小尺度分离（见 p3base/rf_two_state.py）")
    args = ap.parse_args()

    cfg = FoundationConfig()
    if args.mmse:
        cfg = cfg.with_(estimation=replace(cfg.estimation, mmse_stage=True))
    if args.rf_state:
        cfg = cfg.with_(estimation=replace(cfg.estimation, rf_state=args.rf_state))
    if args.rf_period or args.vlc_period:
        cfg = cfg.with_(estimation=replace(
            cfg.estimation,
            rf_pilot_period_s=args.rf_period or cfg.estimation.rf_pilot_period_s,
            vlc_pilot_period_s=args.vlc_period or cfg.estimation.vlc_pilot_period_s))

    scene = Scene(find_scene_root(CODE_ROOT, args.n), cfg.vehicle)
    contract.check_scene(scene, args.n)
    block_dt = cfg.estimation.block_dt_s
    total_blocks = scene.block_count(block_dt)
    warm_blocks = int(round(args.warmup / block_dt))
    run_blocks = args.blocks if args.blocks is not None else int(round(args.duration / block_dt))
    if warm_blocks < 0 or run_blocks <= 0:
        raise SystemExit("warmup 不能为负，blocks/duration 必须给出正的块数")
    run_blocks = min(run_blocks, total_blocks - warm_blocks)
    if run_blocks <= 0:
        raise SystemExit("预热占满全部轨迹，没有可统计的块")

    estimator = ChannelEstimator(scene.n, cfg)
    contract.check_pilot_plan(estimator.plan)
    oracle = TruthOracle(scene, cfg, args.seed)

    keys = ("rf_candidates", "vlc_candidates", "vlc_visible_ratio", "vlc_matching_cover",
            "rf_nlos_ratio", "rf_pilot_snr_db", "rf_coherence_ms", "rf_state_mse", "rf_state_nmse",
            "vlc_state_mse", "rf_gain_rel_mse", "vlc_gain_rel_mse",
            "rf_gain_abs_db_err", "vlc_gain_abs_db_err",
            "rf_age_ms", "vlc_age_ms", "block_sec")
    stats = {k: [] for k in keys}
    off_diagonal = ~np.eye(scene.n, dtype=bool)

    for k in range(warm_blocks + run_blocks):
        t0 = time.perf_counter()
        t = k * block_dt
        oracle.advance_to(t)
        rel = oracle.rel_speed(t)
        estimator.advance(t, oracle, rf_prior_var=oracle.rf_prior_var(),
                          rf_tau=link_coherence_s(rel, cfg.rf))
        belief = estimator.gain_belief(
            oracle.rf_geom["gain"], oracle.vlc_geom["gain"], t,
            rf_candidate=oracle.rf_geom["candidate"],
            vlc_candidate=oracle.vlc_geom["candidate"])
        contract.check_belief(belief)
        if k < warm_blocks:
            continue

        truth = oracle.truth()
        rf, vlc = oracle.rf_geom, oracle.vlc_geom
        stats["rf_candidates"].append(int(rf["candidate"].sum()))
        stats["vlc_candidates"].append(int(vlc["candidate"].sum()))
        stats["rf_nlos_ratio"].append(float((~rf["los"])[off_diagonal].mean()))
        stats["vlc_visible_ratio"].append(
            float(vlc["gain"].reshape(scene.n, scene.n, -1).any(axis=-1).mean()))
        stats["vlc_matching_cover"].append(matching_cover(vlc["candidate"]))

        rf_var = oracle.pilot_observation_var("rf")
        snr = -10.0 * np.log10(np.maximum(rf_var[rf["candidate"]], 1e-300))
        stats["rf_pilot_snr_db"].append(float(np.median(snr)) if snr.size else np.nan)
        tau = link_coherence_s(rel, cfg.rf)[rf["candidate"]]
        stats["rf_coherence_ms"].append(float(np.median(tau)) * 1e3 if tau.size else np.nan)

        rf_err = np.abs(belief["rf_state_mean"] - truth["rf_amp"]) ** 2
        stats["rf_state_mse"].append(float(rf_err[off_diagonal].mean()))
        stats["rf_state_nmse"].append(
            float(rf_err[off_diagonal].sum()
                  / max((np.abs(truth["rf_amp"][off_diagonal]) ** 2).sum(), 1e-300)))
        # 增益比较必须用"绝对增益 = 几何增益 × 归一化状态"，否则量纲不对。
        # 归一化 MSE = sum(误差^2)/sum(真值^2)，对真值极小不影响量级。
        mask = rf["candidate"]
        if mask.any():
            est, tru = belief["rf_gain_mean"][mask], truth["rf"][mask]
            stats["rf_gain_rel_mse"].append(
                float(np.sum((est - tru) ** 2) / max(np.sum(tru ** 2), 1e-300)))
            stats["rf_gain_abs_db_err"].append(
                float(np.median(np.abs(10 * np.log10(np.maximum(est, 1e-300)
                                                    / np.maximum(tru, 1e-300))))))
        else:
            stats["rf_gain_rel_mse"].append(np.nan)
            stats["rf_gain_abs_db_err"].append(np.nan)

        visible = vlc["gain"] >= cfg.estimation.vlc_observable_gain_min
        if visible.any():
            stats["vlc_state_mse"].append(
                float(((belief["vlc_state_mean"] - truth["vlc_log"]) ** 2)[visible].mean()))
            est, tru = belief["vlc_gain_mean"][visible], truth["vlc"][visible]
            stats["vlc_gain_rel_mse"].append(
                float(np.sum((est - tru) ** 2) / max(np.sum(tru ** 2), 1e-300)))
            stats["vlc_gain_abs_db_err"].append(
                float(np.median(np.abs(10 * np.log10(np.maximum(est, 1e-300)
                                                    / np.maximum(tru, 1e-300))))))
        else:
            stats["vlc_state_mse"].append(np.nan)
            stats["vlc_gain_rel_mse"].append(np.nan)
            stats["vlc_gain_abs_db_err"].append(np.nan)

        age_rf = belief["rf_age_s"][~belief["rf_missing"]]
        age_vlc = belief["vlc_age_s"][~belief["vlc_missing"]]
        stats["rf_age_ms"].append(float(np.median(age_rf)) * 1e3 if age_rf.size else np.nan)
        stats["vlc_age_ms"].append(float(np.median(age_vlc)) * 1e3 if age_vlc.size else np.nan)
        stats["block_sec"].append(time.perf_counter() - t0)

    summary = {
        "n": scene.n,
        "blocks": run_blocks,
        "warmup_blocks": warm_blocks,
        "block_dt_s": block_dt,
        "config": cfg.to_dict(),
        "pilot_plan": estimator.plan.as_dict(),
        "scene": scene.summary(),
        "estimator": {"rf_state": estimator.rf.state, "vlc_state": estimator.vlc.state,
                      "mmse_stage": estimator.mmse_stage},
    }
    for key in keys:
        summarise(key, stats[key], summary)
    summary["note"] = ("只说明基础层接口与估计误差，不含资源分配或 EE 结论；轨迹固定，"
                       "跨规模比较的是车辆数与导频开销，不是统计重复实验。")

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / f"foundation_n{scene.n}.json"
        target.write_text(text, encoding="utf-8")
        print(f"已写入 {target}")
    print(text)


if __name__ == "__main__":
    main()
