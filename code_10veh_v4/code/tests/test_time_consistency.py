"""时间一致性验收（2026-10-23 第一轮短验收，第 1 项）。

两项独立性必须成立，否则任何"导频加密改善估计"的结论都不成立：

A. 调用粒度不变性：同一批已到达观测，一次 advance 到 t 与逐 1 ms advance 到 t，
   后验的均值与方差必须一致（旧实现因负 dt 被跳过、时间回绕而不同）。
B. 真值推进与导频解耦：相同 seed、不同导频周期，底层衰落状态在共同时刻必须完全相同
   （旧实现按导频事件切分推进区间，改变周期等于换了一条信道实现）。

运行：
    python -B -I tests/test_time_consistency.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p3base.config import FoundationConfig                       # noqa: E402
from p3base.estimator import ChannelEstimator                     # noqa: E402
from p3base.fading import link_coherence_s                        # noqa: E402
from p3base.scene import Scene, find_scene_root                   # noqa: E402
from run_foundation import CODE_ROOT, TruthOracle                 # noqa: E402

TOL = 1e-10


class SyntheticProvider:
    """解析观测：y(p) = exp(j(300p + 170p))，与真实场景无关，用来只检验时间处理。"""

    def __init__(self):
        self.calls = {"rf": 0, "vlc": 0}

    def rf(self, p):
        self.calls["rf"] += 1
        y = np.full((2, 2), np.exp(1j * (300.0 * p + 170.0 * p)), complex)
        var = np.full((2, 2), 0.1)
        valid = np.ones((2, 2), bool)
        return y, var, valid

    def vlc(self, p):
        self.calls["vlc"] += 1
        y = np.full((2, 2, 2, 2), -0.5 + 0.2 * np.cos(50.0 * p))
        var = np.full((2, 2, 2, 2), 0.05)
        valid = np.ones((2, 2, 2, 2), bool)
        return y, var, valid


def _cfg(period=None, vlc_period=None):
    cfg = FoundationConfig()
    return cfg.with_(estimation=replace(
        cfg.estimation,
        rf_pilot_period_s=period or cfg.estimation.rf_pilot_period_s,
        vlc_pilot_period_s=vlc_period or cfg.estimation.vlc_pilot_period_s))


def test_call_granularity():
    cfg = _cfg(period=0.001)          # 1 ms 导频、2 ms 时延：旧实现必触发回绕
    tau = np.full((2, 2), 0.030)
    prior = np.full((2, 2), 1.25)
    T = 0.100

    a = ChannelEstimator(2, cfg)
    a.advance(T, SyntheticProvider(), rf_prior_var=prior, rf_tau=tau)
    b = ChannelEstimator(2, cfg)
    for k in range(1, 101):
        b.advance(k * 1e-3, SyntheticProvider(), rf_prior_var=prior, rf_tau=tau)
    c = ChannelEstimator(2, cfg)
    for t in (0.013, 0.047, 0.100):    # 非网格、非导频时刻的查询
        c.advance(t, SyntheticProvider(), rf_prior_var=prior, rf_tau=tau)

    ba, bb, bc = a.belief(T), b.belief(T), c.belief(T)
    dm = np.abs(ba["rf_state_mean"] - bb["rf_state_mean"]).max()
    dv = np.abs(ba["rf_state_var"] - bb["rf_state_var"]).max()
    dm2 = np.abs(ba["rf_state_mean"] - bc["rf_state_mean"]).max()
    dv2 = np.abs(ba["rf_state_var"] - bc["rf_state_var"]).max()
    print(f"  A1 一次推进 vs 逐 1 ms：均值差 {dm:.3e}，方差差 {dv:.3e}")
    print(f"  A2 一次推进 vs 不规则查询：均值差 {dm2:.3e}，方差差 {dv2:.3e}")
    assert dm < TOL and dv < TOL, "调用粒度改变了后验：时间处理仍不一致"
    assert dm2 < TOL and dv2 < TOL, "不规则查询时刻改变了后验"


def test_truth_independent_of_pilot_period():
    scene = Scene(find_scene_root(CODE_ROOT, 10), FoundationConfig().vehicle)
    checks = []
    for period in (0.020, 0.001):
        oracle = TruthOracle(scene, _cfg(period=period), seed=20261019)
        oracle.advance_to(0.3)
        checks.append(oracle.fading.snapshot())
    for key in ("rf_amp", "rf_shadow_db", "vlc_log"):
        d = np.abs(checks[0][key] - checks[1][key]).max()
        print(f"  B  {key:14s} 20 ms vs 1 ms 真值最大差 {d:.3e}")
        assert d < TOL, f"真值随导频周期改变：{key}"


def test_truth_forward_only():
    """推进必须是"一次到位 == 分段到位"，否则时间网格仍有隐藏状态。"""
    scene = Scene(find_scene_root(CODE_ROOT, 10), FoundationConfig().vehicle)
    a = TruthOracle(scene, _cfg(), seed=7)
    a.advance_to(0.5)
    b = TruthOracle(scene, _cfg(), seed=7)
    for k in range(1, 51):
        b.advance_to(k * 1e-2)
    d = np.abs(a.fading.rf_amp - b.fading.rf_amp).max()
    print(f"  C  一次推进到 0.5 s vs 每 10 ms 推进：真值最大差 {d:.3e}")
    assert d < TOL, "查询次数改变了真值"


if __name__ == "__main__":
    failures = 0
    for fn in (test_call_granularity, test_truth_independent_of_pilot_period,
               test_truth_forward_only):
        print(f"[{fn.__name__}]")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    raise SystemExit(1 if failures else 0)
