"""瓶颈能效（二分 t + 二次变换 + SCA）与 S1/S2/S3 的单元测试。

对齐的是**能验证的性质**，不是最优性：SCA 只用局部求解器，模块 docstring 已写明
不声称收敛到驻点。所以这里测四件能断言的事：

* 返回的功率永远在逐模式功率域内，且不比起点更差；
* 在二维小例子上不劣于 90 点的对数网格暴力解（退路：说明求解器坏了）；
* 单链路 EE 峰值不劣于量程两端（``link_ee_optimum`` 的定义性质）；
* S1/S2/S3 的排序与合法性：排序只保留正增益，候选永远合法。
"""
import numpy as np
import pytest

from p3base.config import FoundationConfig
from p3learn.bottleneck_power import (bottleneck_sca, bottleneck_upper_bound, link_ee)
from p3learn.dinkelbach_sca import _Physics
from p3learn.weak_link import (interference_sensitivity, link_ee_optimum,
                               relief_ranking, service_power_probe, switch_candidates)


def _domain(n=4, seed=0, rf=1e-3, vlc=1e-5, links=None):
    cfg = FoundationConfig()
    rng = np.random.default_rng(seed)
    gains = {"RF": rng.uniform(.3, 3., (n, n)) * rf,
             "VLC": rng.uniform(.3, 3., (n, n)) * vlc}
    ports = {"tx": np.zeros((n, n), int), "rx": np.zeros((n, n), int)}
    return cfg, (links or [(0, 1, "RF"), (2, 3, "RF")]), gains, ports


def _brute(links, gains, ports, cfg, grid=90):
    phys = _Physics(links, gains, ports, cfg)
    axes = [np.logspace(np.log10(lo), np.log10(hi), grid)
            for lo, hi in zip(phys.lo, phys.hi)]
    best = -1.0
    for a in axes[0]:
        for b in axes[1]:
            best = max(best, float(link_ee(phys, np.array([a, b])).min()))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bottleneck_sca_not_worse_than_grid(seed):
    cfg, links, gains, ports = _domain(seed=seed)
    phys = _Physics(links, gains, ports, cfg)
    p0 = np.clip(np.array([0.3, 0.01]), phys.lo, phys.hi)
    p, info = bottleneck_sca(links, gains, ports, cfg, p_start=p0)
    assert np.all(p >= phys.lo - 1e-12) and np.all(p <= phys.hi + 1e-12)
    assert info.final_min_ee >= info.start_min_ee
    assert info.final_min_ee >= 0.95 * _brute(links, gains, ports, cfg)


@pytest.mark.parametrize("seed", [0, 3])
def test_bottleneck_upper_bound_is_really_an_upper_bound(seed):
    cfg, links, gains, ports = _domain(seed=seed)
    phys = _Physics(links, gains, ports, cfg)
    bound = bottleneck_upper_bound(phys)
    rng = np.random.default_rng(seed + 11)
    for _ in range(200):
        p = rng.uniform(phys.lo, phys.hi)
        assert link_ee(phys, p).min() <= bound * (1.0 + 1e-9)


def test_bottleneck_sca_empty_domain():
    cfg = FoundationConfig()
    p, info = bottleneck_sca([], {}, {}, cfg, p_start=np.zeros(0))
    assert p.size == 0 and info.status == "empty"


def test_link_ee_optimum_beats_both_endpoints():
    cfg, links, gains, ports = _domain(seed=1)
    phys = _Physics(links, gains, ports, cfg)
    powers = np.clip(np.array([0.5, 0.02]), phys.lo, phys.hi)
    for k in range(len(links)):
        best = link_ee_optimum(links, gains, ports, cfg, powers, k)
        lo_p, hi_p = powers.copy(), powers.copy()
        lo_p[k], hi_p[k] = phys.lo[k], phys.hi[k]
        assert link_ee(phys, best)[k] >= link_ee(phys, lo_p)[k] - 1e-9
        assert link_ee(phys, best)[k] >= link_ee(phys, hi_p)[k] - 1e-9


def test_service_power_probe_shrinks_power_at_loose_qos():
    cfg, links, gains, ports = _domain(seed=2)
    phys = _Physics(links, gains, ports, cfg)
    powers = np.clip(np.array([0.5, 0.5]), phys.lo, phys.hi)
    probe = service_power_probe(links, gains, ports, cfg, powers, 1e3)
    assert probe.ratio <= 1.0 + 1e-12
    assert np.all(probe.p_star <= powers + 1e-12)
    assert np.all(probe.slack_w >= 0.0)


def test_relief_ranking_drops_non_positive_gains():
    # sensitivity[c, k] = dEE_k / dp_c，weakest = 0，所以看第 0 列。
    sensitivity = np.array([[1.0, 5.0], [-3.0, 2.0]])
    ranking = relief_ranking(sensitivity, np.array([1.0, 1.0]), weakest=0)
    assert ranking == [(3.0, 1)]           # 只有 -sens[1,0]*slack 为正
    # 全为正导数（降功率反而拖累最弱链路）时排序必须为空，不能编造候选。
    assert relief_ranking(np.full((2, 2), 1.0), np.ones(2), weakest=0) == []


def test_switch_candidates_stay_inside_the_power_domain():
    cfg, links, gains, ports = _domain(seed=2)
    masks = {"RF": np.ones((4, 4), bool), "VLC": np.ones((4, 4), bool)}
    np.fill_diagonal(masks["RF"], False)
    np.fill_diagonal(masks["VLC"], False)

    class _A:
        pass

    action = _A()
    action.links = links
    action.powers = np.clip(np.array([0.3, 0.01]), 0.01, 20.0)
    action.ports = ports
    out = switch_candidates(action, masks, cfg, gains=gains, ports=ports, limit=3)
    assert out, "至少应有 A 候选"
    for _tag, cand_links, cand_powers, _value in out:
        assert len(cand_links) == len(links)
        for (_i, _j, mode), power in zip(cand_links, cand_powers):
            floor = getattr(cfg, mode.lower()).power_min_w
            cap = getattr(cfg, mode.lower()).power_max_w
            assert floor - 1e-12 <= power <= cap + 1e-12


def test_weak_link_repair_stage_submits_three_tuples():
    """回归：S2 候选曾经以 2 元组入库，S3 循环按 3 元组解包会直接崩溃。

    这里只验证**控制流与候选合法性**，不验证收益（收益要真值评价器，见实跑记录）。
    """
    from p3learn.allocation import Action, weak_link_repair_stage

    cfg, links, gains, ports = _domain(seed=0, rf=1e-4, vlc=1e-6)
    n = 4
    masks = {"RF": np.ones((n, n), bool), "VLC": np.ones((n, n), bool)}
    np.fill_diagonal(masks["RF"], False)
    np.fill_diagonal(masks["VLC"], False)
    action = Action(links, np.array([0.4, 0.05]), ports)
    seen = []

    def consider(candidate):
        seen.append(candidate)
        return dict(calls=1, accepted=False, min_pair_ee=1.0, budget_violation=0.0,
                    qos_violation_links=None, fallback=True)

    diag = []
    attempts = weak_link_repair_stage(action, {}, cfg, believe=gains, masks=masks,
                                      ports=ports, qos={"q_min_bps": 1e4}, limit=3,
                                      consider=consider, diagnostics=diag)
    assert attempts == len(seen)
    for candidate in seen:
        assert isinstance(candidate, Action)
        for (_i, _j, mode), power in zip(candidate.links, candidate.powers):
            floor = getattr(cfg, mode.lower()).power_min_w
            cap = getattr(cfg, mode.lower()).power_max_w
            assert floor - 1e-12 <= power <= cap + 1e-12
    assert any(r.get("stage") == "s1_probe" for r in diag)
    # 必须真的走到 S2 分支，否则这个回归用例根本没覆盖到曾经崩溃的那一行。
    assert any(str(r.get("stage", "")).startswith("S2_") for r in diag)


def test_interference_sensitivity_diagonal_is_positive():
    """∂EE_k/∂p_k 在功率还远低于峰值时为正——S3-A 依赖这一点。"""
    cfg, links, gains, ports = _domain(seed=0, rf=1e-4, vlc=1e-6)
    powers = np.clip(np.array([0.02, 0.02]), 0.01, 20.0)
    sensitivity, weakest = interference_sensitivity(links, gains, ports, cfg, powers)
    assert sensitivity.shape == (2, 2)
    assert weakest in (0, 1)
    assert np.all(np.diag(sensitivity) > 0)
