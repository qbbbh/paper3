"""弱链路保障的三个模块 S1 / S2 / S3：探针 -> 排序 -> 判据。

问题在哪：目标 ``mean_k EE_k`` 里，最弱那条链路拖住的是**下界**那一栏，而功率
再分配与匹配搜索都是"整块一起动"的启发式，没有针对最弱链路的定向手段。三个
模块各回答一个问题，且都不直接改结果——它们只**产生候选**，是否采纳一律由
调用方的预算门 + QoS 门 + 条件轨迹预测器判定：

* **S1 服务功率探针**：在其余链路功率固定时，逐链路二分出"刚好满足速率门限
  ``q``"的最小电功率 ``p*_k``。它给出两个量：``Σp*/Σp``（当前动作浪费了多少
  功率）与逐链路可回收的功率 ``p_k - p*_k``。这是**解析口径**（块初信念的瞬时
  速率），不是概率口径的 p05，因此只用于排序与生成候选，不用于报表里的 QoS 判定。
* **S2 干扰灵敏度排序**：用 ``_Physics.rate_jac`` 取 ``∂r_k/∂p_c``，据此算
  ``∂EE_j/∂p_c``（j = 当前最弱链路），再乘 S1 给出的可回收功率，得到"把链路 c
  降到其服务功率，最弱链路能抬多少"的一阶预测。对线性干扰部分，这个雅可比
  就是 ``(I - F)^{-1}`` 展开后的总和（F 为干扰耦合矩阵），所以排序里已经含了
  "间接减压"，不需要另算逆矩阵；平方律 VLC 的 F 随 p 变化，这里取当前点的一阶。
* **S3 切换判据**：对最弱链路比较三类候选的解析预测——(A) 原配对加功率、
  (B) 与另一条链路交换接收端、(C) 该链路换模式——按预测的最低逐链路 EE 排序，
  只把最好的少数几条交给评价器。

**边界（必须与结果一起报）**：三者都用块初信念的解析模型，和真实条件轨迹预测
不是同一个量；S2 是一阶近似，链路功率大幅变动时线性化会失真；S3 的候选集只是
"每条其它链路换一次接收端 + 换一次模式"，不是全部完整匹配，不能据此声称枚举
了最优匹配。任何一条候选都没被采纳是正常结果，此时算法行为与不启用这三个模块
完全一致。
"""
from dataclasses import dataclass, field

import numpy as np

from .allocation import MODES
from .bottleneck_power import link_ee
from .dinkelbach_sca import _Physics

__all__ = ["ServicePowerProbe", "service_power_probe", "interference_sensitivity",
           "relief_ranking", "switch_candidates", "predict_min_ee", "link_ee_optimum"]

_BISECT = 48          # S1 的单链路二分步数：功率域跨 3 个数量级，48 步足够到 1e-12 相对精度


@dataclass
class ServicePowerProbe:
    """S1 的输出：逐链路服务功率、可回收功率、以及聚合的探针统计。"""

    p_star: np.ndarray
    feasible: np.ndarray
    used_w: float
    service_w: float
    ratio: float
    slack_w: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self):
        return dict(used_w=float(self.used_w), service_w=float(self.service_w),
                    ratio=float(self.ratio),
                    infeasible_links=[int(i) for i in np.flatnonzero(~self.feasible)],
                    p_star=[float(x) for x in self.p_star],
                    slack_w=[float(x) for x in self.slack_w])


def _rate_of(phys, p, k):
    return float(phys.rates(p)[k])


def service_power_probe(links, gains, ports, cfg, powers, q_min_bps, *, rounds=1,
                        bisect=_BISECT):
    """S1：逐链路求满足 ``r_k >= q`` 的最小电功率，再做 ``rounds`` 轮耦合迭代。

    速率对自身功率单调（RF 线性、VLC 平方律且散粒噪声随光功率增），所以单链路
    可以二分。干扰在每轮开始时按当时的功率固定，轮末用新的 ``p*`` 重算，因此
    "最小功率"是**给定他人功率**下的最小功率，不是全局可行解——这在文档里写明，
    不冒充联合最优。
    """
    if not links:
        return ServicePowerProbe(np.zeros(0), np.zeros(0, bool), 0., 0., 0., np.zeros(0))
    phys = _Physics(links, gains, ports, cfg)
    q = float(q_min_bps)
    p = np.clip(np.asarray(powers, float), phys.lo, phys.hi)
    if q <= 0:
        return ServicePowerProbe(phys.lo.copy(), np.ones(phys.k, bool),
                                 float(np.sum(p)), float(np.sum(phys.lo)),
                                 float(np.sum(phys.lo) / max(np.sum(p), 1e-12)),
                                 np.maximum(p - phys.lo, 0.))
    for _ in range(max(int(rounds), 0) + 1):
        star = p.copy()
        feasible = np.ones(phys.k, bool)
        for k in range(phys.k):
            trial = p.copy()
            if _rate_of(phys, trial, k) >= q:
                # 已经达标：往下找最小功率。
                lo, hi = float(phys.lo[k]), float(p[k])
            else:
                lo, hi = float(p[k]), float(phys.hi[k])
                if _rate_of(phys, np.where(np.arange(phys.k) == k, hi, p), k) < q:
                    star[k] = hi
                    feasible[k] = False
                    continue
            for _step in range(int(bisect)):
                mid = float(np.sqrt(lo * hi)) if lo > 0 else 0.5 * (lo + hi)
                trial = p.copy()
                trial[k] = mid
                if _rate_of(phys, trial, k) >= q:
                    hi = mid
                else:
                    lo = mid
                if abs(hi - lo) <= 1e-9 * max(hi, 1e-12):
                    break
            star[k] = hi
        if np.allclose(star, p, rtol=0, atol=1e-12):
            p = star
            break
        p = star
    used = float(np.sum(np.clip(np.asarray(powers, float), phys.lo, phys.hi)))
    service = float(np.sum(star))
    return ServicePowerProbe(star, feasible, used, service,
                             float(service / max(used, 1e-12)),
                             np.maximum(np.asarray(powers, float) - star, 0.))


def interference_sensitivity(links, gains, ports, cfg, powers):
    """S2 的原料：``D[c, k] = ∂EE_k/∂p_c``，以及最弱链路下标。

    ``rate_jac`` 已含同模干扰的耦合，所以 ``D`` 就是含间接效应的总导数；
    对线性干扰（RF、线性 VLC）它等于把 ``(I-F)^{-1}`` 展开后的结果。
    """
    if not links:
        return np.zeros((0, 0)), 0
    phys = _Physics(links, gains, ports, cfg)
    p = np.clip(np.asarray(powers, float), phys.lo, phys.hi)
    jac = phys.rate_jac(p)                      # [c, k] = d r_k / d p_c
    denom = phys.static + p
    ee = phys.rates(p) / denom
    grad = jac / denom[None, :]
    idx = np.arange(phys.k)
    grad[idx, idx] -= ee / denom
    return grad, int(np.argmin(ee))


def relief_ranking(sensitivity, slack_w, weakest):
    """按"单位可回收功率能把最弱链路抬多少"排序，返回 ``[(gain, index), ...]``。

    ``gain_c = (p_c - p*_c) * (-∂EE_j/∂p_c)``：把 c 降到服务功率时，最弱链路 j
    的一阶 EE 增量。只对**降功率**有意义（降功率救人），所以取负导数；负导数
    为负（降功率反而拖累最弱链路）的项排在最后且不参与。
    """
    if sensitivity.size == 0:
        return []
    cut = -sensitivity[:, weakest] * np.maximum(np.asarray(slack_w, float), 0.)
    order = np.argsort(-cut)
    return [(float(cut[i]), int(i)) for i in order if cut[i] > 0.]


def link_ee_optimum(links, gains, ports, cfg, powers, index, *, iters=60):
    """S3-A 的功率取值：单链路自身 EE 的最优电功率，其余链路功率固定。

    **为什么不直接抬到功率上限**：``EE_k = r_k/(Pc_k+p_k)`` 对 ``p_k`` 不是单调的。
    平方律 VLC 的速率随光功率超线性增长的时间很短，散粒噪声与分母很快把它压住，
    于是 ``EE`` 先升后降，峰值常在量程中段。抬到上限既可能低于峰值，又会让总功率
    暴涨而触发预算门——实测 A 候选 100% 被预算门拒。这里直接用黄金分割在**对数
    功率域**求单链路峰值，代价约 60 次解析速率评价（不动预测器）。

    只在"其他链路功率固定"下最优；跨模式无干扰时它同时就是瓶颈功率解，同模式有
    耦合时只是候选，是否采纳仍由调用方的评价器决定。
    """
    if not links:
        return np.zeros(0)
    phys = _Physics(links, gains, ports, cfg)
    p = np.clip(np.asarray(powers, float), phys.lo, phys.hi)
    k = int(index)
    lo, hi = float(np.log(phys.lo[k])), float(np.log(phys.hi[k]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return p

    def value(x):
        trial = p.copy()
        trial[k] = float(np.exp(x))
        return float(link_ee(phys, trial)[k])

    # 黄金分割（对数域）：区间固定收缩，iters=60 时相对精度约 (0.618)^60 ~ 3e-13。
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = value(c), value(d)
    for _ in range(int(iters)):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = value(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = value(d)
    best = c if fc > fd else d
    out = p.copy()
    out[k] = float(np.clip(np.exp(best), phys.lo[k], phys.hi[k]))
    return out


def predict_min_ee(links, gains, ports, cfg, powers):
    """解析模型下的 ``min_k EE_k``；候选排序用，不用于最终判定。"""
    if not links:
        return 0.
    phys = _Physics(links, gains, ports, cfg)
    p = np.clip(np.asarray(powers, float), phys.lo, phys.hi)
    return float(link_ee(phys, p).min())


def _legal(masks, mode, tx, rx):
    return bool(masks[mode][tx, rx])


def switch_candidates(action, masks, cfg, *, gains=None, ports=None, limit=3):
    """S3：针对最弱链路生成候选动作（不评价、不采纳，只生成）。

    三类，按解析预测的最低 EE 从高到低返回前 ``limit`` 条：

    * **A 原配对加功率**：最弱链路的功率抬到上限（其余不动）；
    * **B 换接收端**：与另一条链路交换接收端（两链路的模式与发送端不变），
      所以 RF/VLC 数量、活动链路数完全不变，只有几何与干扰关系变了；
    * **C 换模式**：最弱链路在 RF/VLC 之间切换（发送端与接收端不变）。

    每条候选都必须通过块初资格掩码 ``masks``（候选边 & 业务合格）；不合法直接
    丢弃。返回 ``[(tag, action, predicted_min_ee), ...]``，不做任何真值访问。
    """
    links = list(action.links)
    powers = np.asarray(action.powers, float)
    k = len(links)
    if k < 2:
        return []
    modes = [m for _i, _j, m in links]
    lows = np.array([getattr(cfg, m.lower()).power_min_w for m in modes])
    caps = np.array([getattr(cfg, m.lower()).power_max_w for m in modes])
    out = []
    if gains is None or ports is None:
        return out
    weakest = int(np.argmin(predict_all(action, gains, ports, cfg)))
    # A：只动最弱链路的功率，取它**自身 EE 的峰值**而不是量程上限——见
    # :func:`link_ee_optimum`：EE 对功率不单调，抬到上限既可能低于峰值，又几乎
    # 必然触发预算门。
    raised = link_ee_optimum(links, gains, ports, cfg, powers, weakest)
    links_a = [tuple(x) for x in links]
    out.append(("A_raise_power", links_a, raised))
    # B：与每条其它链路交换接收端。
    i, j, m = links[weakest]
    for b in range(k):
        if b == weakest:
            continue
        u, v, mb = links[b]
        if not (_legal(masks, m, i, v) and _legal(masks, mb, u, j)):
            continue
        swapped = [tuple(x) for x in links]
        swapped[weakest] = (i, v, m)
        swapped[b] = (u, j, mb)
        out.append((f"B_swap_rx_{b}", swapped, powers.copy()))
    # C：最弱链路换模式。
    other = "VLC" if m == "RF" else "RF"
    if _legal(masks, other, i, j):
        switched = [tuple(x) for x in links]
        switched[weakest] = (i, j, other)
        p = powers.copy()
        p[weakest] = float(np.clip(p[weakest],
                                   getattr(cfg, other.lower()).power_min_w,
                                   getattr(cfg, other.lower()).power_max_w))
        out.append((f"C_mode_{other}", switched, p))
    scored = []
    for tag, cand_links, cand_powers in out:
        value = predict_min_ee(cand_links, gains, ports, cfg, cand_powers)
        scored.append((tag, cand_links, np.asarray(cand_powers, float), value))
    scored.sort(key=lambda item: -item[3])
    return scored[:max(int(limit), 0)]


def predict_all(action, gains, ports, cfg):
    """逐链路 EE（解析口径），S3 里用来找最弱链路。"""
    phys = _Physics(list(action.links), gains, ports, cfg)
    p = np.clip(np.asarray(action.powers, float), phys.lo, phys.hi)
    return link_ee(phys, p)
