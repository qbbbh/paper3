"""瓶颈能效（逐链路 EE 下界）的二分—二次变换—SCA 求解。

问题（固定匹配与模式，只动连续功率）：

    max_{p,t}  t
    s.t.       r_k(p) / (Pc_k + p_k) >= t       对所有链路 k
               r_k(p) >= q_k                      （可选，QoS 代理约束）
               lo_k <= p_k <= hi_k                （各模式功率域）

**为什么不用无导数搜索。** 目标 ``min_k EE_k`` 是分式的最小值：分式让它对 p
非凹，``min`` 又在两条链路交换"谁是最弱"的邻域里不可导。Nelder-Mead 在这类
脊线目标上要几百次评价，而每次评价在本仓库里是一次**条件轨迹预测**（约 6 ms），
代价最高。这里改用三段结构：

1. **外层二分 t**：``EE_k >= t`` 逐条可判，``t`` 可行集是区间，二分只需约
   ``log2((t_hi-t_lo)/tol)`` 次可行性检验，且每次检验都是独立的可行性问题，
   不需要目标可导。
2. **二次变换（Shen & Yu）**：速率对功率是"信号除以（干扰+噪声）"的分式，先在
   锚点 ``p`` 处取

       r~_k(p') = B/ln2 * ln(1 + 2 y_k x_k(p') - y_k^2 gap_k D_k(p')),

   ``y_k = x_k(p)/(gap_k D_k(p))``，于是 ``r~`` 对 p' 凹、在锚点处与真速率取等号。
3. **内层 SCA**：固定 ``t`` 与 ``y`` 后，约束 ``r~_k(p') - t(Pc_k+p'_k) >= s``
   对 ``(p',s)`` 是凸的，子问题为"最大化最小裕量 s"，用 SLSQP 解。

**必须说清的边界**（与 ``dinkelbach_sca`` 同一口径，不许含糊）：

* ``r~`` 的有效域是 ``1 + A > 0``（A 是 y 的向下抛物线）。子问题在箱约束内可能
  越过边界，所以代码把 A 延拓为 ``max(A, -1/2)``；``max(凹, 常数)`` 一般不再凹，
  因此**子问题不保证是凸规划**，SLSQP 只是局部求解器，不能声称收敛到驻点。
* 因此每一轮都用**原始速率**复验；只有真实 ``EE_k >= t`` 才算该 t 可行。
  二分因此只会保守（把可行 t 判成不可行），不会把不可行点当可行。
* 全部只作用于**块初信念增益**的解析模型，不看真值；最终是否采纳由调用方用
  条件轨迹预测器复核（``maxmin_power_refine`` 里的一致性检查）。

与 ``dinkelbach_sca.dinkelbach_sum_ee`` 的分工：那边是外层 Dinkelbach/二次变换
+ 内层 SCA 最大化**平均** EE；这边是外层二分 t + 内层 SCA 最大化**下界** t。
两者共用 ``_Physics`` 的物理系数，所以不会出现两套口径。
"""
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize

from .dinkelbach_sca import _Physics

_LOG2 = float(np.log(2.0))
# A 的延拓下界：见模块 docstring 的边界说明。
_A_FLOOR = -0.5
# 内层 SLSQP 的迭代预算。**实测**：t 取得过高（不可行）时 SLSQP 会跑满 200 次
# 迭代、2151 次函数评价才放弃（371 ms/次），而可断言的收敛并不重要——真实可行性
# 由 ``_feasible_at`` 用原始速率复验。收到 40 次后不可行分支快 5 倍，可行分支本来
# 只要 6--11 次迭代，不受影响。
_SLSQP_MAXITER = 40
_SLSQP_FTOL = 1e-9


@dataclass
class BottleneckInfo:
    status: str
    feasible: bool
    bisections: int
    sca_rounds: int
    start_min_ee: float
    final_min_ee: float
    t_lo: float
    t_hi: float
    log: list = field(default_factory=list)
    per_link_upper: list = field(default_factory=list)
    per_link_start_ee: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def link_ee(phys, p):
    """逐链路 EE（原始速率，不经任何下界或截断）。"""
    return phys.rates(p) / (phys.static + p)


def bottleneck_upper_bound(phys):
    """``min_k EE_k`` 的一个合法上界：逐链路无干扰、功率上限下的速率上界。

    速率对自身功率单调，且干扰与噪声都 >= 噪声底，所以
    ``r_k(p) <= B log2(1 + x_k(hi)^2/(gap*noise_k))``；再用分母的最小值
    ``Pc_k + lo_k`` 得逐链路上界，取最小者对任何 p 都成立。
    """
    return float(np.min(_per_link_upper(phys)))


def _per_link_upper(phys):
    """逐链路的 ``EE_k`` 上界（无干扰、功率上限）。"""
    noise = np.where(phys.is_rf, phys.noise_rf, phys.noise_base)
    x_max = phys.x(phys.hi)
    rate_max = phys.bw * np.log2(1.0 + x_max ** 2 / (phys.gap * noise))
    return rate_max / (phys.static + phys.lo)


def _dD_coefficients(phys):
    """``dD(p)`` 的仿射分解 ``base + coef * p[:, None]``（对角单独给常量）。

    ``_Physics.dD`` 每调用一次要跑一个长度为 K 的 Python 循环，而 SCA 内层每个
    迭代都要调用两次（函数值与雅可比），是整个求解器的主要开销。它其实对 p
    仿射，所以把常量部分预先算好，内层只剩一次矩阵加法。
    """
    k = phys.k
    base = np.zeros((k, k))
    coef = np.zeros((k, k))
    diag = np.zeros(k)
    idx = np.arange(k)
    for c in range(k):
        if phys.is_rf[c]:
            base[c] = phys.h_rf_off[c]
            continue
        base[c] = phys.shot * phys.eta * phys.h_vlc[c]
        diag[c] = phys.shot * phys.eta * phys.h_vlc[c, c]
        if phys.linear:
            base[c] += phys.eta * phys.h_vlc_off[c]
        else:
            coef[c] = 2.0 * (phys.rho ** 2) * (phys.h_opt[c] ** 2)
    return base, coef, diag, idx


def _linear_rates(phys, anchor):
    """在 ``anchor`` 处把速率线性化，返回凹下界 ``r~(p)`` 及其雅可比。

    雅可比的第一维是**参数**（功率下标 c），第二维是**链路** k，与
    ``_Physics.dx`` 的约定一致。
    """
    y = phys.x(anchor) / (phys.gap * phys.D(anchor))
    yg = y * y * phys.gap
    base, coef, diag, idx = _dD_coefficients(phys)

    def bound(p):
        p = np.asarray(p, float)
        A = 2.0 * y * phys.x(p) - yg * phys.D(p)
        inside = A > _A_FLOOR
        A_eff = np.maximum(A, _A_FLOOR)
        rate = (phys.bw / _LOG2) * np.log1p(A_eff)
        dD = base + coef * p[:, None]
        dD[idx, idx] = diag
        dA = -yg[None, :] * dD
        # dx 是对角矩阵：线性/RF 是 a/(2 sqrt(a p))（随 p 变），平方律 VLC 是常数 c。
        root = np.sqrt(np.maximum(phys.a * p, 1e-300))
        dx_diag = np.where(phys.is_rf | phys.linear, phys.a / (2.0 * root), phys.c)
        dA[idx, idx] += 2.0 * y * dx_diag
        jac = (phys.bw / _LOG2)[None, :] * np.where(inside[None, :],
                                                    dA / (1.0 + A_eff)[None, :], 0.0)
        return rate, jac
    return bound


def _sca_step(phys, p, t, q, trust, denominator):
    """内层一步 SLSQP。``t`` 给定时是可行性检验（约束里减常数 t）。

    ``denominator`` 为 ``None`` 时，速率约束里的分母取 ``Pc_k + p_k`` 本身（与
    t 的差式）；给定时取固定向量 ``d_k``（瓶颈变量 s 的仿射化，见
    :func:`_direct_bottleneck_sca`）。
    """
    bound = _linear_rates(phys, p)
    k = phys.k
    scale = np.maximum(p, 0.01)
    # 约束必须**归一化**：速率与裕量的量级是 1e6~1e9，直接丢给 SLSQP 会让线搜索
    # 产生 NaN（实测全部锚点返回 solver_error）。这里把约束写成无量纲的相对裕量
    # ``g_k = (r_k - t*d_k)/|t*d_k| - sigma``，sigma 就是"相对最弱裕量"。
    if denominator is None:
        cnorm = np.maximum(np.abs(t * (phys.static + p)), 1e-30)
    else:
        cnorm = np.maximum(np.abs(denominator), 1e-30)
    qnorm = None if q is None else np.maximum(np.abs(q), 1.0)

    def constraints(w):
        pw = w[:k] * scale
        r, _jac = bound(pw)
        if denominator is None:
            rows = [(r - t * (phys.static + pw)) / cnorm - w[k]]
        else:
            rows = [r / cnorm - w[k]]
        if q is not None:
            rows.append((r - q) / qnorm)
        return np.concatenate(rows)

    def constraint_jac(w):
        pw = w[:k] * scale
        _r, jac = bound(pw)
        rows = [np.c_[jac * scale[None, :] / cnorm[:, None], -np.ones(k)]]
        if q is not None:
            # QoS 行与 sigma 无关，末列必须补 0，否则与上一行宽度不一致。
            rows.append(np.c_[jac * scale[None, :] / qnorm[:, None], np.zeros(k)])
        return np.vstack(rows)

    def objective(w):
        return -float(w[k]), np.r_[np.zeros(k), -1.0]

    x0 = np.r_[p / scale, 0.0]
    x0[k] = max(float(np.min(constraints(x0)[:k])), 0.0)
    low = np.maximum(phys.lo, p / trust)
    high = np.minimum(phys.hi, p * trust)
    bounds = list(zip(low / scale, np.maximum(high / scale, low / scale))) + [(0.0, None)]
    try:
        res = minimize(objective, x0, jac=True, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": constraints,
                                     "jac": constraint_jac}],
                       options=dict(maxiter=_SLSQP_MAXITER, ftol=_SLSQP_FTOL))
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return None
    cand = np.asarray(res.x[:k], float) * scale
    if not np.isfinite(cand).all():
        return None
    return np.clip(cand, phys.lo, phys.hi)


def _feasible_at(phys, anchor, t, q, max_sca, trust=2.0):
    """外层给定 t 时的一次可行性检验（内层 SCA）。返回 ``(可行?, 点, 轮数)``。

    ``trust`` 是每轮的信任域半径（相对倍数）。凹下界只在锚点附近有效：把 p
    从 0.3 W 一步跳到 0.01 W 时，线性化会把速率算成负数。信任域让每轮只走
    ``[p/trust, p*trust]``，多轮之后再线性化，代价是轮数，换来的是下界可信。
    """
    p = np.clip(np.asarray(anchor, float), phys.lo, phys.hi)
    rounds = 0
    reason = "rounds_exhausted"
    for _ in range(max(int(max_sca), 1)):
        rounds += 1
        cand = _sca_step(phys, p, t, q, trust, None)
        if cand is None:
            return False, p, rounds, "solver_error"
        ee = link_ee(phys, cand)
        rates = phys.rates(cand)
        ok_t = bool(ee.min() >= t * (1.0 - 1e-6))
        ok_q = bool(q is None or rates.min() >= float(np.min(q)) * (1.0 - 1e-6))
        if ok_t and ok_q:
            return True, cand, rounds, "ok"
        if ok_t and not ok_q:
            reason = "qos_short"
        elif float(ee.min()) > float(link_ee(phys, p).min()) + 1e-12:
            # 不可行时只在**真实**最弱裕量变大时才挪锚点，避免被线性化外推带走。
            reason = "step_then_stall"
            p = cand
            continue
        else:
            reason = "no_progress"
        break
    return False, p, rounds, reason


def _direct_bottleneck_sca(phys, anchor, q, max_sca, trust=2.0):
    """直接对瓶颈变量 s 做 SCA：``max s  s.t.  r~_k(p) >= s * (Pc_k + p_k)``。

    存在的原因：二分只回答"t 可行吗"，而 SCA 的线性化在实际最优 t 附近容易假
    不可行（真实速率够、线性下界不够），一旦把 ``t_hi`` 压低就会把真正可行的 t
    判死。这里先用同一个凹下界直接最大化瓶颈，得到一个**已验证可达**的强点，
    再拿它当二分的锚点与 ``t_lo``，从而把假不可行的窗口压到最优 t 的邻域里。

    分母在每轮锚点处固定为 ``d_k = Pc_k + p_k``，于是约束对 ``(p,s)`` 仿射/凹，
    子问题是凸的；这与直接写 ``r >= s*(Pc+p)``（对 s、p 双线性）不同，是标准的
    分母固定化处理，信任域保证它与真式在同一邻域内，最终仍用原始速率复验。
    """
    p = np.clip(np.asarray(anchor, float), phys.lo, phys.hi)
    best_p = p.copy()
    best = float(link_ee(phys, p).min())
    rounds = 0
    for _ in range(max(int(max_sca), 1)):
        rounds += 1
        d = phys.static + p
        cand = _sca_step(phys, p, None, q, trust, d)
        if cand is None:
            break
        value = float(link_ee(phys, cand).min())
        if value > best * (1.0 + 1e-12):
            best, best_p = value, cand.copy()
        if float(np.max(np.abs(cand - p))) <= 1e-9 * max(float(np.max(p)), 1e-9):
            p = cand
            break
        p = cand
    return best, best_p, rounds


def bottleneck_sca(links, gains, ports, cfg, *, p_start, q_min_bps=0.0,
                   max_bisect=12, max_sca=3, t_tol=1e-4):
    """固定匹配下最大化逐链路 EE 的下界。返回 ``(powers, BottleneckInfo)``。

    ``q_min_bps`` 可以是标量或逐链路向量；它只是**解析代理**约束（用块初信念的
    瞬时速率），不是概率口径的 p05 约束——真正的 p05 复核在调用方用条件轨迹
    预测器做。返回的功率一定落在 ``[lo, hi]`` 内，且当二分一次都没成功时等于
    输入（不会把结果改坏）。
    """
    if not links:
        return np.zeros(0), BottleneckInfo("empty", False, 0, 0, 0., 0., 0., 0., [])
    phys = _Physics(links, gains, ports, cfg)
    k = phys.k
    p0 = np.clip(np.asarray(p_start, float), phys.lo, phys.hi)
    if p0.shape != (k,):
        raise ValueError("p_start must have one power per link")
    q = None
    if q_min_bps is not None:
        q_arr = np.asarray(q_min_bps, float)
        if q_arr.ndim == 0:
            q_arr = np.full(k, float(q_arr))
        if q_arr.shape != (k,):
            raise ValueError("q_min_bps must be a scalar or one value per link")
        if np.any(q_arr > 0):
            q = q_arr

    start_min = float(link_ee(phys, p0).min())
    log = []
    sca_rounds = 0
    # 暖启动：多锚点直接对瓶颈变量做 SCA，取最好的可达点。锚点都只花解析评价。
    p = p0.copy()
    for anchor in (p0, phys.lo, phys.hi, np.sqrt(phys.lo * phys.hi)):
        value, cand, rounds = _direct_bottleneck_sca(phys, anchor, q, max_sca)
        sca_rounds += rounds
        if value > float(link_ee(phys, p).min()) * (1.0 + 1e-12):
            p = cand
    t_lo = float(link_ee(phys, p).min())
    t_hi = max(bottleneck_upper_bound(phys), t_lo)
    bisections = 0
    status = "bisection_budget"
    for _ in range(max(int(max_bisect), 0)):
        if t_hi - t_lo <= t_tol * max(abs(t_lo), 1.0):
            status = "interval_tol"
            break
        # EE 跨若干数量级，二分必须在**对数域**：算术中点会让区间在 20 步内
        # 只缩到 (t_hi-t_lo)/2^20，而 t_hi/t_lo 可能有 1e3 倍。
        t = float(np.sqrt(max(t_lo, 1e-30) * max(t_hi, 1e-29)))
        # 多起点：SCA 对锚点敏感，而每个锚点只花解析评价（不调预测器），
        # 从当前点、全域下界、全域上界、几何中点各试一次，任一可行即接受。
        anchors = [p, phys.lo, phys.hi, np.sqrt(phys.lo * phys.hi)]
        ok, cand, rounds, reason = False, p, 0, "no_anchor"
        for anchor in anchors:
            ok, cand, used, why = _feasible_at(phys, anchor, t, q, max_sca)
            rounds += used
            if ok:
                reason = "ok"
                break
            reason = why
        sca_rounds += rounds
        bisections += 1
        log.append(dict(t=float(t), feasible=bool(ok), sca_rounds=int(rounds),
                        reason=reason))
        if ok:
            p = cand
            t_lo = float(link_ee(phys, p).min())
            # 可达值永远不超过上界；若出现，说明上界算错，就地修正而不是让区间翻转。
            t_hi = max(t_hi, t_lo)
        else:
            t_hi = float(t)
    final_min = float(link_ee(phys, p).min())
    feasible = bool(final_min > start_min * (1.0 + 1e-12))
    if not feasible:
        status = "no_improvement"
    if q is not None and phys.rates(p).min() < float(np.min(q)) * (1.0 - 1e-6):
        status = "qos_proxy_violated"
    return p, BottleneckInfo(status, feasible, bisections, sca_rounds,
                             start_min, final_min, float(t_lo), float(t_hi), log,
                             [float(x) for x in _per_link_upper(phys)],
                             [float(x) for x in link_ee(phys, p0)])
