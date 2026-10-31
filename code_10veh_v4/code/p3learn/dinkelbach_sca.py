"""分式和的二次变换（fractional programming）+ SCA 内层：固定匹配下最大化均逐链路 EE。

**术语**：外层用的不是经典 Dinkelbach 的差式 `max sum(r_k - lam_k p_k)`，而是
Shen & Yu 的二次变换（arXiv:1802.10192），下面第 7--21 行说明了为什么差式在这里
是错的。函数名 `dinkelbach_sum_ee` 与 CLI 的 `--power-solver dinkelbach` 是历史
命名，为兼容旧 run 的 provenance 保留，不代表用的是经典 Dinkelbach。

目标（论文口径，也是 `_metrics` 用的口径）：

    F(p) = (1/K) * sum_k r_k(p) / (static_k + p_k)

这不是"总速率/总功率"。**先说一个容易写错的点**：把

    lam_k = r_k(p) / (static_k + p_k),   max_p sum_k [r_k - lam_k p_k]

直接当 Dinkelbach 用是**错的**。它的固定点条件 `sum_k dr_k/dp_c = lam_c` 与
原目标的 `sum_k (dr_k/dp_c)/d_k = r_c/d_c^2` 不是同一个条件（只有当所有 d_k 相等
时才重合）。单比值时两者一致，分式和不一致。本模块用正确的形式：

对 `F = sum_k f_k/g_k`（f_k = r_k，g_k = static_k + p_k）先做二次变换

    max_{p,y>=0}  sum_k [ 2 y_k sqrt(f_k(p)) - y_k^2 g_k(p) ]

对 y 最大化给出 y_k = sqrt(f_k)/g_k，回代得原目标，故与 F 等价；固定 y 后
目标对 p 凹（sqrt 是凹函数，f_k 凹时 sqrt(f_k) 凹）。更新 y 与 p 交替进行，
即 Dinkelbach 型的 fractional programming 外层。

内层 SCA：目标里的 f_k = r_k 因同频干扰对 p 不凹，所以再套一层凹下界。
取 x_k = sqrt(S_k)（RF 与线性 VLC 是 sqrt(a p)，凹；平方律 VLC 的 x = rho*eta*h*p
是仿射），D_k = 干扰 + 噪声（凸：线性项 + 正系数平方项）。固定 y_k 后有

    x_k^2 / (gap_k D_k) >= 2 y_k x_k - y_k^2 gap_k D_k,

右端对 p 凹，在 y_k = x_k/(gap_k D_k) 处取等号，于是

    r_k(p) >= r~_k(p) = B/ln2 * ln(1 + 2 y_k x_k(p) - y_k^2 gap_k D_k(p))

在有效域 `A_k > -1`（A_k = 2 y_k x_k - y_k^2 gap_k D_k）上是凹下界，`sqrt(r~_k)` 也凹。
**但代码为了让子问题处处有定义，对 A 做了 `max(A, -1/2)`、对速率做了 `max(r, 1e-9)`
两处截断，而 `max(凹函数, 常数)` 一般不再是凹函数**（反例：`max(-x^2, -1)` 在
x = 0 与 x = 2 的中点处违反弦条件）。因此内层子问题**不保证是凹规划**，L-BFGS-B
只是局部启发式，**不能声称解出了凹子问题、也不能声称收敛到驻点**。
唯一能保证的是：内层每轮都用**原始速率**复验，只有真实外层目标上升才接受，
所以这个分支不会把结果改坏，但也没有最优性/收敛性证明。
诊断脚本 `e:\\tmp\\scripts\\concavity_check.py`：未截断的 A 弦测试 0 违反，
加 `max(r,1e-9)` 后单条链路的违反率可达 20%。

物理与 `p3base.link_eval.evaluate`、`p3base.block.sample_metrics`、
`p3learn.allocation._rates_and_ee` 同一套（tests 已对拍）。这里只优化功率，
**不改匹配**（模式/配对由外层候选生成器给出）。
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from p3base.energy import static_power_w
from p3base.noise import vlc_noise_base_w2, vlc_shot_coeff

from .allocation import _link_gain_tables, _rates_and_ee


@dataclass
class DinkelbachInfo:
    outer_rounds: int
    sca_rounds: int
    residual: float            # 原始均逐链路 EE 相对上一轮的变化量
    mean_ee: float             # 原始速率的均逐链路 EE（目标本身）
    status: str                # "stationary" 仅指外层不再改进，不是驻点证明
    improved_over_start: bool


class _Physics:
    """把 (links, 块初信念增益) 整理成 SCA 需要的系数。"""

    def __init__(self, links, gains, ports, cfg):
        self.cfg = cfg
        self.links = list(links)
        self.k = k = len(self.links)
        s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
        self.s_gain, self.h_rf, self.h_vlc, self.h_opt = s_gain, h_rf, h_vlc, h_opt
        self.h_rf_off, self.h_vlc_off, self.h_opt_off = (a.copy() for a in (h_rf, h_vlc, h_opt))
        for h in (self.h_rf_off, self.h_vlc_off, self.h_opt_off):
            np.fill_diagonal(h, 0.0)
        self.is_rf = np.array([m == "RF" for _i, _j, m in self.links])
        self.linear = cfg.vlc.sinr_model == "linear"
        self.eta = cfg.vlc.eo_efficiency
        self.rho = cfg.vlc.responsivity
        # v4：RF 的功放效率乘在信号与干扰的**分子**上。``rates()`` 走
        # ``_rates_and_ee``，那里已经乘过一次，所以这里只给 SCA 自己用的系数
        # （``a`` 与 ``h_rf_off``）补上，不能动 ``s_gain`` / ``h_rf``。
        eta_rf = float(cfg.rf.pa_efficiency)
        self.eta_rf = eta_rf
        self.h_rf_off = self.h_rf_off * eta_rf
        # x_k = sqrt(a_k p_k)（RF / 线性 VLC）；平方律 VLC 的 x_k = c_k p_k
        self.a = np.where(self.is_rf, eta_rf * s_gain, self.eta * s_gain)
        self.c = self.rho * self.eta * s_gain
        self.bw = np.where(self.is_rf, cfg.rf.bandwidth_hz, cfg.vlc.bandwidth_hz).astype(float)
        self.gap = np.where(self.is_rf, cfg.rf.gap, cfg.vlc.gap).astype(float)
        self.noise_rf = float(cfg.rf.noise_w)
        self.noise_base = float(vlc_noise_base_w2(cfg.vlc))
        self.shot = float(vlc_shot_coeff(cfg.vlc)) * self.rho
        self.static = np.array([static_power_w(m, cfg) for _i, _j, m in self.links])
        self.hi = np.array([getattr(cfg, m.lower()).power_max_w
                            for _i, _j, m in self.links])
        self.lo = np.array([getattr(cfg, m.lower()).power_min_w for _i, _j, m in links])

    # ---- 基本量 ------------------------------------------------------
    def x(self, p):
        if self.linear:
            return np.sqrt(np.maximum(self.a * p, 1e-300))
        lin = np.sqrt(np.maximum(self.a * p, 1e-300))
        return np.where(self.is_rf, lin, self.c * p)

    def D(self, p):
        p_rf = np.where(self.is_rf, p, 0.0)
        p_vlc = np.where(self.is_rf, 0.0, p)
        optical = self.h_opt.T @ p_vlc                     # eta*h*p，含自身
        inter_rf = self.h_rf_off.T @ p_rf
        if self.linear:
            inter_vlc = self.eta * (self.h_vlc_off.T @ p_vlc)
        else:
            inter_vlc = (self.rho ** 2) * ((self.h_opt_off ** 2).T @ (p_vlc ** 2))
        noise = np.where(self.is_rf, self.noise_rf,
                         self.noise_base + self.shot * optical)
        return np.where(self.is_rf, inter_rf + noise, inter_vlc + noise)

    def dx(self, p):
        """(K_param, K_link)：d x_k / d p_c（只有 c == k 非零）。"""
        out = np.zeros((self.k, self.k))
        root = np.sqrt(np.maximum(self.a * p, 1e-300))
        d_lin = self.a / (2.0 * root)
        for c in range(self.k):
            out[c, c] = d_lin[c] if (self.linear or self.is_rf[c]) else self.c[c]
        return out

    def dD(self, p):
        """(K_param, K_link)：d D_k / d p_c。"""
        k = self.k
        out = np.zeros((k, k))
        idx = np.arange(k)
        p_vlc = np.where(self.is_rf, 0.0, p)
        for c in range(k):
            if self.is_rf[c]:
                row = self.h_rf_off[c].copy()             # RF 只干扰 RF
                out[c] += row
                continue
            row = self.shot * self.eta * self.h_vlc[c]    # 散粒噪声随光功率（含自身）
            if self.linear:
                row = row + self.eta * self.h_vlc_off[c]
            else:
                row = row + 2.0 * (self.rho ** 2) * (self.h_opt[c] ** 2) * p_vlc[c]
                row[idx == c] = self.shot * self.eta * self.h_vlc[c, c]
            out[c] += row
        return out

    # ---- 速率与雅可比 -------------------------------------------------
    def rates(self, p):
        return _rates_and_ee(p, self.links, self.cfg, self.s_gain, self.h_rf,
                             self.h_vlc, self.h_opt)[0]

    def rate_jac(self, p):
        """(K_param, K_link)：d r_k / d p_c。"""
        x, D = self.x(p), self.D(p)
        gD = self.gap * D
        u = x ** 2 / gD
        dx, dD = self.dx(p), self.dD(p)
        du = (2.0 * x[None, :] * dx * gD[None, :]
              - (x ** 2 * self.gap)[None, :] * dD) / (gD ** 2)[None, :]
        return (self.bw / np.log(2.0))[None, :] * du / (1.0 + u)[None, :]

    def mean_ee(self, p):
        return float(np.mean(self.rates(p) / (self.static + p)))

    def fractional_objective(self, p, y):
        """sum_k [2 y_k sqrt(r_k) - y_k^2 (static_k + p_k)]，用**原始速率**。"""
        r = np.maximum(self.rates(p), 1e-9)
        return float(np.sum(2.0 * y * np.sqrt(r) - y ** 2 * (self.static + p)))


def _bound(phys, y, p):
    """凹下界 r~_k(p) 与雅可比。

    下界的有效域是 ``1 + A > 0``（A 是 y 的向下抛物线）。子问题在 p 的箱约束里
    搜索时可能越过边界，所以这里用 ``1 + max(A, -1/2)`` 延拓，越界区变成"平"而不是
    "无定义"。

    **注意：这个延拓不保持凹性。** ``max(凹函数, 常数)`` 一般不是凹函数
    （``max(-x^2, -1)`` 即是反例），所以子问题不是凹规划，见模块 docstring。
    越界点若被采纳，``dinkelbach_sum_ee`` 会用原始速率复验，不会把无效外推当进展。
    """
    x, D = phys.x(p), phys.D(p)
    A = 2.0 * y * x - y ** 2 * phys.gap * D
    inside = A > -0.5
    A_eff = np.maximum(A, -0.5)
    arg = 1.0 + A_eff
    dx, dD = phys.dx(p), phys.dD(p)
    rate = (phys.bw / np.log(2.0)) * np.log1p(A_eff)
    dA = 2.0 * y[None, :] * dx - (y ** 2 * phys.gap)[None, :] * dD
    jac = (phys.bw / np.log(2.0))[None, :] * np.where(inside[None, :], dA / arg[None, :], 0.0)
    return rate, jac, arg


def dinkelbach_sum_ee(links, gains, ports, cfg, *, p_start=None,
                      max_outer=15, max_sca=6, tol=1e-6, verbose=False):
    """固定匹配下最大化 mean_k r_k/(static_k+p_k)。返回 (powers, DinkelbachInfo)。"""
    if not links:
        return np.zeros(0), DinkelbachInfo(0, 0, 0.0, float("nan"), "empty", False)
    phys = _Physics(links, gains, ports, cfg)
    lo, hi = phys.lo, phys.hi
    p = np.clip(np.full(phys.k, 0.05) if p_start is None else np.asarray(p_start, float),
                lo, hi)
    start_ee = phys.mean_ee(p)
    sca_total = 0
    status = "iteration_budget"
    residual = float("inf")
    outer = 0
    prev_ee = start_ee
    for outer in range(max_outer):
        # 外层（分式规划）：固定 y = sqrt(r)/d，内层对 p 最大化二次变换目标。
        y = np.sqrt(np.maximum(phys.rates(p), 1e-9)) / (phys.static + p)
        best_val = phys.fractional_objective(p, y)
        q = p.copy()
        for _sca in range(max_sca):
            yy = phys.x(q) / (phys.gap * phys.D(q))     # 内层 SCA 的速率凹下界

            def sub(v):
                r, j, _a = _bound(phys, yy, v)
                pos = r > 1e-9
                root = np.sqrt(np.where(pos, r, 1e-9))
                val = -float(np.sum(2.0 * y * root - y ** 2 * (phys.static + v)))
                if not np.any(pos):
                    return val, np.zeros_like(v)
                g = (j * np.where(pos, y / root, 0.0)[None, :]).sum(axis=1) - y ** 2
                return val, -g

            try:
                res = minimize(sub, x0=q, jac=True, method="L-BFGS-B",
                               bounds=list(zip(lo, hi)),
                               options=dict(maxiter=500, ftol=1e-15, gtol=1e-12))
            except Exception:                        # noqa: BLE001
                break
            sca_total += 1
            cand = np.clip(np.asarray(res.x, float), lo, hi)
            val = phys.fractional_objective(cand, y)
            if val > best_val + 1e-9:
                best_val, q = val, cand.copy()
            else:
                break
        moved = float(np.max(np.abs(q - p)))
        p = q
        ee = phys.mean_ee(p)
        residual = ee - prev_ee          # 用原始目标的变化量作残差
        prev_ee = ee
        if verbose:
            print("  outer %2d meanEE=%.6e dEE=%.2e move=%.2e"
                  % (outer, ee, residual, moved))
        if moved < tol or (residual <= 1e-12 and outer > 0):
            status = "stationary"
            break
    mean_ee = phys.mean_ee(p)
    return np.clip(p, lo, hi), DinkelbachInfo(
        outer_rounds=outer + 1, sca_rounds=sca_total, residual=residual,
        mean_ee=mean_ee, status=status,
        improved_over_start=bool(mean_ee > start_ee * (1 + 1e-12)))
