"""Complete, causal block allocation with a small frozen pure-EE reference.

The objective is the *mean of the per-link energy efficiencies*: each link's own
trajectory-mean rate divided by its own power, then averaged.  Network EE (total
rate over total power) and the minimum per-link EE are still reported, but only
as diagnostics.  Only block-start eligibility, selected ports, optional
posterior gain means and an injected conditional-trajectory evaluator enter this
module. No Scene, future truth or Jain term is accepted. An explicit predicted
block-average QoS threshold is supported by the continuous solver. Proxy
link scores propose complete actions; *all final comparisons* use the
evaluator's trajectory-mean block rate. The reference is a fixed three-candidate
heuristic, not a global EE optimum.
"""
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Callable

import networkx as nx
import numpy as np

from p3base.block import validate_action
from p3base.energy import link_electrical_power_w, static_power_w
from p3base.noise import vlc_noise_base_w2, vlc_noise_linear_w, vlc_shot_coeff


MODES = ("RF", "VLC")
# v2: electrical denominators, linear VLC, equivalent-SINR candidate matchings,
# and coordinate ascent over powers on every candidate.  Any change here must
# bump the version, because eta_ref (the budget) changes with it.
# v3 (2026-10-26): static power moved to the communication front-end convention
# (RF 0.1159 W, VLC 1.745 W).  The search strategy is unchanged, but every
# eta_ref moves, so outputs of v2 and v3 must never be compared or mixed.
# v4 (2026-10-31): the RF power amplifier efficiency eta_PA is applied to the
# signal numerator, so the action power p means **electrical input power** and
# the RF range becomes 0.01-20 W (= 1 mW-2 W radiated).  Every eta_ref moves
# again; v3 outputs are not comparable.
REFERENCE_VERSION = "pure-mean-electrical-ee-joint-mode-power-v4"
EQU_SINR_GAP = 3.3   # dink10.m 的等效 SINR 常数，固定值，不随 cfg.vlc.gap 变
MIN_USABLE_RATE_BPS = 1e4   # 低于此速率的模式切换视为把链路牺牲掉，直接拒绝


class CoverageFailure(ValueError):
    """The full eligibility graph has no complete matching; no cars are dropped."""


@dataclass(frozen=True)
class Action:
    """One fixed block action; arrays are link powers and full NxN port choices."""

    links: list[tuple[int, int, str]]
    powers: np.ndarray
    ports: dict[str, np.ndarray]


@dataclass(frozen=True)
class ReferencePolicy:
    """Frozen low-cost proposal/search budget, identical for every vehicle count."""

    version: str = REFERENCE_VERSION
    powers_w: tuple[float, ...] = (0.01, 0.3, 3.0)

    def __post_init__(self):
        # Changing this strategy requires an explicit version, recorded in output.
        p = np.asarray(self.powers_w, float)
        if (not isinstance(self.version, str) or not self.version or p.shape != (3,)
                or not np.isfinite(p).all() or np.any(p < .01) or np.any(p > 20)):
            raise ValueError("ReferencePolicy requires a version and three finite powers in [0.01, 20] W")
        if self.version == REFERENCE_VERSION and tuple(self.powers_w) != (.01, .3, 3.0):
            raise ValueError("A changed reference power grid requires a new version")


@dataclass(frozen=True)
class AllocationResult:
    """An executed candidate or explicit block failure, plus independently auditable cost."""

    action: Action | None
    forecast: dict | None
    eta_ref: float | None
    network_ee: float | None
    min_pair_ee: float | None
    budget_violation: float | None = None
    fallback: bool = False
    coverage_failure: bool = False
    reason: str = ""
    solve_time_s: float = 0.
    evaluator_calls: int = 0
    reference_version: str = REFERENCE_VERSION
    reference_solve_time_s: float = 0.
    reference_evaluator_calls: int = 0
    delta: float | None = None
    attempted_candidates: int = 0
    rejected_candidates: int = 0
    evaluator_id: int | tuple[int, int] | None = None
    policy_config: dict = field(default_factory=dict)
    # Mean of per-link EE.  ``eta_ref`` and the budget both use this quantity;
    # ``network_ee`` is reported separately and no longer defines the budget.
    mean_pair_ee: float | None = None
    # Number of links whose predicted QoS variable misses q_min_bps.  Only set on
    # the maxmin/protect_budget path; ``None`` means "not checked" (the reference
    # already encodes QoS, so re-checking it would double-count).
    qos_violation_links: int | None = None

    @property
    def success(self):
        return self.action is not None

    @property
    def proposal_budget_violation(self):
        """Attempted action shortfall, retained even if the legal guard falls back."""
        return self.budget_violation

    @property
    def executed_budget_violation(self):
        """Predicted shortfall of the action actually returned; None on failure/reference."""
        if not self.success or self.delta is None or self.eta_ref is None:
            return None
        value = self.mean_pair_ee if self.mean_pair_ee is not None else self.network_ee
        return max(0., (1 - self.delta) * self.eta_ref - value)


def copy_action(action):
    """Snapshot an action so later actor edits cannot alter a frozen reference."""
    return Action(list(action.links), np.array(action.powers, dtype=float, copy=True),
                  {key: np.array(action.ports[key], copy=True) for key in ("tx", "rx")})


def _bool_matrix(value, n, name):
    matrix = np.asarray(value)
    if matrix.shape != (n, n) or matrix.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean NxN matrix")
    return matrix


def _domain(n, cfg, candidates, business, ports):
    """Validate the complete block-start domain, without imposing a neighbor cap."""
    if not isinstance(n, (int, np.integer)) or n < 2 or n % 2:
        raise ValueError("n must be a positive even vehicle count of at least 2")
    if not isinstance(candidates, dict) or any(mode not in candidates for mode in MODES):
        raise ValueError("candidates must contain RF and VLC masks")
    bm = _bool_matrix(business, n, "business")
    masks = {mode: _bool_matrix(candidates[mode], n, mode) & bm for mode in MODES}
    for mode in MODES:
        masks[mode] = masks[mode].copy()
        np.fill_diagonal(masks[mode], False)
        cap = getattr(cfg, mode.lower()).power_max_w
        floor = getattr(cfg, mode.lower()).power_min_w
        if (not np.isfinite(cap) or not np.isfinite(floor) or not 0 < floor <= cap):
            raise ValueError(f"{mode} data power range must be finite and 0 < min <= max")
    if not isinstance(ports, dict) or any(key not in ports for key in ("tx", "rx")):
        raise ValueError("ports must contain full NxN tx and rx choices")
    port_copy = {}
    for key in ("tx", "rx"):
        matrix = np.asarray(ports[key])
        if matrix.shape != (n, n) or not np.isfinite(matrix).all():
            raise ValueError("port choices must be finite NxN matrices")
        if np.any(~np.isin(matrix, (0, 1))):
            raise ValueError("full fixed port choices must be 0 or 1")
        port_copy[key] = matrix.astype(int, copy=True)
    return masks, port_copy


def has_perfect_matching(mask):
    """Exact feasibility of an entire directed boolean endpoint graph.

    Used by an autoregressive actor after tentatively fixing an edge. Both
    directions represent the same endpoint conflict. The empty remainder is
    feasible; an odd remainder is not. This never prunes business candidates.
    """
    value = np.asarray(mask)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.dtype.kind != "b":
        raise ValueError("matching mask must be a square boolean matrix")
    n = len(value)
    if n % 2:
        return False
    adjacent = value | value.T
    adjacent = adjacent.copy()
    np.fill_diagonal(adjacent, False)
    # A completed greedy matching is an exact feasibility certificate. This
    # accelerates repeated actor lookahead on dense 50/100-car graphs. An
    # incomplete greedy run never declares infeasibility: blossom decides it.
    remaining = np.ones(n, dtype=bool)
    degree = adjacent.sum(axis=1)
    while remaining.any():
        i = int(np.argmin(np.where(remaining, degree, n + 1)))
        neighbors = adjacent[i] & remaining
        if not neighbors.any():
            break
        j = int(np.argmin(np.where(neighbors, degree, n + 1)))
        remaining[[i, j]] = False
        degree -= adjacent[:, i].astype(int) + adjacent[:, j].astype(int)
    if not remaining.any():
        return True
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    i, j = np.where(np.triu(adjacent, 1))
    graph.add_edges_from(zip(i.tolist(), j.tolist()))
    return 2 * len(nx.max_weight_matching(graph, maxcardinality=True)) == n


def decode_complete_pairing(n, cfg, candidates, business, ports, edge_scores=None,
                            power_w=.01):
    """Decode maximum-weight *complete* matching over every legal directed mode.

    edge_scores maps RF/VLC to finite NxN scores (only legal entries are read).
    power_w is a scalar proposal shared by links, clipped to each mode's cap.
    Scores select direction/mode within each endpoint pair before an exact
    maximum-cardinality matching. A failed graph raises CoverageFailure.
    """
    masks, port_copy = _domain(n, cfg, candidates, business, ports)
    # v4：两种模式都是电输入功率，区间由配置给出（RF 0.01–20 W、VLC 0.1–30 W）。
    # 这里的标量提议只要落在两种模式的并集内即可，逐链路裁剪在下面按模式做。
    lo = min(cfg.rf.power_min_w, cfg.vlc.power_min_w)
    hi = max(cfg.rf.power_max_w, cfg.vlc.power_max_w)
    if not np.isscalar(power_w) or not np.isfinite(power_w) or not lo <= power_w <= hi:
        raise ValueError(f"power_w must be finite and in [{lo:g}, {hi:g}] W")
    if edge_scores is None:
        scores = {mode: np.zeros((n, n)) for mode in MODES}
    else:
        if not isinstance(edge_scores, dict) or any(mode not in edge_scores for mode in MODES):
            raise ValueError("edge_scores must contain RF and VLC matrices")
        scores = {mode: np.asarray(edge_scores[mode], float) for mode in MODES}
        if any(scores[mode].shape != (n, n) or not np.isfinite(scores[mode][masks[mode]]).all()
               for mode in MODES):
            raise ValueError("all legal edge scores must be finite; eligibility must not be hidden in scores")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            options = [(float(scores[mode][tx, rx]), tx, rx, mode)
                       for tx, rx in ((i, j), (j, i)) for mode in MODES if masks[mode][tx, rx]]
            if options:
                # Stable tie break, including direction and mode; scores can be negative.
                option = max(options, key=lambda item: (item[0], -item[1], -item[2], item[3] == "RF"))
                graph.add_edge(i, j, weight=option[0], link=option[1:])
    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    if 2 * len(matching) != n:
        raise CoverageFailure("coverage_failure: no complete matching in the full candidate/business graph")
    links = sorted((tuple(graph.edges[i, j]["link"]) for i, j in matching), key=lambda link: min(link[:2]))
    # v4：标量提议按模式裁剪到各自的 [power_min_w, power_max_w]。
    p = np.array([float(np.clip(power_w, getattr(cfg, mode.lower()).power_min_w,
                                getattr(cfg, mode.lower()).power_max_w))
                  for _, _, mode in links])
    action = Action(links, p, port_copy)
    _legal(action, n, cfg, candidates, business)
    return action


def random_legal_action(n, cfg, candidates, business, ports, rng):
    """在块初合法域上**均匀**抽一个完整配对动作（随机下界用）。

    * 配对：每个端点对 (i, j) 先在合法有向 (tx, rx, mode) 里等概率抽一条，
      再做最大基数匹配；这样"随机"不会偏向任何模式或方向；
    * 功率：在各自模式的 log 域区间里均匀抽，两侧都不会被系统性高估；
    * 每块只抽一次。抽多次再取最好的会变成带选择偏差的"随机搜索"，那就不是下界了。

    ``rng`` 必须是 ``np.random.Generator``，同一 seed 必须复现同一动作。
    """
    import networkx as nx

    masks, port_copy = _domain(n, cfg, candidates, business, ports)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            options = [(tx, rx, mode) for tx in (i, j) for rx in (i, j) if tx != rx
                       for mode in MODES if masks[mode][tx, rx]]
            if options:
                pick = options[int(rng.integers(len(options)))]
                graph.add_edge(i, j, weight=float(rng.random()), link=pick)
    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    if 2 * len(matching) != n:
        raise CoverageFailure("coverage_failure: no complete matching for the random baseline")
    links = sorted((tuple(graph.edges[i, j]["link"]) for i, j in matching),
                   key=lambda link: min(link[:2]))
    powers = []
    for _i, _j, mode in links:
        model = getattr(cfg, mode.lower())
        powers.append(float(10.0 ** rng.uniform(np.log10(model.power_min_w),
                                                np.log10(model.power_max_w))))
    action = Action(links, np.asarray(powers, float), port_copy)
    _legal(action, n, cfg, candidates, business)
    return action


def random_allocate(n, cfg, candidates, business, ports, evaluator, *, delta, seed, reference=None):
    """随机下界：一次随机合法动作 + 一次预测评价，**不**走预算/QoS 守卫。

    它就是要展示"不做任何优化会怎样"，所以不享守卫；但预算与 QoS 的违约会由
    输出行如实记录，不能当成达标。
    """
    start = perf_counter()
    _check_delta(delta)
    rng = np.random.default_rng(seed)
    try:
        action = random_legal_action(n, cfg, candidates, business, ports, rng)
    except CoverageFailure as exc:
        return _failure(str(exc), start, coverage=True, reference=reference, delta=delta)
    forecast = _forecast(evaluator, action)
    network, mean_pair, weakest = _metrics(action, forecast, cfg)
    eta_ref = None if reference is None else reference.eta_ref
    return AllocationResult(
        action, forecast, eta_ref, network, weakest,
        budget_violation=(None if eta_ref is None else max(0., (1 - delta) * eta_ref - mean_pair)),
        reason="uniform_random_legal_action", solve_time_s=perf_counter() - start,
        evaluator_calls=1, delta=delta, attempted_candidates=1,
        evaluator_id=_evaluator_identity(evaluator), mean_pair_ee=mean_pair,
        reference_version=(REFERENCE_VERSION if reference is None else reference.reference_version),
        reference_solve_time_s=0. if reference is None else reference.reference_solve_time_s,
        reference_evaluator_calls=0 if reference is None else reference.reference_evaluator_calls,
        policy_config={} if reference is None else dict(reference.policy_config))


def _legal(action, n, cfg, candidates, business):
    """Apply the common physical action contract with extra array-shape checks."""
    if not isinstance(action, Action) or np.asarray(action.powers).shape != (n // 2,):
        raise ValueError("expected an Action with one power per complete link")
    if any(np.asarray(action.ports.get(key)).shape != (n, n) for key in ("tx", "rx")):
        raise ValueError("action ports must be full NxN matrices")
    validate_action(action.links, action.powers, action.ports, n, cfg,
                    candidates=candidates, business=business)


def _forecast(evaluator, action):
    """Enforce trajectory-mean scoring; quantiles remain diagnostics only."""
    value = evaluator(copy_action(action))
    if not isinstance(value, dict):
        raise ValueError("evaluator must return a prediction dictionary")
    k = len(action.links)
    samples = np.asarray(value.get("block_rate_samples_bps"), float)
    mean = np.asarray(value.get("mean_rate_bps"), float)
    # The point-information ablation is a degenerate predictive distribution: it
    # deliberately carries a single trajectory and is only usable for the mean
    # objective (a 5% quantile of one sample is not a quantile).  Any real
    # predictor must still supply at least two draws.
    min_draws = 1 if value.get("information") == "point" else 2
    if samples.ndim != 2 or samples.shape[0] < min_draws or samples.shape[1] != k:
        raise ValueError(f"conditional block-rate samples must have shape (draws>={min_draws}, links)")
    if (mean.shape != (k,) or not np.isfinite(samples).all() or np.any(samples < 0)
            or not np.isfinite(mean).all() or np.any(mean < 0)
            or not np.allclose(mean, samples.mean(axis=0), rtol=1e-9, atol=1e-6)):
        raise ValueError("mean_rate_bps must equal the Monte Carlo mean of complete block trajectory rates")
    result = dict(value)
    result["mean_rate_bps"] = samples.mean(axis=0).copy()
    result["block_rate_samples_bps"] = samples.copy()
    for key in ("low_rate_bps", "high_rate_bps"):
        diagnostic = np.asarray(value.get(key), float)
        if diagnostic.shape != (k,) or not np.isfinite(diagnostic).all() or np.any(diagnostic < 0):
            raise ValueError("predictor must provide finite nonnegative link-rate quantile diagnostics")
        result[key] = diagnostic.copy()
    if np.any(result["low_rate_bps"] > result["high_rate_bps"]):
        raise ValueError("lower prediction diagnostic cannot exceed upper diagnostic")
    return result


def action_metrics(action, forecast, cfg=None):
    """(network EE, mean per-link EE, minimum per-link EE).

    ``network EE`` is total throughput over total *transmit* power.  ``mean
    per-link EE`` is each link's own rate divided by its own electrical power
    (``static + p``), then averaged -- that is the objective.  ``minimum
    per-link EE`` is kept only as a fairness diagnostic.

    Without ``cfg`` the denominator falls back to the transmit power alone,
    which is only for legacy diagnostics; every decision path passes ``cfg``.

    Public so diagnostics can score counterfactual actions without importing a
    private helper.
    """
    rates = np.asarray(forecast["mean_rate_bps"], float)
    powers = np.asarray(action.powers, float)
    if cfg is None:
        denom = powers
    else:
        denom = link_electrical_power_w(powers, [m for _i, _j, m in action.links], cfg)
    pair_ee = rates / denom
    return (float(rates.sum() / powers.sum()), float(pair_ee.mean()), float(pair_ee.min()))


def _metrics(action, forecast, cfg=None):
    return action_metrics(action, forecast, cfg)


def _evaluator_identity(evaluator):
    """Bound methods preserve identity across repeated attribute lookups."""
    owner, function = getattr(evaluator, "__self__", None), getattr(evaluator, "__func__", None)
    return (id(owner), id(function)) if owner is not None and function is not None else id(evaluator)


def _flat_gain(n, mode, gain, ports):
    """把 VLC 的 (n,n,2,2) 端口增益压成块初选定的 (n,n)。"""
    gain = np.asarray(gain, float)
    if mode == "VLC" and gain.shape == (n, n, 2, 2):
        tx, rx = np.indices((n, n))
        gain = gain[tx, rx, ports["tx"], ports["rx"]]
    if gain.shape != (n, n) or not np.isfinite(gain).all() or np.any(gain < 0):
        raise ValueError("belief gains must be finite nonnegative NxN (or VLC NxNx2x2)")
    return gain


def _vlc_optical_sinr(p, gain, cfg, noise=None):
    """单链路 VLC 的 SINR（不含其它 VLC 链路的干扰）。

    接收光功率 P_opt = eta*p*h，光电检测给出 i = R*P_opt：
    平方律 SINR = (R P_opt)^2 / sigma^2；``linear`` 口径为 P_opt / N，
    仅在把 P_opt 与噪声视作同一单位时成立（见 ``noise.vlc_noise_linear_w``）。
    """
    eta = cfg.vlc.eo_efficiency
    optical = eta * np.asarray(p, float) * np.asarray(gain, float)
    nz = vlc_noise_linear_w(optical, cfg.vlc) if noise is None else noise
    if cfg.vlc.sinr_model == "linear":
        return optical / nz
    rho = cfg.vlc.responsivity
    return (rho * optical) ** 2 / nz


def _vlc_optical_rate(p, gain, cfg):
    return cfg.vlc.bandwidth_hz * np.log2(1.0 + _vlc_optical_sinr(p, gain, cfg) / cfg.vlc.gap)


def _proxy_scores(n, cfg, ports, belief_gains, power_w):
    """Cheap isolated-link EE means propose actions; these never define the budget.

    Isolated-link EE uses the same electrical denominator as the objective,
    ``static_power + p``; otherwise strong links would be systematically
    preferred at every power.
    """
    if belief_gains is None:
        return {mode: np.ones((n, n)) for mode in MODES}
    output = {}
    for mode in MODES:
        if mode not in belief_gains:
            raise ValueError("belief_gains must contain RF power gain and VLC optical gain")
        gain = _flat_gain(n, mode, belief_gains[mode], ports)
        model = cfg.rf if mode == "RF" else cfg.vlc
        p = float(np.clip(power_w, model.power_min_w, model.power_max_w))
        if mode == "RF":
            snr = cfg.rf.pa_efficiency * p * gain / model.noise_w
        else:
            snr = _vlc_optical_sinr(p, gain, cfg)
        rate = model.bandwidth_hz * np.log1p(snr / model.gap) / np.log(2.)
        output[mode] = rate / (static_power_w(mode, cfg) + p)
        if not np.isfinite(output[mode]).all():
            raise ValueError("nonfinite isolated-link proposal scores")
    return output


def _equ_sinr_scores(n, cfg, ports, belief_gains, power_w):
    """dink10.m 的等效 SINR 选模规则，作为候选匹配的来源。

    ``equ = 2*((1 + SINR_VLC/3.3)^(B_VLC/B_RF) - 1)`` 把 VLC 的 SINR 换算成
    "同一速率下需要的 RF SINR"，再与 SINR_RF 直接比较：VLC 分高就选 VLC。
    这条规则只看速率，不看能耗，所以在 EE 口径下会选错（实测掉 9%）；
    这里只让它**生成候选**，最终由联合评价按 EE 挑。
    """
    if belief_gains is None:
        return {mode: np.ones((n, n)) for mode in MODES}
    rf_gain = _flat_gain(n, "RF", belief_gains["RF"], ports)
    vlc_gain = _flat_gain(n, "VLC", belief_gains["VLC"], ports)
    p_rf = float(np.clip(power_w, cfg.rf.power_min_w, cfg.rf.power_max_w))
    p_vlc = float(np.clip(power_w, cfg.vlc.power_min_w, cfg.vlc.power_max_w))
    sinr_rf = cfg.rf.pa_efficiency * p_rf * rf_gain / cfg.rf.noise_w
    sinr_vlc = _vlc_optical_sinr(p_vlc, vlc_gain, cfg)
    ratio = cfg.vlc.bandwidth_hz / cfg.rf.bandwidth_hz
    with np.errstate(over="ignore", invalid="ignore"):
        equ = 2.0 * ((1.0 + sinr_vlc / EQU_SINR_GAP) ** ratio - 1.0)
    equ = np.where(np.isfinite(equ), equ, 1e300)
    return {"RF": sinr_rf, "VLC": equ}


def _link_gain_tables(links, gains, ports, cfg):
    """返回 (s_gain, H_rf, H_vlc, H_opt)，全是 KxK 或长度 K。

    ``s_gain[k]`` 是第 k 条链路的信号增益；``H_mode[b, k]`` 是第 b 条链路的
    发射端到第 k 条链路接收端的增益（同模才非零，对角为信号）。所有量都取自
    块初信念均值，是因果的。

    VLC 的交叉项必须用 **b 的发射端口 + k 的接收端口**：``_flat_gain`` 压出来的
    ``vlc[u, j]`` 用的是 ``(u, j)`` 自己配对的端口，对 ``u`` 的另一个接收端 ``j``
    是错的。``link_eval`` / ``block.sample_metrics`` 都按前一种取，所以这里必须
    显式索引端口，否则搜索层与执行层在多条 VLC 链路时不一致。
    """
    n = np.asarray(ports["tx"]).shape[0]
    rf = _flat_gain(n, "RF", gains["RF"], ports)
    vlc_grid = np.asarray(gains["VLC"], float)
    vlc = _flat_gain(n, "VLC", vlc_grid, ports)
    k = len(links)
    s_gain = np.zeros(k)
    h_rf = np.zeros((k, k))
    h_vlc = np.zeros((k, k))
    for a, (i, j, mode) in enumerate(links):
        s_gain[a] = rf[i, j] if mode == "RF" else vlc[i, j]
    for b, (u, v, mb) in enumerate(links):
        tx_b = int(ports["tx"][u, v])
        for a, (i, j, ma) in enumerate(links):
            if mb != ma:
                continue
            if mb == "RF":
                h_rf[b, a] = rf[u, j]
            elif vlc_grid.shape == (n, n, 2, 2):
                h_vlc[b, a] = vlc_grid[u, j, tx_b, int(ports["rx"][i, j])]
            else:
                h_vlc[b, a] = vlc[u, j]
    return s_gain, h_rf, h_vlc, cfg.vlc.eo_efficiency * h_vlc


def _rates_and_ee(p, links, cfg, s_gain, h_rf, h_vlc, h_opt):
    """给定功率向量，返回 (速率, 逐链路 EE)。干扰按同模全量累加。

    必须与 ``p3base.link_eval.evaluate``（执行层）逐位一致：
    ``tests/test_allocation.py::test_search_layer_matches_execution_layer_...``
    用非对称交叉增益对拍。H[b, a] 是"发送端 b -> 接收端 a"，
    接收端 a 的干扰是 ``H.T @ p``，写成 ``H @ p`` 会选错功率。
    """
    # v4：p 是电输入功率，RF 的辐射功率是 eta_PA*p。eta 乘在**信号与同模干扰**
    # 上（两者都是"辐射功率 × 增益"），不乘在噪声上。与 ``link_eval`` 同构。
    eta_rf = float(cfg.rf.pa_efficiency)
    modes = np.array([mode for _i, _j, mode in links])
    is_rf = modes == "RF"
    p_rf = np.where(is_rf, p, 0.0)
    p_vlc = np.where(is_rf, 0.0, p)
    diag_rf, diag_vlc = np.diag(h_rf), np.diag(h_vlc)
    inter_rf = eta_rf * (h_rf.T @ p_rf - p * diag_rf)
    inter_vlc = h_vlc.T @ p_vlc - p * diag_vlc
    # 到达接收端 a 的光功率之和（期望 + 干扰）。同样必须按 H[b, a] 取，
    # 写成 ``h_opt @ p_vlc`` 会把"a 的发射端打到 b 的接收端"当成 a 收到的光，
    # 使散粒噪声项落到错误的链路。
    optical = h_opt.T @ p_vlc
    bw = np.where(is_rf, cfg.rf.bandwidth_hz, cfg.vlc.bandwidth_hz)
    gap = np.where(is_rf, cfg.rf.gap, cfg.vlc.gap)
    if cfg.vlc.sinr_model == "linear":
        # dink10.m 口径：S = eta*h*p（光功率）。只有把光功率与噪声看作同一单位时
        # 才成立，见 ``noise.vlc_noise_linear_w`` 的量纲说明；不推荐作为论文默认。
        signal = np.where(is_rf, eta_rf * p * s_gain, p * diag_vlc * cfg.vlc.eo_efficiency)
        inter = inter_rf + cfg.vlc.eo_efficiency * inter_vlc
        noise = np.where(is_rf, cfg.rf.noise_w, vlc_noise_linear_w(optical, cfg.vlc))
    else:
        # 平方律：光电检测给出 i = R*P_opt，S = (R P_sig)^2，I = sum (R P_int)^2，
        # 噪声是 A^2 的方差。与 ``link_eval`` 完全同构。
        rho = cfg.vlc.responsivity
        sig_opt = p * np.diag(h_opt)                      # eta*h_des*p
        signal = np.where(is_rf, eta_rf * p * s_gain, (rho * sig_opt) ** 2)
        # 干扰必须按 H[b, a] 取（发送端 b -> 接收端 a），即 ``(H**2).T @ p**2``。
        # 之前写成 ``(H**2) @ p**2`` 再减去自身项：既乘反了方向，又因两项相差
        # 十几个数量级而灾难性相消（自干扰项 5.19e-12 与剩余 1.8e-14 相减）。
        # 先把对角清零再转置相乘，既对方向也无相消。
        h2 = h_opt ** 2
        np.fill_diagonal(h2, 0.0)
        inter_sq = (rho ** 2) * (h2.T @ (p_vlc ** 2))
        inter = np.where(is_rf, inter_rf, inter_sq)
        noise = np.where(is_rf, cfg.rf.noise_w, vlc_noise_linear_w(optical, cfg.vlc))
    sinr = signal / np.maximum(noise + inter, 1e-300)
    rate = bw * np.log2(1.0 + sinr / gap)
    ee = rate / link_electrical_power_w(p, modes, cfg)
    return rate, ee


def _rates_and_ee_batch(p, links, cfg, s_gain, h_rf, h_vlc, h_opt):
    """``_rates_and_ee`` 的批量版：``p`` 形状 (B, K)，返回同形状的 (速率, 逐链路 EE)。

    公式与 ``_rates_and_ee`` 逐项相同，只是把多组功率放进前导维度。存在的原因
    是性能：单次 ``_rates_and_ee`` 在 K=5 上也只要几十微秒，但坐标上升每条链路
    要调用 8 次、整个匹配搜索要跑上千次，NumPy 的调用开销就成了主项。批量后
    一次算完整条网格。``tests/test_dinkelbach.py`` 用 ``np.allclose(..., rtol=1e-13)``
    钉住两条路径的一致性。
    """
    pb = np.atleast_2d(np.asarray(p, float))
    eta_rf = float(cfg.rf.pa_efficiency)              # v4：见 ``_rates_and_ee``
    modes = np.array([mode for _i, _j, mode in links])
    is_rf = modes == "RF"
    p_rf = np.where(is_rf, pb, 0.0)
    p_vlc = np.where(is_rf, 0.0, pb)
    diag_rf, diag_vlc = np.diag(h_rf), np.diag(h_vlc)
    # 用 einsum 而不是 ``@``：这些矩阵只有 K×K（K<=100）量级，走 BLAS 没有收益，
    # 而 MKL 在"torch 已导入 + 部分线程池被设置过"的进程里对 (B,K)@(K,K) 这种
    # 小批量转置乘会直接 abort（本机 scipy+torch 环境实测）。einsum 是纯 C 循环。
    inter_rf = eta_rf * (np.einsum("bc,ca->ba", p_rf, h_rf) - pb * diag_rf)
    inter_vlc = np.einsum("bc,ca->ba", p_vlc, h_vlc) - pb * diag_vlc
    optical = np.einsum("bc,ca->ba", p_vlc, h_opt)
    bw = np.where(is_rf, cfg.rf.bandwidth_hz, cfg.vlc.bandwidth_hz)
    gap = np.where(is_rf, cfg.rf.gap, cfg.vlc.gap)
    if cfg.vlc.sinr_model == "linear":
        signal = np.where(is_rf, eta_rf * pb * s_gain, pb * diag_vlc * cfg.vlc.eo_efficiency)
        inter = inter_rf + cfg.vlc.eo_efficiency * inter_vlc
        noise = np.where(is_rf, cfg.rf.noise_w, vlc_noise_linear_w(optical, cfg.vlc))
    else:
        rho = cfg.vlc.responsivity
        sig_opt = pb * np.diag(h_opt)
        signal = np.where(is_rf, eta_rf * pb * s_gain, (rho * sig_opt) ** 2)
        h2 = h_opt ** 2
        np.fill_diagonal(h2, 0.0)
        inter_sq = (rho ** 2) * np.einsum("bc,ca->ba", p_vlc ** 2, h2)
        inter = np.where(is_rf, inter_rf, inter_sq)
        noise = np.where(is_rf, cfg.rf.noise_w, vlc_noise_linear_w(optical, cfg.vlc))
    sinr = signal / np.maximum(noise + inter, 1e-300)
    rate = bw * np.log2(1.0 + sinr / gap)
    static = np.where(is_rf, cfg.rf.circuit_power_w, cfg.vlc.circuit_power_w)
    return rate, rate / (static + pb)


def power_ascent(links, gains, ports, cfg, sweeps=3, points=12, start_w=None):
    """在块初信念增益上做单调坐标上升，最大化逐链路 EE 的均值。

    每一步只有在整条动作的均值提高时才落地（单调接受），所以不会回退；
    ``sweeps`` 是上限，未收敛也返回 best-seen。这是**提议**用的搜索，
    最终比较仍由条件轨迹评估器给出。

    v4：起始点与网格都按模式取 ``power_min_w``/``power_max_w``，不再假定
    两种模式共用 0.001–2 W。
    """
    if not links:
        return np.zeros(0)
    k = len(links)
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    floors = [getattr(cfg, m.lower()).power_min_w for _i, _j, m in links]
    p = np.array(floors if start_w is None
                 else [float(np.clip(start_w, fl, getattr(cfg, m.lower()).power_max_w))
                       for fl, (_i, _j, m) in zip(floors, links)], float)
    cur, ee = _rates_and_ee(p, links, cfg, s_gain, h_rf, h_vlc, h_opt)
    cur = float(ee.mean())
    grids = [np.logspace(np.log10(getattr(cfg, m.lower()).power_min_w),
                         np.log10(getattr(cfg, m.lower()).power_max_w), points)
             for _i, _j, m in links]
    for _ in range(sweeps):
        moved = 0
        for a in range(k):
            old, best_p, best_v = p[a], p[a], cur
            # 整条网格一次算完：逐点调用 `_rates_and_ee` 时 NumPy 的调用开销
            # 比算术本身还大（K=5 时 8 点要 0.5 ms，批量后约 0.06 ms）。
            grid = np.repeat(p[None, :], points, axis=0)
            grid[:, a] = grids[a]
            values = _rates_and_ee_batch(grid, links, cfg, s_gain, h_rf, h_vlc, h_opt)[1].mean(axis=1)
            pick = int(np.argmax(values))
            if values[pick] > best_v + 1e-9:
                best_v, best_p = float(values[pick]), float(grids[a][pick])
            p[a] = best_p
            if best_p != old:
                cur = best_v
                moved += 1
        if moved == 0:
            break
    return p


def repair_unusable_modes(links, powers, gains, ports, cfg, masks):
    """把速率低于 ``MIN_USABLE_RATE_BPS`` 的链路切到另一种模式（若合法且更快）。

    均值目标对"少一条链路"是麻木的：一条 0 速率的链路只减少一项，却能把干扰
    让给其余链路，目标反而上升。这里是显式守卫，不是目标的一部分。
    """
    links = list(links)
    powers = np.asarray(powers, float).copy()
    n = np.asarray(ports["tx"]).shape[0]
    rf = _flat_gain(n, "RF", gains["RF"], ports)
    vlc = _flat_gain(n, "VLC", gains["VLC"], ports)
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    rate, _ = _rates_and_ee(powers, links, cfg, s_gain, h_rf, h_vlc, h_opt)
    changed = False
    for a, (i, j, mode) in enumerate(links):
        if rate[a] >= MIN_USABLE_RATE_BPS:
            continue
        other = "VLC" if mode == "RF" else "RF"
        if not masks[other][i, j]:
            continue
        if other == "VLC":
            alt = _vlc_optical_rate(powers[a], vlc[i, j], cfg)
        else:
            alt = cfg.rf.bandwidth_hz * np.log2(
                1 + powers[a] * rf[i, j] / cfg.rf.noise_w / cfg.rf.gap)
        if alt > rate[a]:
            links[a] = (i, j, other)
            changed = True
    return links, powers, changed


def _failure(reason, start, calls=0, coverage=False, policy=None, delta=None, reference=None):
    policy = policy or ReferencePolicy()
    return AllocationResult(None, None, None if reference is None else reference.eta_ref,
                            None, None, coverage_failure=coverage, reason=reason,
                            solve_time_s=perf_counter() - start, evaluator_calls=calls,
                            reference_version=policy.version if reference is None else reference.reference_version,
                            reference_solve_time_s=0. if reference is None else reference.reference_solve_time_s,
                            reference_evaluator_calls=0 if reference is None else reference.reference_evaluator_calls,
                            delta=delta)


def _candidate_matchings(n, cfg, candidates, business, ports, belief_gains, policy,
                         extra_belief_gains=()):
    """等效 SINR 与 EE 两类分数各生成一组完整匹配，去重后返回。

    dink10.m 的选模规则只按速率排序，直接采用会降低 EE（实测 1 mW 下 -9%），
    所以它只负责"提出候选"；是否采用由条件轨迹评估器按均逐链路 EE 决定。

    ``extra_belief_gains``：额外的增益字典，只用它们**提出候选匹配**（分数与解码），
    不参与功率求解、修复或邻域排序。给 oracle 上界用：真值信念与估计信念各出一组
    起点，避免"同一启发式在不同起点上落到不同局部最优"把上界压低。
    """
    seen, output = set(), []
    for gains in (belief_gains, *extra_belief_gains):
        for power in policy.powers_w:
            for scores in (_proxy_scores(n, cfg, ports, gains, power),
                           _equ_sinr_scores(n, cfg, ports, gains, power)):
                action = decode_complete_pairing(n, cfg, candidates, business, ports, scores, power)
                key = tuple(action.links)
                if key not in seen:
                    seen.add(key)
                    output.append(action)
    return output


def _belief_mean_ee(action, cfg, gains, ports):
    """在块初信念增益上算均逐链路 EE（功率搜索用的廉价目标）。"""
    if not action.links:
        return -np.inf
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(action.links, gains, ports, cfg)
    return float(np.mean(_rates_and_ee(np.asarray(action.powers, float), action.links,
                                       cfg, s_gain, h_rf, h_vlc, h_opt)[1]))


def _swap_neighbours(links, cfg, candidates, business, masks):
    """一对一匹配的合法 1-邻域：换伙伴（可同时换模式）、换模式、反向。

    每个邻域动作都保持端点集合不变（仍是完美匹配），并逐一检查业务方向与
    候选域；因此不会产生"偷偷删一条链路"或"两条链路共用端点"的伪改进——
    均值目标对少一条链路是麻木的（见 ``repair_unusable_modes`` 的注释）。

    返回 ``(候选匹配, 触及的链路下标)``，供上层按"先动最弱链路"截断。
    """
    out = []
    for a, (i, j, mode) in enumerate(links):
        other = "VLC" if mode == "RF" else "RF"
        if masks[other][i, j]:
            out.append((links[:a] + [(i, j, other)] + links[a + 1:], (a,)))
        if masks[mode][j, i]:
            out.append((links[:a] + [(j, i, mode)] + links[a + 1:], (a,)))
        for b in range(a + 1, len(links)):
            i2, j2, mode2 = links[b]
            if masks[mode][i, j2] and masks[mode2][i2, j]:
                out.append((links[:a] + [(i, j2, mode)] + links[a + 1:b]
                            + [(i2, j, mode2)] + links[b + 1:], (a, b)))
            if masks[mode2][i, j2] and masks[mode][i2, j]:
                out.append((links[:a] + [(i, j2, mode2)] + links[a + 1:b]
                            + [(i2, j, mode)] + links[b + 1:], (a, b)))

    def one_to_one(candidate):
        endpoints = [endpoint for link in candidate for endpoint in link[:2]]
        return len(endpoints) == len(set(endpoints))

    return [(candidate, touched) for candidate, touched in out if one_to_one(candidate)]


def _joint_step(start, cfg, candidates, business, ports, masks, gains, max_neighbours=0,
                power_solver="ascent", solver_log=None, q_min_bps=0.):
    """匹配空间的一轮最陡上升：返回严格更好的邻居，没有就返回 ``None``。

    功率一律用坐标上升：匹配层每轮要看几十到上千个邻域，SCA 的 0.2 s/次用不起，
    末端对最终动作再精修一次（见 ``pure_ee_reference``）。判据是块初信念的均逐
    链路 EE，因此这一步只是**提议**；是否采纳由调用方用条件轨迹评估器复评决定。

    ``max_neighbours`` > 0 时按"先动最弱链路"截断：触及当前最弱链路的邻居优先。
    n=50/100 时邻居数是 O(K²)=300/1225，不截断会让单块跑到秒级。
    """
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(start.links, gains, ports, cfg)
    per_link = _rates_and_ee(np.asarray(start.powers, float), start.links, cfg,
                             s_gain, h_rf, h_vlc, h_opt)[1]
    weakest_rank = {a: r for r, a in enumerate(np.argsort(per_link))}
    neighbours = _swap_neighbours(list(start.links), cfg, candidates, business, masks)
    if max_neighbours and len(neighbours) > max_neighbours:
        neighbours.sort(key=lambda item: min(weakest_rank[a] for a in item[1]))
        neighbours = neighbours[:max_neighbours]
    def rank(action):
        ss, rr, vv, oo = _link_gain_tables(action.links, gains, ports, cfg)
        rates, ee = _rates_and_ee(action.powers, action.links, cfg, ss, rr, vv, oo)
        deficit = float(np.maximum(q_min_bps-rates,0.).sum())
        return (deficit <= max(1e-6,q_min_bps*1e-8), -deficit, float(ee.mean())) if deficit > max(1e-6,q_min_bps*1e-8) else (True,0.,float(ee.mean()))
    best, best_rank = None, rank(start)
    for links, _touched in neighbours:
        action = _optimize_action(Action(links, np.zeros(len(links)), ports), cfg, gains,
                                  ports, masks, power_solver, solver_log=solver_log, q_min_bps=q_min_bps)
        value = rank(action)
        if value > best_rank:
            best, best_rank = action, value
    return best


def _raise_targets(links, gains, ports, cfg, powers, targets):
    """把欠速链路的功率抬到目标速率（单调抬升的可行性修复）。

    分式规划（二次变换）+ SCA 的功率解**不含 QoS 项**。为了让文献基线在同一个
    1 Mbit/s 约束下比较，这里只做抬升、不重新优化——与连续求解器的 Phase I 同语义，
    避免把"文献算法"偷换成别的算法。达到模式上限仍欠速时原样返回，由上层的可行性
    判定如实记为不可行，不在这里造假。
    """
    q = np.asarray(targets, float)
    if q.ndim == 0:
        q = np.full(len(links), float(q))
    p = np.asarray(powers, float).copy()
    if p.shape != (len(links),) or q.shape != (len(links),):
        raise ValueError("powers and QoS targets must match the link count")
    if not len(links) or np.all(q <= 0):
        return p
    caps = np.array([getattr(cfg, mode.lower()).power_max_w for _i, _j, mode in links], float)
    p = np.minimum(p, caps)
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    for _ in range(80):
        rates = _rates_and_ee(p, links, cfg, s_gain, h_rf, h_vlc, h_opt)[0]
        short = rates < q * (1 - 1e-9)
        if not short.any():
            break
        p = np.where(short, np.minimum(p * 1.6 + 1e-3, caps), p)
    return p


def _power_solve(links, gains, ports, cfg, solver, solver_log=None, q_min_bps=0.):
    """ascent is the legacy grid baseline; continuous uses unclipped QT/SCA.

    dinkelbach is the legacy clipped refinement, retained only for reproduction.
    All branches are local numerical methods, with no global-optimum claim.
    """
    if solver == "continuous":
        from .continuous_power import continuous_sum_ee
        base = power_ascent(links, gains, ports, cfg, sweeps=2, points=8)
        refined, info = continuous_sum_ee(links, gains, ports, cfg, p_start=base, q_min_bps=q_min_bps)
        if solver_log is not None:
            solver_log.append(dict(links=[list(x) for x in links], **info.to_dict()))
        return refined
    if solver == "ascent":
        return power_ascent(links, gains, ports, cfg, sweeps=2, points=8)
    if solver != "dinkelbach":
        raise ValueError(f"未知功率求解器 {solver!r}")
    from .dinkelbach_sca import dinkelbach_sum_ee          # 避免循环导入
    base = power_ascent(links, gains, ports, cfg, sweeps=2, points=8)
    # 文献基线要满足与主算法相同的 QoS：分式规划解完之后再做单调抬升。
    refined = dinkelbach_sum_ee(links, gains, ports, cfg,
                                p_start=_raise_targets(links, gains, ports, cfg, base, q_min_bps))[0]
    refined = _raise_targets(links, gains, ports, cfg, refined, q_min_bps)
    s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(links, gains, ports, cfg)
    ee_base = float(np.mean(_rates_and_ee(base, links, cfg, s_gain, h_rf, h_vlc, h_opt)[1]))
    ee_ref = float(np.mean(_rates_and_ee(refined, links, cfg, s_gain, h_rf, h_vlc, h_opt)[1]))
    return refined if ee_ref > ee_base else base


def _optimize_action(action, cfg, belief_gains, ports, masks, power_solver="ascent",
                     solver_log=None, q_min_bps=0.):
    """对给定匹配做功率优化 + 不可用链路修复；不改动匹配本身。"""
    if belief_gains is None:
        return copy_action(action)
    links = list(action.links)
    powers = _power_solve(links, belief_gains, ports, cfg, power_solver, solver_log=solver_log, q_min_bps=q_min_bps)
    links, powers, changed = repair_unusable_modes(links, powers, belief_gains, ports, cfg, masks)
    if changed:
        powers = _power_solve(links, belief_gains, ports, cfg, power_solver, solver_log=solver_log, q_min_bps=q_min_bps)
    # v4：按模式裁剪到 [power_min_w, power_max_w]；不再有共享的 0.001/2 W 夹子。
    caps = np.array([getattr(cfg, m.lower()).power_max_w for _i, _j, m in links])
    lows = np.array([getattr(cfg, m.lower()).power_min_w for _i, _j, m in links])
    return Action(links, np.clip(powers, lows, caps), action.ports)


def inflate_low(low, mean, factor):
    """In log domain, scale the predictive lower-tail spread by ``factor``.

    The posterior spread of the rate forecast is only as good as its assumptions
    (independent pilot samples, ``LOG_EXP_VAR``); measured on truth it is about
    half the actual error.  ``factor > 1`` widens the predictive interval around
    its own mean, so that the same nominal 5% quantile becomes conservative.
    ``factor == 1`` is the identity.
    """
    low = np.asarray(low, float)
    mean = np.asarray(mean, float)
    if factor == 1.0:
        return low
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("chance inflation must be a positive finite number")
    floor = 1e-9
    m = np.maximum(mean, floor)
    ratio = np.maximum(low, floor) / m
    return m * ratio ** float(factor)


def chance_targets(low, mean, q_min_bps, hold_mean=False):
    """Per-link deterministic rate targets for the chance repair.

    A short link (predicted 5% quantile below Q) is escalated by its own
    ``Q/low`` ratio, applied to its predicted mean.  Links that already clear
    the chance threshold are held at ``q_min_bps`` -- the constraint they were
    already judged against -- unless ``hold_mean`` asks to pin them at their
    pre-repair mean too.
    """
    low = np.asarray(low, float)
    mean = np.asarray(mean, float)
    ratio = np.clip(q_min_bps/np.maximum(low, 1.0), 1.0, 100.0)
    targets = np.maximum(mean*ratio, q_min_bps)
    if not hold_mean:
        targets = np.where(low < q_min_bps*(1-1e-8), targets, q_min_bps)
    return targets


def pure_ee_reference(n, cfg, candidates, business, ports, evaluator: Callable,
                      *, belief_gains=None, policy=ReferencePolicy(),
                      power_solver="ascent", search="candidates", joint_rounds=8,
                      joint_max_neighbours=0, q_min_bps=0., qos_metric="mean",
                      qos_fallback_mean=False, chance_rounds=2, chance_hold_mean=False,
                      chance_inflation=1.0, extra_belief_gains=(),
                      literature_nominal_w=None):
    """Freeze the best mean per-link EE of a small set of joint mode/power candidates.

    候选 = {EE 代理分数, 等效 SINR 分数} × ``policy.powers_w``（去重，最多 6 个）。
    每个候选先做功率坐标上升（单调接受）并修复不可用链路，再用条件轨迹评估器
    打分；最终按**电输入功率**口径的均逐链路 EE 取最好。

    ``search="joint"`` 打开**匹配+功率联合优化**：以候选集的最好动作为起点，在
    一对一匹配的 1-邻域（换伙伴/换模式/反向）上做最陡上升；邻域按块初信念的
    均 EE 排序（廉价），走到的终点再用**同一个**条件轨迹评估器复评，只有复评
    更好才采纳。启发式候选集的端点集合是固定的，可改进空间实测很大（10 车上有
    +18%~+25%），功率子问题只占其中约 1%。

    ``search="literature"`` 是文献算法复现：关联**只做一次**（等效 SINR 选模 + 匈牙利
    解成完美匹配，标称功率取参考网格中位数），功率交给分式规划
    （``power_solver="dinkelbach"``，二次变换 + SCA）。不生成多起点、不按评估器挑候选；
    评估次数因此固定为 1（+ 机会约束修复轮）。这样主算法与文献基线的差别只剩
    "搜索自由度与功率求解器"，而不是"谁多试了几个动作"。

    EE = mean_k(rate_k / (static_mode + p_k))，静态项从cfg读取。
    ``search="candidates"`` 时评估次数最多 6 次（与 N 无关）；``joint`` 时再多 1 次。
    预测失败与覆盖失败都显式返回失败。

    ``qos_metric="chance"`` 把 QoS 变量从预测**均值**换成同一批后验样本的
    **5% 分位**（``low_rate_bps``）：要求每条链路在 95% 的预测样本下达标。
    代价是每个候选的判决更严；若所有候选都不可行，``qos_fallback_mean=True``
    时退回均值口径下的最优动作并置 ``chance_fallback``，否则返回失败。

    修复口径（``chance_hold_mean``）：默认只抬高**真正违约**那几条链路的目标速率，
    其余链路仍只要求 ``q_min_bps``；``chance_hold_mean=True`` 则把所有链路都钉在
    "修复前的预测均值"，等于给未违约链路加了超出约束语义的额外要求。同一块
    （b83）实测：钉均值 EE −67.6%，只抬违约链路 −2.4%。默认取后者。

    ``chance_inflation``：对数域把预测下尾围绕自身均值放大 ``α`` 倍，用同一个
    5% 名义分位实现更保守的判决。它不改变轨迹样本，只改约束变量；默认 1.0 是
    "未校准"的原语义。实测后验 sd 约为真实误差 sd 的一半，α 用来补偿这一点。
    """
    if search not in ("candidates", "joint", "literature"):
        raise ValueError(f"未知匹配搜索 {search!r}")
    if search == "literature" and power_solver not in ("dinkelbach", "continuous"):
        raise ValueError("文献复现需要分式规划功率求解器（dinkelbach 或 continuous）")
    if qos_metric not in ("mean", "chance"):
        raise ValueError(f"未知 QoS 口径 {qos_metric!r}")
    if q_min_bps < 0 or not np.isfinite(q_min_bps):
        raise ValueError("q_min_bps must be nonnegative")
    if qos_metric == "chance" and chance_rounds < 0:
        raise ValueError("chance_rounds must be nonnegative")
    if not np.isfinite(chance_inflation) or chance_inflation <= 0:
        raise ValueError("chance_inflation must be positive")
    if q_min_bps > 0 and power_solver not in ("continuous", "dinkelbach"):
        # dinkelbach 是文献基线：分式规划本身不含约束，靠 _raise_targets 单调抬升
        # 满足与主算法相同的 QoS；网格上升（ascent）没有约束手段，仍然禁止。
        raise ValueError("positive QoS requires continuous or dinkelbach power solver")
    start = perf_counter()
    calls = 0
    solver_log = []
    masks, checked_ports = _domain(n, cfg, candidates, business, ports)
    if search == "literature":
        # 文献算法复现（关联 + 分式规划）：关联只做**一次**——按等效 SINR 选模、
        # 匈牙利解成完美匹配；功率随后由 dinkelbach/SCA 求。这里不生成多起点、
        # 也不按条件轨迹评估器在候选之间挑——那是本文主算法的自由度，
        # 复现不能借用，否则比较的是"我们的搜索"而不是"文献的算法"。
        # 关联需要一个标称功率来算 SINR：默认取参考网格的中位数（0.3 W），
        # 可用 ``literature_nominal_w`` 覆盖做敏感性分析，写进 solver_log。
        nominal = (float(literature_nominal_w) if literature_nominal_w is not None
                   else float(np.median(np.asarray(policy.powers_w, float))))
        try:
            scores = _equ_sinr_scores(n, cfg, checked_ports, belief_gains, nominal)
            link = decode_complete_pairing(n, cfg, candidates, business, checked_ports,
                                           scores, nominal)
        except CoverageFailure as exc:
            return _failure(str(exc), start, calls, coverage=True, policy=policy)
        proposals = [link]
        solver_log.append(dict(stage="literature_association", nominal_power_w=nominal,
                               rule="equivalent_sinr", matching="hungarian"))
    else:
        try:
            proposals = _candidate_matchings(n, cfg, candidates, business, checked_ports,
                                             belief_gains, policy, extra_belief_gains)
        except CoverageFailure as exc:
            return _failure(str(exc), start, calls, coverage=True, policy=policy)
    incumbent = None
    best = None
    best_mean = None          # only tracked in chance mode: the fallback incumbent
    if power_solver == "continuous" and q_min_bps == 0:
        # Retain the *whole* previous grid-search action under the same trajectory
        # evaluator. Surrogate improvement alone does not ensure forecast gain.
        incumbent = pure_ee_reference(n, cfg, candidates, business, ports, evaluator,
                                     belief_gains=belief_gains, policy=policy,
                                     power_solver="ascent", search=search,
                                     joint_rounds=joint_rounds,
                                     joint_max_neighbours=joint_max_neighbours)
        calls += incumbent.evaluator_calls
        if incumbent.success:
            best = (incumbent.mean_pair_ee, incumbent.network_ee, incumbent.min_pair_ee,
                    copy_action(incumbent.action), incumbent.forecast)
    def qos_rates(forecast, metric="mean"):
        """Constraint variable: predictive mean (old) or 5% posterior quantile.

        ``mean`` is the previous semantics: the constraint holds for the Monte
        Carlo mean of the block-average rate. ``chance`` requires the 5% lower
        quantile of the same posterior samples to clear the threshold, i.e. a
        per-link 95% chance constraint under the forecast posterior.
        """
        key = "low_rate_bps" if metric == "chance" else "mean_rate_bps"
        rates = np.asarray(forecast[key], float)
        if metric == "chance" and chance_inflation != 1.0:
            rates = inflate_low(rates, forecast["mean_rate_bps"], chance_inflation)
        return rates

    def forecast_rank(forecast, mean_ee, metric=None):
        rates = qos_rates(forecast, qos_metric if metric is None else metric)
        deficit = float(np.maximum(q_min_bps-rates,0.).sum())
        feasible = bool(np.all(rates >= q_min_bps*(1-1e-8)))
        return (feasible, 0. if feasible else -deficit, mean_ee)
    for proposal in proposals:
        try:
            action = _optimize_action(proposal, cfg, belief_gains, checked_ports, masks,
                                      power_solver=power_solver, solver_log=solver_log, q_min_bps=q_min_bps)
        except (ValueError, KeyError, IndexError) as exc:
            return _failure(f"reference_power_search_failure: {exc}", start, calls, policy=policy)
        calls += 1
        try:
            forecast = _forecast(evaluator, action)
        except (ValueError, FloatingPointError) as exc:
            # A failed predictor is not a zero-rate valid reference.
            return _failure(f"reference_prediction_failure: {exc}", start, calls, policy=policy)
        network, mean_pair, weakest = _metrics(action, forecast, cfg)
        if best is None or forecast_rank(forecast,mean_pair) > forecast_rank(best[4],best[0]):
            best = mean_pair, network, weakest, copy_action(action), forecast
        mean_rank = forecast_rank(forecast, mean_pair, "mean")
        if best_mean is None or mean_rank > forecast_rank(best_mean[4], best_mean[0], "mean"):
            best_mean = mean_pair, network, weakest, copy_action(action), forecast
    if search == "joint" and belief_gains is not None and best[3].links:
        # 每一轮都复评：信念上升但预测不升就停在此前最好的一轮。这样联合搜索
        # 的结果**按预测口径单调不劣于**启发式候选集的最优者，不会出现
        # "信念涨、真值跌"还照单全收的情况。
        current = best[3]
        for _round in range(max(1, joint_rounds)):
            try:
                step = _joint_step(current, cfg, candidates, business, checked_ports, masks,
                                   belief_gains, max_neighbours=joint_max_neighbours,
                                   power_solver=power_solver, solver_log=solver_log, q_min_bps=q_min_bps)
            except (ValueError, KeyError, IndexError) as exc:
                return _failure(f"reference_joint_search_failure: {exc}", start, calls,
                                policy=policy)
            if step is None:
                break
            try:
                forecast = _forecast(evaluator, step)
            except (ValueError, FloatingPointError) as exc:
                return _failure(f"reference_prediction_failure: {exc}", start, calls,
                                policy=policy)
            calls += 1
            network, mean_pair, weakest = _metrics(step, forecast, cfg)
            mean_rank = forecast_rank(forecast, mean_pair, "mean")
            if best_mean is None or mean_rank > forecast_rank(best_mean[4], best_mean[0], "mean"):
                best_mean = mean_pair, network, weakest, copy_action(step), forecast
            if forecast_rank(forecast,mean_pair) <= forecast_rank(best[4],best[0]):
                break
            best = mean_pair, network, weakest, copy_action(step), forecast
            current = step
        # 末端精修：搜索内部为了速度只用坐标上升，这里对最终匹配解一次
        # 指定的功率求解器（如 dinkelbach），只在其真实均 EE 更高时采纳。
        if power_solver != "ascent":
            try:
                refined = Action(list(best[3].links),
                                 _power_solve(list(best[3].links), belief_gains, checked_ports,
                                              cfg, power_solver, solver_log=solver_log, q_min_bps=q_min_bps),
                                 best[3].ports)
                refined_forecast = _forecast(evaluator, refined)
                calls += 1
                network, mean_pair, weakest = _metrics(refined, refined_forecast, cfg)
                mean_rank = forecast_rank(refined_forecast, mean_pair, "mean")
                if best_mean is None or mean_rank > forecast_rank(best_mean[4], best_mean[0], "mean"):
                    best_mean = mean_pair, network, weakest, copy_action(refined), refined_forecast
                if forecast_rank(refined_forecast,mean_pair) > forecast_rank(best[4],best[0]):
                    best = mean_pair, network, weakest, refined, refined_forecast
            except (ValueError, KeyError, IndexError) as exc:
                return _failure(f"reference_power_search_failure: {exc}", start, calls,
                                policy=policy)
    mean_pair, network, weakest, action, forecast = best
    chance_log = []
    if qos_metric == "chance" and not forecast_rank(forecast, mean_pair)[0]:
        # 机会约束的修复：内层 SCA 仍然只认确定性速率（约等于预测均值），
        # 外圈按**每条违约链路自己的 low/mean 比**抬高它的目标速率，再解一次。
        # 只有机会口径的排序真的变好才采纳；这一层是局部数值步骤，不是新求解器。
        for _round in range(chance_rounds):
            low = qos_rates(forecast, "chance")
            if np.all(low >= q_min_bps*(1-1e-8)):
                break
            mean_rate = np.asarray(forecast["mean_rate_bps"], float)
            targets = chance_targets(low, mean_rate, q_min_bps, chance_hold_mean)
            try:
                powers = _power_solve(list(action.links), belief_gains, checked_ports, cfg,
                                      power_solver, solver_log=solver_log, q_min_bps=targets)
            except (ValueError, KeyError, IndexError) as exc:
                chance_log.append(dict(round=_round, error=str(exc)))
                break
            caps = np.array([getattr(cfg, m.lower()).power_max_w for _i, _j, m in action.links])
            lows = np.array([getattr(cfg, m.lower()).power_min_w for _i, _j, m in action.links])
            repaired = Action(list(action.links), np.clip(powers, lows, caps), action.ports)
            try:
                repaired_forecast = _forecast(evaluator, repaired)
            except (ValueError, FloatingPointError) as exc:
                chance_log.append(dict(round=_round, error=str(exc)))
                break
            calls += 1
            r_network, r_mean, r_weakest = _metrics(repaired, repaired_forecast, cfg)
            chance_log.append(dict(round=_round,
                                   targets_bps=np.asarray(targets).tolist(),
                                   low_bps_before=low.tolist(),
                                   low_bps_after=np.asarray(repaired_forecast["low_rate_bps"]).tolist(),
                                   mean_ee_before=float(mean_pair),
                                   mean_ee_after=float(r_mean)))
            mean_rank = forecast_rank(repaired_forecast, r_mean, "mean")
            if best_mean is None or mean_rank > forecast_rank(best_mean[4], best_mean[0], "mean"):
                best_mean = r_mean, r_network, r_weakest, copy_action(repaired), repaired_forecast
            if forecast_rank(repaired_forecast, r_mean) > forecast_rank(forecast, mean_pair):
                mean_pair, network, weakest, action, forecast = r_mean, r_network, r_weakest, repaired, repaired_forecast
            if forecast_rank(forecast, mean_pair)[0]:
                break
    chance_fallback = False
    if not forecast_rank(forecast, mean_pair)[0]:
        if (qos_metric == "chance" and qos_fallback_mean and best_mean is not None
                and forecast_rank(best_mean[4], best_mean[0], "mean")[0]):
            mean_pair, network, weakest, action, forecast = best_mean
            chance_fallback = True
        else:
            if search == "literature" and action.links and belief_gains is not None:
                # 诊断：这条固定关联在**功率上限**下能不能达标。若不能，失败是
                # "关联本身不可行"（该换匹配），不是分式规划的问题；若能，则失败
                # 来自求解器/预算修复，需要另查。只写进 policy_config，不改判决。
                caps = np.array([getattr(cfg, m.lower()).power_max_w
                                 for _i, _j, m in action.links])
                s_gain, h_rf, h_vlc, h_opt = _link_gain_tables(list(action.links),
                                                               belief_gains, checked_ports, cfg)
                cap_rates = _rates_and_ee(caps, list(action.links), cfg,
                                          s_gain, h_rf, h_vlc, h_opt)[0]
                solver_log.append(dict(
                    stage="literature_qos_diagnostic",
                    powers_w=np.asarray(action.powers, float).tolist(),
                    cap_powers_w=caps.tolist(),
                    belief_rates_at_caps_bps=np.asarray(cap_rates, float).tolist(),
                    predicted_mean_bps=np.asarray(forecast["mean_rate_bps"], float).tolist(),
                    predicted_low_bps=np.asarray(forecast["low_rate_bps"], float).tolist(),
                    q_min_bps=float(q_min_bps),
                    cap_feasible_on_belief=bool(np.all(cap_rates >= q_min_bps*(1-1e-9)))))
            failure = _failure("qos_not_found: no complete candidate met the predicted minimum rate",
                               start,calls,policy=policy)
            return replace(failure,policy_config={"q_min_bps":q_min_bps,
                           "qos_metric":qos_metric,
                           "chance_inflation":float(chance_inflation),
                           "continuous_solver_log":solver_log,
                           "chance_repairs":chance_log,
                           "best_predicted_rates_bps":np.asarray(forecast["mean_rate_bps"]).tolist(),
                           "best_predicted_low_rates_bps":np.asarray(forecast["low_rate_bps"]).tolist()})
    elapsed = perf_counter() - start
    # eta_ref is the reference's *mean per-link EE*: the budget and every
    # comparison below use the same definition the objective uses.
    return AllocationResult(action, forecast, mean_pair, network, weakest,
                            reason="pure_mean_pair_ee_reference",
                            mean_pair_ee=mean_pair,
                            solve_time_s=elapsed, evaluator_calls=calls,
                            reference_version=(policy.version + "+continuous-qt-sca-v1"
                                               if power_solver == "continuous"
                                               else policy.version + "+fractional-programming-sca-v1"
                                               if power_solver == "dinkelbach" else policy.version),
                            reference_solve_time_s=elapsed, reference_evaluator_calls=calls,
                            attempted_candidates=calls, evaluator_id=_evaluator_identity(evaluator),
                            policy_config={"powers_w": list(policy.powers_w),
                                           "matching_candidates": len(proposals),
                                           "search": ("joint_matching_power_local_search"
                                                      if search == "joint"
                                                      else "literature_equ_sinr_association"
                                                      if search == "literature"
                                                      else "equ_sinr_and_ee_proposals")
                                                     + "+power_" + power_solver,
                                           "joint_rounds": joint_rounds if search == "joint" else 0,
                                           "q_min_bps": q_min_bps,
                                           "qos_metric": qos_metric,
                                           "chance_targets": ("hold_mean" if chance_hold_mean else "fair"),
                                           "chance_inflation": float(chance_inflation),
                                           "chance_fallback": chance_fallback,
                                           "chance_repairs": chance_log,
                                           "continuous_solver_log": solver_log,
                                           "grid_incumbent_mean_ee": (incumbent.mean_pair_ee if incumbent and incumbent.success else None),
                                           "denominator": "electrical_static_plus_p",
                                           "vlc_sinr": cfg.vlc.sinr_model,
                                           "vlc_noise": cfg.vlc.noise_model,
                                           "score_scope": "conditional_trajectory_mean_block_rate"})


def _check_delta(delta):
    if not np.isscalar(delta) or not np.isfinite(delta) or not 0 <= delta <= 1:
        raise ValueError("delta must be explicitly supplied, finite and in [0, 1]")


def _reference_error(reference, evaluator, n, cfg, candidates, business):
    """Reject stale/foreign/illegal reference snapshots before examining proposals."""
    if not isinstance(reference, AllocationResult) or not reference.success:
        return "reference_failure: no legal complete reference action"
    try:
        _legal(reference.action, n, cfg, candidates, business)
        # Validate cached prediction through the same consistency checks without
        # consuming another expensive forecast. Never trust cached scalar EE.
        checked = _forecast(lambda _: reference.forecast, reference.action)
        _, mean_pair, _ = _metrics(reference.action, checked, cfg)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return f"reference_failure: {exc}"
    # eta_ref carries the reference's mean per-link EE, i.e. the objective.
    if (reference.eta_ref is None or not np.isfinite(reference.eta_ref)
            or not np.isclose(reference.eta_ref, mean_pair, rtol=1e-10, atol=1e-9)):
        return "reference_failure: cached eta_ref disagrees with its trajectory-mean prediction"
    if (reference.mean_pair_ee is not None
            and not np.isclose(reference.mean_pair_ee, mean_pair, rtol=1e-10, atol=1e-9)):
        return "reference_failure: cached mean pair EE disagrees with its prediction"
    if reference.evaluator_id != _evaluator_identity(evaluator):
        return "reference_failure: reference and proposal must use the same conditional evaluator"
    return None


def predicted_qos_rates(forecast, metric="mean", inflation=1.0):
    """约束变量：预测均值，或校准后的 5% 分位。

    与 ``pure_ee_reference`` 内部的 ``qos_rates`` 保持同一语义；这里提到模块级，
    是为了让 ``protect_budget`` 能用同一个口径复核候选动作。
    """
    if metric not in ("mean", "chance"):
        raise ValueError(f"未知 QoS 口径 {metric!r}")
    rates = np.asarray(forecast["low_rate_bps" if metric == "chance" else "mean_rate_bps"], float)
    if metric == "chance" and inflation != 1.0:
        rates = inflate_low(rates, forecast["mean_rate_bps"], inflation)
    return rates


def qos_shortfall(forecast, *, q_min_bps=0., metric="mean", inflation=1.0):
    """(是否全部达标, 总缺口, 违约链路数)。``q_min_bps<=0`` 视为无约束。"""
    if q_min_bps <= 0:
        return True, 0., 0
    rates = predicted_qos_rates(forecast, metric, inflation)
    deficit = np.maximum(q_min_bps - rates, 0.)
    tol = max(1e-6, q_min_bps * 1e-8)
    bad = int(np.count_nonzero(deficit > tol))
    return bad == 0, float(deficit.sum()), bad


def protect_budget(proposed, reference, *, delta, n, cfg, candidates, business, evaluator, qos=None):
    """Evaluate one actor action; use its legal frozen reference if budget fails.

    This deployment guard accepts any legal proposal meeting the predicted
    *mean per-link EE* budget -- the same quantity the objective maximises. It
    does not insert an EE/Jain/QoS reward, and provides no guarantee about true
    future EE. An illegal reference fails the entire block.

    传入 ``qos=dict(q_min_bps=, metric=, inflation=)`` 时额外复核候选动作的 QoS
    约束变量（mean 或校准 p05）。与预算一样，这是**预测口径**的检查；真实违约
    仍由执行后的物理评价给出。E4 的 B/D 两组靠这个参数保证"公平模块不牺牲 QoS"。
    """
    start = perf_counter()
    _check_delta(delta)
    error = _reference_error(reference, evaluator, n, cfg, candidates, business)
    if error:
        return _failure(error, start, reference=reference if isinstance(reference, AllocationResult) else None,
                        coverage=reference.coverage_failure if isinstance(reference, AllocationResult) else False, delta=delta)
    calls = 0
    attempted_violation = None
    attempted_qos_bad = None
    reason = ""
    try:
        _legal(proposed, n, cfg, candidates, business)
        action = copy_action(proposed)
        calls = 1
        forecast = _forecast(evaluator, action)
        network, mean_pair, weakest = _metrics(action, forecast, cfg)
        # The guard protects the same quantity the objective maximises.
        attempted_violation = max(0., (1 - delta) * reference.eta_ref - mean_pair)
        if mean_pair + 1e-9 < (1 - delta) * reference.eta_ref:
            reason = "reference_fallback_budget_violation"
        # QoS 复核只在显式传入 qos 时进行（maxmin / policy 提议）；reference 本身
        # 已经在 pure_ee_reference 里处理过 QoS，重复检查会把同一约束数两遍。
        if qos and float(qos.get("q_min_bps", 0.)) > 0:
            ok, _, attempted_qos_bad = qos_shortfall(
                forecast, q_min_bps=float(qos["q_min_bps"]),
                metric=qos.get("metric", "mean"), inflation=float(qos.get("inflation", 1.0)))
            if not ok:
                reason = ("reference_fallback_budget_and_qos_violation" if reason
                          else "reference_fallback_qos_violation")
    except (ValueError, TypeError, KeyError, IndexError, FloatingPointError) as exc:
        reason = f"reference_fallback_invalid_proposal: {exc}"
    if reason:
        action, forecast = copy_action(reference.action), dict(reference.forecast)
        network, mean_pair, weakest = _metrics(action, forecast, cfg)
    return AllocationResult(action, forecast, reference.eta_ref, network, weakest,
                            mean_pair_ee=mean_pair,
                            # Violation belongs to the attempted proposal, even after fallback.
                            budget_violation=attempted_violation,
                            qos_violation_links=attempted_qos_bad, fallback=bool(reason),
                            reason=reason or "proposal_satisfies_predicted_budget", solve_time_s=perf_counter() - start,
                            evaluator_calls=calls, reference_version=reference.reference_version,
                            reference_solve_time_s=reference.reference_solve_time_s,
                            reference_evaluator_calls=reference.reference_evaluator_calls, delta=delta,
                            attempted_candidates=1, rejected_candidates=int(bool(reason)), evaluator_id=_evaluator_identity(evaluator),
                            policy_config=dict(reference.policy_config))


def rebalance_proposal(action, forecast, cfg):
    """把"最强"链路降到 0.25 倍功率，给最弱链路腾干扰余量。

    "最强"必须用目标函数自己的分母 ``R/(static+p)``；用 ``R/p`` 时 VLC 链路
    会被高估约一个数量级，第一个被降功率，结果代价是速率、省下的电功率几乎为零
    （VLC 电气功率由 1.745 W 静态项主导）。
    """
    powers = np.asarray(action.powers, float)
    denom = link_electrical_power_w(powers, [mode for _i, _j, mode in action.links], cfg)
    pair_ee = np.asarray(forecast["mean_rate_bps"], float) / denom
    factors = np.where(pair_ee > pair_ee.min(), .25, 1.)
    lows = np.array([getattr(cfg, m.lower()).power_min_w for _i, _j, m in action.links])
    caps = np.array([getattr(cfg, m.lower()).power_max_w for _i, _j, m in action.links])
    return Action(action.links, np.clip(powers * factors, lows, caps), action.ports)


def maxmin_power_refine(action, forecast, cfg, evaluator, *, delta, eta_ref, qos=None,
                        maxiter=240, restarts=2, diagnostics=None):
    """固定当前匹配，连续优化发射功率以最大化块内**最低**逐链路 EE。

    只改功率，不改匹配、模式与端口。约束与部署守卫完全一致：

    * 均逐链路 EE ≥ ``(1-delta)*eta_ref`` —— 与 :func:`protect_budget` 同一个
      预算量，不是另一套口径；
    * QoS 约束变量（``mean`` 或校准后的 p05）不低于门限；
    * ``各模式功率下界 ≤ p ≤ 各模式功率上限``（v4：RF 0.01–20 W、VLC 0.1–30 W 电输入）。

    搜索在 log 功率空间用 Nelder-Mead，目标只有一项：块内最低 EE。评价全部走
    传入的预测评价器，不看真值；因此这只是**预测口径**的局部数值解，不保证
    真实最低 EE 一定提升。返回 ``(action, forecast, evaluator_calls)``；找不到
    可行点或没有改善时原样返回输入动作。

    ``diagnostics`` 若给一个 dict，写入求解终止状态：每次重启的
    ``nit/status/message/nfev``、搜索内遇到的可行点数、不可行点里最接近约束的
    余量。这是为了区分"有可行点但没改进"和"搜索内根本没有可行点"——只有后者
    才谈得上约束是否太紧，前者只说明这次搜索没找到，不能反推问题无解。
    """
    from scipy.optimize import minimize

    links = tuple(action.links)
    modes = [mode for _i, _j, mode in links]
    cap = np.array([cfg.rf.power_max_w if m == "RF" else cfg.vlc.power_max_w for m in modes], float)
    # v4：下界按模式取配置值（RF 0.01 W、VLC 0.1 W），两种模式不再共用 1 mW。
    lo = np.array([cfg.rf.power_min_w if m == "RF" else cfg.vlc.power_min_w for m in modes], float)
    hi = np.maximum(cap, lo)
    budget = (1. - delta) * eta_ref
    q = float((qos or {}).get("q_min_bps", 0.))
    metric = (qos or {}).get("metric", "mean")
    inflation = float((qos or {}).get("inflation", 1.0))

    cache: dict[tuple, float] = {}
    calls = 0
    best = {"min_ee": -np.inf, "action": None, "forecast": None}
    feasible_points = 0
    best_infeasible_slack = -np.inf      # 不可行点里 max(min(预算余量, QoS 余量))

    def objective(x):
        nonlocal calls, feasible_points, best_infeasible_slack
        key = tuple(np.round(np.clip(np.exp(x), lo, hi), 12))
        if key in cache:
            return cache[key]
        candidate = Action(links, np.array(key), action.ports)
        value = _forecast(evaluator, candidate)
        calls += 1
        rates = np.asarray(value["mean_rate_bps"], float)
        ee = rates / link_electrical_power_w(candidate.powers, modes, cfg)
        # 预算与 QoS 罚项按各自尺度归一，任何一项越界都压过目标改进。
        penalty = 1e3 * max(0., budget - float(ee.mean())) / max(eta_ref, 1e-12)
        if q > 0.:
            qos_rates = predicted_qos_rates(value, metric, inflation)
            penalty += 1e3 * max(0., float(np.max(q - qos_rates))) / max(q, 1e-12)
        qos_slack = (float(np.min(qos_rates) - q) if q > 0. else np.inf)
        if penalty == 0.:
            feasible_points += 1
            if float(ee.min()) > best["min_ee"]:
                best.update(min_ee=float(ee.min()), action=candidate, forecast=value)
        else:
            best_infeasible_slack = max(best_infeasible_slack,
                                        min(float(ee.mean()) - budget, qos_slack))
        score = float(ee.min()) / max(eta_ref, 1e-12) - penalty
        cache[key] = -score
        return -score

    x0 = np.log(np.clip(np.asarray(action.powers, float), lo, hi))
    starts = [x0]
    rng = np.random.default_rng(0)
    for _ in range(max(0, restarts)):
        starts.append(np.log(np.clip(np.exp(x0) * rng.uniform(.3, 3., x0.size), lo, hi)))
    terminations = []
    for index, start in enumerate(starts):
        result = minimize(objective, start, method="Nelder-Mead",
                          options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-4, adaptive=True))
        terminations.append(dict(restart=index, nit=int(getattr(result, "nit", -1)),
                                 nfev=int(getattr(result, "nfev", -1)),
                                 status=int(getattr(result, "status", -1)),
                                 message=str(getattr(result, "message", ""))))
    if diagnostics is not None:
        input_min_ee = float(np.min(np.asarray(forecast["mean_rate_bps"], float) /
                                    link_electrical_power_w(action.powers, modes, cfg)))
        best_min_ee = None if best["action"] is None else float(best["min_ee"])
        diagnostics.clear()
        diagnostics.update(
            objective_evaluations=calls, distinct_points=len(cache),
            feasible_points=feasible_points,
            # ``improved`` 指"比输入更好"；搜索里出现过 best 不等于有改进——
            # 起点本身就会被记成 best，块 3/4 正是"有可行点但没改进"。
            improved=(best_min_ee is not None and best_min_ee > input_min_ee + 1e-9),
            best_min_ee=best_min_ee, input_min_ee=input_min_ee,
            best_infeasible_slack=(None if best_infeasible_slack == -np.inf
                                   else float(best_infeasible_slack)),
            budget=float(budget), q_min_bps=q, metric=metric,
            maxiter=maxiter, restarts=restarts, terminations=terminations)
    if best["action"] is None:
        return copy_action(action), dict(forecast), calls
    return best["action"], best["forecast"], calls


def max_min_allocate(n, cfg, candidates, business, ports, evaluator, *, delta,
                     belief_gains=None, reference=None, policy=ReferencePolicy(), qos=None,
                     power_solver="ascent", search="candidates", joint_rounds=8,
                     joint_max_neighbours=0, power_refine=False, refine_maxiter=240,
                     refine_restarts=2, refine_diagnostics=None):
    """Bounded heuristic: maximize predicted minimum pair EE among full actions.

    This is the explicit max-min *baseline*, not the paper's objective: it ranks
    by the minimum conditional-mean pair EE while the budget guard uses the mean
    per-link EE.  Three full matching proposals use log isolated EE scores
    (weak-link oriented), plus one reference-topology power rebalance.  At most
    four new conditional predictions follow the three-call reference.

    ``power_refine=True`` 在四个离散候选之后加一步**连续功率 Max-Min**：
    固定当前最好的匹配，只在功率上最大化最低逐链路 EE，约束是同一个预算与
    QoS。四个候选回答"要不要改动"，这一步回答"在不动匹配的前提下功率还能
    往哪走"；两者不是二选一。
    """
    start = perf_counter()
    _check_delta(delta)
    masks, checked_ports = _domain(n, cfg, candidates, business, ports)
    if reference is None:
        reference = pure_ee_reference(n, cfg, candidates, business, checked_ports, evaluator,
                                      belief_gains=belief_gains, policy=policy,
                                      q_min_bps=float((qos or {}).get("q_min_bps", 0.)),
                                      qos_metric=(qos or {}).get("metric", "mean"),
                                      chance_inflation=float((qos or {}).get("inflation", 1.0)),
                                      power_solver=power_solver, search=search,
                                      joint_rounds=joint_rounds,
                                      joint_max_neighbours=joint_max_neighbours)
    error = _reference_error(reference, evaluator, n, cfg, candidates, business)
    if error:
        checked_reference = reference if isinstance(reference, AllocationResult) else None
        return _failure(error, start, coverage=False if checked_reference is None else reference.coverage_failure,
                        reference=checked_reference, delta=delta)
    proposals = []
    for power in policy.powers_w:
        raw = _proxy_scores(n, cfg, checked_ports, belief_gains, power)
        scores = {mode: np.log(np.maximum(raw[mode], 1e-300)) for mode in MODES}
        try:
            proposals.append(decode_complete_pairing(n, cfg, candidates, business, checked_ports, scores, power))
        except CoverageFailure as exc:
            return _failure(str(exc), start, coverage=True, reference=reference, delta=delta)
    # Reducing stronger links can relieve interference at the weakest pair. This
    # is a bounded proposal, not an assertion that EE is monotone in data power.
    proposals.append(rebalance_proposal(reference.action, reference.forecast, cfg))
    best_action, best_forecast = copy_action(reference.action), dict(reference.forecast)
    best_network, best_mean, best_min = _metrics(best_action, best_forecast, cfg)
    calls = 0
    rejected = 0
    violations = []
    qos_bad = []
    improved = False

    def consider(proposal):
        nonlocal calls, rejected, best_action, best_forecast, best_network, best_mean, best_min, improved
        guarded = protect_budget(proposal, reference, delta=delta, n=n, cfg=cfg,
                                 candidates=candidates, business=business, evaluator=evaluator,
                                 qos=qos)
        calls += guarded.evaluator_calls
        rejected += guarded.rejected_candidates
        if guarded.budget_violation is not None:
            violations.append(guarded.budget_violation)
        if guarded.qos_violation_links is not None:
            qos_bad.append(guarded.qos_violation_links)
        if guarded.success and not guarded.fallback and guarded.min_pair_ee > best_min:
            best_action, best_forecast = guarded.action, guarded.forecast
            best_network, best_mean = guarded.network_ee, guarded.mean_pair_ee
            best_min = guarded.min_pair_ee
            improved = True

    for proposal in proposals:
        consider(proposal)
    if power_refine:
        # 上面四个候选只是"要不要动"的判断；这一步才在固定匹配上连续搜索功率。
        # 起点取当前最好的动作，避免从参考解重复走一遍相同的路径。
        refined, _refined_forecast, refine_calls = maxmin_power_refine(
            best_action, best_forecast, cfg, evaluator, delta=delta,
            eta_ref=reference.eta_ref, qos=qos, maxiter=refine_maxiter,
            restarts=refine_restarts, diagnostics=refine_diagnostics)
        calls += refine_calls
        if not np.array_equal(np.asarray(refined.powers, float), np.asarray(best_action.powers, float)):
            consider(refined)
    return AllocationResult(best_action, best_forecast, reference.eta_ref, best_network, best_min,
                            mean_pair_ee=best_mean,
                            budget_violation=max(violations, default=0.),
                            qos_violation_links=(min(qos_bad) if qos_bad else None),
                            fallback=not improved,
                            reason="max_min_candidate_selected" if improved else "reference_fallback_no_better_feasible_action",
                            solve_time_s=perf_counter() - start, evaluator_calls=calls,
                            reference_version=reference.reference_version,
                            reference_solve_time_s=reference.reference_solve_time_s,
                            reference_evaluator_calls=reference.reference_evaluator_calls,
                            delta=delta,
                            attempted_candidates=len(proposals) + (1 if power_refine else 0),
                            rejected_candidates=rejected,
                            evaluator_id=_evaluator_identity(evaluator), policy_config=dict(reference.policy_config))
