"""路线 B：反事实审计（counterfactual audit）。

回答两个问题：

1. **判决性问题**：v5_C 里 334 个被拒候选（S1/S2/S3），在**真值信道**上是否真的
   都不优于实际执行动作？若是，0/334 采纳是"判据/精修正确"；若不是，说明存在被
   守卫误拒的更好动作。
2. **命题 2 前提覆盖率**：最弱链路的自身 EE 峰值功率 ``p_ee`` 是否低于其服务功率
   ``p_svc``（即"瓶颈钉住"的前提）。这是主线二结构定理的适用条件，必须自己数，
   不能引用别人的 70/100。
3. **"全都采纳"对照**（用户提出的反向思路）：把每块预测 min EE 最高的候选**强制
   采纳**，在真值上与原执行动作比较。这直接量出守卫的价值（或代价）。

做法：
    复用 v5 源码树，按 ``run_compare_10s_v5.ps1`` 的 ``C_ref_sca_s`` 参数逐块重建
    决策；用 monkeypatch 捕获 ``weak_link_repair_stage`` 提交的每个候选动作（含
    links/ports/powers），再把这些动作放到**真值信道**上重新打分。

复现可信度：
    每块的最终动作功率与候选功率都会与已归档的 ``results.csv`` / ``weak_link.json``
    逐元素比对；不一致的块记 ``reconstruction_mismatch`` 并**排除出结论**。

用法::

    D:\\anaconda\\envs\\pymarl\\python.exe -B -I counterfactual_audit.py --limit 3
    D:\\anaconda\\envs\\pymarl\\python.exe -B -I counterfactual_audit.py --blocks 100

产出：``out_routeB/blocks.csv``、``out_routeB/summary.json``、终端报告。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # ...\第3篇-修改版
V5 = ROOT / "代码" / "开发_10车机制定型_v5_20261101"
CODE = V5 / "code"
sys.path.insert(0, str(CODE))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

import p3base.block as block_mod                     # noqa: E402
import p3learn.allocation as A                       # noqa: E402
from p3base import channel                           # noqa: E402
from p3base.config import FoundationConfig           # noqa: E402
from p3base.estimator import ChannelEstimator        # noqa: E402
from p3base.fading import link_coherence_s           # noqa: E402
from p3base.scene import Scene, find_scene_root      # noqa: E402
from p3learn.forecast import ConditionalTrajectoryBank  # noqa: E402
from p3learn.weak_link import (interference_sensitivity, link_ee_optimum,
                               service_power_probe)  # noqa: E402
from run_allocation import _current_geometry         # noqa: E402
from run_foundation import TruthOracle               # noqa: E402

N = 10
DRAWS = 32
DELTA = 0.1
Q_MIN = 1.0e6
QOS = dict(q_min_bps=Q_MIN, metric="chance", inflation=1.35)
CHANNEL_SEED = 2026102401
FORECAST_SEED = 2026102501
START_BLOCK = 2


# --------------------------------------------------------------------------- #
# 读已归档结果，用于验证复现
# --------------------------------------------------------------------------- #
def load_archived(run_dir):
    """返回 {(block, method): [(link_index, tx, rx, mode, power), ...]}、weak_link 候选、
    以及归档的块级 EE（用于校验本脚本的真值口径）。"""
    executed = {}
    archived_ee = {}
    path = os.path.join(run_dir, "results.csv")
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("row_type") != "link":
                continue
            key = (int(row["block"]), row["method"])
            executed.setdefault(key, []).append(
                (int(row["link_index"]), int(row["tx"]), int(row["rx"]), row["mode"],
                 float(row["power_w"])))
            if row["method"] == "maxmin" and int(row["block"]) not in archived_ee:
                try:
                    archived_ee[int(row["block"])] = (
                        float(row["min_pair_ee_bit_per_j"]),
                        float(row["mean_pair_ee_bit_per_j"]))
                except (KeyError, TypeError, ValueError):
                    pass
    for key in executed:
        executed[key].sort()
    candidates = {}
    weak_path = os.path.join(run_dir, "weak_link.json")
    if os.path.isfile(weak_path):
        with open(weak_path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        for entry in data.get("blocks") or []:
            items = []
            for item in entry.get("repair") or []:
                stage = item.get("stage", "")
                if stage.startswith("s1_probe"):
                    continue
                items.append({"stage": stage, "powers": [float(x) for x in item.get("powers") or []],
                              "accepted": bool(item.get("accepted"))})
            candidates[int(entry["block"])] = items
    return executed, candidates, archived_ee


# --------------------------------------------------------------------------- #
# 逐块重建（照抄 run_allocation.py 411-463 的 causal 分支）
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self):
        # 归档 run 的 summary.json 记录 ``vlc.sinr_model = square_law``，但当前
        # ``config.py`` 默认是 ``linear``，且 ``run_compare_10s_v5.ps1`` 没有传
        # ``--vlc-sinr``。**必须显式覆盖**，否则复现不出归档的决策
        # （block 2：linear 给 680.4 M，square_law 给 445.9 M，归档是后者）。
        self.cfg = FoundationConfig()
        self.cfg = self.cfg.with_(vlc=replace(self.cfg.vlc, sinr_model="square_law"))
        directory = find_scene_root(V5, N)
        self.scene = Scene(directory, self.cfg.vehicle)
        self.context = block_mod.GeometryContext.from_scene(self.scene)
        self.oracle = TruthOracle(self.scene, self.cfg, CHANNEL_SEED)
        self.estimator = ChannelEstimator(self.scene.n, self.cfg)
        self.business = ~np.eye(self.scene.n, dtype=bool)

    def advance(self, k):
        """推进 oracle 与估计器到块 k，返回决策所需的全部量。"""
        cfg = self.cfg
        t = k * cfg.estimation.block_dt_s
        self.oracle.advance_to(t)
        state = self.scene.knot(k)
        rg, vg = _current_geometry(self.context, state, cfg)
        v = state["speed_mps"][:, None] * np.stack(
            [np.cos(state["heading_rad"]), np.sin(state["heading_rad"])], -1)
        rel = np.linalg.norm(v[:, None] - v[None], axis=-1)
        sigma = np.where(rg["los"], cfg.rf.los_shadow_sigma_db, cfg.rf.nlos_shadow_sigma_db)
        prior = np.exp((sigma * np.log(10.) / 10.) ** 2 / 2.)
        self.estimator.advance(t, self.oracle, rf_prior_var=prior,
                               rf_tau=link_coherence_s(rel, cfg.rf))
        active = self.estimator
        belief = active.gain_belief(rg["gain"], vg["gain"], t,
                                    rf_candidate=rg["candidate"],
                                    vlc_candidate=vg["candidate"])
        ports = channel.select_ports(belief["vlc_gain_mean"])
        candidates = {"RF": rg["candidate"], "VLC": vg["candidate"]}
        gains = {"RF": belief["rf_gain_mean"], "VLC": belief["vlc_gain_mean"]}
        bank = ConditionalTrajectoryBank(active, state, self.context, t, draws=DRAWS,
                                         seed=FORECAST_SEED + k,
                                         information="trajectory")
        return dict(k=k, t=t, ports=ports, candidates=candidates, gains=gains, bank=bank)

    def truth_window(self, t):
        """块 t 的 1 ms 真值。审计用，绝不回流到决策。"""
        cfg = self.cfg
        dt = float(cfg.estimation.fine_dt_s)
        steps = int(round(float(cfg.estimation.block_dt_s) / dt))
        oracle = TruthOracle(self.scene, cfg, CHANNEL_SEED)
        rf, vlc = [], []
        for off in range(steps):
            oracle.advance_to(t + off * dt)
            truth = oracle.truth()
            rf.append(truth["rf"])
            vlc.append(truth["vlc"])
        return dict(rf=np.stack(rf), vlc=np.stack(vlc), dt_s=dt)


def truth_metrics(session, window, action, eta_ref):
    cfg = session.cfg
    fine = block_mod.sample_metrics(action.links, action.powers, action.ports,
                                    window["rf"], window["vlc"], cfg)
    summary = block_mod.summarize_block(action.links, action.powers, fine,
                                        window["dt_s"], session.scene.n)
    # **不要**用 summary["mean_pair_ee_bit_per_j"]：它是 ``rate / p`` 的旧口径
    # （``probe_common.action_metrics`` 也是这个口径）。论文口径是
    # ``rate / (P_c + p)``，与 ``allocation._metrics`` 和归档 results.csv 一致。
    rate = np.asarray(summary["mean_rate_bps"], float)
    modes = [m for _i, _j, m in action.links]
    denom = np.asarray(A.link_electrical_power_w(np.asarray(action.powers, float), modes, cfg), float)
    ee = rate / denom
    mean_ee = float(ee.mean())
    min_ee = float(ee.min())
    return dict(mean_ee=mean_ee, min_ee=min_ee,
                network_ee=float(rate.sum() / float(np.sum(denom))),
                mean_rate_bps=float(np.mean(rate)),
                min_rate_bps=float(np.min(rate)),
                budget_ok=bool(mean_ee + 1e-9 >= (1.0 - DELTA) * eta_ref),
                total_power_w=float(np.sum(np.asarray(action.powers, float))))


# --------------------------------------------------------------------------- #
# 捕获候选动作
# --------------------------------------------------------------------------- #
_ORIG_STAGE = A.weak_link_repair_stage


def _patched_stage_factory(sink):
    def patched(action, forecast, cfg, **kw):
        inner = kw["consider"]

        def recording(candidate):
            outcome = inner(candidate)
            sink.append({"action": A.copy_action(candidate), "guard": dict(outcome)})
            return outcome

        kw["consider"] = recording
        return _ORIG_STAGE(action, forecast, cfg, **kw)
    return patched


def kendall_tau(a, b):
    """候选之间的预测排序与真值排序一致性；n<3 时返回 None。"""
    n = len(a)
    if n < 3:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(run_dir, blocks, limit, out_dir):
    executed, archived_candidates, archived_ee = load_archived(run_dir)
    session = Session()
    A.weak_link_repair_stage = _patched_stage_factory([])   # 占位，循环内替换

    rows, records = [], []
    t0 = time.perf_counter()
    for step in range(blocks):
        k = START_BLOCK + step
        info = session.advance(k)
        if limit and step >= limit:
            continue

        sink = []
        A.weak_link_repair_stage = _patched_stage_factory(sink)

        reference = A.pure_ee_reference(
            N, session.cfg, info["candidates"], session.business, info["ports"],
            info["bank"].evaluate, belief_gains=info["gains"],
            power_solver="continuous", search="joint", joint_rounds=8,
            joint_max_neighbours=0, q_min_bps=Q_MIN, qos_metric="chance",
            qos_fallback_mean=True, chance_rounds=2, chance_hold_mean=False,
            chance_inflation=1.35)
        if not reference.success:
            records.append({"block": k, "error": f"reference infeasible: {reference.reason}"})
            continue

        diag = []
        decision = A.max_min_allocate(
            N, session.cfg, info["candidates"], session.business, info["ports"],
            info["bank"].evaluate, delta=DELTA, belief_gains=info["gains"],
            reference=reference, qos=QOS, power_solver="continuous", search="joint",
            joint_rounds=8, joint_max_neighbours=0, power_refine=True,
            refine_maxiter=240, refine_restarts=2, refine_solver="sca",
            weak_link_repair=True, repair_limit=3, repair_diagnostics=diag)
        if decision.action is None:
            records.append({"block": k, "error": f"maxmin failed: {decision.reason}"})
            continue

        # ---- 复现校验：与 results.csv 的 maxmin 行比对 --------------------
        archived = executed.get((k, "maxmin"))
        mismatch = None
        if archived is None:
            mismatch = "no archived maxmin rows"
        else:
            got = sorted([(i, tx, rx, mode, float(p)) for (i, tx, rx, mode, p)
                          in zip(range(len(decision.action.links)),
                                 [l[0] for l in decision.action.links],
                                 [l[1] for l in decision.action.links],
                                 [l[2] for l in decision.action.links],
                                 decision.action.powers)])
            want = [(i, tx, rx, mode, p) for (i, tx, rx, mode, p) in archived]
            if len(got) != len(want):
                mismatch = f"link count {len(got)} vs {len(want)}"
            else:
                for (gi, gtx, grx, gmode, gp), (_, wtx, wrx, wmode, wp) in zip(got, want):
                    if (gtx, grx, gmode) != (wtx, wrx, wmode):
                        mismatch = f"link {gi} identity differs"
                        break
                    if abs(gp - wp) > 1e-6:
                        mismatch = f"link {gi} power {gp:.9f} vs {wp:.9f}"
                        break
        # ---- 候选功率与 weak_link.json 比对 ------------------------------
        arch_cands = archived_candidates.get(k, [])
        if mismatch is None and arch_cands:
            if len(arch_cands) != len(sink):
                mismatch = f"candidate count {len(sink)} vs {len(arch_cands)}"
            else:
                for got_c, want_c in zip(sink, arch_cands):
                    gp = np.asarray(got_c["action"].powers, float)
                    wp = np.asarray(want_c["powers"], float)
                    if gp.shape != wp.shape or not np.allclose(gp, wp, rtol=0, atol=1e-6):
                        mismatch = f"candidate {want_c['stage']} powers differ"
                        break

        # ---- 真值打分 ----------------------------------------------------
        window = session.truth_window(info["t"])
        eta_ref = decision.eta_ref
        base = truth_metrics(session, window, decision.action, eta_ref)
        # 真值基准：Reference 动作在同一条真值信道上的均逐链路 EE。
        # 有了它才能定义"真值口径的 δ 预算"，而不是拿决策时的预测门槛去套真值。
        ref_truth = truth_metrics(session, window, reference.action, eta_ref)
        eta_ref_true = ref_truth["mean_ee"]
        # 口径自检：本脚本的真值 EE 必须与归档的块级 EE 一致。不一致说明用了
        # 别的口径（例如 summary 的 rate/p），此时该块的全部结论都不可用。
        ref_ee = archived_ee.get(k)
        if ref_ee:
            ee_dev = max(abs(base["min_ee"] - ref_ee[0]) / max(ref_ee[0], 1e-12),
                         abs(base["mean_ee"] - ref_ee[1]) / max(ref_ee[1], 1e-12))
            if ee_dev > 1e-6 and mismatch is None:
                mismatch = f"truth EE口径不一致 rel_dev={ee_dev:.3e}"
        else:
            ee_dev = None
        cands = []
        for item in sink:
            met = truth_metrics(session, window, item["action"], eta_ref)
            guard = item["guard"]
            cands.append({
                "stage": arch_cands[len(cands)]["stage"] if len(arch_cands) > len(cands) else "?",
                "accepted": bool(guard.get("accepted")),
                "pred_min_ee": float(guard.get("min_pair_ee") or 0.0),
                "pred_budget_violation": float(guard.get("budget_violation") or 0.0),
                "truth_min_ee": met["min_ee"],
                "truth_mean_ee": met["mean_ee"],
                # ``truth_budget_ok`` 用**预测** η_ref（决策时确定的门槛）判定；
                # ``truth_budget_ok_true_ref`` 用同一条真值信道上的 Reference 均 EE 判定。
                # 两者都要报：前者是"执行后是否守住决策时承诺的门槛"，后者是"相对真值参考是否守预算"。
                "truth_budget_ok": met["budget_ok"],
                "truth_budget_ok_true_ref": bool(
                    met["mean_ee"] + 1e-9 >= (1.0 - DELTA) * eta_ref_true),
                "truth_min_rate_bps": met["min_rate_bps"],
                "truth_qos_ok": bool(met["min_rate_bps"] + 1e-9 >= Q_MIN),
                "d_min_vs_exec_pct": 100.0 * (met["min_ee"] - base["min_ee"]) / max(base["min_ee"], 1e-12),
                "d_mean_vs_exec_pct": 100.0 * (met["mean_ee"] - base["mean_ee"]) / max(base["mean_ee"], 1e-12),
            })

        # ---- 命题 2 前提：p_ee < p_svc ? ---------------------------------
        links = tuple(decision.action.links)
        powers = np.asarray(decision.action.powers, float)
        grad, weakest = interference_sensitivity(links, info["gains"], info["ports"],
                                                 session.cfg, powers)
        probe = service_power_probe(links, info["gains"], info["ports"], session.cfg,
                                    powers, Q_MIN, rounds=1)
        opt = link_ee_optimum(links, info["gains"], info["ports"], session.cfg,
                              powers, weakest)
        premise = {
            "weakest_index": int(weakest),
            "weakest_mode": links[weakest][2],
            "p_cur_w": float(powers[weakest]),
            "p_svc_w": float(probe.p_star[weakest]),
            "p_ee_w": float(opt[weakest]),
            "service_ratio": float(probe.ratio),
        }
        premise["premise_holds"] = bool(premise["p_ee_w"] < premise["p_svc_w"])
        premise["over_powered"] = bool(premise["p_cur_w"] > premise["p_svc_w"])

        # ---- 排序一致性（命题 6） ----------------------------------------
        tau = kendall_tau([c["pred_min_ee"] for c in cands],
                          [c["truth_min_ee"] for c in cands])
        # ---- "全都采纳"对照：取预测 min EE 最高的候选强制采纳 ------------
        forced = None
        if cands:
            best_idx = max(range(len(cands)), key=lambda i: cands[i]["pred_min_ee"])
            forced = dict(cands[best_idx])
            forced["is_better_than_exec"] = bool(forced["truth_min_ee"] > base["min_ee"] + 1e-9)

        record = {
            "block": k, "mismatch": mismatch, "ee_rel_dev": ee_dev,
            "executed": base, "executed_pred_min_ee": float(decision.min_pair_ee),
            "reference_truth": ref_truth, "eta_ref_true": eta_ref_true,
            "eta_ref_pred": float(eta_ref),
            "candidates": cands, "premise": premise, "kendall_tau": tau,
            "forced_best": forced,
            "refine": diag[0] if diag else None,
        }
        records.append(record)
        rows.append(record)

        any_better = bool(cands) and max(c["truth_min_ee"] for c in cands) > base["min_ee"] + 1e-9
        print(f"  k={k:3d} mismatch={'-' if mismatch is None else mismatch[:28]:28s}"
              f" exec_min={base['min_ee']/1e6:7.2f}M"
              f" cands={len(cands):2d} any_better={int(any_better)}"
              f" premise={int(premise['premise_holds'])}"
              f" tau={'n/a' if tau is None else f'{tau:+.2f}'}")

    elapsed = time.perf_counter() - t0
    A.weak_link_repair_stage = _ORIG_STAGE
    summary = summarise(rows, elapsed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "blocks": rows}, handle, ensure_ascii=False, indent=1)
    write_csv(os.path.join(out_dir, "blocks.csv"), rows)
    report(summary)
    return summary


def summarise(rows, elapsed):
    ok = [r for r in rows if r.get("mismatch") is None and "error" not in r]
    bad = [r for r in rows if r.get("mismatch") is not None or "error" in r]
    s = {"blocks_total": len(rows), "blocks_verified": len(ok), "blocks_excluded": len(bad),
         "elapsed_s": elapsed,
         "excluded_detail": [{"block": r["block"], "why": r.get("mismatch") or r.get("error")}
                             for r in bad]}
    if not ok:
        return s

    def frac(pred, seq):
        seq = [x for x in seq if x is not None]
        return float(np.mean([pred(x) for x in seq])) if seq else None

    s["premise_holds_rate"] = frac(lambda p: p["premise_holds"],
                                   [r["premise"] for r in ok])
    s["over_powered_rate"] = frac(lambda p: p["over_powered"],
                                  [r["premise"] for r in ok])
    devs = [r["ee_rel_dev"] for r in ok if r.get("ee_rel_dev") is not None]
    s["truth_ee_max_rel_dev"] = float(np.max(devs)) if devs else None
    s["candidates_total"] = sum(len(r["candidates"]) for r in ok)
    s["candidates_accepted"] = sum(1 for r in ok for c in r["candidates"] if c["accepted"])
    s["blocks_with_any_truly_better"] = sum(
        1 for r in ok if r["candidates"]
        and max(c["truth_min_ee"] for c in r["candidates"]) > r["executed"]["min_ee"] + 1e-9)
    s["blocks_with_any_truly_better_frac"] = s["blocks_with_any_truly_better"] / len(ok)

    taus = [r["kendall_tau"] for r in ok if r["kendall_tau"] is not None]
    s["kendall_tau_mean"] = float(np.mean(taus)) if taus else None
    s["kendall_tau_positive_frac"] = float(np.mean([t > 0 for t in taus])) if taus else None

    by_stage = {}
    for r in ok:
        for c in r["candidates"]:
            agg = by_stage.setdefault(c["stage"].split("_")[0] if c["stage"] != "?" else "?",
                                      {"n": 0, "better": 0, "d_min": [], "d_mean": []})
            agg["n"] += 1
            agg["better"] += int(c["truth_min_ee"] > r["executed"]["min_ee"] + 1e-9)
            agg["d_min"].append(c["d_min_vs_exec_pct"])
            agg["d_mean"].append(c["d_mean_vs_exec_pct"])
    for agg in by_stage.values():
        agg["d_min_median"] = float(np.median(agg.pop("d_min")))
        agg["d_mean_median"] = float(np.median(agg.pop("d_mean")))
    s["by_stage"] = by_stage

    # ---- 真值口径的"合法改进"：三条同时成立才算 ------------------------
    #   1) 真值 min EE 更高
    #   2) 真值 QoS 通过（所有链路真值平均速率 >= q_min）
    #   3) 真值预算通过（对本块真值 Reference 均 EE 的 (1-delta) 门槛）
    # 旧口径（只查 1 + 预测门槛预算）会把不满足真值 QoS 的候选也算进来。
    legit = []
    for r in ok:
        base = r["executed"]
        good = [c for c in r["candidates"]
                if c["truth_min_ee"] > base["min_ee"] + 1e-9
                and c.get("truth_qos_ok") and c.get("truth_budget_ok_true_ref")]
        if good:
            best = max(good, key=lambda c: c["truth_min_ee"])
            legit.append({
                "block": r["block"], "stage": best["stage"],
                "gain_pct": 100.0 * (best["truth_min_ee"] - base["min_ee"]) / max(base["min_ee"], 1e-12),
                "d_mean_pct": best["d_mean_vs_exec_pct"],
                "exec_min_ee": base["min_ee"],
                "cand_min_ee": best["truth_min_ee"],
            })
    gains = sorted(item["gain_pct"] for item in legit)
    s["legit_v2"] = {
        "definition": "truth min-EE better AND truth QoS ok AND truth budget ok (vs truth reference)",
        "blocks": len(legit),
        "frac": len(legit) / len(ok) if ok else None,
        "gain_median_pct": float(np.median(gains)) if gains else None,
        "gain_mean_pct": float(np.mean(gains)) if gains else None,
        "gain_p90_pct": float(np.percentile(gains, 90)) if gains else None,
        "n_gain_gt10": int(sum(1 for g in gains if g > 10)),
        "n_gain_gt50": int(sum(1 for g in gains if g > 50)),
        "top5": sorted(legit, key=lambda x: -x["gain_pct"])[:5],
    }

    # ---- 每条候选按"真值是否三条全过"重新分类 -------------------------
    tri = {"better_qos_ok_budget_pred": 0, "better_qos_ok_budget_true": 0,
           "better_qos_fail": 0, "better_budget_true_fail": 0, "total": 0}
    for r in ok:
        base = r["executed"]
        for c in r["candidates"]:
            if c["truth_min_ee"] <= base["min_ee"] + 1e-9:
                continue
            tri["total"] += 1
            if not c.get("truth_qos_ok"):
                tri["better_qos_fail"] += 1
            if not c.get("truth_budget_ok_true_ref"):
                tri["better_budget_true_fail"] += 1
            if c.get("truth_qos_ok") and c.get("truth_budget_ok"):
                tri["better_qos_ok_budget_pred"] += 1
            if c.get("truth_qos_ok") and c.get("truth_budget_ok_true_ref"):
                tri["better_qos_ok_budget_true"] += 1
    s["truth_three_conditions"] = tri

    # ---- 真值参考与预测参考的偏差（说明预算门槛本身的误差） ------------
    deviations = [r["eta_ref_true"] / r["eta_ref_pred"] - 1.0
                  for r in ok if r.get("eta_ref_pred")]
    s["eta_ref_true_over_pred_median"] = float(np.median(deviations)) if deviations else None

    forced = [r["forced_best"] for r in ok if r["forced_best"]]
    if forced:
        s["force_accept"] = {
            "n": len(forced),
            "wins_on_truth_min": int(sum(1 for f in forced if f["is_better_than_exec"])),
            "d_min_median_pct": float(np.median([f["d_min_vs_exec_pct"] for f in forced])),
            "d_mean_median_pct": float(np.median([f["d_mean_vs_exec_pct"] for f in forced])),
            "truth_budget_ok_frac": float(np.mean([f["truth_budget_ok"] for f in forced])),
        }
    s["executed_mean_min_ee"] = float(np.mean([r["executed"]["min_ee"] for r in ok]))
    s["executed_mean_mean_ee"] = float(np.mean([r["executed"]["mean_ee"] for r in ok]))
    return s


def write_csv(path, rows):
    fields = ["block", "mismatch", "exec_min_ee", "exec_mean_ee", "exec_pred_min_ee",
              "weakest_index", "weakest_mode", "p_cur_w", "p_svc_w", "p_ee_w",
              "premise_holds", "over_powered", "kendall_tau",
              "n_candidates", "n_better_truth", "best_cand_truth_min", "forced_stage",
              "forced_d_min_pct", "forced_d_mean_pct", "forced_budget_ok"]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            if "error" in r:
                writer.writerow({"block": r["block"], "mismatch": r["error"]})
                continue
            p = r["premise"]
            cands = r["candidates"]
            forced = r["forced_best"] or {}
            writer.writerow({
                "block": r["block"], "mismatch": r["mismatch"] or "",
                "exec_min_ee": r["executed"]["min_ee"],
                "exec_mean_ee": r["executed"]["mean_ee"],
                "exec_pred_min_ee": r["executed_pred_min_ee"],
                "weakest_index": p["weakest_index"], "weakest_mode": p["weakest_mode"],
                "p_cur_w": p["p_cur_w"], "p_svc_w": p["p_svc_w"], "p_ee_w": p["p_ee_w"],
                "premise_holds": p["premise_holds"], "over_powered": p["over_powered"],
                "kendall_tau": r["kendall_tau"],
                "n_candidates": len(cands),
                "n_better_truth": sum(1 for c in cands
                                      if c["truth_min_ee"] > r["executed"]["min_ee"] + 1e-9),
                "best_cand_truth_min": max([c["truth_min_ee"] for c in cands], default=None),
                "forced_stage": forced.get("stage"),
                "forced_d_min_pct": forced.get("d_min_vs_exec_pct"),
                "forced_d_mean_pct": forced.get("d_mean_vs_exec_pct"),
                "forced_budget_ok": forced.get("truth_budget_ok"),
            })


def report(s):
    print("\n" + "=" * 78)
    print("路线 B 反事实审计")
    print(f"  块：总 {s['blocks_total']}，复现校验通过 {s['blocks_verified']}，排除 {s['blocks_excluded']}")
    print(f"  耗时 {s['elapsed_s']:.1f}s")
    for item in s["excluded_detail"][:10]:
        print(f"    排除块 {item['block']}: {item['why']}")
    if not s["blocks_verified"]:
        print("  !! 没有可用的块，复现失败，不产出结论")
        return
    print(f"\n  命题 2 前提 p_ee < p_svc：{s['premise_holds_rate']:.1%}"
          f"（当前功率 > p_svc：{s['over_powered_rate']:.1%}）")
    print(f"  候选 {s['candidates_total']} 个，采纳 {s['candidates_accepted']} 个")
    print(f"  存在'真值更好'被拒候选的块：{s['blocks_with_any_truly_better']}"
          f" ({s['blocks_with_any_truly_better_frac']:.1%})")
    if s.get("kendall_tau_mean") is not None:
        print(f"  预测排序 vs 真值排序 Kendall tau 均值 {s['kendall_tau_mean']:+.3f}"
              f"（正相关块占比 {s['kendall_tau_positive_frac']:.1%}）")
    print("\n  按族（相对执行动作的真值差，中位）：")
    for name, agg in sorted(s["by_stage"].items()):
        print(f"    {name:12s} n={agg['n']:4d} 真值更优={agg['better']:3d}"
              f"  Δmin={agg['d_min_median']:+8.2f}%  Δmean={agg['d_mean_median']:+8.2f}%")
    f = s.get("force_accept")
    if f:
        print(f"\n  '全都采纳'对照（取预测 min EE 最高者强制采纳）：n={f['n']}"
              f" 真值更优 {f['wins_on_truth_min']} 块"
              f" Δmin中位 {f['d_min_median_pct']:+.2f}% Δmean中位 {f['d_mean_median_pct']:+.2f}%"
              f" 真值预算内 {f['truth_budget_ok_frac']:.1%}")
    lv = s.get("legit_v2")
    if lv:
        print(f"\n  真值口径的合法改进（更优 + 真值QoS通过 + 真值参考预算通过）：")
        print(f"    块数 {lv['blocks']} ({lv['frac']:.1%})  中位增益 {lv['gain_median_pct']:+.2f}%"
              f"  均值 {lv['gain_mean_pct']:+.2f}%  p90 {lv['gain_p90_pct']:+.2f}%"
              f"  >10%:{lv['n_gain_gt10']}  >50%:{lv['n_gain_gt50']}")
        for item in lv["top5"]:
            print(f"      block {item['block']:3d} {item['stage']:14s}"
                  f" minEE {item['exec_min_ee']/1e6:8.2f}→{item['cand_min_ee']/1e6:8.2f} M"
                  f" (+{item['gain_pct']:.1f}%)  Δmean {item['d_mean_pct']:+.2f}%")
    tri = s.get("truth_three_conditions")
    if tri:
        print(f"\n  真值更优的候选共 {tri['total']} 个：真值QoS失败 {tri['better_qos_fail']}，"
              f"真值参考预算失败 {tri['better_budget_true_fail']}，"
              f"三条全过 {tri['better_qos_ok_budget_true']}")
    if s.get("eta_ref_true_over_pred_median") is not None:
        print(f"  真值 Reference 均 EE 相对预测 η_ref 的中位偏差："
              f"{s['eta_ref_true_over_pred_median']:+.2%}")
    print(f"\n  执行动作：均 min EE {s['executed_mean_min_ee']/1e6:.2f} M，"
          f"均 mean EE {s['executed_mean_mean_ee']/1e6:.2f} M")


def main(argv=None):
    parser = argparse.ArgumentParser(description="路线 B 反事实审计")
    parser.add_argument("--run", default=str(V5 / "code" / "runs" / "v5_C_ref_sca_s_v5a"))
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="只审计前 N 块（冒烟）")
    parser.add_argument("--out", default=str(HERE / "out_routeB"))
    args = parser.parse_args(argv)
    print(f"run: {args.run}")
    run(args.run, args.blocks, args.limit, args.out)
    print("写出:", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
