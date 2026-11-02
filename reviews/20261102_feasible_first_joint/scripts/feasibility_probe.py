"""可行性探针：给定通信配置，**最小化预算违反量**，判断"求解器没找到"还是"配置不可行"。

前几轮的 `maxmin_power_refine` 是"最大化 min EE + 罚项"，它的失败不能区分两种情况：

- (a) 存在可行功率解，但局部搜索没找到；
- (b) 在该配置 + QoS + 功率箱下，预算**根本无法满足**。

本脚本直接对**违反量**做最小化（多起点 SLSQP），得到"可达的最小违反量"：

- `min_violation ≈ 0` → 配置可行，之前只是没找到 → 应由"可行域优先"求解接管；
- `min_violation > 0`    → 在该 QoS 与功率箱下，预算对该配置**不可满足**（数值证据，非证明）。

目标（归一化，便于跨块比较）::

    f(p) = max(0, (1-δ)η_ref − meanEE(p)) / η_ref  +  10 · mean_k max(0, q − R̃_k(p)) / q

用法::

    python -B -I feasibility_probe.py --run <v5 run> --channel-seed 2026102401 ^
        --phase0 out_phase0/phase0.json --out feas_design.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

from counterfactual_audit import (                      # noqa: E402
    Session, load_archived, _patched_stage_factory, _ORIG_STAGE,
    truth_metrics, N, DELTA, Q_MIN, QOS, START_BLOCK,
)
import p3learn.allocation as A                          # noqa: E402
from feasible_first_joint import map_reference_powers, prefilter, predicted_metrics  # noqa: E402

MAX_EVALS = 400


def probe_configuration(cand, cfg, evaluator, eta_ref, starts):
    """多起点最小化归一化违反量；返回最小违反量与对应功率。"""
    links = tuple(cand.links)
    modes = [m for _i, _j, m in links]
    lo = np.array([cfg.rf.power_min_w if m == "RF" else cfg.vlc.power_min_w for m in modes])
    hi = np.array([cfg.rf.power_max_w if m == "RF" else cfg.vlc.power_max_w for m in modes])
    cache = {}
    calls = {"n": 0}

    def evaluate(p):
        key = tuple(np.round(np.clip(p, lo, hi), 12))
        if key in cache:
            return cache[key]
        fc = evaluator(A.Action(links, np.asarray(key, float), cand.ports))
        calls["n"] += 1
        rate = np.asarray(fc["mean_rate_bps"], float)
        denom = np.asarray(A.link_electrical_power_w(np.asarray(key, float), modes, cfg), float)
        ee = rate / denom
        qrates = A.predicted_qos_rates(fc, "chance", QOS["inflation"])
        budget_v = max(0.0, (1.0 - DELTA) * eta_ref - float(ee.mean())) / max(eta_ref, 1e-12)
        qos_v = float(np.mean(np.maximum(Q_MIN - qrates, 0.0)) / max(Q_MIN, 1e-12))
        out = (budget_v, qos_v, float(ee.mean()), float(ee.min()))
        cache[key] = out
        return out

    def objective(u):
        p = np.clip(np.exp(u), lo, hi)
        b, qq, _, _ = evaluate(p)
        return b + 10.0 * qq

    best = None
    for start in starts:
        u0 = np.log(np.clip(np.asarray(start, float), lo, hi))
        try:
            res = minimize(objective, u0, method="SLSQP",
                           bounds=[(np.log(a), np.log(b)) for a, b in zip(lo, hi)],
                           options={"maxiter": 120, "ftol": 1e-10})
            u = res.x
        except Exception:                                    # noqa: BLE001
            u = u0
        p = np.clip(np.exp(u), lo, hi)
        b, qq, mean_ee, min_ee = evaluate(p)
        score = b + 10.0 * qq
        if best is None or score < best["score"]:
            best = {"score": float(score), "budget_violation": float(b),
                    "qos_violation": float(qq), "mean_ee": mean_ee, "min_ee": min_ee,
                    "powers": [float(x) for x in p], "start": [float(x) for x in start]}
        if calls["n"] > MAX_EVALS:
            break
    best["evaluator_calls"] = calls["n"]
    return best


def run(args):
    wanted, n_bc = prefilter(args.phase0)
    n_cand = sum(len(v) for v in wanted.values())
    print(f"预登记：B/C {n_bc} → 命中 {n_cand}（{len(wanted)} 块）")
    if not wanted:
        return None

    _, archived_candidates, _ = load_archived(args.run)
    session = Session(channel_seed=args.channel_seed)
    records = []
    t0 = time.perf_counter()

    for step in range(100):
        k = START_BLOCK + step
        info = session.advance(k)
        if k not in wanted:
            continue
        sink = []
        A.weak_link_repair_stage = _patched_stage_factory(sink)
        reference = A.pure_ee_reference(
            N, session.cfg, info["candidates"], session.business, info["ports"],
            info["bank"].evaluate, belief_gains=info["gains"],
            power_solver="continuous", search="joint", joint_rounds=8,
            joint_max_neighbours=0, q_min_bps=Q_MIN, qos_metric="chance",
            qos_fallback_mean=True, chance_rounds=2, chance_hold_mean=False,
            chance_inflation=QOS["inflation"])
        decision = A.max_min_allocate(
            N, session.cfg, info["candidates"], session.business, info["ports"],
            info["bank"].evaluate, delta=DELTA, belief_gains=info["gains"],
            reference=reference, qos=QOS, power_solver="continuous", search="joint",
            joint_rounds=8, joint_max_neighbours=0, power_refine=True,
            refine_maxiter=240, refine_restarts=2, refine_solver="sca",
            weak_link_repair=True, repair_limit=3)
        if decision.action is None:
            continue
        eta_ref = float(decision.eta_ref)
        arch = archived_candidates.get(k, [])
        for idx, item in enumerate(sink):
            stage = arch[idx]["stage"] if idx < len(arch) else "?"
            if stage not in wanted[k]:
                continue
            cand = item["action"]
            modes = [m for _i, _j, m in cand.links]
            lo = np.array([session.cfg.rf.power_min_w if m == "RF" else session.cfg.vlc.power_min_w
                           for m in modes])
            hi = np.array([session.cfg.rf.power_max_w if m == "RF" else session.cfg.vlc.power_max_w
                           for m in modes])
            mapped = map_reference_powers(reference.action, cand.links, session.cfg)
            starts = [np.asarray(cand.powers, float), mapped,
                      np.sqrt(lo * hi), lo.copy(), hi.copy()]
            best = probe_configuration(cand, session.cfg, info["bank"].evaluate,
                                       eta_ref, starts)
            before = predicted_metrics(info["bank"].evaluate(cand), session.cfg,
                                       cand.powers, cand.links, eta_ref)
            records.append({"block": k, "stage": stage,
                            "before": before, "probe": best,
                            "feasible": bool(best["budget_violation"] <= 1e-6
                                             and best["qos_violation"] <= 1e-6)})
            print(f"  k={k:3d} {stage:14s} before viol {before['margin_pred_ref']/1e6:+8.2f}M "
                  f"→ min budget viol {100*best['budget_violation']:6.3f}% "
                  f"qos viol {100*best['qos_violation']:6.3f}% "
                  f"minEE {best['min_ee']/1e6:7.2f}M feasible={records[-1]['feasible']}")

    A.weak_link_repair_stage = _ORIG_STAGE
    return {"run": args.run, "channel_seed": args.channel_seed,
            "prefilter": {"bc": n_bc, "matched": n_cand},
            "elapsed_s": time.perf_counter() - t0, "records": records}


def summarise(payload):
    R = payload["records"]
    if not R:
        return
    feas = [r for r in R if r["feasible"]]
    viol = np.array([r["probe"]["budget_violation"] for r in R])
    print("\n" + "=" * 78)
    print(f"可行性探针（{len(R)} 候选，{payload['elapsed_s']:.0f}s）")
    print(f"  最小化违反量后仍不可行：{len(R)-len(feas)}/{len(R)}")
    print(f"  可达的最小预算违反量：中位 {100*np.median(viol):.3f}%  "
          f"最大 {100*viol.max():.3f}%  ≤1e-6 的 {int((viol<=1e-6).sum())}")
    q = np.array([r["probe"]["qos_violation"] for r in R])
    print(f"  可达的最小 QoS 违反量：中位 {100*np.median(q):.3f}%  最大 {100*q.max():.3f}%")
    if feas:
        gains = [100 * (r["probe"]["min_ee"] - r["before"]["min_ee"]) / max(r["before"]["min_ee"], 1e-12)
                 for r in feas]
        print(f"  可行者的 min EE 相对候选初值：中位 {np.median(gains):+.2f}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description="可行性探针：区分'没找到'与'不可行'")
    parser.add_argument("--run", required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
    parser.add_argument("--phase0", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = run(args)
    if not payload:
        return 1
    summarise(payload)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
