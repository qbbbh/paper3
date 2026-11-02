"""预算可行性上界筛选：在进候选池之前就剪掉"结构上不可能满足预算"的配置。

**动机**：前几轮发现，B/C 候选的预测预算缺口巨大（中位 24% η_ref），
即使最小化违反量（多起点 SLSQP）后，28/31 仍差 **12.6%（中位）**。
也就是说：这些配置**在任何功率向量下都不可能满足 (1−δ)η_ref**。

那我们不应该"先生成再修复"，而应该在生成时就用一个**解析上界**剪掉它们。

**上界（必要条件，方向严格）**：

对固定配置 m，逐链路在自己的功率上取能效峰值、**并让其他链路压到功率下界**
（干扰最小 ⇒ 该链路速率最大 ⇒ 其 EE 最大）。记

    U_k(m) = max_{p_k ∈ [lo_k, hi_k]}  r_k(p_k; others = lo) / (P_c,k + p_k)

则任何功率向量 p 都满足 EE_k(m,p) ≤ U_k(m)（others 的干扰只增不减），
故

    meanEE(m,p) ≤ mean_k U_k(m) =: U(m)

**若 U(m) < (1−δ)η_ref，则配置 m 一定不可行**——这是一个**解析必要条件**，
不需要预测评估器、不需要优化，代价约 5×40 次解析速率计算。

用法::

    python -B -I budget_ceiling_screen.py --run <v5 run> --channel-seed 2026102401 ^
        --phase0 out_phase0/phase0.json --out ceiling_design.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

from counterfactual_audit import Session, load_archived, _patched_stage_factory, _ORIG_STAGE, \
    N, DELTA, Q_MIN, QOS, START_BLOCK                       # noqa: E402
import p3learn.allocation as A                              # noqa: E402
from p3learn.dinkelbach_sca import _Physics                  # noqa: E402
from feasible_first_joint import prefilter                   # noqa: E402

GRID = 40


def budget_ceiling(links, gains, ports, cfg):
    """U(m)：逐链路在他人取功率下界时的能效峰值上界，再对链路取平均。"""
    phys = _Physics(links, gains, ports, cfg)
    lo = np.asarray(phys.lo, float)
    hi = np.asarray(phys.hi, float)
    base = lo.copy()
    ceilings = []
    for k in range(phys.k):
        grid = np.geomspace(max(lo[k], 1e-9), hi[k], GRID)
        best = -np.inf
        for p in grid:
            trial = base.copy()
            trial[k] = p
            ee = float(phys.rates(trial)[k] / (phys.static[k] + p))
            if ee > best:
                best = ee
        ceilings.append(best)
    return float(np.mean(ceilings)), [float(x) for x in ceilings]


def run(args):
    wanted, n_bc = prefilter(args.phase0)
    n_cand = sum(len(v) for v in wanted.values())
    print(f"预登记：B/C {n_bc} → 命中 {n_cand}（{len(wanted)} 块）")
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
        threshold = (1.0 - DELTA) * eta_ref
        arch = archived_candidates.get(k, [])
        for idx, item in enumerate(sink):
            stage = arch[idx]["stage"] if idx < len(arch) else "?"
            if stage not in wanted[k]:
                continue
            cand = item["action"]
            U, per_link = budget_ceiling(list(cand.links), info["gains"],
                                         info["ports"], session.cfg)
            records.append({"block": k, "stage": stage,
                            "ceiling_mean_ee": U, "per_link_ceiling": per_link,
                            "threshold": threshold,
                            "pruned": bool(U < threshold),
                            "headroom": U / max(threshold, 1e-12) - 1.0})
            print(f"  k={k:3d} {stage:14s} U {U/1e6:8.2f}M  阈值 {threshold/1e6:8.2f}M  "
                  f"余量 {(U/max(threshold,1e-12)-1)*100:+7.2f}%  pruned={records[-1]['pruned']}")

    A.weak_link_repair_stage = _ORIG_STAGE
    return {"run": args.run, "channel_seed": args.channel_seed,
            "prefilter": {"bc": n_bc, "matched": n_cand},
            "elapsed_s": time.perf_counter() - t0, "records": records}


def summarise(payload, probe_path=None):
    R = payload["records"]
    pruned = [r for r in R if r["pruned"]]
    print("\n" + "=" * 78)
    print(f"预算上界筛选（{len(R)} 候选，{payload['elapsed_s']:.1f}s）")
    print(f"  解析上界 U < (1-δ)η_ref ⇒ 剪掉：{len(pruned)}/{len(R)}")
    head = np.array([r["headroom"] for r in R])
    print(f"  U 相对阈值的余量：中位 {100*np.median(head):+.1f}%  最大 {100*head.max():+.1f}%")
    if probe_path and os.path.isfile(probe_path):
        with open(probe_path, "r", encoding="utf-8") as handle:
            probe = json.load(handle)
        feas = {(r["block"], r["stage"]) for r in probe["records"] if r["feasible"]}
        tp = sum(1 for r in R if r["pruned"] and (r["block"], r["stage"]) not in feas)
        fp = sum(1 for r in R if r["pruned"] and (r["block"], r["stage"]) in feas)
        fn = sum(1 for r in R if not r["pruned"] and (r["block"], r["stage"]) in feas)
        print(f"\n  与可行性探针对照（真可行 = SLSQP 找到 ≤1e-6 违反量）：")
        print(f"    正确剪掉: {tp}   误剪（真可行却剪掉）: {fp}   漏剪: {fn}")
        print(f"    剪枝精度 {tp/max(tp+fp,1):.1%}，可行配置保留率 {fn/ max(len(feas),1):.1%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="预算可行性解析上界筛选")
    parser.add_argument("--run", required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
    parser.add_argument("--phase0", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe", default=None, help="可行性探针 JSON，用于算剪枝精度")
    args = parser.parse_args(argv)
    payload = run(args)
    summarise(payload, args.probe)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
