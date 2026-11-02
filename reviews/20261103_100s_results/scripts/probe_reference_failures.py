"""探针：找出该种子上参考解失败的块（`reference.eta_ref is None`）。

R2 在 block 587 崩溃于 `run_allocation.py:649`
（`budget = (1-args.delta) * reference.eta_ref`，而 `eta_ref is None`）。
本探针只读、不修改任何代码，用于回答两个问题：

1. 失败是**孤立块**还是**成片出现**？
2. 除了 587，还有哪些块会触发同一个崩溃？

用法::

    python -B -I probe_reference_failures.py --channel-seed 2026102801 --start 580 --end 600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

from counterfactual_audit import Session, N, Q_MIN, QOS, START_BLOCK  # noqa: E402
import p3learn.allocation as A                                     # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="探测参考解失败的块")
    ap.add_argument("--channel-seed", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args(argv)

    session = Session(channel_seed=args.channel_seed)
    bad = []
    for k in range(START_BLOCK, args.end + 1):
        info = session.advance(k)
        if k < args.start:
            continue
        try:
            ref = A.pure_ee_reference(
                N, session.cfg, info["candidates"], session.business, info["ports"],
                info["bank"].evaluate, belief_gains=info["gains"],
                power_solver="continuous", search="joint", joint_rounds=8,
                joint_max_neighbours=0, q_min_bps=Q_MIN, qos_metric="chance",
                qos_fallback_mean=True, chance_rounds=2, chance_hold_mean=False,
                chance_inflation=QOS["inflation"])
        except Exception as exc:                                    # noqa: BLE001
            print(f"  block {k}: 参考解抛异常 {type(exc).__name__}: {exc}")
            bad.append(k)
            continue
        eta = getattr(ref, "eta_ref", "MISSING")
        ok = eta is not None
        flag = "" if ok else "   <-- eta_ref = None，会触发 run_allocation.py:649 崩溃"
        print(f"  block {k}: success={getattr(ref,'success',None)} "
              f"reason={getattr(ref,'reason','')!r} eta_ref="
              f"{'None' if eta is None else (f'{eta/1e6:.2f} M' if eta != 'MISSING' else 'MISSING')}{flag}")
        if not ok:
            bad.append(k)
    print(f"\n窗口 {args.start}..{args.end}：失败块 {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
