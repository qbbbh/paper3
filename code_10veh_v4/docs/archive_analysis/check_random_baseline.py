"""`random_baseline.py` 的自检：不依赖 SUMO，只用合成合法域。

运行：D:\\anaconda\\envs\\pymarl\\python.exe -B -I _staging_20261031/check_random_baseline.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
sys.path.insert(0, str(HERE))

import numpy as np
from p3base.config import FoundationConfig
from p3base.block import validate_action
from random_baseline import random_legal_action


def main():
    cfg = FoundationConfig()
    n = 6
    candidates = {m: np.ones((n, n), bool) for m in ("RF", "VLC")}
    business = np.ones((n, n), bool)
    ports = dict(tx=np.zeros((n, n), int), rx=np.zeros((n, n), int))
    rng = np.random.default_rng(20261031)
    seen = set()
    for _ in range(200):
        action = random_legal_action(n, cfg, candidates, business, ports, rng)
        validate_action(action.links, action.powers, action.ports, n, cfg,
                        candidates=candidates, business=business)
        floors = np.array([getattr(cfg, m.lower()).power_min_w for _i, _j, m in action.links])
        caps = np.array([getattr(cfg, m.lower()).power_max_w for _i, _j, m in action.links])
        assert np.all(action.powers >= floors) and np.all(action.powers <= caps)
        seen.add(tuple(sorted(m for _i, _j, m in action.links)))
    print("ok: 200 samples, distinct mode patterns =", len(seen))
    # 同种子必须复现
    a = random_legal_action(n, cfg, candidates, business, ports, np.random.default_rng(7))
    b = random_legal_action(n, cfg, candidates, business, ports, np.random.default_rng(7))
    assert a.links == b.links and np.array_equal(a.powers, b.powers)
    print("ok: deterministic under a fixed rng")
    # 无完整配对时必须抛 CoverageFailure，而不是给出残缺动作
    holey = {"RF": np.ones((n, n), bool), "VLC": np.zeros((n, n), bool)}
    holey["RF"][0, :] = False
    holey["RF"][:, 0] = False
    try:
        random_legal_action(n, cfg, holey, business, ports, np.random.default_rng(1))
    except Exception as exc:
        print("ok: infeasible domain ->", type(exc).__name__)
    else:
        raise AssertionError("expected coverage failure for an infeasible domain")


if __name__ == "__main__":
    main()
