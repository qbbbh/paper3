"""10 车信道估计：导频周期扫描（预测 vs 陈旧观测）。只读代码树。

默认配置下 RF 导频 1 ms、VLC 50 ms，CSI 年龄几乎是常数，所以"误差 vs 年龄"
没有横轴变化。要回答"历史信息是否改善估计"，需要人为拉开导频间隔。

用法:
    python -B -I estimation_sweep_v1.py --code <code_dir> --out <out_dir>
"""
import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=50)
    ap.add_argument("--start-block", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026102401)
    return ap.parse_args()


def run(cfg, Scene, ChannelEstimator, TruthOracle, link_coherence_s, root, n,
        seed, start, blocks):
    scene = Scene(root, cfg.vehicle)
    oracle = TruthOracle(scene, cfg, seed)
    est = ChannelEstimator(scene.n, cfg)
    g_min = cfg.estimation.vlc_observable_gain_min
    acc = {"RF": [], "VLC": []}
    ages = {"RF": [], "VLC": []}
    for k in range(start, start + blocks):
        t = k * cfg.estimation.block_dt_s
        oracle.advance_to(t)
        est.advance(t, oracle, rf_prior_var=oracle.rf_prior_var(),
                    rf_tau=link_coherence_s(oracle.rel_speed(t), cfg.rf))
        grf = np.asarray(oracle.rf_geom["gain"], float)
        gvlc = np.asarray(oracle.vlc_geom["gain"], float)
        b = est.gain_belief(grf, gvlc, t, rf_candidate=oracle.rf_geom["candidate"],
                            vlc_candidate=oracle.vlc_geom["candidate"])
        truth = oracle.truth()
        rm, rv = est._predict_copy(est.rf, est.state_time["rf"], est.state_time["rf"], None, None)
        vm, vv = est._predict_copy(est.vlc, est.state_time["vlc"], est.state_time["vlc"], None, None)
        srf = grf * est.belief_builder.rf(rm, rv, 1.0)["mean"]
        svlc = gvlc * est.belief_builder.vlc(vm, vv, 1.0)["mean"]

        cand = np.asarray(oracle.rf_geom["candidate"], bool) & (grf > 0)
        for i, j in zip(*np.nonzero(cand)):
            acc["RF"].append((float(truth["rf"][i, j]), float(b["rf_gain_mean"][i, j]),
                              float(srf[i, j]), float(b["rf_gain_low"][i, j]),
                              float(b["rf_gain_high"][i, j])))
            ages["RF"].append(float(b["rf_age_s"][i, j]))

        flat = gvlc.reshape(scene.n, scene.n, 4)
        vis = flat.max(axis=2) >= g_min
        for i, j in zip(*np.nonzero(vis)):
            tp, rp = divmod(int(flat[i, j].argmax()), 2)
            g = gvlc[i, j, tp, rp]
            tv = g * float(np.exp(truth["vlc_log"][i, j, tp, rp]))
            acc["VLC"].append((tv, float(b["vlc_gain_mean"][i, j, tp, rp]),
                               float(svlc[i, j, tp, rp]), float(b["vlc_gain_low"][i, j, tp, rp]),
                               float(b["vlc_gain_high"][i, j, tp, rp])))
            ages["VLC"].append(float(b["vlc_age_s"][i, j, tp, rp]))

    out = {}
    for mode in ("RF", "VLC"):
        a = np.array(acc[mode], float)
        ok = np.isfinite(a).all(axis=1) & (a[:, 0] > 0)
        a = a[ok]
        if len(a) < 100:
            out[mode] = None
            continue
        true, pred, stale, lo, hi = a.T
        e_pred = np.abs(10 * np.log10(np.maximum(pred, 1e-300) / true))
        e_stale = np.abs(10 * np.log10(np.maximum(stale, 1e-300) / true))
        out[mode] = dict(
            n=int(len(a)), age_ms=float(np.median(ages[mode]) * 1e3),
            pred=float(np.median(e_pred)), pred_p90=float(np.percentile(e_pred, 90)),
            stale=float(np.median(e_stale)), stale_p90=float(np.percentile(e_stale, 90)),
            cover=float(np.mean((lo <= true) & (true <= hi))))
    return out


def main():
    a = parse()
    code = a.code.resolve()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code))
    from p3base.config import FoundationConfig
    from p3base.estimator import ChannelEstimator
    from p3base.fading import link_coherence_s
    from p3base.scene import Scene, find_scene_root
    from run_foundation import TruthOracle

    root = find_scene_root(code.parent, a.n)
    base = FoundationConfig()
    RF = [1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 40e-3]
    VLC = [5e-2, 1e-1, 2e-1, 4e-1, 8e-1]
    rows = []

    print("== RF pilot period sweep ==")
    for p in RF:
        cfg = base.with_(estimation=replace(base.estimation, rf_pilot_period_s=p))
        r = run(cfg, Scene, ChannelEstimator, TruthOracle, link_coherence_s, root,
                a.n, a.seed, a.start_block, a.blocks)
        if r["RF"] is None:
            print(f"  {p*1e3:5.1f} ms: too few links")
            continue
        d = r["RF"]
        rows.append(dict(mode="RF", period_ms=p * 1e3, **d))
        print(f"  {p*1e3:5.1f} ms  n={d['n']:5d} age={d['age_ms']:6.1f}ms "
              f"pred={d['pred']:.2f}/{d['pred_p90']:.2f} stale={d['stale']:.2f}/{d['stale_p90']:.2f} "
              f"cover={d['cover']:.3f}")

    print("== VLC pilot period sweep ==")
    for p in VLC:
        cfg = base.with_(estimation=replace(base.estimation, vlc_pilot_period_s=p))
        r = run(cfg, Scene, ChannelEstimator, TruthOracle, link_coherence_s, root,
                a.n, a.seed, a.start_block, a.blocks)
        if r["VLC"] is None:
            print(f"  {p*1e3:5.1f} ms: too few links")
            continue
        d = r["VLC"]
        rows.append(dict(mode="VLC", period_ms=p * 1e3, **d))
        print(f"  {p*1e3:5.1f} ms  n={d['n']:5d} age={d['age_ms']:6.1f}ms "
              f"pred={d['pred']:.2f}/{d['pred_p90']:.2f} stale={d['stale']:.2f}/{d['stale_p90']:.2f} "
              f"cover={d['cover']:.3f}")

    with (out / "sweep.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (mode, title) in enumerate((("RF", "(a) RF：误差随导频周期"),
                                       ("VLC", "(b) VLC：误差随导频周期"))):
        s = [r for r in rows if r["mode"] == mode]
        x = [r["period_ms"] for r in s]
        ax[k].plot(x, [r["pred"] for r in s], "o-", label="因果预测（中位）")
        ax[k].plot(x, [r["stale"] for r in s], "s--", label="陈旧观测（中位）")
        ax[k].plot(x, [r["pred_p90"] for r in s], "o:", color="C0", alpha=.6, label="预测 90 分位")
        ax[k].plot(x, [r["stale_p90"] for r in s], "s:", color="C1", alpha=.6, label="陈旧 90 分位")
        ax[k].set_xscale("log")
        ax[k].set_xlabel("导频周期 (ms)")
        ax[k].set_ylabel("|dB 误差|")
        ax[k].set_title(title)
        ax[k].grid(alpha=.3, which="both")
        ax[k].legend(fontsize=8)
    for mode, c in (("RF", "C0"), ("VLC", "C1")):
        s = [r for r in rows if r["mode"] == mode]
        ax[2].plot([r["period_ms"] for r in s], [r["cover"] for r in s], "o-", color=c,
                   label=f"{mode}")
    ax[2].axhline(.95, color="k", ls=":", label="名义 95%")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("导频周期 (ms)")
    ax[2].set_ylabel("经验覆盖率")
    ax[2].set_ylim(0, 1.02)
    ax[2].set_title("(c) 后验 95% 区间的实际覆盖率")
    ax[2].grid(alpha=.3, which="both")
    ax[2].set_axisbelow(True)
    ax[2].legend(loc="lower left", framealpha=.95)
    fig.tight_layout()
    fig.savefig(out / "fig_sweep.png", dpi=150)
    print("wrote", out / "fig_sweep.png")


if __name__ == "__main__":
    main()
