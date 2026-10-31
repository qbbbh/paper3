"""10 车信道估计质量：预测 vs 陈旧观测，按 CSI 年龄统计。只读代码树，不改任何源码。

用法:
    python -B -I estimation_figure_v1.py --code <code_dir> --out <out_dir>

输出:
    estimation_links.csv   每条(块,链路,模式)的真值/预测/陈旧/区间/年龄
    coverage.csv           按模式与年龄分箱的覆盖率
    fig_estimation.png     三个子图
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--start-block", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026102401)
    return ap.parse_args()


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

    cfg = FoundationConfig()
    scene = Scene(find_scene_root(code.parent, a.n), cfg.vehicle)
    oracle = TruthOracle(scene, cfg, a.seed)
    est = ChannelEstimator(scene.n, cfg)
    n = scene.n
    g_min = cfg.estimation.vlc_observable_gain_min

    rows = []
    for k in range(a.start_block, a.start_block + a.blocks):
        t = k * cfg.estimation.block_dt_s
        oracle.advance_to(t)
        est.advance(t, oracle, rf_prior_var=oracle.rf_prior_var(),
                    rf_tau=link_coherence_s(oracle.rel_speed(t), cfg.rf))
        geom_rf = np.asarray(oracle.rf_geom["gain"], float)
        geom_vlc = np.asarray(oracle.vlc_geom["gain"], float)
        b = est.gain_belief(geom_rf, geom_vlc, t,
                            rf_candidate=oracle.rf_geom["candidate"],
                            vlc_candidate=oracle.vlc_geom["candidate"])
        truth = oracle.truth()

        # 陈旧观测 = 后验停在最后一次观测时刻，不再传播。与 belief() 的预测副本
        # 用同一个 PredictiveBelief 做变换，保证两种口径只差"是否传播"。
        rf_m, rf_v = est._predict_copy(est.rf, est.state_time["rf"], est.state_time["rf"], None, None)
        vl_m, vl_v = est._predict_copy(est.vlc, est.state_time["vlc"], est.state_time["vlc"], None, None)
        stale_rf = geom_rf * est.belief_builder.rf(rf_m, rf_v, 1.0)["mean"]
        stale_vlc = geom_vlc * est.belief_builder.vlc(vl_m, vl_v, 1.0)["mean"]

        cand = np.asarray(oracle.rf_geom["candidate"], bool) & (geom_rf > 0)
        for i, j in zip(*np.nonzero(cand)):
            rows.append(dict(block=k, mode="RF", i=int(i), j=int(j), port="",
                             age_s=float(b["rf_age_s"][i, j]),
                             true=float(truth["rf"][i, j]),
                             pred=float(b["rf_gain_mean"][i, j]),
                             stale=float(stale_rf[i, j]),
                             low=float(b["rf_gain_low"][i, j]),
                             high=float(b["rf_gain_high"][i, j])))

        flat = geom_vlc.reshape(n, n, 4)
        vis = flat.max(axis=2) >= g_min
        for i, j in zip(*np.nonzero(vis)):
            tp, rp = divmod(int(flat[i, j].argmax()), 2)
            g = geom_vlc[i, j, tp, rp]
            tv = g * float(np.exp(truth["vlc_log"][i, j, tp, rp]))
            rows.append(dict(block=k, mode="VLC", i=int(i), j=int(j), port=f"{tp}{rp}",
                             age_s=float(b["vlc_age_s"][i, j, tp, rp]),
                             true=tv, pred=float(b["vlc_gain_mean"][i, j, tp, rp]),
                             stale=float(stale_vlc[i, j, tp, rp]),
                             low=float(b["vlc_gain_low"][i, j, tp, rp]),
                             high=float(b["vlc_gain_high"][i, j, tp, rp])))

    with (out / "estimation_links.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"links: {len(rows)}  (RF {sum(r['mode']=='RF' for r in rows)}, "
          f"VLC {sum(r['mode']=='VLC' for r in rows)})")

    # ---- 分箱统计 ----
    # RF 导频 1 ms、延迟 2 ms，age 几乎恒为 2 ms，按年龄分箱没有意义，改画整体 CDF。
    # VLC 导频 50 ms、延迟 5 ms，age 在 5–55 ms 之间变化，按 10 ms 分箱。
    stats = []
    for mode in ("RF", "VLC"):
        sub = [r for r in rows if r["mode"] == mode and r["true"] > 0 and np.isfinite(r["age_s"])]
        ages = np.array([r["age_s"] for r in sub])
        print(f"{mode}: n={len(sub)} age min={ages.min()*1e3:.1f}ms "
              f"median={np.median(ages)*1e3:.1f}ms max={ages.max()*1e3:.1f}ms")
        if mode == "RF":
            groups = [("all", sub)]
        else:
            groups = []
            step = 10e-3
            hi_max = np.ceil(ages.max() / step) * step
            edges = np.arange(0, hi_max + step, step)
            for lo, hi in zip(edges[:-1], edges[1:]):
                groups.append(((lo, hi), [r for r in sub if lo <= r["age_s"] < hi]))
        for label, grp in groups:
            if len(grp) < 30:
                continue
            lo, hi = (0., 0.) if mode == "RF" else label
            rec = dict(mode=mode, age_lo_ms=lo * 1e3, age_hi_ms=hi * 1e3, count=len(grp))
            for key in ("pred", "stale"):
                e = np.array([10 * np.log10(max(r[key], 1e-300) / r["true"]) for r in grp])
                rec[key + "_med_abs_db"] = float(np.median(np.abs(e)))
                rec[key + "_bias_db"] = float(np.median(e))
            rec["cover95"] = sum(r["low"] <= r["true"] <= r["high"] for r in grp) / len(grp)
            stats.append(rec)

    with (out / "coverage.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    for s in stats:
        print(f"  {s['mode']} {s['age_lo_ms']:5.1f}-{s['age_hi_ms']:5.1f} ms n={s['count']:5d} "
              f"pred={s['pred_med_abs_db']:.2f}dB stale={s['stale_med_abs_db']:.2f}dB "
              f"cover95={s['cover95']:.3f}")

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    for k, (mode, title) in enumerate((("RF", "(a) RF：误差分布（年龄 2 ms）"),
                                       ("VLC", "(b) VLC：误差分布（年龄 50 ms）"))):
        for key, style, name in (("pred", "-", "因果预测"), ("stale", "--", "陈旧观测")):
            sub = [r for r in rows if r["mode"] == mode and r["true"] > 0]
            e = np.sort(np.abs(np.array([10 * np.log10(max(r[key], 1e-300) / r["true"])
                                         for r in sub])))
            ax[k].plot(e, np.arange(1, len(e) + 1) / len(e), style, label=name)
        ax[k].set_xlabel("|dB 误差|")
        ax[k].set_ylabel("累积比例")
        ax[k].set_xlim(0, 12 if mode == "RF" else 3)
        ax[k].set_title(title)
        ax[k].grid(alpha=.3)
        ax[k].legend()

    # (c) 固定周期下的经验覆盖率 vs 名义 95%
    modes = [q for q in stats]
    labels = [f"{q['mode']}\n(age {q['age_lo_ms']:.0f}–{q['age_hi_ms']:.0f} ms)"
              if q["age_hi_ms"] > 0 else f"{q['mode']}\n(age 2 ms)" for q in modes]
    ax[2].bar(labels, [q["cover95"] for q in modes], color=["C0", "C1", "C1"][:len(modes)],
              width=.5)
    ax[2].axhline(.95, color="k", ls=":")
    for idx, q in enumerate(modes):
        ax[2].text(idx, q["cover95"] + .005, f"{q['cover95']*100:.1f}%", ha="center", fontsize=9)
    ax[2].text(2.45, 1.005, "虚线：名义 95%", fontsize=8, va="bottom", ha="right")
    ax[2].set_ylim(0, 1.10)
    ax[2].set_ylabel("经验覆盖率")
    ax[2].set_title("(c) 后验 95% 区间实际覆盖率")
    ax[2].grid(alpha=.3, axis="y")
    ax[2].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig_estimation.png", dpi=150)
    print("wrote", out / "fig_estimation.png")


if __name__ == "__main__":
    main()
