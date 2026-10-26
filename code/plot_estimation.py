"""画出"真实信道 - 估计信道 - 观测信道"的块级曲线。

三条线：
  真实   g_true  = 几何增益 × 真值衰落        （只有评价器知道）
  估计   g_est   = 几何增益 × 卡尔曼后验均值
  观测   g_obs   = 导频时刻几何增益 × 该次观测  （画在导频自己的时刻上）

用法：
    python -B -I plot_estimation.py --n 10 --duration 100
    python -B -I plot_estimation.py --n 10 --duration 100 --i 0 --j 3
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3base.config import FoundationConfig
from p3base.estimator import ChannelEstimator
from p3base.fading import link_coherence_s
from p3base.scene import Scene, find_scene_root
from run_foundation import CODE_ROOT, TruthOracle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
PORT_COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))


def to_db(x):
    x = np.asarray(x, float)
    return np.where(x > 0, 10.0 * np.log10(np.maximum(x, 1e-300)), np.nan)


def probe(scene, cfg, seed, blocks=20):
    """挑一条 RF 全程可达、VLC 尽量可见的链路，并固定收发端口组合。"""
    estimator = ChannelEstimator(scene.n, cfg)
    oracle = TruthOracle(scene, cfg, seed)
    n = scene.n
    rf_ok = np.zeros((n, n))
    vlc_ok = np.zeros((n, n))
    port_ok = np.zeros((n, n, 4))
    dist = np.zeros((n, n))
    for k in range(blocks):
        t = k * cfg.estimation.block_dt_s
        oracle.advance_to(t)
        estimator.advance(t, oracle, rf_prior_var=oracle.rf_prior_var(),
                          rf_tau=link_coherence_s(oracle.rel_speed(t), cfg.rf))
        rf_ok += oracle.rf_geom["candidate"]
        vlc_ok += oracle.vlc_geom["candidate"]
        g_vlc = oracle.vlc_geom["gain"]
        port_ok += (g_vlc >= cfg.estimation.vlc_observable_gain_min).reshape(
            g_vlc.shape[0], g_vlc.shape[1], 4)
        c = oracle.state["center_xy_m"]
        dist += np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    rf_ok /= blocks
    vlc_ok /= blocks
    dist /= blocks
    np.fill_diagonal(rf_ok, -1)
    np.fill_diagonal(vlc_ok, -1)
    score = np.minimum(rf_ok, vlc_ok) - 1e-3 * dist
    i, j = np.unravel_index(np.argmax(score), score.shape)
    i, j = int(i), int(j)
    flat = port_ok[i, j]
    port = PORT_COMBOS[int(np.argmax(flat))] if flat.max() > 0 else None
    return i, j, port, float(rf_ok[i, j]), float(vlc_ok[i, j])


def run_series(scene, cfg, seed, blocks, i, j):
    """按块记录真值/估计/观测。VLC 端口按块取几何增益最大的组合。"""
    estimator = ChannelEstimator(scene.n, cfg)
    oracle = TruthOracle(scene, cfg, seed)
    # 逐导频的真值/观测由 oracle.trace 单独记录（与估计器无关），不依赖最近缓存，
    # 因此 1 ms 导频周期下也不会因为缓存淘汰而漏点。
    oracle.track = (i, j)
    rec = {k: [] for k in ("t", "rf_true", "rf_est", "rf_low", "rf_high", "rf_obs_t",
                           "rf_obs", "vlc_true", "vlc_est", "vlc_low", "vlc_high",
                           "vlc_obs_t", "vlc_obs", "vlc_tp", "vlc_rp",
                           "rf_fine_t", "rf_fine_v", "vlc_fine_t", "vlc_fine_v",
                           "rf_obs_true", "vlc_obs_true", "rf_ok", "vlc_ok")}
    cursor = {"rf": 0, "vlc": 0}
    for k in range(blocks):
        t = k * cfg.estimation.block_dt_s
        oracle.advance_to(t)
        estimator.advance(t, oracle, rf_prior_var=oracle.rf_prior_var(),
                          rf_tau=link_coherence_s(oracle.rel_speed(t), cfg.rf))
        belief = estimator.gain_belief(
            oracle.rf_geom["gain"], oracle.vlc_geom["gain"], t,
            rf_candidate=oracle.rf_geom["candidate"],
            vlc_candidate=oracle.vlc_geom["candidate"])
        truth = oracle.truth()

        rec["t"].append(t)
        rec["rf_ok"].append(bool(oracle.rf_geom["candidate"][i, j]))
        # truth["rf"] 已经是绝对功率增益（几何 × 阴影 × 小尺度），不能再乘一次几何。
        rec["rf_true"].append(truth["rf"][i, j])
        rec["rf_est"].append(belief["rf_gain_mean"][i, j])
        rec["rf_low"].append(belief["rf_gain_low"][i, j])
        rec["rf_high"].append(belief["rf_gain_high"][i, j])

        age = belief["rf_age_s"][i, j]
        obs_t = obs_v = obs_true = np.nan
        trace = oracle.trace["rf"]
        while cursor["rf"] < len(trace) and trace[cursor["rf"]][0] <= t + 1e-9:
            p, g, yv, amp, ok = trace[cursor["rf"]]
            cursor["rf"] += 1
            if not ok:
                continue
            rec["rf_fine_t"].append(p)
            rec["rf_fine_v"].append(g * abs(amp) ** 2)
            if p <= t - age + 1e-9:
                obs_t, obs_v = p, g * abs(yv) ** 2
                obs_true = g * abs(amp) ** 2
        rec["rf_obs_t"].append(obs_t)
        rec["rf_obs"].append(obs_v)
        rec["rf_obs_true"].append(obs_true)

        flat = oracle.vlc_geom["gain"][i, j].reshape(4)
        # 与估计器同一可见性判据；否则掠射端口的 -800 dB 点会把纵轴拉平。
        g_min = cfg.estimation.vlc_observable_gain_min
        visible = flat.max() >= g_min
        rec["vlc_ok"].append(bool(visible))
        if visible:
            tp, rp = divmod(int(flat.argmax()), 2)
            rec["vlc_tp"].append(tp)
            rec["vlc_rp"].append(rp)
            h_geom = oracle.vlc_geom["gain"][i, j, tp, rp]
            rec["vlc_true"].append(h_geom * np.exp(truth["vlc_log"][i, j, tp, rp]))
            rec["vlc_est"].append(belief["vlc_gain_mean"][i, j, tp, rp])
            rec["vlc_low"].append(belief["vlc_gain_low"][i, j, tp, rp])
            rec["vlc_high"].append(belief["vlc_gain_high"][i, j, tp, rp])
            age = belief["vlc_age_s"][i, j, tp, rp]
            obs_t = obs_v = obs_true = np.nan
            trace = oracle.trace["vlc"]
            while cursor["vlc"] < len(trace) and trace[cursor["vlc"]][0] <= t + 1e-9:
                p, gp, yv, zv, ok = trace[cursor["vlc"]]
                cursor["vlc"] += 1
                q = int(np.argmax(gp))
                if gp[q] < cfg.estimation.vlc_observable_gain_min or not ok[q]:
                    continue
                rec["vlc_fine_t"].append(p)
                rec["vlc_fine_v"].append(gp[q] * np.exp(zv[q]))
                if p <= t - age + 1e-9:
                    obs_t, obs_v = p, gp[q] * np.exp(yv[q])
                    obs_true = gp[q] * np.exp(zv[q])
            rec["vlc_obs_t"].append(obs_t)
            rec["vlc_obs"].append(obs_v)
            rec["vlc_obs_true"].append(obs_true)
        else:
            rec["vlc_tp"].append(-1)
            rec["vlc_rp"].append(-1)
            for key in ("vlc_true", "vlc_est", "vlc_low", "vlc_high",
                        "vlc_obs_t", "vlc_obs", "vlc_obs_true"):
                rec[key].append(np.nan)
            # 不可见的块也要把该块内产生的导频点消费掉，游标不能停在旧时间。
            trace = oracle.trace["vlc"]
            while cursor["vlc"] < len(trace) and trace[cursor["vlc"]][0] <= t + 1e-9:
                cursor["vlc"] += 1
    return rec


def _panel(ax, t, rec, prefix, colour, title, detail=False):
    """detail=False 只画块级序列；detail=True 叠加导频时刻的细采样真值与单次观测。

    1 ms 导频下细采样点有 2×10^4 个，全塞进一张 20 s 的图会糊成一条黑带，
    既看不清估计也看不清真值。所以总览图只保留块级三条量，细结构放到放大窗里。
    """
    if detail:
        ft = np.asarray(rec[f"{prefix}_fine_t"], float)
        fv = np.asarray(rec[f"{prefix}_fine_v"], float)
        flat = np.isfinite(ft) & np.isfinite(fv)
        ax.plot(ft[flat], to_db(fv[flat]), color="0.35", lw=0.9,
                label="真实信道（细采样）")
        ot = np.asarray(rec[f"{prefix}_obs_t"], float)
        ov = np.asarray(rec[f"{prefix}_obs"], float)
        ok = np.isfinite(ot) & np.isfinite(ov)
        # 导频发生在它自己的时刻，噪声很小；它"不准"是因为拿到块初已经过时。
        ax.plot(ot[ok], to_db(ov[ok]), ls="none", marker="o", ms=3.0,
                mfc="none", mec="tab:orange", mew=0.8, label="单次导频观测（当时）")
    ax.fill_between(t, to_db(rec[f"{prefix}_low"]), to_db(rec[f"{prefix}_high"]),
                    color=colour, alpha=0.16, lw=0, label="90% 预测区间")
    ax.plot(t, to_db(rec[f"{prefix}_true"]), color="black", lw=1.0, ls="--",
            marker="o", ms=3.2, mfc="none", mew=0.9,
            label="真实信道（决策时刻）")
    ax.plot(t, to_db(rec[f"{prefix}_est"]), color=colour, lw=1.8,
            label="Kalman-MMSE 估计")
    # 决策时刻真正可用的那条观测（导频时刻更早，年龄见 age），画在决策时刻上：
    # 这才是"用观测直接当信道"的误差量级。
    ax.plot(t, to_db(rec[f"{prefix}_obs"]), ls="none", marker="s", ms=3.6,
            mfc="none", mec="tab:orange", mew=1.0,
            label="块初可用观测（已过时）")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)


def _default_zoom(t, rec, prefix):
    """取该面板自身最后一段有数据的 0.5 s，避免 VLC 掉线后放大窗为空。"""
    true = np.asarray(rec[f"{prefix}_true"], float)
    good = t[np.isfinite(true)]
    if good.size == 0:
        return None
    end = float(good[-1]) + 1e-9
    return (end - 0.5, end)


def make_plot(rec, i, j, n, out_png, zoom=None):
    t = np.asarray(rec["t"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    panels = (
        (axes[0, 0], axes[0, 1], "rf", "tab:blue", f"n={n} 车，链路 {i}→{j}：RF"),
        (axes[1, 0], axes[1, 1], "vlc", "tab:green",
         "VLC（每块取几何增益最大的端口组合）"),
    )
    for full, zoomax, prefix, colour, title in panels:
        _panel(full, t, rec, prefix, colour, title, detail=False)
        _panel(zoomax, t, rec, prefix, colour,
               "放大窗：灰色细线为细采样真值，黑色虚线为块级真值", detail=True)
        values = np.concatenate([to_db(rec[f"{prefix}_fine_v"]),
                                 to_db(rec[f"{prefix}_true"]),
                                 to_db(rec[f"{prefix}_est"]),
                                 to_db(rec[f"{prefix}_obs"])])
        values = values[np.isfinite(values)]
        if values.size:
            # 可见性已经按 vlc_observable_gain_min 过滤，极端掠射点不再存在，
            # 直接用全范围，避免曲线被裁到坐标轴外。
            lo = float(values.min()) - 1.0
            hi = float(values.max()) + 1.0
            if hi - lo < 2.0:
                lo, hi = lo - 1.0, hi + 1.0
            for ax in (full, zoomax):
                ax.set_ylim(lo, hi)
        for ax in (full, zoomax):
            ax.set_ylabel("功率增益 / dB")
        full.set_xlabel("时间 / s")
        zoomax.set_xlabel("时间 / s")
        window = zoom if zoom is not None else _default_zoom(t, rec, prefix)
        if window is not None:
            zoomax.set_xlim(*window)
    axes[0, 1].legend(loc="upper center", ncol=2, fontsize=8, framealpha=0.9,
                      bbox_to_anchor=(0.5, -0.18))
    axes[1, 1].legend(loc="upper center", ncol=2, fontsize=8, framealpha=0.9,
                      bbox_to_anchor=(0.5, -0.18))
    fig.suptitle(f"真实信道 / Kalman-MMSE 估计 / 单次观测（n={n}，链路 {i}→{j}）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"图已保存：{out_png}")


def write_csv(rec, out_csv):
    keys = ("rf_true", "rf_est", "rf_low", "rf_high",
            "vlc_true", "vlc_est", "vlc_low", "vlc_high")
    header = (["time_s"] + [f"{k}_db" for k in keys]
              + ["rf_obs_time_s", "rf_obs_db", "vlc_obs_time_s", "vlc_obs_db",
                 "rf_candidate", "vlc_visible", "vlc_tx_port", "vlc_rx_port"])
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(len(rec["t"])):
            row = [rec["t"][k]]
            row += [float(to_db([rec[key][k]])[0]) for key in keys]
            row += [rec["rf_obs_t"][k], float(to_db([rec["rf_obs"][k]])[0]),
                    rec["vlc_obs_t"][k], float(to_db([rec["vlc_obs"][k]])[0]),
                    int(rec["rf_ok"][k]), int(rec["vlc_ok"][k]),
                    rec["vlc_tp"][k], rec["vlc_rp"][k]]
            w.writerow(row)
    print(f"数据已保存：{out_csv}")


def report(rec):
    for label, prefix in (("RF", "rf"), ("VLC", "vlc")):
        true_db = to_db(rec[f"{prefix}_true"])
        ok = np.isfinite(true_db)
        if not ok.any():
            print(f"{label}: 该链路全程不可见，无统计")
            continue
        est_db = to_db(rec[f"{prefix}_est"])
        e_est = (est_db - true_db)[ok]
        # 观测在导频时刻，误差必须对该时刻的真值算，不能对决策时刻的真值算。
        obs_db = to_db(rec[f"{prefix}_obs"])
        obs_true_db = to_db(rec[f"{prefix}_obs_true"])
        good = np.isfinite(obs_db) & np.isfinite(obs_true_db)
        e_obs = obs_db[good] - obs_true_db[good]
        obs_text = ("单次观测 与真值一致（无可用点）" if e_obs.size == 0 else
                    f"单次导频观测(当时) RMSE {np.sqrt(np.mean(e_obs ** 2)):.3f} dB"
                    f"（{e_obs.size} 个点）")
        # 同一批观测的"直接使用"误差：在块初把它当作当前信道。
        e_stale = (obs_db - true_db)[ok & np.isfinite(obs_db)]
        stale_text = ("陈旧观测 无可用点" if e_stale.size == 0 else
                      f"块初直接使用观测 RMSE {np.sqrt(np.mean(e_stale ** 2)):.2f} dB")
        print(f"{label}: 可用块 {int(ok.sum())}/{len(true_db)} | "
              f"估计 RMSE {np.sqrt(np.nanmean(e_est ** 2)):.2f} dB，"
              f"偏差 {np.nanmean(e_est):+.2f} dB | {stale_text} | {obs_text}")


def main():
    ap = argparse.ArgumentParser(description="真实/估计/观测信道曲线")
    ap.add_argument("--n", type=int, required=True, choices=(10, 50, 100))
    ap.add_argument("--duration", type=float, default=100.0)
    ap.add_argument("--i", type=int, default=None)
    ap.add_argument("--j", type=int, default=None)
    ap.add_argument("--zoom", type=float, nargs=2, default=None,
                    help="放大窗口的起止秒，例如 20 22")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--rf-period", type=float, default=None, help="RF 导频周期 / s")
    ap.add_argument("--vlc-period", type=float, default=None, help="VLC 导频周期 / s")
    ap.add_argument("--rf-state", default=None,
                    choices=("paper_doppler", "complex_ar1", "two_state"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = FoundationConfig()
    if args.rf_period or args.vlc_period or args.rf_state:
        from dataclasses import replace
        cfg = cfg.with_(estimation=replace(
            cfg.estimation,
            rf_pilot_period_s=args.rf_period or cfg.estimation.rf_pilot_period_s,
            vlc_pilot_period_s=args.vlc_period or cfg.estimation.vlc_pilot_period_s,
            rf_state=args.rf_state or cfg.estimation.rf_state))
    scene = Scene(find_scene_root(CODE_ROOT, args.n), cfg.vehicle)
    blocks = min(int(round(args.duration / cfg.estimation.block_dt_s)),
                 scene.block_count(cfg.estimation.block_dt_s))

    i, j, _, rf_frac, vlc_frac = probe(scene, cfg, args.seed)
    if args.i is not None and args.j is not None:
        i, j = args.i, args.j
    print(f"链路 {i}→{j}（探测 20 块：RF 可达 {rf_frac:.0%}，VLC 可见 {vlc_frac:.0%}）")

    rec = run_series(scene, cfg, args.seed, blocks, i, j)
    out_dir = args.out or (Path(__file__).resolve().parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"channel_trace_n{scene.n}_{i}to{j}"
    tag += f"_rfp{cfg.estimation.rf_pilot_period_s * 1e3:g}ms_{cfg.estimation.rf_state}_s{args.seed}"
    write_csv(rec, out_dir / f"{tag}.csv")
    make_plot(rec, i, j, scene.n, out_dir / f"{tag}.png",
              zoom=tuple(args.zoom) if args.zoom else None)
    report(rec)


if __name__ == "__main__":
    main()
