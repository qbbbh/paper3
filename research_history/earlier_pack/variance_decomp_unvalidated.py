"""把"预测不确定性"拆成 (a) 初始状态估计误差 (b) 块内未来演化，并与 CRLB 对齐。

做法（全部只读，不重跑决策、不写 run 目录、不改仓库代码）：

1. 重放估计器到块初 t，拿到两状态后验 (a, P_aa, L, P_LL) 与 VLC 的 (mean, var)。
   - σ_est(dB) = 4.343*sqrt(Var(|h|^2))/E|h|^2，其中 h=s*a（快分量 + 慢阴影）。
   - CRLB(dB) 用项目自带 crlb_report.local_db_crlb_sd（只对归一化复状态有效）。
2. **同一估计器、同一随机种子**建两个 ConditionalTrajectoryBank：
   - full  ：从后验抽样（正常口径）
   - ideal ：把 P_aa、P_LL、VLC var 置零（"上一次导频处 CSI 完全准确"），
             其余流程与随机流完全一致（公共随机数）。
   由全方差公式，Var_full − Var_ideal = Var(E[rate|state])，即**估计误差**对
   块平均速率预测的贡献；Var_ideal 是**块内未来演化**的贡献。
3. 用 run 里记录的真值速率做标定检查：实测残差 σ 应与 Var_full 的预测 σ 同量级。

用法（任意目录）：
  python -B -I e:/tmp/scripts/variance_decomp_20261026.py <run_dir> [draws]
"""
import copy
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

CODE = Path(r"E:\博士\课题论文\第3篇-修改版\代码\开发_50_100车_v3_20261024\code")
sys.path.insert(0, str(CODE))

from p3base.config import FoundationConfig                          # noqa: E402
from p3base.scene import Scene, find_scene_root                     # noqa: E402
from p3base import block, channel                                   # noqa: E402
from p3base.estimator import ChannelEstimator                       # noqa: E402
from p3base.fading import link_coherence_s                          # noqa: E402
from p3base.pilots import PilotPlan                                 # noqa: E402
from run_foundation import TruthOracle                              # noqa: E402
from p3learn.forecast import ConditionalTrajectoryBank              # noqa: E402
from crlb_report import local_db_crlb_sd                            # noqa: E402

DB = 10.0 / np.log(10.0)


def _collapsed(estimator):
    """Same posterior means, zero variance: the ideal-CSI counterfactual."""
    clone = copy.deepcopy(estimator)
    rf, vf = clone.rf, clone.vlc
    if rf.state == "two_state":
        rf.P_aa = np.zeros_like(rf.P_aa)
        rf.P_LL = np.zeros_like(rf.P_LL)
    else:
        rf.var = np.zeros_like(rf.var)
    vf.var = np.zeros_like(vf.var)
    return clone


def _rf_gain_sigma_db(a, P_aa, L, P_LL):
    """Relative sd of |s*a|^2 in dB from the two-state posterior."""
    var_pow = 2.0 * P_aa ** 2 + 4.0 * np.abs(a) ** 2 * P_aa
    e_pow = np.exp(L + 0.5 * P_LL) * (np.abs(a) ** 2 + P_aa)
    return np.where(e_pow > 0, DB * np.sqrt(np.maximum(var_pow, 0.)) / np.maximum(e_pow, 1e-300), 0.)


def _posterior_db(estimator, t):
    """Block-start gain posterior sd (dB): RF propagated, VLC log-intensity var."""
    dt = t - estimator.state_time["rf"]
    rf = estimator.rf
    a, P_aa, L, P_LL = rf.propagate(rf.a.copy(), rf.P_aa.copy(), rf.L.copy(),
                                    rf.P_LL.copy(), dt, estimator._rf_doppler, estimator._rf_tau)
    rf_sigma = _rf_gain_sigma_db(a, P_aa, L, P_LL)
    vf = estimator.vlc
    return rf_sigma, DB * np.sqrt(np.maximum(vf.var, 0.))     # VLC: (n,n,port,port)


def _pilot_instant_db(estimator):
    """Posterior sd right after the last consumed pilot (the CRLB comparison)."""
    rf = estimator.rf
    return _rf_gain_sigma_db(rf.a, rf.P_aa, rf.L, rf.P_LL)


def main(run_dir, draws=32):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    conf = summary["config"]
    cfg = FoundationConfig()
    cfg = cfg.with_(vlc=replace(cfg.vlc,
                                sinr_model=conf["physical"]["vlc"]["sinr_model"],
                                noise_model=conf["physical"]["vlc"]["noise_model"]))
    directory = find_scene_root(CODE.parent, 10)
    scene = Scene(directory, cfg.vehicle)
    context = block.GeometryContext.from_scene(scene)
    oracle = TruthOracle(scene, cfg, conf["channel_seed"])
    estimator = ChannelEstimator(scene.n, cfg)
    plan = PilotPlan.for_n(scene.n, cfg)
    npz = np.load(run_dir / "fine_samples.npz", allow_pickle=True)
    have = {int(re.search(r"_b(\d+)_", k).group(1))
            for k in npz.files if k.endswith("_forecast_samples_bps")}

    rows = []
    for k in sorted(have):
        t = k * cfg.estimation.block_dt_s
        oracle.advance_to(t)
        state = scene.knot(k)
        rg = channel.rf_geometry(context, state, cfg.rf)
        vg = channel.vlc_geometry(context, context.ports(state), cfg.vlc)
        v = state["speed_mps"][:, None] * np.stack(
            [np.cos(state["heading_rad"]), np.sin(state["heading_rad"])], -1)
        rel = np.linalg.norm(v[:, None] - v[None], axis=-1)
        sigma = np.where(rg["los"], cfg.rf.los_shadow_sigma_db, cfg.rf.nlos_shadow_sigma_db)
        prior = np.exp((sigma * np.log(10.) / 10.) ** 2 / 2.)
        estimator.advance(t, oracle, rf_prior_var=prior, rf_tau=link_coherence_s(rel, cfg.rf))

        from p3learn.allocation import Action
        belief = estimator.gain_belief(rg["gain"], vg["gain"], t,
                                       rf_candidate=rg["candidate"],
                                       vlc_candidate=vg["candidate"])
        # Deterministic port table, same call the run makes; the recorded ports
        # are then checked against it instead of being trusted blindly.
        ports = channel.select_ports(belief["vlc_gain_mean"])
        links = [(int(a), int(b), str(m)) for a, b, m in
                 zip(npz[f"reference_b{k}_tx"], npz[f"reference_b{k}_rx"],
                     npz[f"reference_b{k}_mode"])]
        powers = npz[f"reference_b{k}_power_w"]
        for n, (i, j, mode) in enumerate(links):
            if mode == "VLC":
                assert ports["tx"][i, j] == npz[f"reference_b{k}_tx_port"][n], "port replay mismatch"
                assert ports["rx"][i, j] == npz[f"reference_b{k}_rx_port"][n], "port replay mismatch"
        act = Action(links, powers, ports)

        seed = conf["forecast_seed"] + k
        full = ConditionalTrajectoryBank(estimator, state, context, t, draws=draws, seed=seed)
        ideal = ConditionalTrajectoryBank(_collapsed(estimator), state, context, t,
                                          draws=draws, seed=seed)
        f_full = full.evaluate(act)
        f_ideal = ideal.evaluate(act)
        s_full = np.asarray(f_full["block_rate_samples_bps"], float) / 1e6
        s_ideal = np.asarray(f_ideal["block_rate_samples_bps"], float) / 1e6
        actual = npz[f"reference_b{k}_rate_bps"].mean(0) / 1e6
        pred = s_full.mean(0)
        rf_sigma, vlc_sigma = _posterior_db(estimator, t)
        pilot_sigma = _pilot_instant_db(estimator)
        truth = oracle.fading.snapshot()
        shadow_true_db = np.asarray(truth["rf_shadow_db"], float)
        shadow_hat_db = DB * estimator.rf.L
        shadow_post_sd = DB * np.sqrt(np.maximum(estimator.rf.P_LL, 0.))

        for n, (tx, rx, mode, p, a_true, a_pred) in enumerate(
                zip(npz[f"reference_b{k}_tx"], npz[f"reference_b{k}_rx"],
                    npz[f"reference_b{k}_mode"], powers, actual, pred)):
            tx, rx = int(tx), int(rx)
            var_full = float(np.var(10 * np.log10(np.maximum(s_full[:, n], 1e-3))))
            var_ideal = float(np.var(10 * np.log10(np.maximum(s_ideal[:, n], 1e-3))))
            crlb = (float(local_db_crlb_sd(cfg, rg["gain"][tx, rx], plan.rf_power_w,
                                           plan.rf_slot_s, abs_h=1.0))
                    if mode == "RF" else float("nan"))
            rows.append(dict(
                block=k, link=n, mode=mode, power_w=float(p), tx=tx, rx=rx,
                pred_mean_mbps=float(a_pred), actual_mbps=float(a_true),
                pred_low_mbps=float(np.quantile(s_full[:, n], .05)),
                err_db=float(10 * np.log10(max(a_true, 1e-3) / max(a_pred, 1e-3))),
                low_ratio=float(np.quantile(s_full[:, n], .05) / max(a_pred, 1e-9)),
                actual_ratio=float(max(a_true, 1e-3) / max(a_pred, 1e-9)),
                sd_full_db=float(np.sqrt(var_full)), sd_ideal_db=float(np.sqrt(var_ideal)),
                est_share=float(max(var_full - var_ideal, 0.) / var_full) if var_full > 0 else 0.,
                sigma_block_gain_db=float(rf_sigma[tx, rx]) if mode == "RF" else float(
                    vlc_sigma[tx, rx, ports["tx"][tx, rx], ports["rx"][tx, rx]]),
                sigma_pilot_gain_db=float(pilot_sigma[tx, rx]) if mode == "RF" else float("nan"),
                shadow_err_db=float(shadow_hat_db[tx, rx] - shadow_true_db[tx, rx])
                if mode == "RF" else float("nan"),
                shadow_post_sd_db=float(shadow_post_sd[tx, rx]) if mode == "RF" else float("nan"),
                crlb_db=crlb))

    def agg(sel, label):
        if not sel:
            return None
        g = lambda name: np.array([r[name] for r in sel], float)   # noqa: E731
        full = float(np.mean(g("sd_full_db") ** 2))
        ideal = float(np.mean(g("sd_ideal_db") ** 2))
        crl = g("crlb_db")[np.isfinite(g("crlb_db"))]
        pilot = g("sigma_pilot_gain_db")[np.isfinite(g("sigma_pilot_gain_db"))]
        err = g("err_db")
        return dict(label=label, n=len(sel),
                    sd_pred_db=float(np.sqrt(full)),
                    sd_est_db=float(np.sqrt(max(full - ideal, 0.))),
                    sd_future_db=float(np.sqrt(ideal)),
                    est_share=float(max(full - ideal, 0.) / full) if full > 0 else 0.,
                    err_rms_db=float(np.sqrt(np.mean(err ** 2))),
                    err_p05_db=float(np.percentile(err, 5)),
                    err_p95_db=float(np.percentile(err, 95)),
                    pred_low_ratio=float(np.mean(g("low_ratio"))),
                    actual_ratio_p05=float(np.percentile(g("actual_ratio"), 5)),
                    sigma_block_gain_db=float(np.sqrt(np.mean(g("sigma_block_gain_db") ** 2))),
                    sigma_pilot_gain_db=float(np.sqrt(np.mean(pilot ** 2))) if len(pilot) else None,
                    crlb_median_db=float(np.median(crl)) if len(crl) else None,
                    sigma_pilot_over_crlb=(float(np.sqrt(np.mean(pilot ** 2)) / np.sqrt(np.mean(crl ** 2)))
                                           if len(crl) and len(pilot) else None),
                    shadow_err_rms_db=(float(np.sqrt(np.mean(g("shadow_err_db")[np.isfinite(g("shadow_err_db"))] ** 2)))
                                       if np.isfinite(g("shadow_err_db")).any() else None),
                    shadow_err_mean_db=(float(np.mean(g("shadow_err_db")[np.isfinite(g("shadow_err_db"))]))
                                        if np.isfinite(g("shadow_err_db")).any() else None),
                    shadow_err_sd_db=(float(np.std(g("shadow_err_db")[np.isfinite(g("shadow_err_db"))]))
                                      if np.isfinite(g("shadow_err_db")).any() else None),
                    shadow_post_sd_db=(float(np.sqrt(np.mean(g("shadow_post_sd_db")[np.isfinite(g("shadow_post_sd_db"))] ** 2)))
                                       if np.isfinite(g("shadow_post_sd_db")).any() else None))

    out = dict(run=str(run_dir), draws=draws, blocks=len(have),
               all=agg(rows, "全部"),
               rf=agg([r for r in rows if r["mode"] == "RF"], "RF"),
               vlc=agg([r for r in rows if r["mode"] == "VLC"], "VLC"),
               floor=agg([r for r in rows if r["power_w"] <= 0.0011], "贴 1 mW"),
               near=agg([r for r in rows if r["mode"] == "RF" and r["pred_mean_mbps"] < 1.6],
                        "预测<1.6M(约束边界)"))
    for value in out.values():
        if isinstance(value, dict) and "sd_pred_db" in value:
            print(f"{value['label']:<20} n={value['n']:<4} "
                  f"σpred={value['sd_pred_db']:.2f}dB 估计={value['sd_est_db']:.2f} "
                  f"未来={value['sd_future_db']:.2f} 估计占比={100*value['est_share']:.1f}% "
                  f"实测残差rms={value['err_rms_db']:.2f} "
                  f"预测p05/mean={value['pred_low_ratio']:.2f} 实测ratio_p05={value['actual_ratio_p05']:.2f} "
                  f"阴影误差 rms={value['shadow_err_rms_db'] if value['shadow_err_rms_db'] is None else round(value['shadow_err_rms_db'],2)}"
                  f"(偏={value['shadow_err_mean_db'] if value['shadow_err_mean_db'] is None else round(value['shadow_err_mean_db'],2)},"
                  f"散={value['shadow_err_sd_db'] if value['shadow_err_sd_db'] is None else round(value['shadow_err_sd_db'],2)}) "
                  f"阴影后验={value['shadow_post_sd_db'] if value['shadow_post_sd_db'] is None else round(value['shadow_post_sd_db'],2)} "
                  f"CRLB={None if value['crlb_median_db'] is None else round(value['crlb_median_db'],4)} "
                  f"比={None if value['sigma_pilot_over_crlb'] is None else round(value['sigma_pilot_over_crlb'],1)}")
    Path(r"e:/tmp/scripts/_variance_decomp.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32)
