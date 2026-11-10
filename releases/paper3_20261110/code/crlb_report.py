"""Matched-object CRLB checks. Outputs JSON, not an alleged composite-channel limit.
Complex observation h+n (E|n|^2=R): ML MSE=R.
Scalar complex linear Gaussian AR1: Kalman posterior MSE=P for matched model.
Neither result certifies two_state or dB/block-rate prediction.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3base.config import FoundationConfig
from p3base.scene import Scene, find_scene_root
from p3base import channel


def obs_var(cfg, geom_gain, pilot_power_w, slot_s):
    g = np.asarray(geom_gain, float)
    n0 = cfg.rf.noise_w / cfg.rf.bandwidth_hz
    return np.divide(n0, pilot_power_w * slot_s * g,
                     out=np.full_like(g, np.inf), where=g > 0)


def local_db_crlb_sd(cfg, geom_gain, pilot_power_w, slot_s, abs_h=1.):
    # Transformed-parameter CRLB SD at specified deterministic h != 0.
    # Log of noisy ML estimate attains this only asymptotically at high SNR.
    if np.any(np.asarray(abs_h) <= 0):
        raise ValueError('nonzero deterministic amplitude required')
    return (10/np.log(10)) * np.sqrt(2*obs_var(cfg, geom_gain, pilot_power_w, slot_s)) / abs_h


def required_pilot_energy(cfg, gain, target_db_sd, abs_h=1.):
    if gain <= 0 or target_db_sd <= 0 or abs_h <= 0:
        raise ValueError('positive gain, precision and amplitude required')
    return cfg.rf.noise_w / cfg.rf.bandwidth_hz / gain * (10*np.sqrt(2)/np.log(10)/target_db_sd/abs_h)**2


def validate_bounds(seed=123, repetitions=50000):
    rng = np.random.default_rng(seed)
    def cn():
        return (rng.standard_normal(repetitions)+1j*rng.standard_normal(repetitions))/np.sqrt(2)
    R=.03
    err=np.sqrt(R)*cn()
    obs_mse=float(np.mean(np.abs(err)**2))
    # Matched linear Gaussian synthetic example, NOT the composite shadow channel.
    rho=.85; Q=1-rho*rho
    x=cn(); m=np.zeros(repetitions,complex); P=1.
    rows=[]
    for k in range(20):
        if k:
            x=rho*x+np.sqrt(Q)*cn(); m=rho*m; P=rho*rho*P+Q
        y=x+np.sqrt(R)*cn(); K=P/(P+R)
        m=m+K*(y-m); P=(1-K)*P
        rows.append(dict(step=k, mse=float(np.mean(np.abs(x-m)**2)), bayes_mse=P))
    return dict(seed=seed,repetitions=repetitions, observation_complex_mse=obs_mse,
                observation_crlb=R, observation_ratio=obs_mse/R, linear_gaussian=rows,
                scope='normalized complex state; not composite two-state or dB optimality')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('results/crlb_validation.json'))
    args=ap.parse_args()
    cfg=FoundationConfig(); result=validate_bounds()
    result['scene_local_db_sd_at_unit_state']=[]
    for n in (10,50,100):
        scene=Scene(find_scene_root(Path(__file__).resolve().parents[1],n), cfg.vehicle)
        values=[]
        slot=min(cfg.estimation.rf_pilot_max_slot_s,.2*.001/n)
        for t in (0.,10.,50.,99.9):
            g=channel.rf_geometry(scene,scene.at(t),cfg.rf)
            values.extend(local_db_crlb_sd(cfg,g['gain'][g['candidate']],2.,slot).tolist())
        a=np.asarray(values)
        result['scene_local_db_sd_at_unit_state'].append(dict(n=n,slot_s=slot,snapshot_count=4,
            candidate_samples=len(a),p50=float(np.median(a)),p90=float(np.quantile(a,.9)),worst=float(a.max()),
            note='local bound at normalized |h|=1, excludes realized fades; not full-trajectory measurement error'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(args.out)

if __name__=='__main__':
    main()
