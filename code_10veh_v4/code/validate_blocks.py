"""Development audit, not a proposed scheduler or formal performance benchmark.
Fixed same action evaluated on causal sampled forecasts and real future channels.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass
import numpy as np
import networkx as nx
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3base.config import FoundationConfig
from p3base.scene import Scene, find_scene_root
from p3base.estimator import ChannelEstimator
from p3base.fading import link_coherence_s
from p3base import block, channel
from run_foundation import TruthOracle


def fixture_action(rg, vg, state, mixed):
    n = len(state['speed_mps'])
    graph = nx.Graph(); graph.add_nodes_from(range(n))
    choices = {}
    for i in range(n):
        for j in range(i+1, n):
            options=[]
            for u,v in ((i,j),(j,i)):
                if rg['candidate'][u,v]:
                    options.append((u,v,'RF'))
                if mixed and vg['candidate'][u,v]:
                    options.append((u,v,'VLC'))
            if options:
                options.sort(key=lambda e: (e[2]!='VLC',e[0],e[1]))
                choices[i,j]=options[0]
                distance=np.linalg.norm(state['center_xy_m'][i]-state['center_xy_m'][j])
                graph.add_edge(i,j,weight=(1000 if options[0][2]=='VLC' else 0)-distance)
    matching=nx.max_weight_matching(graph,maxcardinality=True)
    links=[choices[min(i,j),max(i,j)] for i,j in sorted(tuple(sorted(e)) for e in matching)]
    if len(links)!=n//2:
        raise ValueError('fixture search did not find full geometry-qualified matching')
    return links, np.full(len(links),.1), channel.select_ports(vg['gain'])


def run_case(n, blocks, draws, mixed, seed, out):
    cfg=FoundationConfig()
    cfg=cfg.with_(estimation=replace(cfg.estimation, rf_state='two_state',rf_pilot_period_s=.001))
    scene=Scene(find_scene_root(Path(__file__).resolve().parents[1],n),cfg.vehicle)
    context=block.GeometryContext.from_scene(scene)
    oracle=TruthOracle(scene,cfg,seed); est=ChannelEstimator(n,cfg)
    tag=f'n{n}_{"mixed" if mixed else "rf"}'
    rows=[]; raw={}; dt=cfg.estimation.fine_dt_s; T=cfg.estimation.block_dt_s
    started=time.perf_counter()
    for b in range(blocks):
        index=2+b; t=index*T  # 0.2 s development warmup, not stationary calibration
        oracle.advance_to(t)
        est.advance(t,oracle,rf_prior_var=oracle.rf_prior_var(),rf_tau=link_coherence_s(oracle.rel_speed(t),cfg.rf))
        state=scene.knot(index)
        links,p,ports=fixture_action(oracle.rf_geom,oracle.vlc_geom,state,mixed)
        block.validate_action(links,p,ports,n,cfg,candidates={'RF':oracle.rf_geom['candidate'],'VLC':oracle.vlc_geom['candidate']})
        before=hashlib.sha256(repr((links,p.tolist(),ports['tx'].tolist(),ports['rx'].tolist())).encode()).hexdigest()
        pred=block.predict_block(est,state,context,links,p,ports,t,draws=draws,seed=seed+index)
        fine={key:[] for key in ('rate_bps','sinr','signal','interference','noise')}
        for k in range(int(round(T/dt))):
            oracle.advance_to(t+k*dt)
            truth=oracle.truth()
            v=block.sample_metrics(links,p,ports,truth['rf'],truth['vlc'],cfg)
            for key in fine:
                fine[key].append(v[key])
        fine={k:np.asarray(v) for k,v in fine.items()}
        summary=block.summarize_block(links,p,fine,dt,n)  # no unapproved QoS threshold
        after=hashlib.sha256(repr((links,p.tolist(),ports['tx'].tolist(),ports['rx'].tolist())).encode()).hexdigest()
        assert before==after
        np.testing.assert_allclose(summary['link_delivered_bits'].sum(),fine['rate_bps'].sum()*dt)
        np.testing.assert_allclose(summary['link_energy_j'].sum(),T*p.sum())
        np.testing.assert_allclose(summary['duration_s'],T)
        for a,(i,j,m) in enumerate(links):
            rate=summary['mean_rate_bps'][a]
            rows.append(dict(n=n,case=tag,block=b,time_s=t,tx_id=str(scene.vehicle_ids[i]),rx_id=str(scene.vehicle_ids[j]),
                tx=i,rx=j,mode=m,tx_port=int(ports['tx'][i,j]),rx_port=int(ports['rx'][i,j]),
                selected=True,active=True,power_w=float(p[a]),duration_s=T,
                predicted_rate_bps=float(pred['mean_rate_bps'][a]),predicted_p05_bps=float(pred['low_rate_bps'][a]),
                predicted_p95_bps=float(pred['high_rate_bps'][a]),actual_rate_bps=float(rate),
                inside_interval=bool(pred['low_rate_bps'][a]<=rate<=pred['high_rate_bps'][a]),
                delivered_bits=float(summary['link_delivered_bits'][a]),energy_j=float(summary['link_energy_j'][a]),
                pair_ee_bit_per_j=float(summary['pair_ee_bit_per_j'][a]),network_ee_bit_per_j=summary['network_ee_bit_per_j'],
                min_pair_ee_bit_per_j=summary['min_pair_ee_bit_per_j'],vlc_link_fraction=summary['vlc_link_fraction'],
                vlc_rate_fraction=summary['vlc_rate_fraction'],q_eval_bps='',qos_met='',
                action_sha256=before,seed=seed,draws=draws))
        raw.update({f'b{b}_{k}':v for k,v in fine.items()})
        raw.update({f'b{b}_powers_w':p,f'b{b}_offset_s':np.arange(len(fine['rate_bps']))*dt,
                    f'b{b}_forecast_block_rate_samples_bps':pred['block_rate_samples_bps']})
        print(f'{tag} block {b+1}/{blocks} complete',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with (out/f'{tag}.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/f'{tag}_fine.npz',**raw)
    err=np.array([r['predicted_rate_bps']-r['actual_rate_bps'] for r in rows])
    report=dict(n=n,case=tag,blocks=blocks,link_blocks=len(rows),seed=seed,draws=draws,
                fixed_action_all_blocks=True,full_coverage_all_blocks=True,
                integration_energy_checks=True,rmse_bps=float(np.sqrt(np.mean(err**2))),bias_bps=float(err.mean()),
                interval_coverage=float(np.mean([r['inside_interval'] for r in rows])),
                vlc_link_fraction=float(np.mean([r['mode']=='VLC' for r in rows])),
                elapsed_s=time.perf_counter()-started,config=cfg.to_dict(),pilot_plan=est.plan.as_dict(),
                scope='short integration validation; coverage diagnostic only, no reliability claim; data-only gross PHY EE')
    (out/f'{tag}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,choices=(10,50,100),default=10)
    ap.add_argument('--blocks',type=int,default=2)
    ap.add_argument('--draws',type=int,default=32)
    ap.add_argument('--seed',type=int,default=20261019)
    ap.add_argument('--mode',choices=('rf','mixed','both'),default='both')
    ap.add_argument('--out',type=Path,default=Path('results/block_validation'))
    args=ap.parse_args()
    for mixed in ([False,True] if args.mode=='both' else [args.mode=='mixed']):
        print(json.dumps(run_case(args.n,args.blocks,args.draws,mixed,args.seed,args.out),ensure_ascii=False),flush=True)
