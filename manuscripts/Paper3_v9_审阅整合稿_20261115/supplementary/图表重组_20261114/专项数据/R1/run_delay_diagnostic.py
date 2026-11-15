"""R1: shared noisy pilots, delayed causal consumption, fresh two-stage allocation."""
import os
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_v]='1'
import sys, json, gzip, pickle, copy, hashlib, time, argparse, csv, traceback
import psutil
from pathlib import Path
from dataclasses import replace
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=ROOT/'代码/公平功率求解改进_20261110/code'
sys.path.insert(0,str(CODE))
import replay_saved as rs
from p3learn.allocation import pure_ee_reference,max_min_allocate,predicted_qos_rates
from p3base.estimator import ChannelEstimator
from p3base.fading import link_coherence_s
from p3learn.forecast import ConditionalTrajectoryBank
from p3base import channel,block
from run_foundation import TruthOracle

DELAYS=[0,5,10,20]
BLOCKS=json.loads((ROOT/'论文/图表重组_20261114/专项版本核查.json').read_text(encoding='utf-8'))['proposed_uniform_blocks']
INPUT=HERE/'inputs';RUNS=HERE/'runs'

class SharedPilots:
    """No truth accessor; immutable arrays from one generated observation stream."""
    def __init__(self,oracle):self.oracle=oracle
    def _get(self,mode,t):
        y,var,valid=self.oracle.cache[mode][round(float(t),9)][:3]
        for a in [y,var,valid]:a.setflags(write=False)
        return y,var,valid
    def rf(self,t):return self._get('rf',t)
    def vlc(self,t):return self._get('vlc',t)

def dump_pickle(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wb',compresslevel=3) as f:pickle.dump(value,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,path)

def geometry(case,k):
    state=case.scene.knot(k);rg,vg=rs.geometry_now(case.context,state,case.cfg)
    v=state['speed_mps'][:,None]*np.stack([np.cos(state['heading_rad']),np.sin(state['heading_rad'])],-1)
    rel=np.linalg.norm(v[:,None]-v[None,:],axis=-1)
    sigma=np.where(rg['los'],case.cfg.rf.los_shadow_sigma_db,case.cfg.rf.nlos_shadow_sigma_db)
    prior=np.exp((sigma*np.log(10.)/10.)**2/2.)
    return state,rg,vg,prior,link_coherence_s(rel,case.cfg.rf)

def prepare(last=999):
    INPUT.mkdir(parents=True,exist_ok=True)
    case=rs.SavedCase('10B'); oracle=TruthOracle(case.scene,case.cfg,case.config['channel_seed'])
    provider=SharedPilots(oracle); estimators={}
    for d in DELAYS:
        e=ChannelEstimator(case.n,case.cfg)
        e.plan=replace(e.plan,rf_delay_s=e.plan.rf_delay_s+d/1000.,vlc_delay_s=e.plan.vlc_delay_s+d/1000.)
        estimators[d]=e
    started=time.perf_counter();stats=[];obs_hash=hashlib.sha256();hashed={'rf':-1.,'vlc':-1.}
    # Every block advances the estimator, including all 800 nonselected blocks.
    for k in range(last+1):
        t=k*case.cfg.estimation.block_dt_s
        oracle.advance_to(t);state,rg,vg,prior,tau=geometry(case,k)
        for mode in ['rf','vlc']:
            for pt,item in oracle.cache[mode].items():
                if pt<=hashed[mode]:continue
                obs_hash.update(mode.encode());obs_hash.update(np.float64(pt).tobytes())
                for a in item[:3]:obs_hash.update(np.ascontiguousarray(a).tobytes())
                hashed[mode]=pt
        beliefs={}
        for d,e in estimators.items():
            e.advance(t,provider,rf_prior_var=prior,rf_tau=tau)
            beliefs[d]=e.gain_belief(rg['gain'],vg['gain'],t,rf_candidate=rg['candidate'],vlc_candidate=vg['candidate'])
            assert all(pt<=t+1e-12 for pt in e.state_time.values())
        for d in DELAYS[1:]:
            assert np.array_equal(beliefs[0]['vlc_gain_mean'],beliefs[d]['vlc_gain_mean']),('VLC control',k,d)
            assert estimators[0].pilot_index['vlc']==estimators[d].pilot_index['vlc']
        if k in BLOCKS:
            truth=oracle.truth(); rfmask=np.asarray(rg['candidate'],bool)
            vlcmask=vg['candidate'][...,None,None] & (vg['gain']>=case.cfg.estimation.vlc_observable_gain_min)
            errors={}
            for d,b in beliefs.items():
                er={}
                for mode,mask,key in [('rf',rfmask,'rf_gain_mean'),('vlc',vlcmask,'vlc_gain_mean')]:
                    pred=b[key][mask];y=truth[mode][mask]
                    er[mode+'_error_sq_sum']=float(np.square(pred-y).sum())
                    er[mode+'_true_sq_sum']=float(np.square(y).sum())
                    er['n_'+mode+'_domain']=int(mask.sum())
                    er[mode+'_mean_age_s']=float(np.mean(b[mode+'_age_s'][mask])) if mask.any() else None
                er['rf_last_pilot_time_s']=estimators[d].state_time['rf']
                er['vlc_last_pilot_time_s']=estimators[d].state_time['vlc']
                errors[d]=er
            path=INPUT/f'block_{k:06d}.pkl.gz'
            if path.exists():
                with gzip.open(path,'rb') as f:previous=pickle.load(f)
                for d,e in estimators.items():
                    old=previous['estimators'][d]
                    assert e.state_time==old.state_time and e.pilot_index==old.pilot_index
                    for field in ['a','L','P_aa','P_LL']:
                        assert np.array_equal(getattr(e.rf,field),getattr(old.rf,field)),('snapshot replay',k,d,field)
                    for field in ['mean','var']:
                        assert np.array_equal(getattr(e.vlc,field),getattr(old.vlc,field))
                for mode in ['rf','vlc']:assert np.array_equal(previous['truth_start'][mode],truth[mode])
            else:
                dump_pickle(path,{'block':k,'estimators':estimators,'errors':errors,
                    'truth_start':{'rf':truth['rf'],'vlc':truth['vlc']},
                    'masks':{'rf':rfmask,'vlc':vlcmask}})
            rs.atomic_json(path.with_suffix('.commit.json'),{'block':k,'sha256':rs.digest(path),'delays_ms':DELAYS})
            stats.append({'block':k,'errors':errors,'sha256':rs.digest(path)})
        # Delete only observations consumed by ALL arms.
        for mode in ['rf','vlc']:
            period=getattr(oracle.plan,mode+'_period_s')
            next_t=min(e.pilot_index[mode] for e in estimators.values())*period
            for pt in list(oracle.cache[mode]):
                if pt<next_t-1e-12:del oracle.cache[mode][pt]
        if k%25==0 or k==last:
            rs.atomic_json(HERE/'prepare_progress.json',{'block':k,'last_requested':last,'selected_saved':len(stats),'elapsed_s':time.perf_counter()-started,'rss_mb':psutil.Process().memory_info().rss/2**20,'available_gib':psutil.virtual_memory().available/2**30})
            print('prepare',k,'elapsed',round(time.perf_counter()-started,2),flush=True)
    rs.atomic_json(HERE/'prepare_manifest.json',{'group':'10B','blocks':BLOCKS,'prepared_through_block':last,
        'extra_delay_ms':DELAYS,'observation_stream_sha256':obs_hash.hexdigest(),'config':case.config,
        'input':case.provenance(),'script_sha256':rs.digest(__file__),
        'formal_code_hashes':{str(f.relative_to(CODE)):rs.digest(f) for f in sorted(CODE.rglob('*.py'))},
        'domain_rf':'block-start RF geometry candidate mask',
        'domain_vlc':'block-start VLC candidate pairs and observable ports H>=Hmin',
        'vlc_information_equal_all_blocks':True,'elapsed_s':time.perf_counter()-started,'records':stats})

def action_record(a):
    return {'links':a.links,'powers_w':a.powers,'ports':a.ports}

def compare_action(a,b):
    assert sorted(a.links)==sorted(b.links),'links changed in delta-zero reproduction'
    ia=sorted(range(len(a.links)),key=lambda i:a.links[i]);ib=sorted(range(len(b.links)),key=lambda i:b.links[i])
    ap,bp=np.asarray(a.powers)[ia],np.asarray(b.powers)[ib]
    assert np.allclose(ap,bp,rtol=1e-9,atol=1e-10),('powers',float(np.max(abs(ap-bp))))
    for i,j,m in a.links:
        if m=='VLC':assert all(a.ports[s][i,j]==b.ports[s][i,j] for s in ['tx','rx'])
    return {'max_abs_power_w':float(np.max(abs(ap-bp))),'same_links':True}

def calculate(case,k,d,validate=False):
    source=INPUT/f'block_{k:06d}.pkl.gz'
    commit=json.loads(source.with_suffix('.commit.json').read_text(encoding='utf-8'))
    assert rs.digest(source)==commit['sha256']
    with gzip.open(source,'rb') as f:inp=pickle.load(f)
    est=inp['estimators'][d];state,rg,vg,_,_=geometry(case,k);t=k*.1
    b=est.gain_belief(rg['gain'],vg['gain'],t,rf_candidate=rg['candidate'],vlc_candidate=vg['candidate'])
    ports=channel.select_ports(b['vlc_gain_mean']);gains={'RF':b['rf_gain_mean'],'VLC':b['vlc_gain_mean']}
    c=case.config;started=time.perf_counter()
    bank=ConditionalTrajectoryBank(est,state,case.context,t,draws=c['draws'],seed=c['forecast_seed']+k,information=c['forecast_information'])
    evaluator=bank.evaluate
    candidates={'RF':rg['candidate'],'VLC':vg['candidate']};business=~np.eye(case.n,dtype=bool)
    checks={}
    # The fixed probe always uses ORIGINAL-information reference action and ports.
    normal=inp['estimators'][0].gain_belief(rg['gain'],vg['gain'],t,rf_candidate=rg['candidate'],vlc_candidate=vg['candidate'])
    normalports=channel.select_ports(normal['vlc_gain_mean'])
    probe,_=case.saved_action(k,'reference',normalports)
    oldmain,_=case.saved_action(k,'maxmin',normalports)
    probe_pred=evaluator(probe.action) if probe else None
    if d==0 and probe:
        checks['probe_prediction']=rs.assert_same_samples(probe_pred['block_rate_samples_bps'],probe.samples,'zero-delay original reference prediction')
        checks['old_main_prediction']=rs.assert_same_samples(evaluator(oldmain.action)['block_rate_samples_bps'],oldmain.samples,'zero-delay old fair prediction')
    result={'group':'10B','extra_delay_ms':d,'block':k,'forecast_seed':c['forecast_seed']+k,
        'input_sha256':commit['sha256'],'success':False,'reference_success':False,**inp['errors'][d]}
    try:
        ref=pure_ee_reference(case.n,case.cfg,candidates,business,ports,evaluator,
            belief_gains=gains,power_solver=c['power_solver'],search=c['reference_search'],
            joint_rounds=c['joint_rounds'],joint_max_neighbours=c['joint_max_neighbours'],
            q_min_bps=c['q_min_bps'],qos_metric=c['qos_metric'],qos_fallback_mean=c['qos_fallback_mean'],
            chance_rounds=c['qos_chance_rounds'],chance_hold_mean=c['qos_chance_hold_mean'],chance_inflation=c['qos_chance_inflation'])
        result['reference_success']=ref.success;result['reference_reason']=ref.reason
        result['initial_solve_s']=time.perf_counter()-started-bank.build_time_s
        actions={}
        if probe:actions['probe']=probe.action
        if ref.success:
            refined=max_min_allocate(case.n,case.cfg,candidates,business,ports,evaluator,
                delta=c['delta'],belief_gains=gains,reference=ref,qos=case.qos,
                power_solver=c['power_solver'],search=c['reference_search'],joint_rounds=c['joint_rounds'],
                joint_max_neighbours=c['joint_max_neighbours'],power_refine=c['maxmin_power_refine'],
                refine_maxiter=c['maxmin_refine_maxiter'],refine_restarts=c['maxmin_refine_restarts'],
                refine_solver='guarded',weak_link_repair=False,accept_key=c.get('accept_key','min_ee'))
            result['success']=refined.success;result['reason']=refined.reason;result['eta_ref']=ref.eta_ref
            actions['reference']=ref.action
            if refined.success:actions['main']=refined.action
            if d==0 and probe:
                checks['reference_action']=compare_action(ref.action,probe.action)
                checks['reference_eta']=bool(np.isclose(ref.eta_ref,rs.prediction_metrics(probe.action,probe_pred,case,ref.eta_ref)['mean_ee'],rtol=1e-9))
                assert checks['reference_eta']
                checks['main_action']=compare_action(refined.action,oldmain.action)
        else:result['reason']='initial_'+ref.reason
    except Exception as exc:
        if validate or d==0:raise
        result.update(success=False,reason='solver_exception:'+str(exc),traceback=traceback.format_exc());actions={'probe':probe.action} if probe else {}
    result['decision_s']=time.perf_counter()-started;result['forecast_s']=bank.build_time_s
    # Causal decision is complete before future truth access.
    measured,fine,evaluation_s=case.actual_evaluate_after_decision(k,actions)
    result['actual_evaluation_s']=evaluation_s;result['actual']=measured
    result['checks']=checks;result['actions']={name:action_record(a) for name,a in actions.items()}
    result['prediction']={}
    arrays={}
    for name,a in actions.items():
        forecast=evaluator(a)
        result['prediction'][name]=rs.prediction_metrics(a,forecast,case,result.get('eta_ref',0.))
        arrays[name+'_forecast_samples_bps']=forecast['block_rate_samples_bps']
        arrays[name+'_actual_mean_rate_bps']=fine[name]['rate_bps'].mean(axis=0)
        arrays[name+'_predicted_mean_rate_bps']=forecast['mean_rate_bps']
        arrays[name+'_calibrated_quantile_bps']=predicted_qos_rates(forecast,case.qos['metric'],case.qos['inflation'])
    if probe:
        checks['probe_actual']=rs.assert_same_samples(fine['probe']['rate_bps'],probe.actual_rates,'shared original truth')
    if 'main' in measured:
        a=measured['main'];r=measured['reference']
        result.update(avg_ee_mbit_j=a['mean_pair_ee_bit_per_j']/1e6,min_ee_mbit_j=a['min_pair_ee_bit_per_j']/1e6,
            ref_avg_ee_mbit_j=r['mean_pair_ee_bit_per_j']/1e6,ref_min_ee_mbit_j=r['min_pair_ee_bit_per_j']/1e6,
            all_service_pass=a['all_pairs_qos_met'],actual_loss=1-a['mean_pair_ee_bit_per_j']/r['mean_pair_ee_bit_per_j'])
    path=RUNS/f'delay_{d:02d}ms'/f'block_{k:06d}';path.mkdir(parents=True,exist_ok=True)
    rs.atomic_json(path/'record.json',result)
    np.savez_compressed(path/'arrays.npz',**arrays)
    rs.atomic_json(path/'commit.json',{'record.json':rs.digest(path/'record.json'),'arrays.npz':rs.digest(path/'arrays.npz'),'script_sha256':rs.digest(__file__)})
    return result

def worker(shard,workers=2,limit=None,validation=False):
    case=rs.SavedCase('10B');rows=[]
    ids=BLOCKS if limit is None else BLOCKS[:limit]
    for ordinal,k in enumerate(ids):
        if ordinal%workers!=shard:continue
        for d in ([0] if validation else DELAYS):
            path=RUNS/f'delay_{d:02d}ms'/f'block_{k:06d}'/'commit.json'
            if path.exists():continue
            r=calculate(case,k,d,validate=validation);rows.append(r)
            print('completed',d,k,'success',r['success'],'decision_s',round(r['decision_s'],2),flush=True)
            rs.atomic_json(HERE/f'worker_{shard}_progress.json',{'pid':os.getpid(),'block':k,'delay_ms':d,'completed_in_process':len(rows),'last_decision_s':r['decision_s'],'updated_epoch':time.time(),'rss_mb':psutil.Process().memory_info().rss/2**20,'peak_rss_mb':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)/2**20,'available_gib':psutil.virtual_memory().available/2**30})
    rs.atomic_json(HERE/f'worker_{shard}_done.json',{'pid':os.getpid(),'finished_epoch':time.time(),'mode':'validation' if validation else 'full','completed_in_process':len(rows)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','validate','worker']);ap.add_argument('--last',type=int,default=999);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int)
    a=ap.parse_args()
    if a.mode=='prepare':prepare(a.last)
    elif a.mode=='validate':worker(0,1,a.limit or 3,True)
    else:worker(a.shard,a.workers,a.limit)
