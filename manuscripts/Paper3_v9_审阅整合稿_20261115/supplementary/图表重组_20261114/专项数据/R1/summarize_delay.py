"""Aggregate committed R1 outputs; never treat pending periods as solver failures."""
from pathlib import Path
import json,csv,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BLOCKS=json.loads((ROOT/'论文/图表重组_20261114/专项版本核查.json').read_text(encoding='utf-8'))['proposed_uniform_blocks']
DELAYS=[0,5,10,20]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rows=[];completed={};probe=[];probe_links=[];action_links=[]
    repair_manifest=json.loads((HERE/'repairs_manifest.json').read_text(encoding='utf-8')) if (HERE/'repairs_manifest.json').exists() else {'repairs':[]}
    repairs={(r['extra_delay_ms'],r['block']):r for r in repair_manifest['repairs']}
    for d in DELAYS:
        for k in BLOCKS:
            folder=HERE/'runs'/f'delay_{d:02d}ms'/f'block_{k:06d}'
            repair=repairs.get((d,k))
            if repair:
                assert sha(folder/'commit.json')==repair['original_commit_sha256']
                folder=HERE/repair['replacement_directory']
                assert sha(folder/'commit.json')==repair['replacement_commit_sha256']
            commit=folder/'commit.json'
            row={'extra_delay_ms':d,'block':k,'status':'pending','success':None,'boundary_repaired':bool(repair)}
            if commit.exists():
                cm=json.loads(commit.read_text(encoding='utf-8'))
                for n in ['record.json','arrays.npz']:assert sha(folder/n)==cm[n]
                r=json.loads((folder/'record.json').read_text(encoding='utf-8'))
                completed[d,k]=r;row['status']='complete'
                for f in ['success','reference_success','reason','eta_ref','avg_ee_mbit_j','min_ee_mbit_j',
                    'ref_avg_ee_mbit_j','ref_min_ee_mbit_j','all_service_pass','actual_loss',
                    'rf_error_sq_sum','rf_true_sq_sum','vlc_error_sq_sum','vlc_true_sq_sum',
                    'n_rf_domain','n_vlc_domain','rf_mean_age_s','vlc_mean_age_s','decision_s']:
                    row[f]=r.get(f)
                pred=r.get('prediction',{}).get('main',{})
                row['predicted_budget_pass']=pred.get('budget_pass');row['quantile_pass']=pred.get('quantile_pass')
                row['reference_all_service_pass']=r.get('actual',{}).get('reference',{}).get('all_pairs_qos_met')
                for name in ['reference','main']:
                    action=r.get('actions',{}).get(name)
                    if action:
                        actual=r.get('actual',{}).get(name,{})
                        for j,(link,power) in enumerate(zip(action['links'],action['powers_w'])):
                            tx,rx,mode=link
                            action_links.append({'extra_delay_ms':d,'block':k,'method':name,'link_index':j,
                                'tx':tx,'rx':rx,'mode':mode,'electrical_input_power_w':power,
                                'tx_port':action['ports']['tx'][tx][rx] if mode=='VLC' else None,
                                'rx_port':action['ports']['rx'][tx][rx] if mode=='VLC' else None,
                                'actual_rate_bps':actual.get('mean_rate_bps',[None]*len(action['links']))[j],
                                'actual_ee_bit_per_j':actual.get('pair_ee_bit_per_j',[None]*len(action['links']))[j]})
                with np.load(folder/'arrays.npz',allow_pickle=False) as ar:
                    if 'probe_actual_mean_rate_bps' in ar:
                        y=ar['probe_actual_mean_rate_bps'];mu=ar['probe_predicted_mean_rate_bps'];q=ar['probe_calibrated_quantile_bps']
                        probe.append({'extra_delay_ms':d,'block':k,'n_links':len(y),
                            'rate_error_sq_sum':float(np.square(mu-y).sum()),'rate_true_sq_sum':float(np.square(y).sum()),
                            'mean_prediction_bias_bps':float((mu-y).mean()),'quantile_covered_links':int(np.sum(y>=q)),
                            'quantile_service_pass_links':int(np.sum(q>=1e6)),'actual_service_pass_links':int(np.sum(y>=1e6))})
                        for j,link in enumerate(r['actions']['probe']['links']):
                            probe_links.append({'extra_delay_ms':d,'block':k,'link_index':j,'tx':link[0],'rx':link[1],'mode':link[2],
                                'actual_rate_bps':float(y[j]),'predicted_mean_rate_bps':float(mu[j]),'calibrated_quantile_bps':float(q[j])})
            rows.append(row)
    common=[k for k in BLOCKS if all((d,k) in completed and completed[d,k]['success'] for d in DELAYS)]
    sums=[]
    for d in DELAYS:
        rr=[r for (dd,k),r in completed.items() if dd==d]
        ok=[r for r in rr if r['success']];paired=[completed[d,k] for k in common]
        pp=[r for r in probe if r['extra_delay_ms']==d]
        z={'extra_delay_ms':d,'planned_periods':len(BLOCKS),'complete_periods':len(rr),'success_periods':len(ok),
            'failed_periods':len(rr)-len(ok),'pending_periods':len(BLOCKS)-len(rr),'common_success_periods':len(common),
            'all_service_pass_periods':sum(bool(r.get('all_service_pass')) for r in rr),
            'reference_all_service_pass_periods':sum(bool(r.get('actual',{}).get('reference',{}).get('all_pairs_qos_met')) for r in rr),
            'predicted_budget_fail_periods':sum(not r['prediction']['main']['budget_pass'] for r in ok),
            'predicted_quantile_fail_periods':sum(not r['prediction']['main']['quantile_pass'] for r in ok),
            'actual_loss_over_10pct_periods':sum(r['actual_loss']>.1+1e-12 for r in ok)}
        for mode in ['rf','vlc']:
            numerator=sum(r[mode+'_error_sq_sum'] for r in rr);denominator=sum(r[mode+'_true_sq_sum'] for r in rr)
            z[mode+'_nmse']=numerator/denominator if denominator>0 else None
            z['n_'+mode+'_domain']=sum(r['n_'+mode+'_domain'] for r in rr)
        for prefix,subset in [('own',ok),('common',paired)]:
            for f in ['avg_ee_mbit_j','min_ee_mbit_j','ref_avg_ee_mbit_j','ref_min_ee_mbit_j']:
                z[prefix+'_'+f]=float(np.mean([r[f] for r in subset])) if subset else None
        z['probe_rate_nmse']=sum(r['rate_error_sq_sum'] for r in pp)/sum(r['rate_true_sq_sum'] for r in pp) if pp else None
        z['probe_quantile_coverage']=sum(r['quantile_covered_links'] for r in pp)/sum(r['n_links'] for r in pp) if pp else None
        z['probe_periods']=len(pp)
        z['probe_mean_prediction_bias_bps']=float(np.mean([r['mean_prediction_bias_bps'] for r in pp])) if pp else None
        for mode in ['RF','VLC']:
            qq=[r for r in probe_links if r['extra_delay_ms']==d and r['mode']==mode]
            den=sum(r['actual_rate_bps']**2 for r in qq)
            z['probe_'+mode.lower()+'_rate_nmse']=sum((r['predicted_mean_rate_bps']-r['actual_rate_bps'])**2 for r in qq)/den if den else None
            z['probe_'+mode.lower()+'_quantile_coverage']=sum(r['actual_rate_bps']>=r['calibrated_quantile_bps'] for r in qq)/len(qq) if qq else None
            z['probe_'+mode.lower()+'_link_periods']=len(qq)
        links=[r for r in action_links if r['extra_delay_ms']==d and r['method']=='main']
        z['executed_link_periods']=len(links)
        z['vlc_link_fraction']=sum(r['mode']=='VLC' for r in links)/len(links) if links else None
        for mode in ['RF','VLC']:
            mm=[r for r in links if r['mode']==mode]
            z[mode.lower()+'_mean_electrical_input_power_w']=float(np.mean([r['electrical_input_power_w'] for r in mm])) if mm else None
        z['action_comparison_periods']=0;z['mode_changed_periods']=0;z['ports_changed_periods']=0;z['complete_action_changed_periods']=0
        for r in ok:
            zero=completed.get((0,r['block']))
            if not zero or not zero['success']:continue
            aa=r['actions']['main'];bb=zero['actions']['main'];z['action_comparison_periods']+=1
            # Sort by directed link identity, so ordering alone is not a changed allocation.
            def amap(a):
                return {tuple(link):(float(p), a['ports']['tx'][link[0]][link[1]] if link[2]=='VLC' else None,
                    a['ports']['rx'][link[0]][link[1]] if link[2]=='VLC' else None) for link,p in zip(a['links'],a['powers_w'])}
            ma,mb=amap(aa),amap(bb);mode_changed=ma.keys()!=mb.keys()
            ports_changed=not mode_changed and any(ma[q][1:]!=mb[q][1:] for q in ma)
            power_changed=not mode_changed and any(not np.isclose(ma[q][0],mb[q][0],rtol=1e-9,atol=1e-10) for q in ma)
            z['mode_changed_periods']+=int(mode_changed);z['ports_changed_periods']+=int(ports_changed)
            z['complete_action_changed_periods']+=int(mode_changed or ports_changed or power_changed)
        sums.append(z)
    for name,values in [('delay_periods.csv',rows),('delay_summary.csv',sums),('fixed_probe_statistics.csv',probe),
        ('fixed_probe_link_rates.csv',probe_links),('executed_link_actions.csv',action_links)]:
        fields=list(dict.fromkeys(k for r in values for k in r))
        with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
    (HERE/'summary.json').write_text(json.dumps({'complete':len(completed)==800,'planned_blocks':BLOCKS,
        'common_success_blocks':common,'summary':sums,'nmse_definition':'sum squared errors / sum squared true gains; separate modes and shared geometric domains',
        'boundary_repairs':repair_manifest['repairs'],
        'effective_algorithm':{'group':'10B','delta':0.10,'refine_solver':'guarded','frozen_source_config_solver':'nm','explanation':'The source-run configuration is retained in prepare_manifest; the diagnostic explicitly invokes the frozen final guarded solver, as recorded by run_delay_diagnostic.py.'},
        'action_change_definition':'Compare only periods where both arms solve; mode is the unordered set of directed RF/VLC links; complete action includes selected VLC ports and electrical powers (rtol=1e-9,atol=1e-10 W). Ports_changed_periods counts changes only when mode is unchanged.',
        'extra_delay_applies':'RF and VLC observation arrival; VLC arrivals at decision instants unchanged across these four arms'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'completed':len(completed),'expected':800,'common_success':len(common),'summary':sums},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
