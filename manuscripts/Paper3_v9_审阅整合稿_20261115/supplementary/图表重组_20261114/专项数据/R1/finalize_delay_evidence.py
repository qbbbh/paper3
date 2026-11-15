"""Validate the complete R1 diagnostic and index a portable, compact evidence bundle."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    s=read(HERE/'summary.json');m=read(HERE/'prepare_manifest.json')
    assert s['complete'], 'Run summarize_delay.py after all 800 units finish.'
    assert m['prepared_through_block']==999 and m['vlc_information_equal_all_blocks']
    assert sha(HERE/'run_delay_diagnostic.py')==m['script_sha256']
    formal=ROOT/'代码/公平功率求解改进_20261110/code'
    for name,h in m['formal_code_hashes'].items(): assert sha(formal/name)==h,name
    blocks=s['planned_blocks'];n_original=0;n_actual=0;max_power=0.0
    records={};failures=[]
    repairs={(r['extra_delay_ms'],r['block']):r for r in s.get('boundary_repairs',[])}
    for d in [0,5,10,20]:
        for k in blocks:
            p=HERE/'runs'/f'delay_{d:02d}ms'/f'block_{k:06d}'
            if (d,k) in repairs:
                repair=repairs[d,k];assert sha(p/'commit.json')==repair['original_commit_sha256']
                p=HERE/repair['replacement_directory'];assert sha(p/'commit.json')==repair['replacement_commit_sha256']
            cm=read(p/'commit.json')
            for name in ['record.json','arrays.npz']: assert sha(p/name)==cm[name]
            r=read(p/'record.json');assert r['block']==k and r['extra_delay_ms']==d
            records[d,k]=r
            assert r['input_sha256']==sha(HERE/'inputs'/f'block_{k:06d}.pkl.gz')
            checks=r.get('checks',{})
            if 'probe_actual' in checks:
                assert checks['probe_actual']['bitwise_equal'];n_actual+=1
            if d==0 and r['success']:
                assert checks['probe_prediction']['bitwise_equal'] and checks['old_main_prediction']['bitwise_equal']
                assert checks['reference_eta']
                for name in ['reference_action','main_action']:
                    assert checks[name]['same_links']
                    max_power=max(max_power,checks[name]['max_abs_power_w'])
                n_original+=1
            if not r['success']:failures.append({'extra_delay_ms':d,'block':k,'reason':r.get('reason')})
            for name in ['reference','main']:
                action=r.get('actions',{}).get(name);actual=r.get('actual',{}).get(name)
                if not action or not actual:continue
                modes=[link[2] for link in action['links']];powers=np.array(action['powers_w'])
                circuits=np.array([m['config']['physical'][mode.lower()]['circuit_power_w'] for mode in modes])
                ee=np.array(actual['mean_rate_bps'])/(powers+circuits)
                assert np.allclose(ee,actual['pair_ee_bit_per_j'],rtol=1e-12,atol=1e-6)
                assert np.isclose(ee.mean(),actual['mean_pair_ee_bit_per_j'],rtol=1e-12,atol=1e-6)
                assert np.isclose(ee.min(),actual['min_pair_ee_bit_per_j'],rtol=1e-12,atol=1e-6)
                for mode,power in zip(modes,powers):
                    c=m['config']['physical'][mode.lower()]
                    assert c['power_min_w']-1e-12<=power<=c['power_max_w']+1e-12
    for k in blocks:
        for d in [5,10,20]:
            for name in ['vlc_error_sq_sum','vlc_true_sq_sum','n_vlc_domain','rf_true_sq_sum','n_rf_domain']:
                assert records[d,k][name]==records[0,k][name],(d,k,name)
    with (HERE/'common_success_blocks.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['block']);w.writerows([[k] for k in s['common_success_blocks']])
    v={'completed_units':len(records),'zero_delay_successes_exactly_reproduced':n_original,
        'zero_delay_max_power_difference_w':max_power,'fixed_probe_actual_bitwise_checks':n_actual,
        'all_observed_vlc_information_equal':True,'same_nmse_truth_and_domain_across_delays':True,
        'independent_link_ee_recalculation':True,'formal_code_hashes_unchanged':True,
        'failed_units':failures,'source_driver_sha256':sha(HERE/'run_delay_diagnostic.py'),
        'boundary_repairs':list(repairs.values()),
        'effective_refine_solver':'guarded','source_config_refine_solver':m['config']['maxmin_refine_solver']}
    delta_csv=HERE.parent/'delta/runs/10B/full/periods.csv'
    if delta_csv.exists():
        with delta_csv.open(encoding='utf-8-sig',newline='') as f:
            dr={int(q['block']):q for q in csv.DictReader(f) if np.isclose(float(q['delta']),.10)}
        assert set(dr)==set(blocks)
        max_difference=0.0
        for k,q in dr.items():
            r=records[0,k];assert (q['success'].lower()=='true')==r['success']
            if r['success']:
                for key in ['avg_ee_mbit_j','min_ee_mbit_j','ref_avg_ee_mbit_j','ref_min_ee_mbit_j']:
                    difference=abs(float(q[key])-r[key]);max_difference=max(max_difference,difference)
                    assert difference<=1e-10,(k,key,difference)
        v['r4_10B_delta_010_matched_periods']=len(dr)
        v['r4_10B_delta_010_max_ee_difference_mbit_j']=max_difference
        v['r4_10B_delta_010_periods_csv_sha256']=sha(delta_csv)
    (HERE/'diagnostic_validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    files=['delay_periods.csv','delay_summary.csv','fixed_probe_statistics.csv','fixed_probe_link_rates.csv',
        'executed_link_actions.csv','common_success_blocks.csv','summary.json','prepare_manifest.json',
        'diagnostic_validation.json','README_时延专项.md','R1_完成核验与结果.md','run_delay_diagnostic.py','summarize_delay.py',
        'finalize_delay_evidence.py','launch_worker.py']
    if repairs:
        files+=['repairs_manifest.json','diagnose_delay_boundary.py','boundary_261/original/diagnostic.json',
            'boundary_261/repaired/diagnostic.json','boundary_261/repaired/isolated_final_handoff.patch']
    entries=[{'path':name,'bytes':(HERE/name).stat().st_size,'sha256':sha(HERE/name)} for name in files]
    full_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    bundle={'files':entries,'portable_bytes':sum(x['bytes'] for x in entries),'diagnostic_directory_bytes':full_bytes,
        'excluded_from_portable':['inputs/','runs/','worker logs and progress'],
        'index_scope':'All four delays use the same preselected 200 blocks; EE uses the four-arm common success set.',
        'power_scope':'All powers in executed_link_actions.csv are electrical input W.'}
    (HERE/'portable_data_manifest.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'validation':v,'portable_bytes':bundle['portable_bytes'],'diagnostic_directory_bytes':full_bytes},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
