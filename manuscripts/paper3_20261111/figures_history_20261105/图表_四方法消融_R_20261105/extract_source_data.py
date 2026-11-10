"""Read archived outputs only. All visualization is performed by plot_ablation.R."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1] / '代码' / '分析_三条主线_20261101'
RUNS = [
 ('10_long_A', 'paper3_100s/runs/R1p_v5A_d010_100s', False),
 ('10_long_B', 'paper3_100s/runs/R2p_v5A_d010_100s_valseed', False),
 ('10_long_A', 'paper3_100s/runs/R3p1_norefine_d010_100s', True),
 ('10_long_B', 'paper3_100s/runs/R3p2_norefine_d010_100s_valseed', True),
 ('10_short_A', 'paper3_100s/runs/R7_same_topo_100blocks', False),
 ('50_short_A', 'paper3_n50/runs/R1_n50_d010_100blocks', False),
 ('50_short_B', 'paper3_n50/runs/R2_n50_d010_100blocks_valseed', False),
]
frames, manifests, reference_checks = [], [], []
for cohort, rel, no_refine in RUNS:
    path = ROOT / rel
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    cfg = summary['config']
    raw = pd.read_csv(path / 'results.csv', low_memory=False)
    raw = raw[raw.method.isin(['reference', 'maxmin', 'same_topo_maxmin'])].copy()
    links = raw[raw.row_type.eq('link')].copy()
    pc = links['mode'].map({m: cfg['physical'][m.lower()]['circuit_power_w'] for m in ['RF','VLC']})
    links['check_ee'] = links.actual_rate_bps / (links.power_w + pc)
    check = links.groupby(['method','block']).check_ee.agg(['mean','min'])
    blocks = raw.drop_duplicates(['method','block']).copy()
    saved = blocks[blocks.row_type.eq('link')].set_index(['method','block'])
    err = max(((check['mean']-saved.mean_pair_ee_bit_per_j).abs()/saved.mean_pair_ee_bit_per_j).max(),
              ((check['min']-saved.min_pair_ee_bit_per_j).abs()/saved.min_pair_ee_bit_per_j).max())
    assert err < 1e-12, (rel,err)
    blocks['cohort'] = cohort
    blocks['ok'] = blocks.row_type.eq('link')
    blocks['run'] = rel
    blocks['method_plot'] = blocks.method.map({'reference':'Reference','maxmin':'Proposed','same_topo_maxmin':'Fixed configuration'})
    if no_refine:
        reference_checks.append((cohort, blocks[blocks.method.eq('reference')].copy()))
        blocks = blocks[blocks.method.eq('maxmin')].copy()
        blocks['method_plot'] = 'No power refinement'
    frames.append(blocks)
    manifests.append(dict(run=rel,cohort=cohort,source_sha256=summary['source_sha256'],
                          results_sha256=hashlib.sha256((path/'results.csv').read_bytes()).hexdigest(),
                          channel_seed=cfg['channel_seed'],forecast_seed=cfg['forecast_seed'],
                          traffic_hashes=summary['traffic_hashes'],config=cfg,ee_max_relative_error=float(err)))

data = pd.concat(frames,ignore_index=True)
assert not data.duplicated(['cohort','method_plot','block']).any()
for cohort, ref in reference_checks:
    orig = data[(data.cohort==cohort)&(data.method_plot=='Reference')].set_index('block')
    ref = ref.set_index('block')
    assert orig.index.equals(ref.index)
    for col in ['row_type','action_sha256','source_sha256','mobility_sha256','channel_seed','forecast_seed']:
        assert orig[col].fillna('').equals(ref[col].fillna('')), (cohort,col)
    for col in ['mean_pair_ee_bit_per_j','min_pair_ee_bit_per_j']:
        assert np.allclose(orig[col],ref[col],equal_nan=True,rtol=1e-13), (cohort,col)

data['mean_ee'] = data.mean_pair_ee_bit_per_j / 1e6
data['min_ee'] = data.min_pair_ee_bit_per_j / 1e6
data['qos_ok'] = data.all_pairs_qos_met.astype(str).str.lower().eq('true')
data['prediction_ok'] = data.predicted_mean_pair_ee >= (1-data.delta)*data.eta_ref - 1e-6
ref = data[data.method_plot.eq('Reference')][['cohort','block','mean_ee','min_ee','ok']]
ref = ref.rename(columns={'mean_ee':'ref_mean','min_ee':'ref_min','ok':'ref_ok'})
data = data.merge(ref,on=['cohort','block'],validate='many_to_one')
data['paired_ok'] = data.ok & data.ref_ok
data['gain'] = data.min_ee - data.ref_min
data['gain_pct'] = 100*(data.min_ee/data.ref_min-1)
data['loss_pct'] = 100*(1-data.mean_ee/data.ref_mean)
cols = ['cohort','run','method_plot','block','start_time_s','duration_s','ok','paired_ok','mean_ee','min_ee',
        'ref_mean','ref_min','gain','gain_pct','loss_pct','qos_ok','prediction_ok','delta',
        'solve_time_s','fallback','chance_fallback','source_sha256','channel_seed','forecast_seed','action_sha256']
data[cols].to_csv(OUT/'source_blocks.csv',index=False,encoding='utf-8')
stats=[]
for (c,m),g in data.groupby(['cohort','method_plot'],sort=False):
    v=g[g.paired_ok]
    stats.append(dict(cohort=c,method=m,planned=len(g),paired_n=len(v),failed=int((~g.ok).sum()),
                      min_ee_median=v.min_ee.median(),mean_ee_median=v.mean_ee.median(),
                      gain_median=v.gain.median(),gain_pct_median=v.gain_pct.median(),loss_pct_median=v.loss_pct.median(),
                      improved=int((v.gain>1e-9).sum()),equal=int((v.gain.abs()<=1e-9).sum()),
                      worse=int((v.gain< -1e-9).sum()),actual_budget_breaches=int((v.loss_pct>100*v.delta+1e-8).sum()),
                      qos_failed=int((~v.qos_ok).sum())))
pd.DataFrame(stats).to_csv(OUT/'method_summary.csv',index=False,encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(stats).to_string(index=False))
print('\nAll saved EE values verified from individual link rates and mode-specific circuit power.')
print('Reference actions and metrics match exactly across each full / no-refine paired batch.')
print(data.groupby(['cohort','method_plot'])[['loss_pct','gain']].agg(['min','max']).to_string())
