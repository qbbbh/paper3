"""Independent read-only cross-scale summaries from completed saved runs."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
RUNS={
 '10A':ROOT/'对比算法安全运行_20261106/runs/n10_A_100s_1000blocks',
 '10B':ROOT/'对比算法安全运行_20261106/runs/n10_B_100s_1000blocks',
 '50main':ROOT/'长跑保存修复_20261105/runs/R2_n50_d010_full100s_1000blocks',
 '50supp':ROOT/'50车补跑_20261108/runs/n50_B_missing_100s_1000blocks'}
METHODS=['reference','maxmin','same_topo_maxmin','no_refine','aboagye_matching_qt','zappone_mmee_fp_fixed']
BLOCKS=[]; LINKS=[]; provenance={}; original={}
for source,path in RUNS.items():
 d=json.loads((path/'summary.json').read_text(encoding='utf-8')); cfg=d['config']; original[source]=d
 b=pd.DataFrame(d['blocks']); l=pd.read_csv(path/'results.csv',encoding='utf-8-sig',low_memory=False)
 l=l[l.row_type=='link'].copy(); l['block']=l.block.astype(int); l['link_index']=l.link_index.astype(int)
 assert not b.duplicated(['block','method']).any()
 assert all(set(g.block)==set(range(1000)) for _,g in b.groupby('method'))
 l['electrical_w']=l.power_w+l['mode'].map({m:cfg['physical'][m.lower()]['circuit_power_w'] for m in ['RF','VLC']})
 l['ee']=l.actual_rate_bps/l.electrical_w/1e6; l['pred_ee']=l.predicted_rate_bps/l.electrical_w/1e6
 np.testing.assert_allclose(l.ee*1e6,l.pair_ee_bit_per_j,rtol=1e-9)
 mu=np.maximum(l.predicted_rate_bps,1e-9)
 l['calibrated_low']=mu*(np.maximum(l.predicted_p05_bps,1e-9)/mu)**cfg['qos_chance_inflation']
 l['low_bad']=l.calibrated_low<cfg['q_min_bps']*(1-1e-8)
 l['actual_bad']=l.actual_rate_bps<cfg['q_eval_bps']
 l['pred_actual_ratio']=l.actual_rate_bps/l.predicted_rate_bps
 l['emitted_w']=l.power_w*l['mode'].map({'RF':cfg['physical']['rf']['pa_efficiency'],'VLC':cfg['physical']['vlc']['eo_efficiency']})
 l['at_lower']=l.power_w<=l['mode'].map({m:cfg['physical'][m.lower()]['power_min_w'] for m in ['RF','VLC']})*(1+1e-6)
 l['at_upper']=l.power_w>=l['mode'].map({m:cfg['physical'][m.lower()]['power_max_w'] for m in ['RF','VLC']})*(1-1e-6)
 agg=l.groupby(['method','block']).agg(mean=('ee','mean'),minimum=('ee','min'),links=('ee','size'),pred_min=('pred_ee','min'),low_bad=('low_bad','sum'),actual_bad=('actual_bad','sum'),total_input_w=('power_w','sum'),total_electrical_w=('electrical_w','sum'),total_rate=('actual_rate_bps','sum'))
 ok=b[b.success].set_index(['method','block'])
 assert set(ok.index)==set(agg.index); assert agg.links.eq(int(b.n.iloc[0]/2)).all()
 np.testing.assert_allclose(agg['mean']*1e6,ok.loc[agg.index,'mean_pair_ee_bit_per_j'],rtol=1e-9)
 np.testing.assert_allclose(agg.minimum*1e6,ok.loc[agg.index,'min_pair_ee_bit_per_j'],rtol=1e-9)
 np.testing.assert_array_equal(agg.actual_bad.eq(0),ok.loc[agg.index,'all_pairs_qos_met'])
 b=b.join(agg,on=['method','block']); b['mean_ee']=b.mean_pair_ee_bit_per_j/1e6; b['min_ee']=b.min_pair_ee_bit_per_j/1e6
 b['source']=source; l['source']=source
 b['dataset']=l['dataset']='50B' if source.startswith('50') else source
 provenance[source]={'path':str(path),'config':cfg,'traffic_hashes':d['traffic_hashes'],'source_sha256':d['source_sha256'],
  'hashes':{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ['summary.json','results.csv']}}
 if source=='50supp':
  audit=json.loads((path/'paired_reference_audit.json').read_text(encoding='utf-8')); assert audit['success'] and audit['blocks']==1000
  b=b[b.method!='reference'];l=l[l.method!='reference']
 BLOCKS.append(b);LINKS.append(l)
b=pd.concat(BLOCKS,ignore_index=True);l=pd.concat(LINKS,ignore_index=True)
assert not b.duplicated(['dataset','method','block']).any()

def q(s,p):return float(s.quantile(p)) if len(s) else None
summary=[];mode_rows=[];pair_rows=[];diagnostics={};weak_rows=[]
for (dataset,m),g in b.groupby(['dataset','method'],sort=False):
 good=g[g.success];gl=l[(l.dataset==dataset)&(l.method==m)]
 wl=gl.loc[gl.groupby('block').ee.idxmin()]; wp=gl.loc[gl.groupby('block').pred_ee.idxmin()]
 row=dict(dataset=dataset,method=m,vehicles=int(g.n.iloc[0]),draws=original[g.source.iloc[0]]['config']['draws'],planned=len(g),success=len(good),failed=len(g)-len(good),
  mean_ee=good.mean_ee.mean(),min_ee=good.min_ee.mean(),min_ee_p10=q(good.min_ee,.1),min_ee_median=q(good.min_ee,.5),
  service_success_pct=100*good.actual_bad.eq(0).sum()/len(g),qos_bad_blocks=int(good.actual_bad.gt(0).sum()),qos_bad_links=int(gl.actual_bad.sum()),
  vlc_link_pct=100*gl['mode'].eq('VLC').mean(),vlc_rate_share_pct=100*gl.loc[gl['mode']=='VLC','actual_rate_bps'].sum()/gl.actual_rate_bps.sum(),
  mean_total_input_w=good.total_input_w.mean(),mean_total_electrical_w=good.total_electrical_w.mean(),
  mean_total_rate_Mbps=good.total_rate.mean()/1e6,mean_link_rate_Mbps=gl.actual_rate_bps.mean()/1e6,
  weakest_actual_rf_pct=100*wl['mode'].eq('RF').mean(),weakest_predicted_rf_pct=100*wp['mode'].eq('RF').mean(),
  final_low_bad_blocks=int(good.low_bad.gt(0).sum()),mean_fallback_flag_count=int(good.chance_fallback.sum()),
  time_mean_s=g.solve_time_s.mean(),time_median_s=q(g.solve_time_s,.5),time_p90_s=q(g.solve_time_s,.9),
  actual_loss_over_10pct=int(good.actual_mean_pair_loss_fraction_vs_actual_reference.gt(.1+1e-12).sum()))
 for mode,ml in gl.groupby('mode'):
  mode_rows.append(dict(dataset=dataset,method=m,mode=mode,link_observations=len(ml),input_power_mean_w=ml.power_w.mean(),input_power_median_w=q(ml.power_w,.5),input_power_p10_w=q(ml.power_w,.1),input_power_p90_w=q(ml.power_w,.9),input_power_p95_w=q(ml.power_w,.95),input_power_min_w=ml.power_w.min(),input_power_max_w=ml.power_w.max(),emitted_power_mean_w=ml.emitted_w.mean(),at_lower_pct=100*ml.at_lower.mean(),at_upper_pct=100*ml.at_upper.mean(),rate_mean_Mbps=ml.actual_rate_bps.mean()/1e6,ee_mean_MbitJ=ml.ee.mean(),actual_to_predicted_rate_median=q(ml.pred_actual_ratio,.5),actual_to_predicted_rate_p10=q(ml.pred_actual_ratio,.1),actual_to_predicted_rate_p90=q(ml.pred_actual_ratio,.9)))
  row[mode.lower()+'_input_power_mean_w']=ml.power_w.mean()
  row[mode.lower()+'_input_power_median_w']=q(ml.power_w,.5)
  row[mode.lower()+'_emitted_power_mean_w']=ml.emitted_w.mean()
 summary.append(row)
 for which,w in [('actual',wl),('predicted',wp)]:
  for mode,mw in w.groupby('mode'):
   weak_rows.append(dict(dataset=dataset,method=m,selection=which,mode=mode,count=len(mw),power_mean=mw.power_w.mean(),power_median=q(mw.power_w,.5),rate_mean_Mbps=mw.actual_rate_bps.mean()/1e6,actual_ee_mean=mw.ee.mean(),predicted_ee_mean=mw.pred_ee.mean()))

for dataset,db in b.groupby('dataset',sort=False):
 prop=db[db.method=='maxmin'].set_index('block');ref=db[db.method=='reference'].set_index('block')
 for other in METHODS:
  if other=='maxmin':continue
  c=db[db.method==other].set_index('block'); ids=prop.index[prop.success&c.success]
  x=prop.loc[ids];y=c.loc[ids];delta=x.min_ee-y.min_ee
  pair_rows.append(dict(dataset=dataset,other=other,n=len(ids),mean_ee_change_pct=100*(x.mean_ee.mean()/y.mean_ee.mean()-1),min_ee_change_pct=100*(x.min_ee.mean()/y.min_ee.mean()-1),min_win=int(delta.gt(1e-6).sum()),min_tie=int(delta.abs().le(1e-6).sum()),min_loss=int(delta.lt(-1e-6).sum()),min_gain_pct_median=q(100*(x.min_ee/y.min_ee-1),.5)))
 z=db[db.method=='zappone_mmee_fp_fixed'].set_index('block'); ids=prop.index[prop.success&z.success]
 budget=z.loc[ids,'predicted_mean_pair_ee']>=.9*ref.loc[ids,'eta_ref']*(1-1e-9)
 low=z.loc[ids,'low_bad'].eq(0);better=z.loc[ids,'pred_min']>prop.loc[ids,'pred_min']*(1+1e-9)
 awin=z.loc[ids,'min_ee']>prop.loc[ids,'min_ee']+1e-6
 predx=prop.loc[ids,'pred_min'];predz=z.loc[ids,'pred_min']
 diagnostics[dataset]=dict(common=len(ids),power_dimensions=int(prop.n.iloc[0]/2),nm_iteration_limit=original[prop.source.iloc[0]]['config']['maxmin_refine_maxiter'],nm_additional_restarts=original[prop.source.iloc[0]]['config']['maxmin_refine_restarts'],
  z_pass_budget=int(budget.sum()),z_pass_both=int((budget&low).sum()),z_feasible_predicted_better=int((budget&low&better).sum()),z_feasible_actual_better=int((budget&low&awin).sum()),
  z_predicted_better_total=int(better.sum()),z_actual_better_total=int(awin.sum()),z_predicted_better_and_actual_better=int((better&awin).sum()),
  proposed_predicted_min_mean=predx.mean(),z_predicted_min_mean=predz.mean(),proposed_actual_min_mean=prop.loc[ids,'min_ee'].mean(),z_actual_min_mean=z.loc[ids,'min_ee'].mean(),
  proposed_ratio_actual_to_predicted_min_median=q(prop.loc[ids,'min_ee']/predx,.5),z_ratio_actual_to_predicted_min_median=q(z.loc[ids,'min_ee']/predz,.5),
  better_feasible_blocks=list(map(int,ids[budget&low&better])))

keys=['block','tx','rx','mode','tx_port','rx_port']
ll=l[l.dataset=='50B']
main=ll[ll.method=='maxmin']; zz=ll[ll.method=='zappone_mmee_fp_fixed']; rr=ll[ll.method=='reference']
joined=main.merge(zz,on=keys,suffixes=('_main','_z'),validate='one_to_one')
assert len(joined)==len(main)==len(zz)==24975
matched=[]
for mode,g in joined.groupby('mode'):
 matched.append(dict(mode=mode,count=len(g),main_power_mean=g.power_w_main.mean(),z_power_mean=g.power_w_z.mean(),
  z_over_main_power_median=q(g.power_w_z/g.power_w_main,.5),z_power_up_pct=100*(g.power_w_z>g.power_w_main*(1+1e-6)).mean(),z_power_down_pct=100*(g.power_w_z<g.power_w_main*(1-1e-6)).mean(),
  main_rate_mean_Mbps=g.actual_rate_bps_main.mean()/1e6,z_rate_mean_Mbps=g.actual_rate_bps_z.mean()/1e6,
  main_ee_mean=g.ee_main.mean(),z_ee_mean=g.ee_z.mean(),z_ee_up_pct=100*(g.ee_z>g.ee_main+1e-6).mean()))
weakids=main.loc[main.groupby('block').ee.idxmin(),keys]
wj=weakids.merge(joined,on=keys,validate='one_to_one')
weak_matched=[]
for mode,g in wj.groupby('mode'):
 weak_matched.append(dict(mode=mode,count=len(g),main_power_mean=g.power_w_main.mean(),z_power_mean=g.power_w_z.mean(),main_rate_Mbps=g.actual_rate_bps_main.mean()/1e6,z_rate_Mbps=g.actual_rate_bps_z.mean()/1e6,main_ee=g.ee_main.mean(),z_ee_same_link=g.ee_z.mean(),z_ee_improved_count=int((g.ee_z>g.ee_main+1e-6).sum()),power_ratio_median=q(g.power_w_z/g.power_w_main,.5)))

result={'summary':summary,'by_mode':mode_rows,'weakest_links':weak_rows,'paired':pair_rows,'zappone_diagnostic':diagnostics,'matched50_by_mode':matched,'matched50_main_weakest_links':weak_matched,'provenance':provenance}
OUT.mkdir(exist_ok=True)
for name,rows in [('performance',summary),('mode_power',mode_rows),('weakest_links',weak_rows),('paired',pair_rows)]:
 pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
print(pd.DataFrame(summary)[['dataset','method','mean_ee','min_ee','service_success_pct','vlc_link_pct','rf_input_power_mean_w','vlc_input_power_mean_w','weakest_actual_rf_pct']].to_string(index=False))
print('Z_DIAGNOSTICS',json.dumps({k:{a:b for a,b in v.items() if a!='better_feasible_blocks'} for k,v in diagnostics.items()},ensure_ascii=False))
print('MATCHED50',json.dumps(matched,ensure_ascii=False))
print('MAIN_WEAK_LINKS50',json.dumps(weak_matched,ensure_ascii=False))
