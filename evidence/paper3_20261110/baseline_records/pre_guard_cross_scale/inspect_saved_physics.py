"""Compare saved physical components of identical 50-car links; no new actions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
paths={'main':ROOT/'长跑保存修复_20261105/runs/R2_n50_d010_full100s_1000blocks',
       'z':ROOT/'50车补跑_20261108/runs/n50_B_missing_100s_1000blocks'}
methods={'main':'maxmin','z':'zappone_mmee_fp_fixed'}
tables={}
for name,p in paths.items():
 d=pd.read_csv(p/'results.csv',encoding='utf-8-sig',low_memory=False)
 d=d[(d.row_type=='link')&(d.method==methods[name])].copy()
 d['block']=d.block.astype(int);d['link_index']=d.link_index.astype(int)
 tables[name]=d
data=[]; all_ratios=[]
with np.load(paths['main']/'fine_samples.npz') as a,np.load(paths['z']/'fine_samples.npz') as z:
 for k,g in tables['main'].groupby('block'):
  j=tables['z'][tables['z'].block==k].sort_values('link_index');g=g.sort_values('link_index')
  np.testing.assert_array_equal(g[['tx','rx','mode','tx_port','rx_port']].to_numpy(),j[['tx','rx','mode','tx_port','rx_port']].to_numpy())
  idx=int(np.argmin(g.pair_ee_bit_per_j.to_numpy())); mode=g.iloc[idx]['mode']
  row=dict(block=int(k),weak_mode=mode,link_index=idx,main_power=float(g.iloc[idx].power_w),z_power=float(j.iloc[idx].power_w))
  for name,archive in [('main',a),('z',z)]:
   key=f'{methods[name]}_b{k}'
   arrays={f:np.asarray(archive[key+'_'+f],float) for f in ['signal','interference','noise','sinr']}
   for f,ar in arrays.items():row[name+'_'+f]=float(ar[:,idx].mean())
   ratios=np.mean(arrays['interference'],axis=0)/np.mean(arrays['noise'],axis=0)
   for position,link in enumerate(g.itertuples()):
    all_ratios.append(dict(method=name,mode=link.mode,block=int(k),in_ratio=float(ratios[position])))
  for f in ['power','signal','interference','noise','sinr']:
   row[f+'_ratio_z_over_main']=row['z_'+f]/row['main_'+f] if row['main_'+f]>0 else None
  data.append(row)
d=pd.DataFrame(data); ar=pd.DataFrame(all_ratios)
stats=[]
for mode,g in d.groupby('weak_mode'):
 r=dict(mode=mode,count=len(g))
 for field in ['power','signal','interference','noise','sinr']:
  s=g[field+'_ratio_z_over_main'].dropna()
  r[field+'_ratio_median']=s.median();r[field+'_ratio_p10']=s.quantile(.1);r[field+'_ratio_p90']=s.quantile(.9)
 r['interference_decreased_count']=int(g.z_interference.lt(g.main_interference*(1-1e-6)).sum())
 r['sinr_increased_count']=int(g.z_sinr.gt(g.main_sinr*(1+1e-6)).sum())
 stats.append(r)
by_mode=[]
for (m,mode),g in ar.groupby(['method','mode']):
 by_mode.append(dict(method=m,mode=mode,n=len(g),in_ratio_median=g.in_ratio.median(),in_ratio_p10=g.in_ratio.quantile(.1),in_ratio_p90=g.in_ratio.quantile(.9),i_greater_n_pct=100*g.in_ratio.gt(1).mean()))
d.to_csv(OUT/'same_weak_link_physics_50.csv',index=False,encoding='utf-8-sig')
out={'selected_main_weakest_links':stats,'all_link_interference_to_noise':by_mode}
(OUT/'saved_physics_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
