"""Final Python rendering after the local R runtime failed at package unload.
Reads audited block data only; never invokes a simulator or optimizer.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, FixedLocator

OUT=Path(__file__).resolve().parent
R_TRIAL=OUT.parent/'图表_四方法消融_R_20261105'
for name in ['source_blocks.csv','method_summary.csv','provenance.json']:
    if not (OUT/name).exists(): shutil.copy2(R_TRIAL/name,OUT/name)
d=pd.read_csv(OUT/'source_blocks.csv')
d=d[d.paired_ok.astype(str).str.lower().eq('true')].copy()
qs=[]
for c in ['10_long_A','10_long_B']:
    ref=d[d.cohort.eq(c)&d.method_plot.eq('Reference')].sort_values(['ref_min','block']).copy()
    ref['quartile']=np.ceil(np.arange(1,len(ref)+1)*4/len(ref)).astype(int)
    z=d[d.cohort.eq(c)&~d.method_plot.eq('Reference')].merge(ref[['block','quartile']],on='block',validate='many_to_one')
    for (m,k),v in z.groupby(['method_plot','quartile']):
        qs.append(dict(cohort=c,method=m,quartile=k,n=len(v),med=v.gain.median(),lo=v.gain.quantile(.25),
                       hi=v.gain.quantile(.75),improved=(v.gain>1e-9).mean()*100,ref_lo=v.ref_min.min(),ref_hi=v.ref_min.max()))
q=pd.DataFrame(qs);q.to_csv(OUT/'quartile_response.csv',index=False)
ds=[];fs=[]
for c in ['10_short_A','50_short_A','50_short_B']:
    a=d[d.cohort.eq(c)&d.method_plot.eq('Proposed')].set_index('block')
    b=d[d.cohort.eq(c)&d.method_plot.eq('Fixed configuration')].set_index('block').loc[a.index]
    diff=a.min_ee-b.min_ee
    v=pd.DataFrame(dict(cohort=c,block=a.index,t=a.start_time_s.to_numpy(),ref_min=a.ref_min.to_numpy(),
                        full=a.min_ee.to_numpy(),fixed=b.min_ee.to_numpy(),difference=diff.to_numpy(),
                        mean_difference=(a.mean_ee-b.mean_ee).to_numpy()))
    ds.append(v);fs.append(dict(cohort=c,n=len(v),equal=(diff.abs()<=1e-9).sum(),higher=(diff>1e-9).sum(),
                               lower=(diff< -1e-9).sum(),mean_diff=diff.mean(),median_diff=diff.median()))
pd.concat(ds).to_csv(OUT/'fixed_configuration_block_differences.csv',index=False)
pd.DataFrame(fs).to_csv(OUT/'fixed_configuration_summary.csv',index=False)
s=pd.read_csv(OUT/'method_summary.csv')
pal={'Reference':'#88929C','No power refinement':'#BE873E','Fixed configuration':'#9A83B3','Proposed':'#267F98'}
styles={'Reference':(0,(2,2)),'No power refinement':(0,(6,3)),'Fixed configuration':(0,(5,3)),'Proposed':'-'}
short={'Reference':'Mean-EE reference','No power refinement':'No continuous refinement','Fixed configuration':'Fixed configuration','Proposed':'Proposed'}
plt.rcParams.update({'font.family':'Arial','font.size':8.2,'axes.titlesize':9.2,'axes.labelsize':8.2,
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#B8C3CB','axes.labelcolor':'#354551','text.color':'#253A47','xtick.color':'#596974',
 'ytick.color':'#596974','axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white','axes.facecolor':'white'})
def base(ax,label,title,subtitle=None):
    ax.set_axisbelow(True);ax.grid(True,color='#E8EDF0',linewidth=.5)
    ax.text(-.11,1.12,label,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
    ax.set_title(title,loc='left',pad=21 if subtitle else 10,fontweight='bold')
    if subtitle: ax.text(0,1.025,subtitle,transform=ax.transAxes,fontsize=7.2,color='#657580',va='bottom')
def title(fig,t,sub):
    fig.text(.06,.97,t,fontsize=14,fontweight='bold',va='top')
    fig.text(.06,.925,sub,fontsize=8,color='#697A85',va='top')
def legend(fig,methods,y=.885):
    handles=[Line2D([],[],color=pal[m],linestyle=styles[m],lw=1.8,label=short[m]) for m in methods]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,y),ncol=len(methods),frameon=False,
               handlelength=3.1,columnspacing=1.6)
def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=320)
    plt.close(fig)
def ecdf(ax,z,methods):
    for m in methods:
        v=np.sort(z.loc[z.method_plot.eq(m),'min_ee'].to_numpy())
        ax.step(np.r_[0,v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where='post',color=pal[m],
                linestyle=styles[m],lw=1.5,label=short[m])
    ax.set_ylim(0,1.015);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_yticks([0,.25,.5,.75,1])

# 1. Marginal distributions + conditional paired benefit.
fig,axs=plt.subplots(2,2,figsize=(7.6,6.2))
fig.subplots_adjust(left=.10,right=.975,bottom=.19,top=.785,hspace=.78,wspace=.27)
title(fig,'Continuous refinement: who benefits?', '10 vehicles | two channel realizations of the same traffic | prediction budget 10%')
legend(fig,['Reference','No power refinement','Proposed'])
for i,c in enumerate(['10_long_A','10_long_B']):
    z=d[d.cohort.eq(c)]; n=z[z.method_plot.eq('Reference')].shape[0]
    ax=axs[0,i];base(ax,'ab'[i],f'Channel realization {"AB"[i]}',f'{n} paired blocks; 99.9-s planned window')
    ecdf(ax,z,['Reference','No power refinement','Proposed'])
    ax.set_xlim(0,500);ax.set_xticks([0,100,200,300,400,500]);ax.set_xlabel('Actual weakest-link EE (Mbit/J)')
    if i==0:ax.set_ylabel('Cumulative share of blocks')
    ax=axs[1,i];base(ax,'cd'[i],'Benefit across baseline conditions','Median paired gain; whiskers show the middle 50%')
    ax.axhline(0,color='#8C9AA5',lw=.7)
    for m,offset,marker in [('No power refinement',-.07,'D'),('Proposed',.07,'o')]:
        v=q[q.cohort.eq(c)&q.method.eq(m)].sort_values('quartile')
        ax.errorbar(v.quartile+offset,v.med,yerr=np.vstack([v.med-v.lo,v.hi-v.med]),color=pal[m],
                    linestyle=styles[m],marker=marker,mfc='white',mec=pal[m],ms=5,lw=1.4,elinewidth=2,capsize=0)
    ax.set_xticks([1,2,3,4],['Q1\nWeakest','Q2','Q3','Q4\nStrongest']);ax.set_xlim(.75,4.25);ax.set_ylim(-10,100)
    ax.set_yticks([0,25,50,75,100]);ax.set_xlabel('Reference weakest-link EE quartile')
    if i==0:ax.set_ylabel('Weakest-link EE gain (Mbit/J)')
fig.text(.06,.045,'Quartiles use reference values within each realization. Intervals show block variability, not confidence intervals.\nOne failed block in realization B is excluded. All EE values are arithmetic per-link metrics.',fontsize=7,color='#647681')
save(fig,'fig01_distribution_and_weak_blocks')

# 2. Full-range paired trade-off. Asinh avoids hiding negative-loss tails.
fig,axs=plt.subplots(2,2,figsize=(7.6,6.5))
fig.subplots_adjust(left=.10,right=.975,bottom=.16,top=.80,hspace=.65,wspace=.28)
title(fig,'Block-level benefit and efficiency cost','Each point compares one action with the same-block mean-EE reference')
for i,c in enumerate(['10_long_A','10_long_B']):
    for j,m in enumerate(['No power refinement','Proposed']):
        z=d[d.cohort.eq(c)&d.method_plot.eq(m)];ax=axs[i,j]
        improved=(z.gain>1e-9).mean()*100
        good=((z.gain>1e-9)&(z.loss_pct<=10+1e-8)).mean()*100
        exceed=(z.loss_pct>10+1e-8).mean()*100
        base(ax,'abcd'[i*2+j],f'{"AB"[i]}  |  {short[m]}',f'Improved: {improved:.1f}%  |  gain with loss ≤10%: {good:.1f}%')
        ax.fill_between([-200,10],0,280,color='#F0F6F5',zorder=0)
        ax.axhline(0,color='#8C9AA5',lw=.7);ax.axvline(10,color='#A17B70',ls='--',lw=.9)
        ax.scatter(z.loss_pct,z.gain,c=pal[m],s=9,alpha=.32,edgecolors='none')
        bad=z[~z.qos_ok.astype(str).str.lower().eq('true')]
        ax.scatter(bad.loss_pct,bad.gain,marker='x',color='#30424F',s=25,lw=.9,zorder=4)
        ax.scatter([0],[0],marker='o',facecolor='white',edgecolor=pal['Reference'],s=35,zorder=5)
        ax.set_xscale('function',functions=(lambda x:np.arcsinh(x/6),lambda x:6*np.sinh(x)))
        ax.set_xlim(-200,40);ax.set_xticks([-150,-50,-10,0,5,10,25]);ax.set_xticklabels(['−150','−50','−10','0','5','10','25'])
        ax.set_ylim(-100,280);ax.set_yticks([-80,0,80,160,240]);ax.set_xlabel('Mean-EE loss (%) · asinh scale')
        if j==0:ax.set_ylabel('Weakest-link EE gain (Mbit/J)')
        ax.text(.025,.96,f'n = {len(z)}\nMedian gain: {z.gain.median():.1f} Mbit/J\nLoss >10%: {exceed:.1f}%',
                transform=ax.transAxes,va='top',fontsize=7.4,color='#4D626F',linespacing=1.5)
fig.text(.06,.045,'Pale region: positive weakest-link gain with actual mean-EE loss ≤10%. All points, including tails, are shown.\nThe 10% line evaluates execution outcomes; the algorithm constrains predicted loss. Crosses mark block QoS failures.\nAsinh compresses the x-axis tails; tick labels retain actual percentages. Negative loss means mean EE increased.',fontsize=7,color='#647681')
save(fig,'fig02_paired_tradeoff')

# 3. Same-configuration refinement in correctly matched short batches.
dd=pd.read_csv(OUT/'fixed_configuration_block_differences.csv');ss=pd.read_csv(OUT/'fixed_configuration_summary.csv')
fig,axs=plt.subplots(2,3,figsize=(7.6,5.9))
fig.subplots_adjust(left=.09,right=.975,bottom=.20,top=.785,hspace=.78,wspace=.32)
title(fig,'The additional effect of link reconfiguration','Matched 10-s experiments | fixed configuration keeps continuous power refinement')
legend(fig,['Reference','Fixed configuration','Proposed'])
for i,(c,head) in enumerate(zip(['10_short_A','50_short_A','50_short_B'],['10 vehicles | A','50 vehicles | A','50 vehicles | B'])):
    z=d[d.cohort.eq(c)];st=ss[ss.cohort.eq(c)].iloc[0];v=dd[dd.cohort.eq(c)]
    ax=axs[0,i];base(ax,'abc'[i],head,'100 paired blocks')
    ecdf(ax,z,['Reference','Fixed configuration','Proposed'])
    ax.set_xlim(0,340 if i==0 else 120);ax.set_xticks([0,100,200,300] if i==0 else [0,40,80,120])
    ax.set_xlabel('Weakest-link EE (Mbit/J)');ax.set_yticks([0,.5,1])
    if i==0:ax.set_ylabel('Cumulative share')
    ax=axs[1,i];base(ax,'def'[i],f'{int(st.equal)}/100 blocks equal',f'Higher {int(st.higher)} · lower {int(st.lower)} · mean {st.mean_diff:+.3f}')
    ax.axhline(0,color='#8C9AA5',lw=.7)
    ax.vlines(v.t,0,v.difference,color=pal['Fixed configuration'],lw=.7,alpha=.8)
    ax.scatter(v.t,v.difference,s=12,c=pal['Proposed'],edgecolors='white',lw=.3,zorder=3)
    ax.set_ylim(-22,22);ax.set_yticks([-20,-10,0,10,20]);ax.set_xlabel('Block start time (s)')
    if i==0:ax.set_ylabel('Full − fixed configuration\nweakest-link EE (Mbit/J)')
fig.text(.06,.045,'The lower panels show every block, including negative differences. Equality tolerance: 1e−9 Mbit/J.\nThe 10-vehicle batch differs in evaluation window and source snapshot from Figures 1–2 and is shown separately.\nA/B are channel realizations, not independent traffic scenes. No matched 50-vehicle no-refinement batch is available.',fontsize=7,color='#647681')
save(fig,'fig03_fixed_configuration')

# 4. Benefit coverage at increasing effect sizes, rather than counting tiny positive changes alone.
thresholds=[0,1,5,10,20,50];th_rows=[]
for c in ['10_long_A','10_long_B']:
    for m in ['Proposed','No power refinement']:
        z=d[d.cohort.eq(c)&d.method_plot.eq(m)]
        for t in thresholds:
            k=int((z.gain>1e-9).sum()) if t==0 else int((z.gain>=t).sum())
            th_rows.append(dict(cohort=c,method=m,threshold_Mbit_per_J=t,count=k,n=len(z),pct=k/len(z)*100))
th=pd.DataFrame(th_rows);th.to_csv(OUT/'gain_threshold_coverage.csv',index=False)
fig,axs=plt.subplots(2,1,figsize=(7.6,4.65))
fig.subplots_adjust(left=.205,right=.975,bottom=.17,top=.76,hspace=.65)
title(fig,'How much improvement reaches how many blocks?', 'Higher gain thresholds distinguish substantial improvements from tiny positive changes')
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('coverage',['#F2F5F7','#C2DBE0','#5B9BAC'])
for i,c in enumerate(['10_long_A','10_long_B']):
    ax=axs[i];base(ax,'ab'[i],f'Channel realization {"AB"[i]}')
    a=th[th.cohort.eq(c)]
    mat=np.array([a[a.method.eq(m)].pct.to_numpy() for m in ['Proposed','No power refinement']])
    ax.imshow(mat,cmap=cm,vmin=0,vmax=100,aspect='auto',interpolation='none')
    for row,m in enumerate(['Proposed','No power refinement']):
        v=a[a.method.eq(m)].reset_index(drop=True)
        for col,r in v.iterrows():
            ax.text(col,row,f'{r.pct:.1f}%\n{int(r["count"])} blocks',ha='center',va='center',fontsize=8,linespacing=1.45,color='#243B49')
    ax.set_xticks(range(6),['>0','≥1','≥5','≥10','≥20','≥50']);ax.set_yticks([0,1],['Proposed','No continuous\nrefinement'])
    ax.set_xlabel('Paired weakest-link EE gain threshold (Mbit/J)')
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True)
    ax.grid(False);ax.grid(which='minor',color='white',linewidth=4);ax.tick_params(which='both',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
fig.text(.06,.055,'Denominators: A = 999 paired blocks; B = 998 (one failed block excluded). Thresholds describe effect sizes, not constraints.\nEvery cell uses the same reference pairing. Reference gain is zero by definition; unfavorable outcomes remain in the denominator.',fontsize=7,color='#647681')
save(fig,'fig04_gain_coverage')

# Assertions for untruncated point sets and exact input scope.
assert d.loc[d.cohort.isin(['10_long_A','10_long_B']),'min_ee'].max()<500
assert d.loc[d.cohort.isin(['10_long_A','10_long_B']),'gain'].between(-100,280).all()
assert d.loc[d.cohort.isin(['10_long_A','10_long_B']),'loss_pct'].between(-200,40).all()
assert dd.difference.between(-22,22).all()
(OUT/'render_audit.json').write_text(json.dumps({'backend':'Python / matplotlib','reason':'R package-load exit access violation on this Windows host',
  'source':'R-trial extraction; independently verified EE and matched reference actions',
  'no_simulation_run':True,'all_scatter_points_inside_axes':True,'no_smoothing':True,'confidence_intervals':False,
  'quantitative_plots':4,'formats':['png','pdf','svg'],'matplotlib':matplotlib.__version__},indent=2),encoding='utf-8')
print(th.to_string(index=False))
print('Four final figures exported; all point ranges verified.')
