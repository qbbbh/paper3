"""Render the six manuscript figure groups from the frozen supplied CSV tables.

No simulator or scientific solver is called. Run with:
  python redraw_figures.py --data ../data --output ../figures
Requires numpy, matplotlib, and a locally installed Times New Roman font.
"""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument('--data', type=Path, default=Path(__file__).resolve().parent.parent / 'data')
ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent / 'figures')
args = ap.parse_args()
D, F = args.data, args.output
F.mkdir(parents=True, exist_ok=True)
# Windows fonts may have been installed after matplotlib's first font-cache scan.
for path in (Path('C:/Windows/Fonts/times.ttf'), Path('C:/Windows/Fonts/timesbd.ttf'),
             Path('C:/Windows/Fonts/timesi.ttf')):
    if path.exists(): font_manager.fontManager.addfont(str(path))
font_path = font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'legend.fontsize': 8, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': .7, 'lines.linewidth': 1.35,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})
BLUE, RED, GREEN, GRAY = '#35648B', '#A94146', '#568565', '#6B6B6B'

def rows(name):
    with (D/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def vals(rr, key): return np.array([float(r[key]) for r in rr])
def setup(ax, title):
    ax.set_title(title, loc='left', pad=7)
    ax.grid(axis='y', color='#d6d6d6', lw=.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

def finish(fig, name):
    fig.canvas.draw()
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(F/f'{name}.{ext}', dpi=360)
    plt.close(fig)

# Figure 1: estimated gain, fixed-action prediction, new allocation, action response.
r = rows('R1_plot_summary.csv'); x = vals(r, 'extra_delay_ms')
fig, axs = plt.subplots(2, 2, figsize=(7.15, 5.15))
fig.subplots_adjust(left=.105, right=.985, bottom=.105, top=.94, hspace=.55, wspace=.33)
titles = ['(a) Channel-estimation error', '(b) Rate prediction at fixed allocation',
          '(c) Minimum EE after reallocation', '(d) Mode response and VLC use']
for ax, title in zip(axs.flat, titles):
    setup(ax, title); ax.set_xlabel('Additional pilot delay (ms)'); ax.set_xticks(x)
ax = axs[0,0]
for k, label, marker, color, ls in [('rf_nmse','RF power gain','o',BLUE,'-'),
                                 ('vlc_nmse','VLC optical gain','s',RED,'--')]:
    ax.plot(x, vals(r,k), marker=marker, ms=4, color=color, ls=ls, label=label)
ax.set(ylabel='Gain NMSE', ylim=(0,.56)); ax.legend(frameon=False, loc='upper left')
ax = axs[0,1]
for k,label,marker,color,ls in [('probe_rf_rate_nmse','RF','o',BLUE,'-'),
                              ('probe_vlc_rate_nmse','VLC','s',RED,'--')]:
    ax.plot(x,vals(r,k),marker=marker,ms=4,color=color,ls=ls,label=label)
ax.set(ylabel='Period-rate NMSE',ylim=(0,.024)); ax.legend(frameon=False,loc='upper left')
ax.ticklabel_format(axis='y',style='plain')
ax=axs[1,0]
for k,label,marker,color,ls in [('common_ref_min_ee_mbit_j','Mean-EE reference','s',GRAY,'--'),
                              ('common_min_ee_mbit_j','Proposed','o',RED,'-')]:
    ax.plot(x,vals(r,k),marker=marker,ms=4,color=color,ls=ls,label=label)
ax.set(ylabel='Mean minimum EE (Mbit/J)',ylim=(40,115)); ax.legend(frameon=False,loc='lower left')
ax=axs[1,1]
ax.plot(x,vals(r,'mode_changed_periods')/vals(r,'action_comparison_periods')*100,
        'o-',color=BLUE,ms=4,label='Periods with a mode change')
ax.plot(x,vals(r,'vlc_link_fraction')*100,'s--',color=RED,ms=4,label='VLC link share')
ax.set(ylabel='Fraction (%)',ylim=(-3,84)); ax.legend(frameon=False,loc='upper left',fontsize=7.5)
finish(fig,'F1')

# Figure 2: equal-weight A/B ECDFs on six-method common solve-success periods.
cdf=rows('pooled_ecdf.csv')
meths=[('reference','Mean-EE reference',GRAY,'--'),('main_new','Proposed',RED,'-'),
       ('zappone_mmee_fp_fixed','Zappone-adapted',BLUE,'-.'),
       ('aboagye_matching_qt','Aboagye-adapted',GREEN,':')]
fig,axs=plt.subplots(2,2,figsize=(7.15,5.2))
fig.subplots_adjust(left=.09,right=.985,bottom=.105,top=.875,hspace=.47,wspace=.29)
for ax,n,metric,title in [(axs[0,0],10,'min_ee_mbit_j','(a) Minimum EE: 10 vehicles'),
 (axs[0,1],50,'min_ee_mbit_j','(b) Minimum EE: 50 vehicles'),
 (axs[1,0],10,'avg_ee_mbit_j','(c) Mean EE: 10 vehicles'),
 (axs[1,1],50,'avg_ee_mbit_j','(d) Mean EE: 50 vehicles')]:
    setup(ax,title)
    for m,label,color,ls in meths:
        rr=sorted([z for z in cdf if int(z['scale'])==n and z['curve']==m and z['metric']==metric],key=lambda z:float(z['x']))
        # Add only the physical zero origin; retain every frozen observed x and weight.
        ax.step(np.r_[0,vals(rr,'x')],np.r_[0,vals(rr,'cdf')],where='post',ls=ls,color=color,label=label)
    ax.set(xlabel=('Minimum' if metric.startswith('min') else 'Mean')+' link EE (Mbit/J)',
           ylabel='Cumulative period fraction',ylim=(0,1.02),xlim=(0,None))
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.52,.995),columnspacing=1.1)
finish(fig,'F2')

# Figure 3: link-population statistics, not paired-link transitions.
mode=rows('link_mode_statistics.csv'); groups=['10A','10B','50A','50B']; x=np.arange(4)
fig,axs=plt.subplots(1,2,figsize=(7.15,2.85))
fig.subplots_adjust(left=.09,right=.985,bottom=.17,top=.84,wspace=.32)
ax=axs[0]; setup(ax,'(a) RF use and bottleneck ownership')
mr=[next(z for z in mode if z['group']==g and z['method']=='main_new' and z['scope']=='own_success' and z['mode']=='RF') for g in groups]
for offset,key,label,color,hatch in [(-.18,'fraction','RF link share',BLUE,'//'),(.18,'weakest_mode_fraction','RF bottleneck periods',RED,'')]:
    yy=vals(mr,key)*100
    ax.bar(x+offset,yy,width=.32,color=color,label=label,hatch=hatch,edgecolor='white',lw=.5)
    for xx,y in zip(x+offset,yy):ax.text(xx,y+1.5,f'{y:.1f}',ha='center',va='bottom',fontsize=7.5)
ax.set(xticks=x,xticklabels=groups,ylabel='Fraction (%)',ylim=(0,105))
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.03),fontsize=7.5)
ax=axs[1];setup(ax,'(b) Lower tail of RF link EE')
for method,label,marker,color in [('reference','Mean-EE reference','s',GRAY),('main_new','Proposed','o',RED)]:
    rr=[next(z for z in mode if z['group']==g and z['method']==method and z['scope']=='own_success' and z['mode']=='RF') for g in groups]
    yy=vals(rr,'link_ee_p05_mbit_j')
    ax.plot(x,yy,marker=marker,ms=5,ls='none',color=color,label=label)
    for xx,y in zip(x,yy):ax.annotate(f'{y:.2f}',(xx,y),xytext=(0,6 if method=='main_new' else -12),textcoords='offset points',ha='center',fontsize=7.3)
ax.set(xticks=x,xticklabels=groups,ylabel='5th-percentile RF EE (Mbit/J)',ylim=(0,59))
ax.legend(frameon=False,loc='upper left',fontsize=7.5)
finish(fig,'F3')

# Figure 4: three-method common periods, and per-realization reference quartiles.
fig,axs=plt.subplots(2,2,figsize=(7.15,5.25))
fig.subplots_adjust(left=.09,right=.985,bottom=.105,top=.94,hspace=.56,wspace=.29)
for ax,n,title in [(axs[0,0],10,'(a) Paired gains: 10 vehicles'),(axs[0,1],50,'(b) Paired gains: 50 vehicles')]:
    setup(ax,title)
    for m,label,color,ls in [('no_refine','Proposed - no fair power',BLUE,'-'),('same_topo_new','Proposed - fixed mode',RED,'--')]:
        rr=sorted([z for z in cdf if int(z['scale'])==n and z['curve']==m and z['metric']=='delta_min_ee_mbit_j'],key=lambda z:float(z['x']))
        ax.step(vals(rr,'x'),vals(rr,'cdf'),where='post',ls=ls,color=color,label=label)
    ax.axvline(0,ls=':',lw=.75,color=GRAY)
    ax.set(xlabel='Paired minimum-EE gain (Mbit/J)',ylabel='Cumulative period fraction',ylim=(0,1.02))
    ax.legend(frameon=False,loc='lower right',fontsize=7.25)
qr=rows('reference_weakness_quartiles.csv')
for ax,n,title in [(axs[1,0],10,'(c) Reference-state groups: 10 vehicles'),(axs[1,1],50,'(d) Reference-state groups: 50 vehicles')]:
    setup(ax,title)
    for field,label,marker,color,ls in [('proposed_min_ee_gain_mbit_j','Proposed - reference','o',GRAY,'-.'),
       ('fair_power_gain_mbit_j','Proposed - no fair power','s',BLUE,'-'),('mode_gain_mbit_j','Proposed - fixed mode','^',RED,'--')]:
        yy=[np.mean([float(z[field]) for z in qr if z['group'].startswith(str(n)) and int(z['quartile'])==q]) for q in range(1,5)]
        ax.plot(range(1,5),yy,marker=marker,ms=4,ls=ls,color=color,label=label)
    ax.set(xticks=range(1,5),xticklabels=['Q1','Q2','Q3','Q4'],
           xlabel='Initial minimum-EE quartile (low to high)',ylabel='Mean paired gain (Mbit/J)',
           ylim=(-3,68 if n==10 else 40))
    ax.legend(frameon=False,loc='upper left',fontsize=7.25)
finish(fig,'F4')

# Figure 5: separate marginal interquartile ranges, not confidence regions.
rr=rows('R4_plot_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(7.15,2.9))
fig.subplots_adjust(left=.095,right=.985,bottom=.19,top=.84,wspace=.31)
for ax,n,title in [(axs[0],10,'(a) Budget trade-off: 10 vehicles'),(axs[1],50,'(b) Budget trade-off: 50 vehicles')]:
    setup(ax,title); r=sorted([z for z in rr if int(z['scale'])==n],key=lambda z:float(z['delta']))
    for z,color,marker in zip(r,[GRAY,BLUE,RED],['s','o','^']):
        lm,gm=float(z['loss_median']),float(z['gain_median'])
        ax.errorbar([lm],[gm],xerr=[[lm-float(z['loss_p25'])],[float(z['loss_p75'])-lm]],
            yerr=[[gm-float(z['gain_p25'])],[float(z['gain_p75'])-gm]],fmt=marker,color=color,
            capsize=2.5,ms=5,lw=.9)
        ax.annotate('δ = '+f"{float(z['delta']):.2f}",(lm,float(z['gain_p75'])),xytext=(0,5),
                    textcoords='offset points',fontsize=8,ha='center',color=color)
    ax.plot(vals(r,'loss_median'),vals(r,'gain_median'),ls='--',lw=.7,color=GRAY,zorder=0)
    ax.set(xlabel='Actual mean-EE loss (%)',ylabel='Paired minimum-EE gain (Mbit/J)',
           xlim=(-1,21),ylim=(-3,89 if n==10 else 50))
finish(fig,'F5')

# Figure 6: aligned rows couple service and timing while separating timing provenance.
service=rows('service_runtime.csv'); selected=['main_new','zappone_mmee_fp_fixed','aboagye_matching_qt']
s=[next(z for z in service if z['group']==g and z['method']==m) for g in groups for m in selected]
y=np.array([g*3.65+j for g in range(4) for j in range(3)])
fig,axs=plt.subplots(1,2,figsize=(7.15,4.25),sharey=True)
fig.subplots_adjust(left=.135,right=.97,bottom=.13,top=.83,wspace=.22)
ax=axs[0]; setup(ax,'(a) Actual service shortfalls')
fail,short=vals(s,'no_action'),vals(s,'returned_service_failure')
ax.barh(y,fail,height=.62,color=GRAY,hatch='///',edgecolor='white',label='No allocation returned')
ax.barh(y,short,left=fail,height=.62,color=BLUE,label='Returned; service unmet')
for yy,total in zip(y,fail+short):ax.text(total+5,yy,str(int(total)),va='center',fontsize=8)
ax.set(xlabel='Unmet periods out of 1000',xlim=(0,420),yticks=y,
       yticklabels=[g+' / '+m for g in groups for m in ['P','Z','A']])
ax.invert_yaxis();ax.grid(axis='y',visible=False);ax.grid(axis='x',color='#d6d6d6',lw=.4)
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(-.02,1.10),fontsize=7.4)
ax=axs[1];setup(ax,'(b) Median decision time')
for yy,z in zip(y,s):
    composed=z['basis']=='sum_of_components'; sec=float(z['median_seconds'])
    ax.plot(sec,yy,marker='o' if z['method']=='main_new' else 's' if z['method'].startswith('zappone') else '^',
        ms=5,color=RED if z['method']=='main_new' else BLUE if z['method'].startswith('zappone') else GREEN,
        markerfacecolor='white' if composed else None,ls='none',markeredgewidth=1.2)
    ax.annotate(f'{sec:.2f}',(sec,yy),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
ax.set_xscale('log');ax.set(xlabel='Time (s), logarithmic scale',xlim=(.35,330),xticks=[.5,1,5,10,50,100])
ax.set_xticklabels(['0.5','1','5','10','50','100'])
ax.grid(axis='y',visible=False);ax.grid(axis='x',color='#d6d6d6',lw=.4)
ax.legend(handles=[Line2D([0],[0],marker='o',ls='none',color=GRAY,label='Measured end to end'),
                   Line2D([0],[0],marker='o',ls='none',color=GRAY,markerfacecolor='white',label='Sum of timed components')],
          frameon=False,loc='lower left',bbox_to_anchor=(-.02,1.10),fontsize=7.4)
for ax in axs:
    for boundary in [2.825,6.475,10.125]:ax.axhline(boundary,color='#bcbcbc',lw=.5)
finish(fig,'F6')

inputs=['R1_plot_summary.csv','pooled_ecdf.csv','link_mode_statistics.csv',
        'reference_weakness_quartiles.csv','R4_plot_summary.csv','service_runtime.csv']
audit={'font':'Times New Roman','font_file':font_path,'new_simulations':False,
       'figure_groups':6,'panel_counts':[4,4,2,4,2,2],
       'data_sha256':{name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in inputs},
       'format':['PDF','SVG','PNG'],'png_dpi':360,'source_data_unchanged':True,
       'figure_6_timing_provenance':'solid measured; open summed per-period components',
       'figure_4_strata':'per-realization initial-minimum-EE quartiles; A/B conditional means equally weighted'}
(F/'figure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('Six figure groups exported using',font_path)
