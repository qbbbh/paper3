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
text_bounds=[]

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
    renderer=fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():continue
        skip=False
        for ax in fig.axes:
            if artist in ax.get_xticklabels() and not min(ax.get_xlim())<=artist.get_position()[0]<=max(ax.get_xlim()):skip=True
            if artist in ax.get_yticklabels() and not min(ax.get_ylim())<=artist.get_position()[1]<=max(ax.get_ylim()):skip=True
        if skip:continue
        bbox=artist.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width+1 or bbox.y1 > fig.bbox.height+1:
            text_bounds.append({'figure':name,'text':artist.get_text(),'bounds':list(bbox.extents)})
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
for k,label,marker,color,ls in [('common_ref_min_ee_mbit_j','Mean-EE','s',GRAY,'--'),
                              ('common_min_ee_mbit_j','PF-EERA','o',RED,'-')]:
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
meths=[('reference','Mean-EE',GRAY,'--'),('main_new','PF-EERA',RED,'-'),
       ('zappone_mmee_fp_fixed','MMEE-FP',BLUE,'-.'),
       ('aboagye_matching_qt','Matching-QT',GREEN,':')]
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

# Figure 3: the link share and lowest-EE-period share have different physical units.
mode=rows('link_mode_statistics.csv'); groups=['10A','10B','50A','50B']; x=np.array([0,1,2.5,3.5])
fig,axs=plt.subplots(1,2,figsize=(7.15,3.05))
fig.subplots_adjust(left=.095,right=.985,bottom=.24,top=.79,wspace=.35)
ax=axs[0]; setup(ax,'(a) RF use and minimum-EE occurrence')
mr=[next(z for z in mode if z['group']==g and z['method']=='main_new' and z['scope']=='own_success' and z['mode']=='RF') for g in groups]
for offset,key,label,color,hatch in [(-.18,'fraction','Vehicle pairs using RF',BLUE,'//'),(.18,'weakest_mode_fraction','Periods with minimum EE on RF',RED,'')]:
    yy=vals(mr,key)*100
    ax.bar(x+offset,yy,width=.32,color=color,label=label,hatch=hatch,edgecolor='white',lw=.5)
    for xx,y in zip(x+offset,yy):ax.text(xx,y+1.4,f'{y:.1f}',ha='center',va='bottom',fontsize=7.4)
ax.set(ylabel='Fraction (%)',ylim=(0,90))
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(-.06,1.08),fontsize=7.7)
ax=axs[1];setup(ax,'(b) Low-EE RF links')
for method,label,marker,color in [('reference','Mean-EE','s',GRAY),('main_new','PF-EERA','o',RED)]:
    rr=[next(z for z in mode if z['group']==g and z['method']==method and z['scope']=='own_success' and z['mode']=='RF') for g in groups]
    yy=vals(rr,'link_ee_p05_mbit_j')
    ax.plot(x,yy,marker=marker,ms=5,ls='none',color=color,label=label)
    for xx,y in zip(x,yy):ax.annotate(f'{y:.2f}',(xx,y),xytext=(0,7 if method=='main_new' else -13),textcoords='offset points',ha='center',fontsize=7.4)
ax.set(ylabel='RF EE at the 5% quantile (Mbit/J)',ylim=(5,47))
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.08),ncol=2)
for ax in axs:
    ax.set_xticks(x,['Realization\n1','Realization\n2','Realization\n1','Realization\n2'],fontsize=7.4)
    ax.set_xlim(-.55,4.05)
    ax.text(.5,-.25,'10 vehicles',transform=ax.get_xaxis_transform(),ha='center')
    ax.text(3,-.25,'50 vehicles',transform=ax.get_xaxis_transform(),ha='center')
finish(fig,'F3')

# Figure 4: show the mass at zero explicitly, rather than a near-vertical ECDF.
mechanism=rows('R3_mechanism_statistics.csv')
fig,axs=plt.subplots(2,2,figsize=(7.15,5.75))
fig.subplots_adjust(left=.15,right=.985,bottom=.13,top=.88,hspace=.62,wspace=.48)
status=[('positive_fraction','Higher',BLUE),('near_zero_fraction','Similar','#CACACA'),('negative_fraction','Lower',RED)]
for ax,n,title in [(axs[0,0],10,'(a) PF-EERA versus ablations: 10 vehicles'),(axs[0,1],50,'(b) PF-EERA versus ablations: 50 vehicles')]:
    setup(ax,title); yy=np.array([0,1.1]); left=np.zeros(2)
    rr=[next(z for z in mechanism if int(z['scale'])==n and z['group']=='pooled' and z['control']==m) for m in ['no_refine','same_topo_new']]
    for key,label,color in status:
        v=vals(rr,key)*100
        ax.barh(yy,v,left=left,height=.5,color=color,edgecolor='white',lw=.6,label=label)
        for pos,val,lft in zip(yy,v,left):
            if val>=10:
                ax.text(lft+val/2,pos,f'{val:.1f}%',ha='center',va='center',color='white' if key!='near_zero_fraction' else '#222222',fontsize=8)
            elif val>=7:
                ax.annotate(f'{val:.1f}%',(lft+val/2,pos-.25),xytext=(-8,12),textcoords='offset points',ha='center',va='bottom',color=color,fontsize=8,arrowprops={'arrowstyle':'-','color':color,'lw':.7})
        left+=v
    ax.set(yticks=yy,yticklabels=['No-Fair-Power','Fixed-Mode'],xlabel='Periods (%)',xlim=(0,100),ylim=(-.65,1.6))
    ax.tick_params(axis='y',labelsize=7.5); ax.invert_yaxis(); ax.grid(axis='y',visible=False)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.55,.99),title='PF-EERA minimum EE relative to each ablation',title_fontsize=8.5)
qr=rows('reference_weakness_quartiles.csv')
for ax,n,title in [(axs[1,0],10,'(c) Initial-state dependence: 10 vehicles'),(axs[1,1],50,'(d) Initial-state dependence: 50 vehicles')]:
    setup(ax,title)
    for field,label,marker,color,ls in [('proposed_min_ee_gain_mbit_j','vs Mean-EE','o',GRAY,'-.'),('fair_power_gain_mbit_j','vs No-Fair-Power','s',BLUE,'-'),('mode_gain_mbit_j','vs Fixed-Mode','^',RED,'--')]:
        yy=[np.mean([float(z[field]) for z in qr if z['group'].startswith(str(n)) and int(z['quartile'])==q]) for q in range(1,5)]
        ax.plot(range(1,5),yy,marker=marker,ms=4,ls=ls,color=color,label=label)
    ax.set(xticks=range(1,5),xticklabels=['Lowest','Lower\nmiddle','Upper\nmiddle','Highest'],xlabel='Initial minimum-EE group',ylabel='PF-EERA gain (Mbit/J)',ylim=(-3,68 if n==10 else 40))
    ax.legend(frameon=False,loc='upper left',fontsize=7.4)
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

# Figure 6: use actual method names, separate the two vehicle scales.
service=rows('service_runtime.csv'); selected=['main_new','zappone_mmee_fp_fixed','aboagye_matching_qt']
labels=['PF-EERA','MMEE-FP','Matching-QT']; xx=np.arange(3)
fig,axs=plt.subplots(2,2,figsize=(7.15,5.30))
fig.subplots_adjust(left=.105,right=.985,bottom=.11,top=.875,hspace=.53,wspace=.32)
for col,n in enumerate([10,50]):
    a,b=axs[0,col],axs[1,col]
    setup(a,f'({"ab"[col]}) Actual service: {n} vehicles')
    setup(b,f'({"cd"[col]}) Decision time: {n} vehicles')
    for real,code,offset,marker,color in [(1,'A',-.10,'o',BLUE),(2,'B',.10,'D',RED)]:
        rr=[next(z for z in service if z['group']==str(n)+code and z['method']==m) for m in selected]
        rate=vals(rr,'service_pass')/vals(rr,'planned')*100
        a.plot(xx+offset,rate,marker=marker,ms=5,ls='none',color=color,label=f'Realization {real}')
        for i,v in enumerate(rate):a.annotate(f'{v:.1f}',(xx[i]+offset,v),xytext=(-10 if real==1 else 10,7 if real==1 else -14),textcoords='offset points',ha='center',fontsize=7.5,color=color)
        for i,z in enumerate(rr):
            sec=float(z['median_seconds']);composed=z['basis']=='sum_of_components'
            b.plot(xx[i]+offset,sec,marker=marker,ms=5,ls='none',color=color,markerfacecolor='white' if composed else color,markeredgewidth=1.2)
            b.annotate(f'{sec:.2f}',(xx[i]+offset,sec),xytext=(-7 if real==1 else 7,7 if real==1 or sec<1 else -14),textcoords='offset points',ha='center',fontsize=7.5,color=color)
    for ax in [a,b]:ax.set(xticks=xx,xticklabels=labels,xlim=(-.4,2.4))
    a.set(ylabel='All-link service satisfaction (%)',ylim=(58,106))
    b.set_yscale('log'); b.set(ylabel='Median decision time (s)',ylim=(.25,220),yticks=[.5,1,5,10,50,100]);b.set_yticklabels(['0.5','1','5','10','50','100'])
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.52,.995))
finish(fig,'F6')

inputs=['R1_plot_summary.csv','pooled_ecdf.csv','link_mode_statistics.csv',
        'reference_weakness_quartiles.csv','R4_plot_summary.csv','service_runtime.csv','R3_mechanism_statistics.csv']
audit={'font':'Times New Roman','font_file':font_path,'new_simulations':False,
       'figure_groups':6,'panel_counts':[4,4,4,4,2,4],
       'data_sha256':{name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in inputs},
       'format':['PDF','SVG','PNG'],'png_dpi':360,'source_data_unchanged':True,
       'figure_6_timing_provenance':'solid measured; open summed per-period components',
       'figure_4_similar_tolerance_mbit_j':1e-6,
       'text_outside_canvas':text_bounds,
       'figure_4_strata':'per-realization initial-minimum-EE quartiles; A/B conditional means equally weighted'}
from draw_review_panels import draw as draw_panels
draw_panels(D,F/'panels')
audit['independent_panels']={'F2': 'panels/F2a–F2d', 'F3': 'panels/F3a–F3d', 'formats': ['PNG', 'PDF'], 'composite_F2_F3': '上轮合成图留作历史对照，现稿使用独立子图。'}
(F/'figure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('Six figure groups exported using',font_path)
