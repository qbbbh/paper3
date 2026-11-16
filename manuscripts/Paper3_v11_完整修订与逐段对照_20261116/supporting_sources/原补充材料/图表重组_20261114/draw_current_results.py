"""Render current R2/R3 from frozen, paired period statistics only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(ROOT / 'pooled_ecdf.csv')
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 8, 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})
STYLE = {
    'reference': ('Mean-EE reference', '#737373', (0, (4, 2))),
    'main_new': ('Proposed', '#0072B2', '-'),
    'zappone_mmee_fp_fixed': ('Zappone-adapted', '#D55E00', (0, (6, 2, 1.5, 2))),
    'aboagye_matching_qt': ('Aboagye-adapted', '#009E73', ':'),
    'no_refine': ('Fair-power refinement', '#0072B2', '-'),
    'same_topo_new': ('Fair-mode search', '#D55E00', (0, (4, 2))),
}


def setup(ax, tag, title):
    ax.text(-.14, 1.055, f'({tag})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom')
    ax.set_title(title, pad=7)
    ax.set_ylim(0, 1.015)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.00'])
    ax.grid(axis='y', color='#E5E5E5', linewidth=.4)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3, pad=2.5)


def ecdf(ax, scale, curve, metric, xmin, xmax):
    d = DATA[(DATA.scale == scale) & (DATA.curve == curve) & (DATA.metric == metric)]
    d = d.sort_values('x', kind='stable')
    assert len(d) and d.x.notna().all() and np.all(np.diff(d.cdf) >= -1e-12)
    assert abs(d.cdf.iloc[-1] - 1) < 1e-10
    assert xmin <= d.x.min() and xmax >= d.x.max()
    _, color, ls = STYLE[curve]
    ax.step(np.r_[xmin, d.x, xmax], np.r_[0., d.cdf, 1.], where='post',
            color=color, ls=ls, lw=1.45 if curve == 'main_new' else 1.1,
            zorder=4 if curve == 'main_new' else 3)


def legend(fig, curves, y):
    hs = [Line2D([], [], color=STYLE[m][1], ls=STYLE[m][2], lw=1.4,
                 label=STYLE[m][0]) for m in curves]
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(.5, y),
               ncol=2, frameon=False, handlelength=3.1,
               columnspacing=2.5, labelspacing=.5)


def export(fig, name):
    fig.canvas.draw()
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(ROOT / (name + '.' + ext), dpi=320)
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {'name': name, 'width_mm': width * 25.4, 'height_mm': height * 25.4}


fig, axes = plt.subplots(2, 2, figsize=(182 / 25.4, 133 / 25.4))
fig.subplots_adjust(left=.092, right=.985, bottom=.095, top=.825,
                    wspace=.30, hspace=.47)
curves = ['reference', 'main_new', 'zappone_mmee_fp_fixed', 'aboagye_matching_qt']
legend(fig, curves, .987)
for row, metric in enumerate(['min_ee_mbit_j', 'avg_ee_mbit_j']):
    for col, scale in enumerate([10, 50]):
        ax = axes[row, col]
        setup(ax, 'abcd'[row * 2 + col], f'{scale} vehicles')
        xmax = {(0, 10): 500, (0, 50): 180, (1, 10): 900, (1, 50): 450}[row, scale]
        for method in curves:
            ecdf(ax, scale, method, metric, 0, xmax)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(('Minimum link EE' if row == 0 else 'Mean link EE') + ' (Mbit/J)')
        ax.set_ylabel('Cumulative period fraction')
        ax.xaxis.set_major_locator(MultipleLocator({500: 100, 180: 30, 900: 150, 450: 75}[xmax]))
sizes = [export(fig, 'R2_主要方法能效分布')]

fig, axes = plt.subplots(1, 2, figsize=(182 / 25.4, 74 / 25.4))
fig.subplots_adjust(left=.092, right=.985, bottom=.19, top=.77, wspace=.30)
legend(fig, ['no_refine', 'same_topo_new'], .99)
for col, scale in enumerate([10, 50]):
    ax = axes[col]
    setup(ax, 'ab'[col], f'{scale} vehicles')
    xmin, xmax = (-110, 240) if scale == 10 else (-65, 135)
    ax.axvline(0, color='#A5A5A5', lw=.65, zorder=1)
    for control in ['no_refine', 'same_topo_new']:
        ecdf(ax, scale, control, 'delta_min_ee_mbit_j', xmin, xmax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'Paired minimum-EE gain (Mbit/J)')
    ax.set_ylabel('Cumulative period fraction')
    ax.xaxis.set_major_locator(MultipleLocator(50))
sizes.append(export(fig, 'R3_连续功率与模式搜索'))

report = {'backend': 'Python/matplotlib', 'figures': sizes,
          'source_sha256': hashlib.sha256((ROOT / 'pooled_ecdf.csv').read_bytes()).hexdigest(),
          'aggregation': '0.5 F_A + 0.5 F_B; weighted empirical steps',
          'all_support_retained': True, 'zero_and_negative_gains_retained': True,
          'exports': ['PDF vector', 'SVG editable text', 'PNG 320 dpi']}
(ROOT / 'figure_export_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
