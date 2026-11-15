"""Render only completed, version-checked delay and budget diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parents[1] / '代码' / '信息与预算专项_20261114'
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 9, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 8, 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def panel(ax, tag, title):
    ax.text(-.15, 1.065, f'({tag})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom')
    ax.set_title(title, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E5E5', lw=.4)
    ax.tick_params(direction='out', length=3, pad=2.5)


def save(fig, name, source, details):
    fig.canvas.draw()
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(ROOT / f'{name}.{ext}', dpi=320)
    info = dict(details, width_mm=float(fig.get_size_inches()[0] * 25.4),
                height_mm=float(fig.get_size_inches()[1] * 25.4),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    plt.close(fig)
    (ROOT / (name + '_核查.json')).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')


def delay():
    local = ROOT / 'R1_plot_summary.csv'
    src = local if local.exists() else EXPERIMENT / 'delay' / 'delay_summary.csv'
    d = pd.read_csv(src).sort_values('extra_delay_ms')
    assert d.extra_delay_ms.tolist() == [0, 5, 10, 20]
    assert (d.complete_periods == 200).all() and (d.planned_periods == 200).all()
    assert d.common_success_periods.nunique() == 1
    cols = ['rf_nmse', 'vlc_nmse', 'common_ref_min_ee_mbit_j', 'common_min_ee_mbit_j']
    assert np.isfinite(d[cols].to_numpy()).all()
    if src.resolve() != local.resolve():
        shutil.copy2(src, local)
    fig, axes = plt.subplots(1, 2, figsize=(182 / 25.4, 78 / 25.4))
    fig.subplots_adjust(left=.105, right=.985, bottom=.20, top=.82, wspace=.34)
    panel(axes[0], 'a', 'Channel information')
    panel(axes[1], 'b', 'Resource allocation')
    axes[0].plot(d.extra_delay_ms, d.rf_nmse, color='#7B61A8', marker='o', ms=4,
                 lw=1.1, label='RF power gain')
    axes[0].plot(d.extra_delay_ms, d.vlc_nmse, color='#009E73', marker='s', ms=3.7,
                 ls='--', lw=1.1, label='VLC optical gain')
    axes[0].set_ylabel('Normalized mean squared error')
    axes[0].set_ylim(bottom=0)
    axes[1].plot(d.extra_delay_ms, d.common_ref_min_ee_mbit_j,
                 color='#737373', marker='s', ms=3.7, ls='--', lw=1.1,
                 label='Mean-EE reference')
    axes[1].plot(d.extra_delay_ms, d.common_min_ee_mbit_j,
                 color='#0072B2', marker='o', ms=4, lw=1.3, label='Proposed')
    axes[1].set_ylabel('Mean minimum-link EE (Mbit/J)')
    axes[1].set_ylim(0, d.common_min_ee_mbit_j.max() * 1.16)
    for ax in axes:
        ax.set_xlabel('Additional pilot delay (ms)')
        ax.set_xticks([0, 5, 10, 20])
        ax.set_xlim(-.7, 20.7)
        ax.legend(frameon=False, loc='best', handlelength=2.4)
    save(fig, 'R1_信道信息与资源决策', src,
         dict(n_planned=200, n_common=int(d.common_success_periods.iloc[0]),
              nmse_aggregation='sum squared error / sum squared truth',
              ee_aggregation='mean over four-delay common-success periods',
              selected_delays_ms=[0, 5, 10, 20]))


def budget():
    local = ROOT / 'R4_plot_summary.csv'
    src = local if local.exists() else EXPERIMENT / 'delta' / 'analysis' / 'R4_plot_summary.csv'
    d = pd.read_csv(src).sort_values(['scale', 'delta'])
    assert len(d) == 6 and sorted(d.scale.unique()) == [10, 50]
    assert all(np.allclose(g.delta, [.05, .1, .2]) for _, g in d.groupby('scale'))
    if src.resolve() != local.resolve():
        shutil.copy2(src, local)
    fig, axes = plt.subplots(1, 2, figsize=(182 / 25.4, 80 / 25.4))
    fig.subplots_adjust(left=.10, right=.985, bottom=.20, top=.82, wspace=.34)
    colors = ['#737373', '#0072B2', '#D55E00']
    markers = ['o', 's', '^']
    for ax, scale, tag in zip(axes, [10, 50], ['a', 'b']):
        panel(ax, tag, f'{scale} vehicles (B)')
        g = d[d.scale == scale]
        assert g.n_common.nunique() == 1
        ax.plot(g.loss_median, g.gain_median, color='#AAAAAA', lw=.7, zorder=1)
        for i, (_, r) in enumerate(g.iterrows()):
            xe = np.array([[r.loss_median - r.loss_p25], [r.loss_p75 - r.loss_median]])
            ye = np.array([[r.gain_median - r.gain_p25], [r.gain_p75 - r.gain_median]])
            assert (xe >= 0).all() and (ye >= 0).all()
            ax.errorbar(r.loss_median, r.gain_median, xerr=xe, yerr=ye,
                        fmt=markers[i], color=colors[i], ms=4.2,
                        elinewidth=.85, capsize=2.5, capthick=.7, zorder=3+i)
            ax.annotate(r'$\delta=' + f'{r.delta:.2f}' + '$',
                        (r.loss_median, r.gain_p75), xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=7.7, color=colors[i])
        ax.set_xlabel('Actual mean-EE loss (%)')
        ax.set_ylabel('Minimum-EE gain (Mbit/J)')
        ax.margins(x=.22, y=.18)
    save(fig, 'R4_预算与效率公平取舍', src,
         dict(center='marginal medians', whiskers='marginal P25-P75',
              delta_values=[.05, .1, .2], interval_type='empirical interquartile range'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure', choices=['R1', 'R4', 'both'], default='both')
    args = parser.parse_args()
    if args.figure in ('R1', 'both'):
        delay()
    if args.figure in ('R4', 'both'):
        budget()
