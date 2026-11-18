"""Redraw the eight individual panels used in the Word review manuscript.

Run against the unchanged V12 CSV files. Each panel carries its own legend so
that it can be moved or copied from Word without losing the method names.
"""
from pathlib import Path
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FONT = Path("C:/Windows/Fonts")
METHODS = [
    ("reference", "Mean-EE", "#595959", "--"),
    ("main_new", "PF-EERA", "#A03C3F", "-"),
    ("zappone_mmee_fp_fixed", "MMEE-FP", "#2C6089", "-."),
    ("aboagye_matching_qt", "Matching-QT", "#3A704D", ":"),
]
BLUE, RED, GREY = "#2C6089", "#A03C3F", "#595959"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def axes(bottom=.22):
    fig, ax = plt.subplots(figsize=(3.06, 1.92))
    fig.subplots_adjust(left=.19, right=.975, bottom=bottom, top=.77)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.4, pad=2, width=.6)
    ax.grid(axis="y", color="#dadada", linewidth=.4)
    ax.set_axisbelow(True)
    return fig, ax


def legend(ax, columns):
    ax.legend(loc="lower center", bbox_to_anchor=(.50, 1.015), ncol=columns,
              fontsize=7.3, frameon=False, handlelength=2.0, columnspacing=.9,
              handletextpad=.4, borderaxespad=0, labelspacing=.18)


def save(fig, directory, key):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        if any((artist in a.get_xticklabels() and not min(a.get_xlim()) <= artist.get_position()[0] <= max(a.get_xlim()))
               or (artist in a.get_yticklabels() and not min(a.get_ylim()) <= artist.get_position()[1] <= max(a.get_ylim())) for a in fig.axes):
            continue
        x0, y0, x1, y1 = artist.get_window_extent(renderer).extents
        if x0 < -1 or y0 < -1 or x1 > fig.bbox.width+1 or y1 > fig.bbox.height+1:
            raise ValueError((key, artist.get_text(), (x0,y0,x1,y1)))
    fig.savefig(directory / (key + ".png"), dpi=250, facecolor="white")
    fig.savefig(directory / (key + ".pdf"), facecolor="white")
    plt.close(fig)


def draw(data, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for font in ("times.ttf", "timesbd.ttf", "timesi.ttf"):
        if (FONT / font).exists():
            font_manager.fontManager.addfont(str(FONT / font))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family":"Times New Roman", "font.size":8.4,
                         "axes.labelsize":8.3, "xtick.labelsize":7.8, "ytick.labelsize":7.8,
                         "axes.linewidth":.65, "lines.linewidth":1.2,
                         "pdf.fonttype":42, "axes.unicode_minus":False})
    cdf = read_rows(data / "pooled_ecdf.csv")
    # Keep every original observation and the equal-realization empirical weights.
    for key, scale, metric in (("F2a",10,"min"),("F2b",50,"min"),("F2c",10,"avg"),("F2d",50,"avg")):
        fig, ax = axes()
        panel_rows = [r for r in cdf if int(r["scale"])==scale and r["metric"]==metric+"_ee_mbit_j"
                      and r["curve"] in {m[0] for m in METHODS}]
        right = max(float(r["x"]) for r in panel_rows)
        for method, label, color, style in METHODS:
            samples = sorted((float(r["x"]),float(r["cdf"])) for r in panel_rows if r["curve"]==method)
            x, y = np.array(samples).T
            assert x[0]>=0 and np.all(np.diff(y)>=-1e-12) and abs(y[-1]-1)<1e-8
            # An empirical CDF stays at one beyond its last sample.
            ax.step(np.r_[0,x,right], np.r_[0,y,1.], where="post", label=label, color=color, ls=style)
        ax.set(xlim=(0,None), ylim=(0,1.02), yticks=[0,.25,.5,.75,1],
               ylabel="Cumulative fraction", xlabel=("Minimum" if metric=="min" else "Mean")+" link EE (Mbit/J)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
        legend(ax,2)
        save(fig,directory,key)
    records = read_rows(data / "link_mode_statistics.csv")
    lookup = {(r["group"],r["method"]):r for r in records if r["mode"]=="RF" and r["scope"]=="own_success"}
    for i, scale in enumerate((10,50)):
        x=np.arange(2)
        groups=[str(scale)+seed for seed in ("A","B")]
        fig, ax=axes()
        labels=("RF pairs among all pairs","Periods with minimum EE on RF")
        for offset,field,label,color,hatch in zip((-.18,.18),("fraction","weakest_mode_fraction"),labels,(BLUE,RED),("//","")):
            vals=[100*float(lookup[g,"main_new"][field]) for g in groups]
            ax.bar(x+offset,vals,.32,color=color,label=label,hatch=hatch,edgecolor="white",lw=.5)
            for xx,yy in zip(x+offset,vals): ax.text(xx,yy+1.4,f"{yy:.1f}",ha="center",fontsize=7.8)
        ax.set(xlim=(-.56,1.56),ylim=(0,100),yticks=[0,25,50,75,100], xticks=x,
               xticklabels=["1","2"],xlabel="Channel realization",ylabel="Fraction (%)")
        legend(ax,1)
        save(fig,directory,("F3a","F3b")[i])
        fig, ax=axes()
        for method,label,color,marker in (("reference","Mean-EE",GREY,"s"),("main_new","PF-EERA",RED,"o")):
            vals=[float(lookup[g,method]["link_ee_p05_mbit_j"]) for g in groups]
            ax.plot(x,vals,ls="none",color=color,marker=marker,ms=5,label=label)
            for xx,yy in zip(x,vals):
                ax.annotate(f"{yy:.2f}",(xx,yy),xytext=(0,5 if method=="main_new" else -10),
                            ha="center",fontsize=7.6,textcoords="offset points")
        ax.set(xlim=(-.56,1.56), ylim=(0,52),yticks=[0,10,20,30,40,50],xticks=x,xticklabels=["1","2"],
               xlabel="Channel realization",ylabel="RF link EE, 5th percentile (Mbit/J)")
        legend(ax,2)
        save(fig,directory,("F3c","F3d")[i])


if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--data",type=Path,default=ROOT/"data")
    cli.add_argument("--output",type=Path,default=ROOT/"figures"/"panels")
    opt=cli.parse_args()
    draw(opt.data,opt.output)
