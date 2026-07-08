import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .util import ordered_set

MARKERCYCLE = "osv^pd"
FAP_LEVELS = (1e-1, 1e-2, 1e-3)


def _instrument_style(inst_rv, t, cmap=plt.cm.RdBu_r, markercycle=MARKERCYCLE,
                      markers=None, colors=None, labelsdict=None):
    """Per-instrument markers + per-point time-mapped colors + legend labels."""
    insts = ordered_set(inst_rv)
    if markers is None:
        markers = [markercycle[i % len(markercycle)] for i in range(len(insts))]
    if colors is None:
        tn = (t - t.min())/(t.max() - t.min())
        colors = np.array([cmap(v) for v in tn])
    if labelsdict is None:
        labelsdict = {k: k for k in insts}
    return insts, markers, colors, labelsdict


def annotate(ax, txt, annotate_loc=1, annotate_color='k', fontsize=12):
    if annotate_loc == 1:
        xy, ha, va, xytext = (0, 1), 'left', 'top', (5, -5)
    elif annotate_loc == 2:
        xy, ha, va, xytext = (1, 1), 'right', 'top', (-5, -5)
    else:
        raise ValueError(f'annotate_loc must be 1 or 2, got {annotate_loc}')
    ax.annotate(txt, zorder=10, xy=xy, xycoords="axes fraction", ha=ha, va=va,
                xytext=xytext, textcoords="offset points", fontsize=fontsize,
                fontweight='normal', color=annotate_color, bbox=dict(color='w'))


def plot_gls_power(gls, ax, log=True, fap_levels=None, color='k', lw=0.5,
                   highlight=None, annotate_text=None, annotate_color='r',
                   annotate_alpha=0.25):
    x, y = 1/gls.freq, gls.power
    ax.plot(x, y, color=color, lw=lw)
    peak = x[np.argmax(y)]
    ax.axvline(peak, lw=5, alpha=annotate_alpha, color=annotate_color, zorder=0)
    if annotate_text is not None:
        annotate(ax, annotate_text, annotate_loc=1, annotate_color=color, fontsize=12)
    if fap_levels is not None:
        for fl in fap_levels:
            ax.axhline(gls.powerLevel(fl), zorder=-1, color='gray', ls='-', alpha=0.5)
    if highlight is not None:
        for hl in highlight:
            ax.axvline(hl, lw=1, ls=':', color='k', zorder=-10)
    plt.setp(ax, xlim=(x.min(), x.max()), xlabel='Period [days]', ylabel='GLS Power')
    if log:
        plt.setp(ax, xscale='log')
    return ax.figure


def _phase(t, per, T0):
    return ((t - T0) % per)/per - 0.5


def plot_gls_folded(gls, ax, yerr=False, inst_rv=None, colors=None,
                    cmap=plt.cm.RdBu_r, annotate_color='r', lw=1,
                    markers=None, markercycle=MARKERCYCLE):
    fbest, T0 = gls.best["f"], gls.best["T0"]
    per = 1/fbest
    tt = np.arange(T0, T0 + per, 0.01*per)
    yy = gls.sinmod(tt)
    if inst_rv is not None:
        insts, markers, colors, _ = _instrument_style(
            inst_rv, gls.t, cmap, markercycle, markers, colors)
        for i, inst in enumerate(insts):
            ix = inst_rv == inst
            if yerr:
                ax.errorbar(_phase(gls.t[ix], per, T0), gls.y[ix], gls.e_y[ix],
                            fmt='k.', lw=1, ms=0, zorder=0)
            ax.scatter(_phase(gls.t[ix], per, T0), gls.y[ix], c=colors[ix],
                       marker=markers[i], edgecolor='black', linewidth=1)
    else:
        if yerr:
            ax.errorbar(_phase(gls.t, per, T0), gls.y, gls.e_y,
                        fmt='k.', lw=1, ms=0, zorder=0)
        ax.scatter(_phase(gls.t, per, T0), gls.y, c=gls.t, cmap=cmap,
                   edgecolor='black', linewidth=1)
    annotate(ax, f'{per :.2f} days', annotate_loc=2, annotate_color=annotate_color)

    xx = _phase(tt, per, T0)
    ii = np.argsort(xx)
    ax.plot(xx[ii], yy[ii], color=annotate_color, lw=lw)

    plt.setp(ax, xlim=(xx.min(), xx.max()), xlabel='Phase', ylabel='RV [m/s]',
             xticks=[-0.4, 0, 0.4])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    return ax.figure


def plot_gls_timeseries(res, x_offset='auto', cmap=plt.cm.RdBu_r, labelsdict=None,
                        markers=None, colors=None, markercycle=MARKERCYCLE, fp=None):
    x, y, yerr, inst = res['x'], res['y'], res['yerr'], res['inst']
    sinmod, t_hr, sinmod_hr = res['sinmod'], res['t_hr'], res['sinmod_hr']

    if x_offset == 'auto':
        x_offset = 2457000 if np.median(x) > 2.4e6 else 0
    xlabel = f'BJD $-$ {x_offset}' if x_offset else 'Time [days]'

    fig = plt.figure(figsize=(10, 5))
    gs = plt.GridSpec(4, 1)
    ax_data = fig.add_subplot(gs[:3])
    ax_res = fig.add_subplot(gs[3])

    style = None
    if inst is not None:
        style = _instrument_style(inst, x, cmap, markercycle,
                                  markers, colors, labelsdict)

    for ax, yvals in ((ax_data, y), (ax_res, y - sinmod)):
        if inst is not None:
            insts, mk, cl, ld = style
            for i, ins in enumerate(insts):
                ix = inst == ins
                if yerr is not None:
                    ax.errorbar(x[ix] - x_offset, yvals[ix], yerr[ix],
                                fmt='k.', ms=0, zorder=-1)
                ax.scatter(x[ix] - x_offset, yvals[ix], c=cl[ix], marker=mk[i],
                           label=ld[ins], edgecolor='black', linewidth=1)
        else:
            if yerr is not None:
                ax.errorbar(x - x_offset, yvals, yerr, fmt='k.', ms=0, zorder=-1)
            ax.scatter(x - x_offset, yvals, c=x, cmap=cmap,
                       edgecolor='black', linewidth=1)

    if inst is not None:
        insts, mk, _, _ = style
        handles = [Line2D([0], [0], marker=m, markerfacecolor='w',
                          markeredgecolor='k', ls='') for m in mk]
        _, lab = ax_data.get_legend_handles_labels()
        ax_data.legend(handles, lab[:len(insts)])

    ax_data.plot(t_hr - x_offset, sinmod_hr, 'k', lw=0.5, zorder=-10)
    plt.setp(ax_data, xticklabels=[], ylabel='RV [m/s]')
    ax_res.axhline(0, color='k', ls=':', alpha=0.5, zorder=-2)
    plt.setp(ax_res, xlabel=xlabel, ylabel='Residuals')
    fig.align_ylabels()
    if fp is not None:
        fig.savefig(fp, bbox_inches='tight')
    return fig
