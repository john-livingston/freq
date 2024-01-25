import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .util import ordered_set

MARKERCYCLE = "osv^pd"


def annotate(ax, txt, annotate_loc=1, annotate_color='k', fontsize=12):
    if annotate_loc == 1:
        xy = 0,1
        ha, va = "left", "top"
        xytext = 5,-5
    elif annotate_loc == 2:
        xy = 1,1
        ha, va = "right", "top"
        xytext = -5,-5
    ax.annotate(txt, zorder=10,
                xy=xy, xycoords="axes fraction", ha=ha, va=va,
                xytext=xytext, textcoords="offset points", fontsize=fontsize,
                fontweight='normal', color=annotate_color, bbox=dict(color='w'))

def plot_gls_power(gls, ax, log=True, fap_levels=None, color='k', lw=0.5, highlight=None,
                   annotate_text=None, annotate_color='r', annotate_alpha=0.25):

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

def plot_gls_folded(gls, ax, yerr=False, inst_rv=None, colors=None,
                    cmap=plt.cm.RdBu_r, annotate_color='r', lw=1, markers=None, markercycle=MARKERCYCLE):

    def phase(t, per, T0, days=False):
        if days:
            return (t-T0) % per
        else:
            return ((t-T0) % per) / per - 0.5

    if inst_rv is not None and markers is None:
        markers = [markercycle[i] for i in range(len(set(inst_rv)))]
    if inst_rv is not None and colors is None:
        x_rv = gls.t.copy()
        ic = [(i-x_rv.min())/(x_rv.max()-x_rv.min()) for i in x_rv]
        colors = np.array([cmap(i) for i in ic])

    fbest, T0 = gls.best["f"], gls.best["T0"]
    per = 1 / fbest

    tt = np.arange(T0, T0+per, 0.01*per)
    yy = gls.sinmod(tt)
    if inst_rv is not None:
        for i,inst in enumerate(ordered_set(inst_rv)):
            ix = inst_rv == inst
            if yerr:
                ax.errorbar(phase(gls.t[ix], per, T0), gls.y[ix], gls.e_y[ix], 
                            fmt='k.', lw=1, ms=0, zorder=0)
            ax.scatter(phase(gls.t[ix], per, T0), gls.y[ix], c=colors[ix],
                       marker=markers[i], edgecolor='black', linewidth=1)
    else:
        if yerr:
            ax.errorbar(phase(gls.t, per, T0), gls.y, gls.e_y, fmt='k.', lw=1, ms=0, zorder=0)
        ax.scatter(phase(gls.t, per, T0), gls.y, c=gls.t, cmap=cmap, edgecolor='black', linewidth=1)
    annotate(ax, f'{1/fbest :.2f} days', annotate_loc=2, annotate_color=annotate_color)

    xx = phase(tt, per, T0)
    ii = np.argsort(xx)
    ax.plot(xx[ii], yy[ii], color=annotate_color, lw=lw)

    plt.setp(ax, xlim=(xx.min(), xx.max()), xlabel='Phase', ylabel='RV [m/s]', xticks=[-0.4,0,0.4])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

def plot_gls_timeseries(iterative_gls_res, x_rv, y_rv, yerr_rv, inst_rv=None, markers=None, colors=None,
               cmap=plt.cm.RdBu_r, x_offset=2457000, labelsdict=None, markercycle=MARKERCYCLE, fp=None):

    fig = plt.figure(figsize=(10,5))
    gs = plt.GridSpec(4, 1)

    sinmod = iterative_gls_res['sinmod']
    t_hr = iterative_gls_res['t_hr']
    sinmod_hr = iterative_gls_res['sinmod_hr']

    if inst_rv is not None and markers is None:
        markers = [markercycle[i] for i in range(len(set(inst_rv)))]
    if inst_rv is not None and colors is None:
        ic = [(i-x_rv.min())/(x_rv.max()-x_rv.min()) for i in x_rv]
        colors = np.array([cmap(i) for i in ic])
    if inst_rv is not None and labelsdict is None:
        labelsdict = {k:k for k in set(inst_rv)}

    ax = fig.add_subplot(gs[:3])
    if inst_rv is not None:
        for i,inst in enumerate(ordered_set(inst_rv)):
            ix = inst_rv == inst
            ax.errorbar(x_rv[ix]-x_offset, y_rv[ix], yerr_rv[ix], 
                        fmt='k.', ms=0, zorder=-1)
            ax.scatter(x_rv[ix]-x_offset, y_rv[ix], c=colors[ix], 
                       label=labelsdict[inst], 
                       edgecolor='black', linewidth=1, marker=markers[i])
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [
            Line2D([0], [0], marker=m, markerfacecolor='w', markeredgecolor='k', ls='')
            for m in markers
                   ]
        ax.legend(new_handles, labels)
    else:
        ax.errorbar(x_rv-x_offset, y_rv, yerr_rv, 
                    fmt='k.', ms=0, zorder=-1)
        ax.scatter(x_rv-x_offset, y_rv, c=x_rv, 
                   cmap=plt.cm.RdBu_r, edgecolor='black', linewidth=1)

    ax.plot(t_hr-x_offset, sinmod_hr, 'k', lw=0.5, zorder=-10)
    plt.setp(ax, xticklabels=[], ylabel='RV [m/s]')

    ax = fig.add_subplot(gs[3])
    if inst_rv is not None:
        for i,inst in enumerate(ordered_set(inst_rv)):
            ix = inst_rv == inst
            ax.errorbar(x_rv[ix]-x_offset, y_rv[ix]-sinmod[ix], yerr_rv[ix], 
                        fmt='k.', ms=0, zorder=-1)
            ax.scatter(x_rv[ix]-x_offset, y_rv[ix]-sinmod[ix], c=colors[ix], marker=markers[i],
                       label=labelsdict[inst], edgecolor='black', linewidth=1)
    else:
        ax.errorbar(x_rv-x_offset, y_rv-sinmod, yerr_rv, fmt='k.', ms=0, zorder=-1)
        ax.scatter(x_rv-x_offset, y_rv-sinmod, c=x_rv, cmap=plt.cm.RdBu_r, edgecolor='black', linewidth=1)
    ax.axhline(0, color='k', ls=':', alpha=0.5, zorder=-2)
    plt.setp(ax, xlabel=f'BJD $-$ {x_offset}', ylabel='Residuals')

    if fp is not None:
        fig.savefig(fp, bbox_inches='tight')
