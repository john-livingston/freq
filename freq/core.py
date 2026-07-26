import numpy as np
import matplotlib.pyplot as plt

from .gls import Gls
from .util import get_alias, ordered_set
from .plot import plot_gls_power, plot_gls_folded, FAP_LEVELS


def iterative_gls(x_rv, y_rv, yerr_rv=None, inst_rv=None, n=3, pmin=1.0, pmax=100.0,
                  plot=True, highlight=None, labels=None, annotate_color='k', fp=None):
    """Iterative (prewhitening) generalized Lomb-Scargle periodogram.

    Computes a GLS periodogram over periods pmin to pmax, subtracts the
    best-fit sinusoid, and repeats n times, so successive signals are found
    after the strongest ones are removed. When inst_rv is given, per-instrument
    median velocities are subtracted first, since the GLS has no offset model
    (contrast l1_periodogram, which fits offsets as unpenalized vectors).

    With plot=True a figure is built with one row per iteration: the periodogram
    on the left and the phase-folded best period on the right. Pass fp to save it.
    Sinusoid models are evaluated on a fine grid (t_hr) sampled at pmin/20 so
    short-period models are not aliased.

    Returns a dict with:
        gls        list of n Gls objects, one per iteration
        summary    list of n dicts: P, e_P, FAP, alias_day, alias_month
                   (aliases are (plus, minus) tuples from get_alias)
        sinmod     summed model evaluated at x_rv
        t_hr       fine time grid; sinmod_hr the summed model on it
        x, y,      the arrays actually fitted (y median-subtracted per
        yerr, inst instrument when inst_rv was given)
        fig        the figure, or None when plot=False

    The returned dict is what plot_gls_timeseries consumes.
    """
    if labels is None:
        labels = [f'iteration {i+1}' for i in range(n)]
    if len(labels) < n:
        raise ValueError(f'need at least n={n} labels, got {len(labels)}')

    # GLS has no offset model: remove per-instrument zero points up front
    if inst_rv is not None:
        y_rv = np.asarray(y_rv, float).copy()
        for inst in ordered_set(inst_rv):
            ix = inst_rv == inst
            y_rv[ix] -= np.median(y_rv[ix])

    yerr = yerr_rv is not None

    fig = None
    if plot:
        fig = plt.figure(figsize=(10, n*2.5))
        gs = plt.GridSpec(n, 4)

    gls, summary = [], []
    sinmod = np.zeros_like(x_rv)
    t_hr = np.arange(x_rv.min(), x_rv.max(), pmin/20)
    sinmod_hr = np.zeros_like(t_hr)

    for i in range(n):
        g = Gls((x_rv, y_rv - sinmod, yerr_rv), Pbeg=pmin, Pend=pmax)
        gls.append(g)
        P, e_P, fap = g.best['P'], g.best['e_P'], g.FAP()
        alias_day, alias_month = get_alias(P, 1.0), get_alias(P, 29.5)
        summary.append(dict(P=P, e_P=e_P, FAP=fap,
                            alias_day=alias_day, alias_month=alias_month))
        print(f"n = {i+1}: P = {P:.4f} +/- {e_P:.4f} days; FAP = {fap:.2e}; "
              f"1-day aliases = {alias_day[0]:.4f}/{alias_day[1]:.4f} days; "
              f"1-month aliases = {alias_month[0]:.4f}/{alias_month[1]:.4f} days")

        sinmod = sinmod + g.sinmod()
        sinmod_hr = sinmod_hr + g.sinmod(t_hr)

        if plot:
            ax = fig.add_subplot(gs[i, :3])
            plot_gls_power(g, ax, fap_levels=FAP_LEVELS, highlight=highlight,
                           annotate_text=labels[i], annotate_color=annotate_color)
            if i < n-1:
                plt.setp(ax, xlabel='', xticklabels='')
            ax = fig.add_subplot(gs[i, 3])
            plot_gls_folded(g, ax, yerr=yerr, inst_rv=inst_rv,
                            annotate_color=annotate_color)
            if i < n-1:
                plt.setp(ax, xlabel='', xticklabels='')

    if plot:
        fig.align_ylabels()
        if fp is not None:
            fig.savefig(fp, bbox_inches='tight')

    return dict(gls=gls, sinmod=sinmod, t_hr=t_hr, sinmod_hr=sinmod_hr,
                x=x_rv, y=y_rv, yerr=yerr_rv, inst=inst_rv,
                summary=summary, fig=fig)
