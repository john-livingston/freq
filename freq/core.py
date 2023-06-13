import sys
import numpy as np
import matplotlib.pyplot as plt
#sys.path.append("/home/jhl/github/GLS/python/")
from gls import Gls

from .util import get_alias
from .plot import plot_gls_power, plot_gls_folded

def iterative_gls(x_rv, y_rv, yerr_rv=None, inst_rv=None, n=3, pmin=0.2, pmax=200,
                  plot=True, highlight=None, labels=None, annotate_color='k', fp=None):

    if labels is None:
        labels = [f'iteration {i+1}' for i in range(n)]

    yerr = yerr_rv is not None

    if plot:
        fig = plt.figure(figsize=(10,n*2.5))
        gs = plt.GridSpec(n, 4)

    gls = []
    sinmod = np.zeros_like(x_rv)
    t_hr = np.linspace(x_rv.min(), x_rv.max(), 1000)
    sinmod_hr = np.zeros_like(t_hr)

    for i in range(n):

        gls.append(Gls((x_rv, y_rv-sinmod, yerr_rv), Pbeg=pmin, Pend=pmax))
        msg = f"P = {gls[i].best['P'] :.4f} +/- {gls[i].best['e_P'] :.4f} days"
        msg += f"; FAP = {gls[i].FAP() :.2e}"
        msg += f"; 1-day alias = {get_alias(gls[i].best['P'], 1.0) :.4f} days"
        print(msg)

        sinmod += gls[i].sinmod()
        sinmod_hr += gls[i].sinmod(t_hr)

        if plot:
            ax = fig.add_subplot(gs[i,:3])
            plot_gls_power(gls[i], ax, fap_levels=[1e-1,1e-2,1e-3], highlight=highlight,
                           annotate_text=labels[i], annotate_color=annotate_color)
            if i < n-1: plt.setp(ax, xlabel='', xticklabels='')

            ax = fig.add_subplot(gs[i,3])
            plot_gls_folded(gls[i], ax, yerr=yerr, inst_rv=inst_rv, annotate_color=annotate_color)
            if i < n-1: plt.setp(ax, xlabel='', xticklabels='')

    if plot:
        fig.align_ylabels()
        if fp is not None:
            fig.savefig(fp, bbox_inches='tight')

    return dict(gls=gls, sinmod=sinmod, t_hr=t_hr, sinmod_hr=sinmod_hr)
