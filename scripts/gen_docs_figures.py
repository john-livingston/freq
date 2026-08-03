"""Generate the documentation figures from seeded synthetic RV data.

Reproducible (Agg backend, seeded RNG, no network, no real data). Writes only
into docs/assets/. Run from the repo root: `python scripts/gen_docs_figures.py`.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from freq import iterative_gls, l1_periodogram, l1_crossval, plot_gls_timeseries
from freq.plot import plot_l1_cv_peaks

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'docs', 'assets')
os.makedirs(ASSETS, exist_ok=True)


def synthetic():
    """Two instruments, three injected planets, an instrument offset, white noise."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 400, 160))
    inst = np.where(t < 200, 'inst_a', 'inst_b')
    periods, amps, phases = (4.13, 9.74, 21.3), (5.0, 3.0, 2.0), (0.3, 1.1, 2.0)
    y = sum(a*np.sin(2*np.pi*t/p + ph) for p, a, ph in zip(periods, amps, phases))
    y = y + np.where(inst == 'inst_b', 12.0, 0.0)
    yerr = np.full_like(t, 1.2)
    y = y + rng.normal(0, yerr)
    return t, y, yerr, inst


def plot_data(t, y, yerr, inst):
    """Raw RV time series, per-instrument median-subtracted (as GLS sees it)."""
    fig, ax = plt.subplots(figsize=(10, 3))
    for name, marker in (('inst_a', 'o'), ('inst_b', 's')):
        m = inst == name
        yc = y[m] - np.median(y[m])
        ax.errorbar(t[m], yc, yerr[m], fmt=marker, ms=4, lw=1, capsize=0,
                    label=name, alpha=0.85)
    ax.set(xlabel='time [days]', ylabel='RV [m/s]')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, 'data.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    t, y, yerr, inst = synthetic()

    plot_data(t, y, yerr, inst)

    res = iterative_gls(t, y, yerr, inst_rv=inst, n=3, pmin=1, pmax=100,
                        fp=os.path.join(ASSETS, 'gls_stack.png'))
    plot_gls_timeseries(res, fp=os.path.join(ASSETS, 'gls_timeseries.png'))

    l1_periodogram(t, y, yerr, inst_rv=inst, pmin=2, pmax=50, sigmaW=1.0,
                   significance_methods=(),
                   fp=os.path.join(ASSETS, 'l1_periodogram.png'))

    cv = l1_crossval(t, y, yerr, inst_rv=inst, pmin=2,
                     sigmaW=(1.0, 2.0), sigmaR=(0.0, 2.0, 4.0), tau=(20.0,),
                     Prot=(-1.0,), n_sim=50, n_jobs=1, rerun_best=False)
    plot_l1_cv_peaks(cv['table'], perc=100,
                     fp=os.path.join(ASSETS, 'cv_peaks.png'))

    print('wrote data.png, gls_stack.png, gls_timeseries.png, '
          'l1_periodogram.png, cv_peaks.png to', ASSETS)


if __name__ == '__main__':
    main()
