import numpy as np

from .util import ordered_set, get_alias


def _build_MH0(t, inst_rv, trend):
    if inst_rv is None:
        MH0 = np.ones((len(t), 1))
    else:
        insts = ordered_set(inst_rv)
        MH0 = np.zeros((len(t), len(insts)))
        for j, inst in enumerate(insts):
            MH0[:, j] = (inst_rv == inst).astype(float)
    if trend:
        tc = t - t.mean()
        MH0 = np.c_[MH0, tc/np.abs(tc).max()]
    return MH0


def l1_periodogram(x_rv, y_rv, yerr_rv=None, inst_rv=None, pmin=1.0, pmax=None,
                   sigmaW=1.0, sigmaR=0.0, tau=0.0, Prot=-1,
                   trend=False, unpenalized_periods=None,
                   oversampling=10, Nphi=8,
                   significance_methods=('fap', 'evidence_laplace'),
                   max_significance_tests=12, starname='', verbose=0,
                   plot=True, n_peaks_plot=4, highlight=None,
                   annotate_color='k', fp=None):
    """l1 periodogram (Hara et al. 2017) of an RV time series.

    Sibling of iterative_gls: fits all signals at once via basis pursuit.
    White noise + jitter by default; sigmaR/tau/Prot enable the red-noise
    kernel. Instrument offsets (and optional trend) are unpenalized vectors.
    """
    import pandas as pd
    from .l1p import l1periodogram_v1, covariance_matrices

    if yerr_rv is None and sigmaW <= 0:
        raise ValueError('sigmaW must be > 0 when yerr_rv is None')

    x_rv = np.asarray(x_rv, float)
    y_rv = np.asarray(y_rv, float)
    order = np.argsort(x_rv)
    t = x_rv[order]
    y = y_rv[order].copy()
    yerr = (np.zeros_like(y) if yerr_rv is None
            else np.asarray(yerr_rv, float)[order])
    inst = None if inst_rv is None else np.asarray(inst_rv)[order]

    if inst is not None:
        for i in ordered_set(inst):
            ix = inst == i
            y[ix] -= y[ix].mean()
    else:
        y = y - y.mean()

    if sigmaR > 0:
        V = covariance_matrices.covar_mat(t, yerr, sigmaW, sigmaR, 0.0, tau,
                                          Prot=Prot)
    else:
        V = np.diag(yerr**2 + sigmaW**2)

    MH0 = _build_MH0(t, inst, trend)

    c = l1periodogram_v1.l1p_class(t, y)
    c.starname = starname
    if inst is not None:
        c.dataset_names = list(ordered_set(inst))
    c.set_model(omegamax=2*np.pi/pmin, oversampling=oversampling, Nphi=Nphi,
                V=V, MH0=MH0, verbose=verbose)
    if unpenalized_periods:
        c.unpenalize_periods(list(unpenalized_periods), MH0, verbose=verbose)
    c.l1_perio(numerical_method='lars', plot_output=False, verbose=verbose,
               significance_evaluation_methods=list(significance_methods),
               max_n_significance_tests=max_significance_tests)

    periods = 2*np.pi/c.omegas
    power = c.smoothed_solution
    peak_periods = 2*np.pi/np.asarray(c.omega_peaks)
    peak_values = np.asarray(c.peakvalues)

    npk = min(max_significance_tests, len(peak_periods))
    tab = pd.DataFrame({'period_d': peak_periods[:npk],
                        'amplitude': peak_values[:npk]})
    if 'log10faps' in c.significance:
        tab['log10fap'] = np.asarray(c.significance['log10faps'])[:npk]
    if 'log10_bayesf_laplace' in c.significance:
        tab['log10_bayesf_laplace'] = \
            np.asarray(c.significance['log10_bayesf_laplace'])[:npk]
    if pmax is not None:
        tab = tab[tab.period_d <= pmax].reset_index(drop=True)
    alias_d = [get_alias(p, 1.0) for p in tab.period_d]
    alias_m = [get_alias(p, 29.5) for p in tab.period_d]
    tab['alias_1d_plus'] = [a[0] for a in alias_d]
    tab['alias_1d_minus'] = [a[1] for a in alias_d]
    tab['alias_1mo_plus'] = [a[0] for a in alias_m]
    tab['alias_1mo_minus'] = [a[1] for a in alias_m]
    print(tab.to_string(index=False))

    res = dict(l1p=c, periods=periods, power=power,
               peak_periods=peak_periods, peak_values=peak_values,
               significance=c.significance, table=tab)

    if plot:
        from .plot import plot_l1_power
        res['fig'] = plot_l1_power(res, pmax=pmax, n_peaks=n_peaks_plot,
                                   highlight=highlight,
                                   annotate_color=annotate_color, fp=fp)
    return res
