import itertools
import multiprocessing as mp
import os

import numpy as np

from .l1 import (_prep_data, _build_V, _build_MH0, _unpenalized_MH0,
                 _check_noise_args, l1_periodogram)

_BLAS_ENV = ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def build_grid(sigmaW, sigmaR, tau, Prot):
    """Cartesian product of noise hyperparameters; sigmaR=0 combos collapse
    to a single white-noise model (tau/Prot irrelevant)."""
    seen, grid = set(), []
    for w, r, t_, p in itertools.product(sigmaW, sigmaR, tau, Prot):
        if r == 0:
            t_, p = 0.0, -1.0
        key = (w, r, t_, p)
        if key not in seen:
            seen.add(key)
            grid.append(dict(sigmaW=w, sigmaR=r, tau=t_, Prot=p))
    return grid


def _cv_chunk(args):
    """Score a chunk of noise models; runs in a worker process (or inline).

    Builds the sine dictionary once per chunk, then swaps only V between
    models (update_model skips the dictionary rebuild for V-only changes).
    """
    (t, y, yerr, MH0, models, pmin, oversampling, Nphi, kernel, qp, qp_gamma,
     fap_threshold, n_sim, training_prop, seed, max_tests,
     unpenalized_periods) = args
    from .l1p import l1periodogram_v1, significance

    c = l1periodogram_v1.l1p_class(t, y.copy())
    if unpenalized_periods:
        MH0 = _unpenalized_MH0(t, MH0, unpenalized_periods)
    rows = []
    for j, m in enumerate(models):
        V = _build_V(t, yerr, m['sigmaW'], m['sigmaR'], m['tau'], m['Prot'],
                     kernel, qp, qp_gamma)
        if j == 0:
            c.set_model(omegamax=2*np.pi/pmin, oversampling=oversampling,
                        Nphi=Nphi, V=V, MH0=MH0, verbose=0)
        else:
            # MH0 must be passed too: the projection matrix is V-weighted, and
            # upstream only rebuilds it when MH0 is given. Passing it does not
            # rebuild the sine dictionary (only omegamax/oversampling/Nphi do).
            c.update_model(V=V, MH0=MH0, verbose=0)
        c.l1_perio(numerical_method='lars', plot_output=False, verbose=0,
                   significance_evaluation_methods=['fap'],
                   max_n_significance_tests=max_tests)
        faps = np.asarray(c.significance.get('log10faps', []))
        sel = faps < fap_threshold
        omegas = np.asarray(c.omega_peaks)[:len(faps)][sel]
        # same seed for every model -> identical train/test splits,
        # so CV scores are directly comparable across models
        np.random.seed(seed)
        mean_cv, median_cv, ll_all = significance.crossval(
            c.t, c.y_init, c.W2, omegas, c.MH0, Nsim=n_sim,
            method='random', Training_prop=training_prop)
        rows.append(dict(**m, n_selected=int(sel.sum()),
                         selected_periods=list(np.round(2*np.pi/omegas, 4)),
                         selected_log10faps=list(np.round(faps[sel], 2)),
                         median_cv=median_cv, mean_cv=mean_cv,
                         loglike_all=ll_all))
    return rows


def l1_crossval(x_rv, y_rv, yerr_rv=None, inst_rv=None, pmin=1.0,
                sigmaW=(0.5, 1.0, 2.0), sigmaR=(0.0, 2.0, 4.0),
                tau=(41.0, 82.0), Prot=(-1.0,),
                qp='cos', qp_gamma=8.0, kernel='gaussian',
                fap_threshold=-0.5, n_sim=400, training_prop=0.6, seed=0,
                oversampling=10, Nphi=8, trend=False,
                unpenalized_periods=None,
                n_jobs=4, max_significance_tests=10,
                rerun_best=True, rerun_kwargs=None, verbose=0):
    """Rank noise models for the l1 periodogram by cross-validation.

    Hara et al. (2017/2020) procedure: for each hyperparameter combination,
    compute the l1 periodogram, keep peaks with log10 FAP < fap_threshold,
    and score the model (selected sinusoids + offsets, noise covariance) by
    the median held-out log-likelihood over n_sim random
    training_prop/(1-training_prop) splits. Models are ranked by median_cv.

    Parallelism: the grid is split over n_jobs worker processes, each with
    single-threaded BLAS (measured faster than multithreaded even at
    n_jobs=1). Each worker holds its own sine dictionary
    (~3 x Nt x Ngrid x Nphi x 8 bytes) - lower n_jobs if memory-bound.

    Returns dict(table=<ranked DataFrame>, best=<param dict>[, l1=<result of
    l1_periodogram rerun with the best model>]).
    """
    import pandas as pd

    _check_noise_args(kernel, qp)
    t, y, yerr, inst = _prep_data(x_rv, y_rv, yerr_rv, inst_rv)
    MH0 = _build_MH0(t, inst, trend)
    grid = build_grid(sigmaW, sigmaR, tau, Prot)

    n_workers = max(1, min(n_jobs, len(grid)))
    chunks = [grid[i::n_workers] for i in range(n_workers)]
    argss = [(t, y, yerr, MH0, chunk, pmin, oversampling, Nphi, kernel, qp,
              qp_gamma, fap_threshold, n_sim, training_prop, seed,
              max_significance_tests, unpenalized_periods)
             for chunk in chunks]

    if n_workers == 1:
        rows = _cv_chunk(argss[0])
    else:
        old_env = {k: os.environ.get(k) for k in _BLAS_ENV}
        os.environ.update({k: '1' for k in _BLAS_ENV})
        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(n_workers) as pool:
                rows = [r for chunk in pool.map(_cv_chunk, argss)
                        for r in chunk]
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    tab = (pd.DataFrame(rows).sort_values('median_cv', ascending=False)
           .reset_index(drop=True))
    print(tab.drop(columns=['selected_periods', 'selected_log10faps'])
          .to_string(index=False))

    best = {k: float(tab.iloc[0][k]) for k in ('sigmaW', 'sigmaR',
                                               'tau', 'Prot')}
    print(f'best model by median CV score: {best}')
    res = dict(table=tab, best=best)
    if rerun_best:
        kwargs = dict(pmin=pmin, kernel=kernel, qp=qp, qp_gamma=qp_gamma,
                      trend=trend, oversampling=oversampling, Nphi=Nphi,
                      unpenalized_periods=unpenalized_periods,
                      verbose=verbose)
        kwargs.update(rerun_kwargs or {})
        res['l1'] = l1_periodogram(x_rv, y_rv, yerr_rv, inst_rv=inst_rv,
                                   **best, **kwargs)
    return res
