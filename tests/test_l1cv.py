import numpy as np

import pytest

from freq.l1cv import l1_crossval, build_grid


def test_build_grid_dedupes_sigmaR_zero():
    g = build_grid((1.0,), (0.0, 2.0), (10.0, 20.0), (-1.0, 5.0))
    zero = [m for m in g if m['sigmaR'] == 0]
    assert len(zero) == 1
    assert len(g) == 1 + 4


def test_crossval_ranks_and_reruns_best(synth_rv):
    t, y, yerr, (P1, _) = synth_rv
    res = l1_crossval(t, y, yerr, pmin=2.0, sigmaW=(1.0,), sigmaR=(0.0, 2.0),
                      tau=(10.0,), Prot=(-1.0,), n_sim=50, n_jobs=1, seed=1,
                      max_significance_tests=5,
                      rerun_kwargs=dict(significance_methods=(), plot=False))
    tab = res['table']
    assert len(tab) == 2
    assert {'sigmaW', 'sigmaR', 'tau', 'Prot', 'median_cv',
            'n_selected'} <= set(tab.columns)
    assert list(tab.median_cv) == sorted(tab.median_cv, reverse=True)
    assert set(res['best']) == {'sigmaW', 'sigmaR', 'tau', 'Prot'}
    assert abs(res['l1']['peak_periods'][0] - P1) < 0.1


def test_parallel_matches_serial(synth_rv):
    t, y, yerr, _ = synth_rv
    kw = dict(pmin=2.0, sigmaW=(1.0,), sigmaR=(0.0, 2.0), tau=(10.0,),
              Prot=(-1.0,), n_sim=30, seed=2, max_significance_tests=5,
              rerun_best=False)
    r1 = l1_crossval(t, y, yerr, n_jobs=1, **kw)
    r2 = l1_crossval(t, y, yerr, n_jobs=2, **kw)
    key = ['sigmaW', 'sigmaR', 'tau', 'Prot']
    t1 = r1['table'].sort_values(key).reset_index(drop=True)
    t2 = r2['table'].sort_values(key).reset_index(drop=True)
    assert np.allclose(t1.median_cv, t2.median_cv)


def test_cv_results_independent_of_worker_count(synth_rv):
    """Per-model results must not depend on chunking (n_jobs).

    Catches: update_model(V=...) leaving projmat stale, so every model after
    the first in a chunk is scored with the previous model's projection.
    """
    t, y, yerr, _ = synth_rv
    kw = dict(pmin=2.0, sigmaW=(1.0,), sigmaR=(0.0, 2.0, 4.0), tau=(20.0,),
              Prot=(-1.0,), n_sim=30, seed=4, max_significance_tests=5,
              rerun_best=False)
    serial = l1_crossval(t, y, yerr, n_jobs=1, **kw)['table']
    # one worker per model: every model is first-in-chunk, so never stale
    fresh = l1_crossval(t, y, yerr, n_jobs=3, **kw)['table']
    key = ['sigmaW', 'sigmaR', 'tau', 'Prot']
    a = serial.sort_values(key).reset_index(drop=True)
    b = fresh.sort_values(key).reset_index(drop=True)
    assert list(a.n_selected) == list(b.n_selected)
    for pa, pb in zip(a.selected_periods, b.selected_periods):
        assert np.allclose(sorted(pa), sorted(pb), rtol=1e-6), (pa, pb)


def test_unpenalized_periods_reach_the_cv_grid(synth_rv):
    """unpenalized_periods must apply to CV scoring and the best-model rerun.

    Catches: the parameter being accepted but never forwarded, so known
    planet periods stay penalized under --l1_cv.
    """
    t, y, yerr, (P1, _) = synth_rv
    res = l1_crossval(t, y, yerr, pmin=2.0, sigmaW=(1.0,), sigmaR=(0.0,),
                      tau=(0.0,), Prot=(-1.0,), n_sim=20, n_jobs=1, seed=7,
                      max_significance_tests=5,
                      unpenalized_periods=[P1],
                      rerun_kwargs=dict(significance_methods=(), plot=False))
    # P1 is fitted for free in the null model, so it is not a selected peak
    for periods in res['table'].selected_periods:
        assert all(abs(p - P1) > 0.2 for p in periods), periods
    assert all(abs(p - P1) > 0.2 for p in res['l1']['peak_periods'][:3])
