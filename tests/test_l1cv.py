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
