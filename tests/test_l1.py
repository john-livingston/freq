import numpy as np

import pytest

from freq import l1_periodogram


def test_recovers_two_periods(synth_rv):
    t, y, yerr, (P1, P2) = synth_rv
    res = l1_periodogram(t, y, yerr, pmin=2.0, sigmaW=1.0,
                         significance_methods=(), plot=False)
    top2 = sorted(res['peak_periods'][:2])
    assert abs(top2[0] - P1) < 0.1
    assert abs(top2[1] - P2) < 0.5
    assert {'period_d', 'amplitude', 'alias_1d_plus'} <= set(res['table'].columns)


def test_absorbs_instrument_offsets():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 120, 90))
    inst = np.where(t < 60, 'inst_a', 'inst_b')
    P = 7.7
    y = 5*np.sin(2*np.pi*t/P) + np.where(inst == 'inst_a', 0.0, 40.0)
    y = y + rng.normal(0, 1.0, len(t))
    res = l1_periodogram(t, y, np.ones_like(t), inst_rv=inst, pmin=2.0,
                         significance_methods=(), plot=False)
    assert abs(res['peak_periods'][0] - P) < 0.1


def test_significance_keys(synth_rv):
    t, y, yerr, _ = synth_rv
    res = l1_periodogram(t, y, yerr, pmin=2.0,
                         significance_methods=('fap',),
                         max_significance_tests=2, plot=False)
    assert 'log10faps' in res['significance']
    assert 'log10fap' in res['table'].columns


def test_no_yerr_requires_sigmaW(synth_rv):
    t, y, _, _ = synth_rv
    with pytest.raises(ValueError):
        l1_periodogram(t, y, None, sigmaW=0.0, plot=False)


def test_red_noise_path_smoke(synth_rv):
    t, y, yerr, (P1, _) = synth_rv
    res = l1_periodogram(t, y, yerr, pmin=2.0, sigmaW=1.0,
                         sigmaR=2.0, tau=10.0, Prot=30.0,
                         significance_methods=(), plot=False)
    assert abs(res['peak_periods'][0] - P1) < 0.1


def test_exponential_kernel_smoke(synth_rv):
    t, y, yerr, (P1, _) = synth_rv
    res = l1_periodogram(t, y, yerr, pmin=2.0, sigmaW=1.0,
                         sigmaR=2.0, tau=10.0, kernel='exponential',
                         significance_methods=(), plot=False)
    assert abs(res['peak_periods'][0] - P1) < 0.1


def test_unknown_kernel_raises(synth_rv):
    t, y, yerr, _ = synth_rv
    with pytest.raises(ValueError, match='kernel'):
        l1_periodogram(t, y, yerr, sigmaR=2.0, tau=10.0,
                       kernel='matern', plot=False)


def test_ess_qp_kernel_smoke(synth_rv):
    t, y, yerr, (P1, _) = synth_rv
    res = l1_periodogram(t, y, yerr, pmin=2.0, sigmaW=1.0,
                         sigmaR=2.0, tau=10.0, Prot=5.0,
                         qp='ess', qp_gamma=8.0,
                         significance_methods=(), plot=False)
    assert abs(res['peak_periods'][0] - P1) < 0.1


def test_bad_qp_raises(synth_rv):
    t, y, yerr, _ = synth_rv
    with pytest.raises(ValueError, match='qp'):
        l1_periodogram(t, y, yerr, sigmaR=2.0, tau=10.0, Prot=5.0,
                       qp='matern', plot=False)


def test_trend_adds_unpenalized_column_and_absorbs_drift():
    """trend=True appends a time column to MH0 so linear drift is absorbed.

    Catches: the `if trend:` block being skipped -> drift leaks into the
    periodogram and MH0 is one column short.
    """
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 150, 100))
    P = 6.3
    y = 4*np.sin(2*np.pi*t/P) + 0.25*t          # strong linear drift
    y = y + rng.normal(0, 1.0, len(t))
    yerr = np.ones_like(t)
    res = l1_periodogram(t, y, yerr, pmin=2.0, sigmaW=1.0, trend=True,
                         significance_methods=(), plot=False)
    assert res['l1p'].MH0.shape[1] == 2         # 1 offset + 1 trend column
    assert abs(res['peak_periods'][0] - P) < 0.1


def test_unpenalized_period_is_absorbed_into_null_model():
    """A period passed to unpenalized_periods is fitted for free, not as a peak.

    Catches: unpenalize_periods() not being called (or called with the wrong
    periods) -> the known signal still shows up in the l1 solution.
    """
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0, 150, 100))
    P_known, P_other = 3.7, 11.9
    y = (6*np.sin(2*np.pi*t/P_known) + 4*np.sin(2*np.pi*t/P_other + 0.5)
         + rng.normal(0, 1.0, len(t)))
    yerr = np.ones_like(t)
    res = l1_periodogram(t, y, yerr, pmin=2.0, sigmaW=1.0,
                         unpenalized_periods=[P_known],
                         significance_methods=(), plot=False)
    assert min(abs(p - P_other) for p in res['peak_periods']) < 0.2
    assert all(abs(p - P_known) > 0.2 for p in res['peak_periods'])


def test_evidence_laplace_adds_bayes_factor_column(synth_rv):
    """Requesting evidence_laplace puts log10_bayesf_laplace in the table.

    Catches: the significance block that copies the column being dropped.
    """
    t, y, yerr, _ = synth_rv
    res = l1_periodogram(t, y, yerr, pmin=2.0,
                         significance_methods=('fap', 'evidence_laplace'),
                         max_significance_tests=2, plot=False)
    assert 'log10_bayesf_laplace' in res['table'].columns
    assert res['table']['log10_bayesf_laplace'].notna().all()


def test_sigmaR_without_tau_raises(synth_rv):
    """sigmaR > 0 with tau <= 0 is rejected on every kernel/qp path.

    Catches: the cos path silently returning a diagonal covariance, so the
    requested red noise becomes extra white jitter with no warning.
    """
    t, y, yerr, _ = synth_rv
    for qp, Prot in (('cos', -1.0), ('cos', 30.0), ('ess', 30.0)):
        with pytest.raises(ValueError, match='tau'):
            l1_periodogram(t, y, yerr, sigmaR=4.0, tau=0.0, qp=qp, Prot=Prot,
                           plot=False)


def test_unsorted_times_give_same_result_as_sorted(synth_rv):
    """Input need not be time-ordered; the wrapper sorts before fitting.

    Catches: dropping the argsort in _prep_data. Upstream sizes the whole
    frequency grid from Tobs = t[-1] - t[0], so unsorted input silently
    changes the grid.
    """
    t, y, yerr, (P1, _) = synth_rv
    rng = np.random.default_rng(9)
    perm = rng.permutation(len(t))
    ref = l1_periodogram(t, y, yerr, pmin=2.0, sigmaW=1.0,
                         significance_methods=(), plot=False)
    shuf = l1_periodogram(t[perm], y[perm], yerr[perm], pmin=2.0, sigmaW=1.0,
                          significance_methods=(), plot=False)
    assert np.allclose(ref['periods'], shuf['periods'])
    assert np.allclose(ref['power'], shuf['power'])
    assert abs(shuf['peak_periods'][0] - P1) < 0.1
