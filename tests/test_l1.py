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
