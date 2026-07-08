import numpy as np

import pytest

from freq import iterative_gls


@pytest.fixture(scope='module')
def gls_res(synth_rv):
    t, y, yerr, _ = synth_rv
    return iterative_gls(t, y, yerr, n=2, pmin=1.0, pmax=100.0, plot=False)


def test_recovers_two_periods(synth_rv, gls_res):
    _, _, _, (P1, P2) = synth_rv
    found = sorted(s['P'] for s in gls_res['summary'])
    assert abs(found[0] - P1) < 0.05
    assert abs(found[1] - P2) < 0.5


def test_result_carries_data_and_summary(synth_rv, gls_res):
    t, y, yerr, _ = synth_rv
    for k in ('gls', 'sinmod', 't_hr', 'sinmod_hr', 'x', 'y', 'yerr', 'inst', 'summary', 'fig'):
        assert k in gls_res
    assert gls_res['x'] is t and gls_res['fig'] is None
    s = gls_res['summary'][0]
    assert set(s) == {'P', 'e_P', 'FAP', 'alias_day', 'alias_month'}
    assert len(s['alias_day']) == 2


def test_t_hr_resolves_pmin(gls_res):
    assert np.diff(gls_res['t_hr']).max() <= 1.0/20 + 1e-9


def test_short_labels_raise(synth_rv):
    t, y, yerr, _ = synth_rv
    with pytest.raises(ValueError):
        iterative_gls(t, y, yerr, n=2, labels=['only one'], plot=False)


def test_instrument_offsets_auto_subtracted():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 120, 90))
    inst = np.where(t < 60, 'inst_a', 'inst_b')
    P = 7.7
    y = 5*np.sin(2*np.pi*t/P) + np.where(inst == 'inst_a', 0.0, 4000.0)
    y = y + rng.normal(0, 1.0, len(t))
    res = iterative_gls(t, y, np.ones_like(t), inst_rv=inst, n=1,
                        pmin=2.0, pmax=50.0, plot=False)
    assert abs(res['summary'][0]['P'] - P) < 0.1
    assert abs(np.median(res['y'][inst == 'inst_b'])) < 5.0
