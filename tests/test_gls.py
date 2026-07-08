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
