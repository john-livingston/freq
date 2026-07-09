import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest


@pytest.fixture(scope='session')
def synth_rv():
    """Two circular signals, uneven sampling, known periods."""
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 200, 120))
    P1, K1, P2, K2 = 5.234, 8.0, 17.89, 4.0
    y = K1*np.sin(2*np.pi*t/P1) + K2*np.sin(2*np.pi*t/P2 + 1.0)
    yerr = np.full_like(t, 1.5)
    y = y + rng.normal(0, yerr)
    return t, y, yerr, (P1, P2)


@pytest.fixture(scope='session')
def rv_file(tmp_path_factory, synth_rv):
    """synth_rv written as a whitespace RV file (two instruments) for CLI tests."""
    t, y, yerr, _ = synth_rv
    inst = np.where(t < 100, 'carmenes', 'harpsn')
    p = tmp_path_factory.mktemp('data') / 'rv.txt'
    with open(p, 'w') as w:
        w.write('btjd rv rv_err inst_name\n')
        for ti, yi, ei, ii in zip(t, y, yerr, inst):
            w.write(f'{ti:.6f} {yi:.6f} {ei:.6f} {ii}\n')
    return str(p)
