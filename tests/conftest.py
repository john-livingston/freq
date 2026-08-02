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
def rv_file_activity(tmp_path_factory, synth_rv):
    """RV file with activity-indicator columns covering every -ai branch:

    shk      periodic at 11.5 d, has shk_err, contains NaNs (dropna path)
    halpha   periodic at 6.4 d, no error column (2-tuple path)
    zeroerr  periodic at 8.2 d, error column is all zeros (2-tuple fallback)
    flat     identically zero (skipped)
    """
    t, _, _, _ = synth_rv
    rng = np.random.default_rng(11)
    inst = np.where(t < 100, 'carmenes', 'harpsn')
    shk = 1.0 + 0.3*np.sin(2*np.pi*t/11.5) + rng.normal(0, 0.01, len(t))
    shk[::17] = np.nan                      # gaps only dropna() can handle
    halpha = 0.5 + 0.2*np.sin(2*np.pi*t/6.4) + rng.normal(0, 0.01, len(t))
    zeroerr = 0.8 + 0.25*np.sin(2*np.pi*t/8.2) + rng.normal(0, 0.01, len(t))
    p = tmp_path_factory.mktemp('data') / 'rv_activity.txt'
    with open(p, 'w') as w:
        w.write('btjd rv rv_err inst_name shk shk_err halpha zeroerr '
                'zeroerr_err flat\n')
        for i in range(len(t)):
            w.write(f'{t[i]:.6f} {shk[i] if np.isfinite(shk[i]) else 0.0:.6f} '
                    f'1.0 {inst[i]} {shk[i]:.6f} 0.01 {halpha[i]:.6f} '
                    f'{zeroerr[i]:.6f} 0.0 0.0\n')
    return str(p)


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


@pytest.fixture(scope='session')
def rv_file_messy(tmp_path_factory, synth_rv):
    """RV file with a constant-velocity instrument, NaN rows, and outliers.

    inst_c has identical velocities (MAD 0); two rows carry NaN rv/rv_err;
    inst_a gets two large outliers.
    """
    t, y, yerr, _ = synth_rv
    y, yerr = y.copy(), yerr.copy()
    inst = np.where(t < 100, 'inst_a', 'inst_b')
    inst[:12] = 'inst_c'
    y[inst == 'inst_c'] = 5.0            # zero MAD
    y[20], y[21] = 500.0, -500.0         # outliers in inst_a
    y[30], yerr[31] = np.nan, np.nan     # non-finite rows
    p = tmp_path_factory.mktemp('data') / 'rv_messy.txt'
    with open(p, 'w') as w:
        w.write('btjd rv rv_err inst_name\n')
        for i in range(len(t)):
            w.write(f'{t[i]:.6f} {y[i]:.6f} {yerr[i]:.6f} {inst[i]}\n')
    return str(p)
