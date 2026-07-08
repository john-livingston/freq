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
