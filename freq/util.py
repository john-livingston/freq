import numpy as np


def ordered_set(x):
    res, ind = np.unique(x, return_index=True)
    return res[np.argsort(ind)]


def get_alias(preal, psamp):
    """Alias periods of `preal` given sampling period `psamp`.

    Returns (p_plus, p_minus) for f_alias = f_real +/- f_samp;
    p_minus is inf when the frequencies coincide.
    """
    fr, fs = 1.0/preal, 1.0/psamp
    p_plus = 1.0/(fr + fs)
    p_minus = np.inf if fr == fs else abs(1.0/(fr - fs))
    return p_plus, p_minus
