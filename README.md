# freq

Frequency analysis of unevenly sampled RV time series: iterative GLS
(Zechmeister & Kürster 2009) + l1 periodogram (Hara et al. 2017, MNRAS 464, 1220).

## Install

    pip install -e .

## CLI

    freq data/toi1410.txt -c btjd rv rv_err inst_name -sm -mu 4 \
        -n 3 --pmin 1 --pmax 100 --l1 -o results

Outputs: `periodogram.png` (GLS stack), `timeseries.png`, `l1_periodogram.png`,
`l1_peaks.csv`, `l1_periodogram.npz`, `activity_indicators-<inst>.png` (with `-ai`).

## Library

    from freq import iterative_gls, l1_periodogram, plot_gls_timeseries
    res = iterative_gls(t, rv, rv_err, inst_rv=inst, n=3)
    plot_gls_timeseries(res)
    l1res = l1_periodogram(t, rv, rv_err, inst_rv=inst, sigmaW=1.0)

See `examples/toi1410.ipynb`.

![](plots/gls1.png)

![](plots/gls2.png)

## License / citations

GPL-3 (vendored l1periodogram is GPL-3; vendored gls.py is MIT).
If you use this, cite Zechmeister & Kürster (2009) for GLS and
Hara, Boué, Laskar & Correia (2017) for the l1 periodogram.
