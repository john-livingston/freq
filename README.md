# freq

Frequency analysis of unevenly sampled RV time series: iterative GLS
(Zechmeister & Kürster 2009) + l1 periodogram (Hara et al. 2017, MNRAS 464, 1220).

## Install

    pip install -e .

## CLI

Default mode is the iterative GLS stack; `--l1` switches to the l1 periodogram:

    freq data/toi1410.txt -c btjd rv rv_err inst_name -mu 4 \
        -n 3 --pmin 1 --pmax 100 -o results
    freq data/toi1410.txt -c btjd rv rv_err inst_name -mu 4 \
        --pmin 1 --pmax 100 --l1 -o results

GLS outputs: `periodogram.png`, `timeseries.png`. l1 outputs: `l1_periodogram.png`,
`l1_peaks.csv`, `l1_periodogram.npz`. Both: `args.txt`,
`activity_indicators-<inst>.png` (with `-ai`). Per-instrument median RVs are
subtracted automatically for GLS (it has no offset model); the l1 fit instead
models offsets as unpenalized vectors.

## Noise model (l1)

The l1 periodogram requires a noise covariance:

    V = diag(rv_err²) + σW² I + σR² · k(Δt) · q(Δt)

- `k(Δt)`: `--l1_kernel gaussian` (default) `exp(−Δt²/2τ²)`, or
  `--l1_kernel exponential` `exp(−Δt/τ)`
- `q(Δt) = [1 + cos(2πΔt/Prot)]/2` when `--l1_Prot > 0`, else 1
  (quasi-periodic modulation for rotation)
- Flags/params: `--l1_sigmaW` (σW, jitter, m/s; default 1), `--l1_sigmaR`
  (σR, correlated amplitude, m/s; default 0 = white only), `--l1_tau`
  (τ, decay timescale, days), `--l1_Prot` (days; −1 disables)

Example fiducial red-noise model: `--l1_sigmaW 1 --l1_sigmaR 4 --l1_tau 82 --l1_Prot 30`.
Library equivalent: `l1_periodogram(..., sigmaW=1, sigmaR=4, tau=82, Prot=30, kernel='gaussian')`.

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
