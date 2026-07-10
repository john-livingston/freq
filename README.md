# freq

Frequency analysis of unevenly sampled RV time series: iterative GLS
(Zechmeister & Kürster 2009) + ℓ₁ periodogram (Hara et al. 2017, MNRAS 464, 1220).

Documentation: https://john-livingston.github.io/freq

## Install

    pip install git+https://github.com/john-livingston/freq

or for development:

    git clone https://github.com/john-livingston/freq && cd freq && pip install -e .

## CLI

Default mode is the iterative GLS stack; `--l1` switches to the ℓ₁ periodogram.
Input is a whitespace- or delimiter-separated table; name the time / RV / error /
instrument columns with `-c`:

    freq rvs.csv -c bjd rv rv_err inst_name --sep ',' -mu 4 \
        -n 3 --pmin 1 --pmax 100 -o results
    freq rvs.csv -c bjd rv rv_err inst_name --sep ',' \
        --pmin 1 --pmax 100 --l1 -o results

GLS outputs: `periodogram.png`, `timeseries.png`. ℓ₁ outputs: `l1_periodogram.png`,
`l1_peaks.csv`, `l1_periodogram.npz`. Both: `args.txt`,
`activity_indicators-<inst>.png` (with `-ai`). Per-instrument median RVs are
subtracted automatically for GLS (it has no offset model); the ℓ₁ fit instead
models offsets as unpenalized vectors.

## Noise model (ℓ₁)

The ℓ₁ periodogram requires a noise covariance:

    V = diag(rv_err²) + σW² I + σR² · k(Δt) · q(Δt)

- `k(Δt)`: `--l1_kernel gaussian` (default) `exp(−Δt²/2τ²)`, or
  `--l1_kernel exponential` `exp(−Δt/τ)`
- `q(Δt)` when `--l1_Prot > 0` (else 1): `--l1_qp cos` (default)
  `[1 + cos(2πΔt/Prot)]/2` (cosine bell — fundamental only), or `--l1_qp ess`
  `exp(−Γ sin²(πΔt/Prot))` (standard exp-sine-squared; `--l1_qp_gamma` = Γ = 2/λ²,
  default 8 — higher Γ = more harmonic content)
- Flags/params: `--l1_sigmaW` (σW, jitter, m/s), `--l1_sigmaR` (σR, correlated
  amplitude, m/s; 0 = white only), `--l1_tau` (τ, decay timescale, days),
  `--l1_Prot` (days; −1 disables)

Library equivalent: `l1_periodogram(t, rv, rv_err, inst_rv=inst, sigmaW=1,
sigmaR=4, tau=82, Prot=30, kernel='gaussian')`.

## Noise-model selection by cross-validation

`--l1_cv` ranks a grid of noise models by the Hara et al. (2020) procedure —
for each model, peaks with log₁₀ FAP < `--l1_cv_fap_threshold` (−0.5) are kept
and the model is scored by the median held-out log-likelihood over
`--l1_cv_nsim` (400) random 60/40 train/test splits — then reruns the full ℓ₁
with the best model:

    freq rvs.csv -c bjd rv rv_err inst_name --sep ',' -o results --l1_cv \
        --l1_cv_sigmaW 0.5 1 2 --l1_cv_sigmaR 0 2 4 --l1_cv_tau 41 82 \
        --l1_cv_Prot -1 30

Extra outputs: `l1_cv.csv` (ranked models), `l1_cv_best.json` (winner), and
`l1_cv_peaks.png` (peak stability across the top 20% of models). The grid is
scored in parallel (`--l1_cv_jobs`, default 4) with single-threaded-BLAS workers.

## Library

    from freq import iterative_gls, l1_periodogram, l1_crossval, plot_gls_timeseries
    res = iterative_gls(t, rv, rv_err, inst_rv=inst, n=3)
    plot_gls_timeseries(res)
    l1res = l1_periodogram(t, rv, rv_err, inst_rv=inst, sigmaW=1.0)

## License / citations

GPL-3 (vendored ℓ₁periodogram is GPL-3; vendored gls.py is MIT).
If you use this, cite Zechmeister & Kürster (2009) for GLS and
Hara, Boué, Laskar & Correia (2017) for the ℓ₁ periodogram.

## Documentation

Full docs live at https://john-livingston.github.io/freq (MkDocs + Material).
To build locally:

```bash
pip install -e '.[docs]'
mkdocs serve
```

Figures are regenerated with `python scripts/gen_docs_figures.py`; publish with
`mkdocs gh-deploy`.
