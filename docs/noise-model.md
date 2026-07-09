# Noise model

The \(\ell_1\) periodogram requires a noise covariance matrix. freq builds

\[
V = \operatorname{diag}(\sigma_{\mathrm{err}}^2) + \sigma_W^2\, I
    + \sigma_R^2\, k(\Delta t)\, q(\Delta t),
\]

where \(\sigma_{\mathrm{err}}\) are the measurement errors, \(\sigma_W\) is a white
jitter term, and \(\sigma_R\) is the amplitude of a correlated ("red") component
with decay kernel \(k\) and optional quasi-periodic factor \(q\). Setting
\(\sigma_R = 0\) gives a pure white-plus-jitter model.

## Decay kernels

\(k(\Delta t)\) is selected with `kernel`:

- `gaussian` (default): \(k(\Delta t) = e^{-\Delta t^2 / 2\tau^2}\)
- `exponential`: \(k(\Delta t) = e^{-\Delta t / \tau}\)

with correlation timescale \(\tau\) (days).

## Quasi-periodic factor

When \(P_\mathrm{rot} > 0\), a rotation-modulation factor \(q(\Delta t)\) multiplies
the kernel (otherwise \(q = 1\)), selected with `qp`:

- `cos` (default): \(q(\Delta t) = \tfrac12\left[1 + \cos(2\pi\Delta t / P_\mathrm{rot})\right]\)
  — a cosine bell (fundamental only).
- `ess`: \(q(\Delta t) = e^{-\Gamma \sin^2(\pi \Delta t / P_\mathrm{rot})}\)
  — the standard exp-sine-squared kernel, with \(\Gamma = 2/\lambda^2\) set by
  `qp_gamma` (default 8). Larger \(\Gamma\) puts more power in the harmonics
  (\(P_\mathrm{rot}/2\), \(P_\mathrm{rot}/3\), …).

## Parameters

| Parameter | `l1_periodogram` arg | CLI flag | Meaning |
|-----------|----------------------|----------|---------|
| \(\sigma_W\) | `sigmaW` | `--l1_sigmaW` | white jitter (m/s) |
| \(\sigma_R\) | `sigmaR` | `--l1_sigmaR` | correlated amplitude (m/s); 0 = white only |
| \(\tau\) | `tau` | `--l1_tau` | decay timescale (days) |
| \(P_\mathrm{rot}\) | `Prot` | `--l1_Prot` | rotation period (days); −1 disables \(q\) |
| kernel | `kernel` | `--l1_kernel` | `gaussian` or `exponential` |
| \(q\) form | `qp` | `--l1_qp` | `cos` or `ess` |
| \(\Gamma\) | `qp_gamma` | `--l1_qp_gamma` | ess harmonic content |

## Selecting the model by cross-validation

Choosing \((\sigma_W, \sigma_R, \tau, P_\mathrm{rot})\) by hand is error-prone.
`l1_crossval` automates it following Hara et al. (2020): for each model on a grid it

1. computes the \(\ell_1\) periodogram and keeps peaks with
   \(\log_{10}\mathrm{FAP} < \) `fap_threshold` (default \(-0.5\));
2. scores the model by the **median held-out log-likelihood** over `n_sim` random
   train/test splits (default 400 splits, 60/40);

then ranks models by that score and reruns the \(\ell_1\) periodogram with the best
one. Because the score is comparable across models (same split seed), the ranking is
reproducible. Trust the peaks that recur across the top-ranked models — see the
peak-stability plot in the [Guide](usage.md).
