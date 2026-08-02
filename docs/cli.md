# CLI reference

`freq` runs the GLS stack by default; `--l1` switches to the \(\ell_1\) periodogram,
and `--l1_cv` runs cross-validated noise-model selection (implies `--l1`). Outputs
are written to `--outdir`.

## Input

| Flag | Default | Meaning |
|------|---------|---------|
| `input` | n/a | input table (positional) |
| `--sep` | `\s+` | column separator regex |
| `-c`, `--columns` | `time mnvel errvel tel` | time / rv / err / instrument column names (exactly 4) |
| `-o`, `--outdir` | `.` | output directory |

## Filtering

| Flag | Default | Meaning |
|------|---------|---------|
| `-mu`, `--max_unc` | none | drop points with error above this |
| `-oc`, `--outlier_cond` | none | per-instrument MAD clip factor (applied before GLS and l1) |
| `-i`, `--instruments` | all | restrict to these instruments |

## GLS

| Flag | Default | Meaning |
|------|---------|---------|
| `-n`, `--n_iter` | 3 | prewhitening iterations |
| `--pmin` | 1.0 | minimum period (days) |
| `--pmax` | 100.0 | maximum period (days) |

## ℓ₁ periodogram

| Flag | Default | Meaning |
|------|---------|---------|
| `--l1` | off | compute the \(\ell_1\) periodogram instead of the GLS stack |
| `--l1_sigmaW` | 1.0 | white jitter σW (m/s) |
| `--l1_sigmaR` | 0.0 | correlated amplitude σR (m/s) |
| `--l1_tau` | 0.0 | decay timescale τ (days) |
| `--l1_Prot` | −1 | rotation period (days); −1 disables the QP factor |
| `--l1_kernel` | `gaussian` | decay kernel (`gaussian` or `exponential`) |
| `--l1_qp` | `cos` | QP factor (`cos` or `ess`) |
| `--l1_qp_gamma` | 8.0 | ess harmonic-content Γ |
| `--l1_trend` | off | add a linear trend as an unpenalized vector |
| `--l1_unpenalized` | none | periods to leave unpenalized (known planets) |
| `--l1_max_tests` | 12 | max peaks to assess significance for |
| `--l1_no_significance` | off | skip FAP / evidence evaluation |
| `--l1_n_peaks_plot` | 4 | peaks to annotate on the plot |

## ℓ₁ cross-validation

| Flag | Default | Meaning |
|------|---------|---------|
| `--l1_cv` | off | rank noise models by CV, then run \(\ell_1\) with the best |
| `--l1_cv_sigmaW` | `0.5 1 2` | σW grid |
| `--l1_cv_sigmaR` | `0 2 4` | σR grid |
| `--l1_cv_tau` | `41 82` | τ grid |
| `--l1_cv_Prot` | `-1` | Prot grid |
| `--l1_cv_fap_threshold` | −0.5 | log₁₀ FAP threshold for keeping peaks |
| `--l1_cv_nsim` | 400 | train/test splits |
| `--l1_cv_training_prop` | 0.6 | training fraction |
| `--l1_cv_seed` | 0 | split RNG seed |
| `--l1_cv_jobs` | 4 | parallel workers (single-threaded BLAS each) |

## Activity

| Flag | Default | Meaning |
|------|---------|---------|
| `-ai`, `--activity_indicators` | none | indicator columns to periodogram per instrument |

Uses `<name>_err` as uncertainties when that column exists. Best periods are
printed per instrument; plots go to `activity_indicators-<inst>.png`. Indicators
that are missing or identically zero are skipped with a note.

## Plotting

| Flag | Default | Meaning |
|------|---------|---------|
| `-hl`, `--highlight` | none | periods to mark on periodograms |
| `--annotate_color` | `k` | annotation colour |
| `--x_offset` | `auto` | time offset for plots (`auto` or a number) |

## Examples

```bash
# GLS stack
freq rvs.csv -c bjd rv rv_err inst_name --sep ',' -mu 4 -n 3 --pmin 1 --pmax 100 -o results

# l1 periodogram with a red-noise model
freq rvs.csv -c bjd rv rv_err inst_name --sep ',' --l1 \
    --l1_sigmaW 1 --l1_sigmaR 4 --l1_tau 82 --l1_Prot 30 -o results

# cross-validated noise-model selection
freq rvs.csv -c bjd rv rv_err inst_name --sep ',' --l1_cv \
    --l1_cv_sigmaW 0.5 1 2 --l1_cv_sigmaR 0 2 4 --l1_cv_tau 41 82 --l1_cv_Prot -1 30 -o results
```
