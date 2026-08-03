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

Lines beginning with `#` are treated as comments and skipped. Rows with a
non-finite time, velocity, or error are dropped before anything else runs, and
the number dropped is reported.

## Filtering

| Flag | Default | Meaning |
|------|---------|---------|
| `-mu`, `--max_unc` | none | drop points with error above this |
| `-oc`, `--outlier_cond` | none | per-instrument MAD clip factor (applied before GLS and l1) |
| `-i`, `--instruments` | all | restrict to these instruments |

The MAD clip keeps points with \(|y - \mathrm{med}| < \) `cond` \(\times\) MAD/0.67,
per instrument. An instrument whose velocities are all identical (MAD of zero) is
left untouched rather than discarded.

## Period range

`--pmin` and `--pmax` apply to every mode, not just the GLS.

| Flag | Default | Meaning |
|------|---------|---------|
| `--pmin` | 1.0 | minimum period (days); also sets the \(\ell_1\) frequency-grid maximum |
| `--pmax` | 100.0 | maximum period (days); trims the \(\ell_1\) peak table and plot |

For the \(\ell_1\) periodogram, `--pmin` bounds the dictionary through
\(\omega_\mathrm{max} = 2\pi/p_\mathrm{min}\), while `--pmax` only trims the
reported peaks and the plot: it does not bound the frequency grid, and it does
not apply while `--l1_cv` is ranking models, only to the final rerun. Both also
set the period range of the `-ai` activity periodograms.

## GLS

| Flag | Default | Meaning |
|------|---------|---------|
| `-n`, `--n_iter` | 3 | prewhitening iterations |

## \(\ell_1\) periodogram

| Flag | Default | Meaning |
|------|---------|---------|
| `--l1` | off | compute the \(\ell_1\) periodogram instead of the GLS stack |
| `--l1_sigmaW` | 1.0 | white jitter σW (m/s) |
| `--l1_sigmaR` | 0.0 | correlated amplitude σR (m/s) |
| `--l1_tau` | 0.0 | decay timescale τ (days); required when σR > 0 |
| `--l1_Prot` | −1 | rotation period (days); −1 disables the QP factor |
| `--l1_kernel` | `gaussian` | decay kernel (`gaussian` or `exponential`) |
| `--l1_qp` | `cos` | QP factor (`cos` or `ess`) |
| `--l1_qp_gamma` | 8.0 | ess harmonic-content Γ |
| `--l1_trend` | off | add a linear trend as an unpenalized vector |
| `--l1_unpenalized` | none | periods to leave unpenalized (known planets) |
| `--l1_max_tests` | 12 | max peaks to assess significance for |
| `--l1_no_significance` | off | skip FAP / evidence evaluation |
| `--l1_n_peaks_plot` | 4 | peaks to annotate on the plot |

Setting `--l1_sigmaR` without `--l1_tau` is an error: the correlated term
degenerates to a diagonal when τ = 0, which would quietly turn the requested red
noise into extra white jitter.

## \(\ell_1\) cross-validation

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
| `--l1_cv_jobs` | 4 | parallel workers, each with single-threaded BLAS |

`--l1_trend` and `--l1_unpenalized` apply to both the model ranking and the final
rerun. `--l1_max_tests` and `--pmax` reach only the rerun; while ranking, the
number of peaks tested comes from `l1_crossval`'s own default of 10 and no period
trimming is applied. Results do not depend on `--l1_cv_jobs`.

## Activity

| Flag | Default | Meaning |
|------|---------|---------|
| `-ai`, `--activity_indicators` | none | indicator columns to periodogram per instrument |

Uses `<name>_err` as uncertainties when that column exists, and runs unweighted
when it does not or when it is identically zero. Rows where the indicator is
missing are dropped. Best periods are printed per instrument; plots go to
`activity_indicators-<inst>.png`. A named column that does not exist is skipped
with a note; an indicator that is identically zero, or has no rows left after
dropping missing values, is skipped silently.

## Plotting

| Flag | Default | Meaning |
|------|---------|---------|
| `-hl`, `--highlight` | none | periods to mark on periodograms |
| `--annotate_color` | `k` | annotation color |
| `--x_offset` | `auto` | time offset for the GLS time-series plot (`auto` or a number) |

## Outputs

Every run writes `args.txt` (the command line) into `--outdir`. The rest depends
on the mode:

| File | Written when | Contents |
|------|--------------|----------|
| `periodogram.png` | default (GLS) | the prewhitening stack, one row per iteration |
| `timeseries.png` | default (GLS) | summed model over the data, plus residuals |
| `l1_periodogram.png` | `--l1` | the \(\ell_1\) periodogram with peaks marked |
| `l1_peaks.csv` | `--l1` | peak periods, amplitudes, FAP, Bayes factors, aliases (trimmed by `--pmax`) |
| `l1_periodogram.npz` | `--l1` | `periods`, `power`, `peak_periods`, `peak_values` (not trimmed) |
| `l1_cv.csv` | `--l1_cv` | every noise model ranked by median CV score |
| `l1_cv_best.json` | `--l1_cv` | the winning model's parameters and score |
| `l1_cv_peaks.png` | `--l1_cv` | peak stability across the top 20% of models |
| `activity_indicators-<inst>.png` | `-ai` | one periodogram panel per indicator |

The \(\ell_1\) peak table and, under `--l1_cv`, the ranked model table are also
printed to stdout.

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
