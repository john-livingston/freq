# Vendored l1periodogram

Source: https://github.com/nathanchara/l1periodogram
Commit: 65ecc8103cb61998613251bee13b094659fd7cf6
License: GPL-3 (see repo-root LICENSE). Cite Hara, Boué, Laskar & Correia 2017, MNRAS 464, 1220.

Modules: l1periodogram_v1, lars_l1p, significance, fastlinsquare_cholesky,
covariance_matrices, gglasso_basis_pursuit_l1p.
Not vendored: combine_timeseries, filter_poly (freq builds datasets itself).

Local patches (only these):
1. flat imports -> package-relative (`import significance` -> `from . import significance`).
2. gglasso_basis_pursuit_l1p: f2py hint print at import time replaced with `pass`
   (LARS is the only solver freq supports).
3. gglasso_basis_pursuit_l1p.py: upstream file uses CRLF line endings; normalized
   to LF as an incidental byproduct of the patch tooling (no content change).
