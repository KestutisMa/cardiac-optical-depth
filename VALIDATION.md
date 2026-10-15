# Release Validation

Initial release checks, 14 October 2026:

- All 100 retained original NPY, CSV and checkpoint artifacts match their
  pre-curation SHA-256 hashes.
- Independent recomputation of archived predictions reproduces the primary,
  optical multi-setting, factorial, channel-ablation and feature-baseline metrics.
- All non-control rows occur once in the pooled test folds. Train, validation
  and test indices are disjoint; exact-depth groups do not leak in grouped CV.
- All 279 factorial combinations are present exactly once.
- Frozen optical-sensitivity checkpoint hashes match the primary models.
- The feature baseline reruns to RMSE 0.455 mm.
- CPU inference was exercised on all five folds of each principal analysis.
  Pooled RMSE was 0.274 mm (primary, stochastic), 0.188 mm (optical, stochastic),
  and 0.476 mm (factorial, deterministic). Archived stochastic predictions retain
  the original reported 0.298 / 0.190 mm values; CPU reruns need not reproduce
  the same dropout samples as the original runs.
- Python source compilation and CFF 1.2.0 citation-schema validation pass.
- Training and physical simulations were not rerun for this release.

CPU tests used Python 3.12 with PyTorch 2.14.1, NumPy 2.5.4, SciPy 1.18.1,
PyYAML 6.0.3 and Matplotlib 3.11.2. The historical analysis environment is
recorded separately in the README. Compatibility with every combination
allowed by the dependency ranges has not been tested.
