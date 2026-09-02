# Forensic review — Tail Estimator Audit, Track B

- **Reviewer / model:** Codex (GPT-5), independent of the Claude builder.
- **Candidate:** `75714e8502d202b56934d30bff2c9a5257e634e6`.
- **Date:** 2026-09-02.
- **Scope:** full candidate review: controlled DGP, estimator and metric
  semantics, reproducibility, result claims, tests, figures, and public-ready
  documentation. No implementation, result, figure, dependency, parent
  Portfolio, site, or Sekiya file was changed by this review.

## Evidence and checks

- Read the research contract, README, estimator, experiment, aggregation and
  metric modules, the test suite, and the candidate diff.
- Ran `.venv/bin/pytest -q`: **27 passed**.
- Ran `git diff --check`: clean.
- Independently aggregated the committed `results/main_grid.csv` with
  `aggregate_tail_index`. For Student-t, tail index 6, `n=500`, the GPD row
  contains `n_reps=400`, `n_used=201`, and `n_dropped=199`; the raw grid has
  1,158 records with `gpd_converged=False`.
- Visually inspected `figures/bias_rmse_heatmaps.png`; its displayed GPD
  values match the documented conditional-on-finite-estimate behavior and its
  colour-cap note does not conceal the printed values.

## Findings

### TAIL-R01-001 — REVISE: denominator claim conflicts with implementation

`README.md` says that GPD-POT failures are “counted in the denominator
throughout,” and `docs/research_contract.md` says they are “counted and
reported, not silently dropped from the denominator.” That is not true for
the reported GPD bias, RMSE, quantile-error, or interval-coverage metrics:
`src/tail_estimator_audit/metrics.py` intentionally excludes non-finite
estimates, while preserving `n_used` and `n_dropped`. The inspected
Student-t/6/500 cell scores 201 of 400 replications, not all 400.

This is scientifically defensible if stated precisely: the study reports a
separate all-replication failure rate, while estimator-quality metrics are
conditional on successful heavy-tail recovery and must display their scoring
denominator. The existing language overstates what the metrics measure and
can mislead a public reader, especially where failure rates are large.

**Required correction:** revise only the contract/README/result wording and,
where the existing figures or result presentation summarise GPD metrics,
make the conditional denominator explicit. Do not alter DGPs, estimators,
seeds, raw results, metrics, or headline numerical values. Describe
`xi_hat <= 0` as a defined *heavy-tail-recovery failure*, not as a literal
optimizer non-convergence unless the optimiser itself failed.

### TAIL-R01-002 — non-blocking: result provenance is reproducible but not
checked in

The 7.2 MB primary result CSV and diagnostic sweep outputs are intentionally
ignored and must be regenerated. This is reasonable for a compact portfolio
repository because the seeds, code, commands, and committed figures are
present, but future publication should retain a machine-readable summary or
artifact digest if independent audit without a re-run matters.

## Assessment

The known-truth Pareto/Student-t DGPs, Hill and GPD-POT implementations,
threshold diagnostic, Gaussian failure reference, seeded grid, tests, and
figures form a serious, bounded prestudy. The implementation records failed
fits rather than silently coercing them, and it reports unstable/negative
results rather than marketing the methods. The only blocking issue is the
incorrect public wording about denominators and the resulting ambiguity
between operational failure rate and conditional estimator quality.

## Blind spots

- This review did not independently re-run the 16,000-replication grid; it
  inspected the committed raw result artifact and ran the supplied unit suite.
- It did not audit SciPy's GPD-MLE numerical implementation or third-party
  wheel provenance.
- It did not test another operating system or a real dependent time series;
  those are outside the declared controlled-DGP scope.

## Verdict and next boundary

**REVISE.** Apply the narrowly defined reporting correction above, then
obtain a fresh Codex confirmation limited to `TAIL-R01-001`: truthful
denominator/failure terminology, unchanged scientific contract and candidate
results, and clean documentation-only correction. No publication, site edit,
or broader Portfolio work is authorized by this review.
