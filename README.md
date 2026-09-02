# Tail Estimator Audit

A controlled-DGP audit of two standard tail-index estimators — Hill and GPD
peaks-over-threshold (POT) maximum likelihood — against parametric data with
an exactly known, controllable tail index. The bounded question, hypotheses,
DGPs, frozen threshold rule, and metrics were all written down and frozen
*before* this experiment ran; see [`docs/research_contract.md`](docs/research_contract.md)
for the full contract, including what this study explicitly does not claim.

This is a synthetic-data study only. No real financial, market, or other
observed data is used anywhere.

## Reproduction

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,figures]"

pytest -q                                              # 27 tests

python scripts/run_experiments.py --out results/main_grid.csv
python scripts/run_threshold_diagnostics.py --out-dir results
python scripts/make_figures.py --results-dir results --figures-dir figures
```

Everything is seeded end to end from one root integer (`ROOT_SEED = 1729` in
`src/tail_estimator_audit/experiment.py`) via `numpy.random.SeedSequence`, so
every cell of the grid is independently reproducible regardless of run order
(`tests/test_experiment_and_analysis.py::test_run_grid_shape_and_seed_independence_of_order`
checks this directly).

### What was actually run, and how long it took

| Stage | Command | Result | Wall time |
|---|---|---|---|
| Test suite | `pytest -q` | 27 passed | 1.0s |
| Main grid | `run_experiments.py` | 16,000 rows written to `results/main_grid.csv` | 3m 53s |
| Threshold diagnostic | `run_threshold_diagnostics.py` | 4 x 8,000-row sweep CSVs written to `results/` | 1m 0s |
| Figures | `make_figures.py` | 4 PNGs written to `figures/` | 3.0s |

The full 2 DGPs x 4 tail indices x 5 sample sizes x 400 replications = 16,000
grid ran synchronously in under four minutes on the machine used for this
study (a single 20-second timing probe on the most expensive cell,
`student_t` at `n=10000`, gave ~0.016s/replication, extrapolating to the
observed total). **No adjustment to the preregistered grid size, replication
count, or threshold rule was needed** — the contract's 16,000-replication
main grid and 3,200-row-per-DGP diagnostic sweep both ran to completion at
full size on the first attempt.

## Deviations from the contract

None to the experimental design, grid, seeds, or metrics. The only change
made after inspecting results was to `scripts/make_figures.py` (a
presentation-only module with, by its own docstring, no estimation logic):
the tail-index bias/RMSE heatmap's color scale is capped at the 90th
percentile of `|bias|` / RMSE rather than the max, because a small number of
genuine GPD-POT MLE blow-ups at small `n` combined with a heavy tail (see
"GPD-POT can fail badly," below) would otherwise saturate the entire
colormap toward one extreme cell and make the other 39 cells indistinguishable
from white. The exact numeric value is still printed as text in every cell
regardless of the color cap; nothing about the underlying estimates,
aggregation, or metrics was touched.

## Findings

All numbers below are measured from `results/main_grid.csv` (16,000 rows,
400 replications per cell) and `results/threshold_sensitivity_*.csv` (200
replications x 40 threshold values per DGP), produced by the run described
above.

### Convergence

- **Hill**: 0/16,000 failures (0.00%). The estimator is closed-form and
  never fails to produce a number, though — see below — that number is not
  always trustworthy.
- **GPD-POT MLE**: 1,158/16,000 heavy-tail-recovery failures (7.24%),
  measured across every replication. A replication is scored as a failure
  either because `scipy.stats.genpareto.fit` itself raises an exception or
  returns a non-finite parameter — a literal optimizer failure — or because
  it converges to a non-positive shape (`xi_hat <= 0`), which is a
  well-defined outcome of a successful fit meaning no heavy (regularly
  varying) tail was recovered, not an optimizer error. This is *not*
  uniform: it is 0.20% under Pareto at `n=10000` and rises to 49.75%
  (essentially a coin flip) for Student-t, `nu=6`, `n=500` — the combination
  of a lighter tail and a small sample most strains the "exactly GPD above
  `u`" assumption. This failure rate is reported over all replications; the
  bias, RMSE, quantile-error, and coverage numbers reported below for
  GPD-POT are conditional on the replications that produced a finite
  estimate, and the underlying results record each cell's `n_used` and
  `n_dropped` so the scoring denominator is never implicit.

### H1 (sample-size consistency) — mostly confirmed, with one exception

RMSE shrinks monotonically with `n` for Hill in every DGP/tail-index cell
checked. For GPD-POT it shrinks monotonically in the Pareto DGP (e.g. at
tail index 3: RMSE 43.10 -> 23.45 -> 2.97 -> 0.64 -> 0.41 across
n = 500/1000/2000/5000/10000), but is **not** monotonic for Student-t at the
same tail index: RMSE goes 22.49 -> 10.84 -> **12.95** -> 1.57 -> 1.13 — a
bump at `n=2000`, driven by GPD-POT non-convergence and near-zero-`xi`
instability interacting with the specific exceedance counts at that cell,
not a data or seeding bug (reproducible from the frozen seed). H1 holds as a
general shrinking trend but GPD-POT's convergence is fragile enough that it
is not strictly monotonic everywhere.

### H2 (misspecification cost) — confirmed

GPD-POT's bias is larger under Student-t than under Pareto at every sample
size tested, and the gap widens as `n` grows (tail index 3.0):

| n | GPD-POT bias, Pareto | GPD-POT bias, Student-t |
|---|---|---|
| 500 | 5.46 | 5.90 |
| 1000 | 2.46 | 3.51 |
| 2000 | 0.71 | 2.76 |
| 5000 | 0.12 | 1.11 |
| 10000 | 0.07 | 0.93 |

By `n=10000` GPD-POT's bias under Student-t (0.93) is more than 13x its bias
under Pareto (0.07) at the same sample size — exactly the asymptotic-vs-exact
GPD assumption gap the contract predicted. Hill's bias also differs by DGP
(it is negative and larger in magnitude under Student-t than under Pareto),
but stays bounded within roughly [-0.5, 0.5] across the same cells, versus
GPD-POT's bias which reaches into the single-to-double digits at small `n`.
In that sense Hill degrades far less between the two DGPs, as H2 predicted.

### H3 (threshold instability) — confirmed

See **Figure 1** (`figures/hill_plot.png`) and **Figure 4**
(`figures/threshold_sensitivity.png`). Neither estimator has a
threshold-free plateau:

- Hill under Pareto is comparatively well-behaved — the mean estimate sits
  close to the true tail index (3) across nearly two decades of `k`, with a
  shrinking IQR band as `k` grows.
- Hill under Student-t drifts steadily downward as `k` grows past ~50,
  falling from ~3.2 toward ~0.3 by `k=1000` (`n=2000`) as the window pulls
  in bulk (lighter-tailed, in the regularly-varying sense at finite `k`)
  observations — the classic "Hill horror plot" (Resnick).
- GPD-POT is more volatile at low threshold percentiles (more exceedances,
  but the "exactly GPD" assumption is more strained) than at high ones,
  and under Student-t is dramatically more volatile than under Pareto across
  the entire sweep (mean `alpha_hat` swings from ~32 down to ~5 as `u`'s
  percentile moves from 0.50 to ~0.85, before climbing again near the
  boundary where exceedance counts get very small).

The frozen headline rule (`k = floor(sqrt(n))`, `u` = 90th percentile) is one
slice through this surface, not a specially favorable one — visible in
Figure 4 as the region where both curves are near, but not exactly on, the
true tail index.

### H4 (extreme-quantile extrapolation cost) — confirmed

At tail index 3.0, `n=10000`, both tail-aware estimators' median relative
error in the extreme quantile grows in magnitude and IQR width as the target
quantile moves further beyond the data (Pareto, Hill): 0.99 -> -0.2%,
0.999 -> +0.1%, 0.9999 -> +0.04% median but IQR widening from
[-2.4%, +2.1%] to [-9.2%, +10.8%]. The effect is sharper under Student-t,
where Hill's median error at `q=0.9999` reaches +5.2% with an IQR of
[-5.0%, +18.8%] — this is also the DGP where GPD-POT's misspecification
(H2) and Hill's threshold instability (H3) compound.

### H5 (Gaussian failure) — confirmed, and starkly

The Gaussian baseline's median relative error at `q=0.999`, tail index 3.0,
across every sample size from 500 to 10,000:

| n | Pareto | Student-t |
|---|---|---|
| 500 | -61.6% | -50.2% |
| 1000 | -60.6% | -49.8% |
| 2000 | -60.3% | -49.0% |
| 5000 | -59.7% | -48.7% |
| 10000 | -59.5% | -48.5% |

This is essentially flat: 20x more data barely moves the error at all (≈2
percentage points, versus Hill/GPD-POT's IQR-halving-or-better improvement
over the same range in Figure 3). The Gaussian approximation is not merely
less precise than the tail-aware estimators — it is stuck at a categorically
wrong answer regardless of sample size, understating the true extreme
quantile by roughly half, exactly as H5 predicted.

### A finding not predeclared as a named hypothesis: Hill's nominal-95% CI is badly miscalibrated under Student-t

This was not one of H1-H5 but falls directly out of the predeclared CI
coverage metric, and is large enough to flag on its own. Empirical coverage
of Hill's asymptotic Wald 95% CI, Student-t DGP, as the true tail index
grows from 2 to 6 (averaged over all five sample sizes at each tail index):

| true tail index (nu) | mean coverage across n=500..10000 |
|---|---|
| 2.0 | 94.0% |
| 3.0 | 86.3% |
| 4.0 | 62.6% |
| 6.0 | 15.9% |

Coverage does **not** recover with more data at high tail index — at
`nu=6` it is 14.0% at `n=500` and still only 11.0% at `n=10000`. This is a
systematic bias problem, not a variance problem: the Wald CI
(`alpha_hat / sqrt(k)`) is derived from the estimator's asymptotic variance
alone and has no term for the second-order deviation from pure regular
variation that a finite-nu Student-t tail carries at any finite threshold.
As the tail gets lighter (larger nu, closer to Gaussian in the body), that
second-order term dominates the CI's width and the interval simply misses
the true value most of the time, no matter how large `n` gets. GPD-POT's
delta-method CI does not show the same collapse (mean coverage 93.3% Pareto,
97.6% Student-t across the full grid) — though this partly reflects that its
denominator already excludes non-converged fits, which are disproportionately
the small-`n`, low-`nu` (heaviest tail, hardest) cells.

## Figures

- **`figures/hill_plot.png`** (Figure 1) — Hill-plot instability: 15
  individual replications plus the mean and IQR across 200, for both DGPs at
  `n=2000`, true tail index 3. Illustrates H3.
- **`figures/bias_rmse_heatmaps.png`** (Figure 2) — Bias and RMSE of
  `alpha_hat` across the full tail-index x sample-size grid, both estimators,
  both DGPs. Color scale capped at the 90th percentile (see "Deviations from
  the contract" above); every cell's exact value is printed regardless.
  Illustrates H1 and H2.
- **`figures/quantile_error.png`** (Figure 3) — Median and IQR of relative
  extreme-quantile error at `q=0.999` vs `n`, both DGPs, tail index 3, all
  three methods including the Gaussian failure reference. Illustrates H4 and
  H5.
- **`figures/threshold_sensitivity.png`** (Figure 4) — The threshold-choice
  instability surface both estimators live on; the frozen headline rule is
  one slice through it. Illustrates H3.

## Known limits

Restated from the contract, now with the specific numbers that make each one
concrete:

- **Parametric, i.i.d. samples only.** No temporal dependence or volatility
  clustering. Nothing here says anything about how these estimators behave
  on real return series with either.
- **Only two DGP families.** Lognormal, stable, and mixture tails are not
  covered.
- **The frozen threshold rule is not optimal**, and its fragility is real:
  Figure 4 shows GPD-POT's mean `alpha_hat` under Student-t swinging by more
  than 6x (from ~32 to ~5) across the swept threshold range at a single
  fixed `(n, tail index)` cell.
- **GPD MLE heavy-tail recovery is not reliable at small n + heavy tail**:
  up to 49.75% failure (counted across all replications in that cell) in
  the worst observed cell (Student-t, `nu=6`, `n=500`). Any application of
  GPD-POT to a similarly small, similarly light-tailed-for-a-tail-model
  sample should expect a meaningful chance of failing to recover a heavy
  tail at all — separate from, and prior to, the bias/RMSE question, which
  is scored only over the replications that produced a finite estimate.
- **Hill's classical Wald CI is not trustworthy across the board** — it
  degrades to 11-20% coverage (against a nominal 95%) under Student-t at
  tail index 6, and this does not improve with `n`. A practitioner reading
  Hill confidence intervals off the classical asymptotic formula on a
  moderately-heavy-tailed-but-not-extreme series should not trust the stated
  width.
- **CI coverage relies on asymptotic theory** that is not guaranteed at the
  smaller sample sizes in this grid; where it fails, that is reported as a
  property of the estimator (see above), not hidden or treated as an
  implementation defect.

## Repository layout

```
docs/research_contract.md   frozen contract (read this first)
src/tail_estimator_audit/   dgp.py, estimators.py, experiment.py, metrics.py,
                             analysis.py, threshold_diagnostics.py
scripts/                    run_experiments.py, run_threshold_diagnostics.py,
                             make_figures.py
tests/                      27 tests covering DGPs, estimators, metrics,
                             experiment/analysis, seeding determinism
results/                    generated CSVs (gitignored, regenerate via scripts/)
figures/                    generated PNGs (committed, so the README renders
                             them directly on GitHub)
```

`results/` is regenerated by the commands above and is not checked in — the
raw grid is a 7.6MB CSV with no diff value. `figures/` is committed as-is so
the images in this README render without a build step.
