# Research contract: Tail Estimator Audit

Written and frozen before any experiment ran. Deviations from this contract,
if any, are logged in the README under "Deviations from the contract," not
edited into this file after the fact.

## Bounded question

Across generative distributions with an exactly known, controllable tail
index, which of two standard tail estimators — Hill and GPD peaks-over-
threshold (POT) maximum likelihood — recover that tail index and derived
extreme quantiles reliably, and at what combination of threshold choice and
sample size does each estimator's reliability break down?

## What this study does not claim

- No claim about real financial returns, real market microstructure, or any
  other observed real-world series. Every dataset here is drawn from a
  parametric random-number generator with a tail index fixed by the
  experimenter, never estimated from or fit to real data.
- No claim that Hill or GPD-POT are "the best" tail estimators in general, or
  that the two families of generating process used here (Pareto, Student-t)
  exhaust "heavy-tailed behaviour." They are two tractable, well-understood
  cases chosen because the true tail index is known exactly, which is the
  only way to score recovery error against ground truth.
- No claim that the frozen threshold rule used for headline results is
  optimal. It is one predeclared, defensible rule. Its fragility relative to
  the choice of threshold is the subject of a dedicated diagnostic, not
  swept under the headline numbers.

## Hypotheses (predeclared)

- **H1 (sample-size consistency).** For both estimators, bias and RMSE of the
  recovered tail index shrink as sample size `n` grows, at fixed tail index.
- **H2 (misspecification cost).** GPD-POT, which assumes the tail is *exactly*
  generalized Pareto beyond the threshold, is closer to unbiased under Pareto
  data (where that assumption is exactly true beyond any positive threshold)
  than under Student-t data (where it is only asymptotically true as the
  threshold grows). Hill, which only assumes regular variation, degrades less
  between the two DGPs.
- **H3 (threshold instability).** Both estimators are unstable in the
  threshold/order-statistic count `k`: too small a `k` gives high variance,
  too large a `k` pulls in bulk observations and biases the tail index toward
  the (lighter) bulk behaviour. Neither estimator has a threshold-free
  regime — the "Hill horror plot" (Resnick) shows visible level-drift and
  variance rather than a stable plateau, especially at small-to-moderate `n`.
- **H4 (extreme-quantile extrapolation cost).** Relative error in an extreme
  quantile estimate (e.g. the 1-in-10,000 quantile from at most 10,000
  observations) grows as the target quantile moves further beyond the range
  of the observed data, for every estimator including the true-model ones.
- **H5 (Gaussian failure).** A Gaussian (mean + z-score x sd) approximation to
  the same extreme quantile is not merely worse but *categorically* worse —
  its relative error does not shrink with `n` the way the tail-aware
  estimators' errors do, because it is fit to the wrong family regardless of
  sample size. It is included only to demonstrate this failure, not as a
  competitor.

## Data-generating processes

Both are simulated with NumPy's `Generator` under explicit, recorded seeds —
no data is read from disk or the network.

1. **Pareto(alpha, x_m=1).** Density `alpha * x_m^alpha / x^(alpha+1)` for
   `x >= x_m`. Exact tail index is `alpha` at *every* threshold `>= x_m`: the
   Pareto distribution is itself a generalized Pareto tail with shape
   `xi = 1/alpha`, everywhere. This is the DGP where GPD-POT's assumption is
   exactly correct, not merely asymptotically correct — the "clean room" case.
2. **Student-t(nu).** Density with `nu` degrees of freedom, symmetric,
   two-sided. The tail index equals `nu` (the classical result that a
   Student-t tail is regularly varying with index `nu`), but the GPD
   approximation to that tail is only a first-order asymptotic statement —
   at any finite threshold there is a second-order correction term the GPD
   fit ignores. Right tail only is used for tail-index/quantile recovery
   (the fold `|X|` is not used, to avoid conflating two tails).

Sweep grid (frozen before running):

- Tail index: `alpha` (Pareto) / `nu` (Student-t) in `{2.0, 3.0, 4.0, 6.0}` —
  from a barely-finite-variance regime (`2.0`) to a moderately heavy one
  (`6.0`; Student-t with `nu=6` is close to, but still distinguishable from,
  Gaussian in the body).
- Sample size `n` in `{500, 1000, 2000, 5000, 10000}`.
- Monte Carlo replications per (DGP, tail index, n) cell: `400`, each with an
  independently seeded draw from a fixed `numpy.random.SeedSequence` root, so
  the entire experiment is reproducible from one integer seed.

That is `2 DGPs x 4 tail indices x 5 sample sizes x 400 reps = 16,000`
simulated datasets for the main grid, plus a separate, smaller grid for the
threshold-sensitivity diagnostic (below).

## Estimators under audit

- **Hill.** `alpha_hat = k / sum_{i=1..k} (log X_(n-i+1) - log X_(n-k))` on
  the descending order statistics, for the top `k` exceedances. Assumes only
  regular variation of the tail (broadly applicable, weak assumption).
  Asymptotic standard error `alpha_hat / sqrt(k)` gives a Wald CI.
- **GPD peaks-over-threshold, MLE.** Fit a generalized Pareto distribution by
  maximum likelihood to exceedances over a threshold `u`, via
  `scipy.stats.genpareto`. Assumes the tail is *exactly* GPD above `u`
  (an asymptotic theorem — the Pickands-Balkema-de Haan theorem — for large
  enough `u`; at finite `u` this is an approximation whose quality depends on
  the DGP). Shape parameter `xi = 1/alpha`. Delta-method CI from the observed
  Fisher information of the fit.
- **Gaussian extreme-quantile baseline (failure reference only).** Fit a
  Normal to the full sample, compute the target quantile as
  `mean + z_p * sd`. Included specifically because it is the wrong tool for
  this job; never reported as a serious tail-index competitor, and it has no
  associated tail-index estimate (a Gaussian has no tail index in this
  sense).

## Frozen threshold rule (headline results)

Predeclared before any result was inspected:

- **Hill:** `k = floor(n^0.5)` top order statistics.
- **GPD-POT:** `u` = the empirical 90th percentile of the sample.

These are standard, defensible, unremarkable choices — not tuned to make
either estimator look good. Their sensitivity is then measured directly and
reported honestly in the threshold-sensitivity diagnostic (below), which
sweeps `k` and `u` across a wide range for a fixed representative cell. The
headline bias/RMSE/quantile-error numbers use only the frozen rule.

## Metrics (predeclared)

For the tail-index estimate `alpha_hat` against the true `alpha`:
- Bias: `mean(alpha_hat - alpha)` over replications.
- RMSE: `sqrt(mean((alpha_hat - alpha)^2))`.
- Empirical coverage of the nominal 95% Wald/delta-method CI (should be near
  0.95 if the asymptotic theory is trustworthy at that `n`; systematic
  under-coverage is itself a finding, not an implementation bug, and is
  reported as one).

For the extreme quantile estimate `q_hat` against the true `q` (closed-form
for both DGPs) at levels `p in {0.99, 0.999, 0.9999}`:
- Relative error: `(q_hat - q) / q`.
- Median and interquartile range across replications (reported, not just the
  mean, because tail-quantile-error distributions are themselves skewed).

Threshold-sensitivity diagnostic (secondary, exploratory by design):
- For one fixed, representative cell per DGP (`n=2000`, mid tail index), sweep
  `k` (Hill) / `u`-as-percentile (GPD) across a grid and report the
  estimate's mean and spread across replications as a function of the
  threshold choice — this is the instability surface the frozen headline
  rule is a single slice through.

## Known limits, stated up front

- Parametric, i.i.d. samples only. No temporal dependence, volatility
  clustering, or regime change — real return series have both, and tail
  estimators behave differently under dependence. This study says nothing
  about that case.
- Only two DGP families. Lognormal, stable, and mixture tails are not
  covered and may behave differently.
- Threshold selection is frozen a priori for headline numbers, not chosen by
  a data-driven rule (e.g. minimizing AMSE, double bootstrap). Those
  adaptive methods are out of scope; the diagnostic exists precisely because
  a fixed rule is known to be imperfect.
- GPD MLE can fail to converge, especially at small `n` combined with a
  heavy tail. Failures are counted and reported, not silently dropped from
  the denominator.
- Reported CI coverage relies on asymptotic theory that is not guaranteed to
  hold at the smaller sample sizes in the grid; where coverage is poor, that
  is reported as a result about the estimator, not treated as an
  implementation defect to be hidden.

## References

- Hill, B. M. (1975). *A Simple General Approach to Inference About the Tail
  of a Distribution.* Annals of Statistics 3(5), 1163-1174.
- Pickands, J. (1975). *Statistical Inference Using Extreme Order
  Statistics.* Annals of Statistics 3(1), 119-131.
- Balkema, A. A. and de Haan, L. (1974). *Residual Life Time at Great Age.*
  Annals of Probability 2(5), 792-804.
- Embrechts, P., Kluppelberg, C., and Mikosch, T. (1997). *Modelling
  Extremal Events for Insurance and Finance.* Springer.
- Resnick, S. I. (2007). *Heavy-Tail Phenomena: Probabilistic and
  Statistical Modeling.* Springer. (Source of the "Hill horror plot"
  instability diagnostic used here.)
- de Haan, L. and Ferreira, A. (2006). *Extreme Value Theory: An
  Introduction.* Springer.

## Plain-language model

- **What the mechanism does:** generates data with a tail shape we already
  know exactly, then checks whether two standard estimators can recover that
  known shape from a finite sample, and how far a wrong-but-common shortcut
  (fit a Gaussian) misses it.
- **Why it is needed:** in the real world nobody knows the true tail index,
  so the only way to grade an estimator's honesty is to give it a case where
  the answer is already known and see how close it gets, and how that
  closeness degrades as the input choices (sample size, threshold) change.
- **One concrete example:** draw 2,000 points from a Pareto distribution with
  a true tail index of 3. Hill and GPD should recover something near 3; a
  Gaussian fit has no concept of "3" at all and will systematically
  understate how large the largest values can get.
- **What fails if it is incorrect:** if the estimators looked reliable here
  but the DGPs, thresholds, or metrics were chosen to flatter them, a reader
  would walk away trusting Hill/GPD estimates on real data more than the
  evidence supports.
- **What remains uncertain:** how these results carry over to dependent
  (non-i.i.d.) data, to DGPs outside this study's two families, and to
  threshold-selection rules more sophisticated than the frozen one used for
  headline numbers.
