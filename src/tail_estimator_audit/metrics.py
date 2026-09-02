"""Predeclared scoring metrics: bias, RMSE, CI coverage, relative quantile error.

All functions take arrays of per-replication values and a scalar ground
truth. NaNs (from non-converged fits) are excluded explicitly and the drop
count is returned alongside the metric, so a failure rate never silently
shrinks a denominator without being visible in the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoredSample:
    n_used: int
    n_dropped: int
    bias: float
    rmse: float


def bias_rmse(estimates: np.ndarray, truth: float) -> ScoredSample:
    estimates = np.asarray(estimates, dtype=float)
    finite = np.isfinite(estimates)
    used = estimates[finite]
    n_dropped = int((~finite).sum())
    if used.size == 0:
        return ScoredSample(0, n_dropped, np.nan, np.nan)
    errors = used - truth
    return ScoredSample(
        n_used=int(used.size),
        n_dropped=n_dropped,
        bias=float(np.mean(errors)),
        rmse=float(np.sqrt(np.mean(errors**2))),
    )


def ci_coverage(ci_low: np.ndarray, ci_high: np.ndarray, truth: float) -> tuple[float, int, int]:
    """Empirical coverage of a nominal interval: fraction with truth inside.

    Returns ``(coverage, n_used, n_dropped)``.
    """
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)
    finite = np.isfinite(ci_low) & np.isfinite(ci_high)
    n_dropped = int((~finite).sum())
    lo, hi = ci_low[finite], ci_high[finite]
    if lo.size == 0:
        return np.nan, 0, n_dropped
    covered = (lo <= truth) & (truth <= hi)
    return float(np.mean(covered)), int(lo.size), n_dropped


def relative_error(estimates: np.ndarray, truth: float) -> np.ndarray:
    """Elementwise ``(estimate - truth) / truth``; NaNs pass through."""
    estimates = np.asarray(estimates, dtype=float)
    return (estimates - truth) / truth
