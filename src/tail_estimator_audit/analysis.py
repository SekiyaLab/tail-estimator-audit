"""Aggregation of raw per-replication results into the predeclared metrics.

Reads the raw grid produced by ``experiment.py`` (one row per Monte Carlo
replication) and reduces it to bias/RMSE/coverage and quantile relative-error
tables, grouped by (DGP, tail index, sample size). Kept separate from
plotting: this module has no matplotlib dependency, so the metrics themselves
can be tested and consumed without a display backend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tail_estimator_audit.metrics import bias_rmse, ci_coverage, relative_error

TAIL_INDEX_METHODS = ("hill", "gpd")
QUANTILE_METHODS = ("hill", "gpd", "gaussian")


def aggregate_tail_index(df: pd.DataFrame) -> pd.DataFrame:
    """Bias, RMSE, and 95% CI coverage of alpha_hat, by DGP/method/cell."""
    rows = []
    for (dgp, tail_index, n), group in df.groupby(["dgp", "tail_index", "n"]):
        truth = group["true_alpha"].iloc[0]
        for method in TAIL_INDEX_METHODS:
            est_col = f"{method}_alpha_hat"
            lo_col = f"{method}_ci_low"
            hi_col = f"{method}_ci_high"
            scored = bias_rmse(group[est_col].to_numpy(), truth)
            coverage, cov_used, cov_dropped = ci_coverage(
                group[lo_col].to_numpy(), group[hi_col].to_numpy(), truth
            )
            rows.append(
                dict(
                    dgp=dgp,
                    method=method,
                    tail_index=tail_index,
                    n=n,
                    n_reps=len(group),
                    n_used=scored.n_used,
                    n_dropped=scored.n_dropped,
                    bias=scored.bias,
                    rmse=scored.rmse,
                    ci_coverage=coverage,
                    ci_n_used=cov_used,
                    ci_n_dropped=cov_dropped,
                )
            )
    return pd.DataFrame(rows)


def aggregate_quantile_error(df: pd.DataFrame, quantile_tags: tuple[str, ...]) -> pd.DataFrame:
    """Median and IQR of relative extreme-quantile error, by DGP/method/cell/q."""
    rows = []
    for (dgp, tail_index, n), group in df.groupby(["dgp", "tail_index", "n"]):
        for tag in quantile_tags:
            truth = group[f"true_q_{tag}"].iloc[0]
            for method in QUANTILE_METHODS:
                est_col = f"{method}_q_{tag}"
                rel_err = relative_error(group[est_col].to_numpy(), truth)
                finite = rel_err[np.isfinite(rel_err)]
                n_dropped = int(np.size(rel_err) - finite.size)
                if finite.size == 0:
                    median = q25 = q75 = np.nan
                else:
                    median = float(np.median(finite))
                    q25 = float(np.percentile(finite, 25))
                    q75 = float(np.percentile(finite, 75))
                rows.append(
                    dict(
                        dgp=dgp,
                        method=method,
                        tail_index=tail_index,
                        n=n,
                        q_tag=tag,
                        n_used=int(finite.size),
                        n_dropped=n_dropped,
                        median_rel_err=median,
                        q25_rel_err=q25,
                        q75_rel_err=q75,
                    )
                )
    return pd.DataFrame(rows)
