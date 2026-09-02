import numpy as np

from tail_estimator_audit.metrics import bias_rmse, ci_coverage, relative_error


def test_bias_rmse_basic():
    estimates = np.array([2.0, 3.0, 4.0])
    scored = bias_rmse(estimates, truth=3.0)
    assert scored.n_used == 3
    assert scored.n_dropped == 0
    assert scored.bias == 0.0
    assert scored.rmse == np.sqrt((1.0 + 0.0 + 1.0) / 3)


def test_bias_rmse_drops_nans_and_counts_them():
    estimates = np.array([2.0, np.nan, 4.0, np.nan])
    scored = bias_rmse(estimates, truth=3.0)
    assert scored.n_used == 2
    assert scored.n_dropped == 2
    assert scored.bias == 0.0


def test_bias_rmse_all_nan_returns_nan_not_crash():
    scored = bias_rmse(np.array([np.nan, np.nan]), truth=3.0)
    assert scored.n_used == 0
    assert scored.n_dropped == 2
    assert np.isnan(scored.bias)
    assert np.isnan(scored.rmse)


def test_ci_coverage_all_covered():
    lo = np.array([1.0, 1.5, 2.0])
    hi = np.array([5.0, 5.5, 6.0])
    coverage, n_used, n_dropped = ci_coverage(lo, hi, truth=3.0)
    assert coverage == 1.0
    assert n_used == 3
    assert n_dropped == 0


def test_ci_coverage_partial_and_nan_handling():
    lo = np.array([1.0, 2.5, np.nan])
    hi = np.array([2.0, 3.5, np.nan])
    coverage, n_used, n_dropped = ci_coverage(lo, hi, truth=3.0)
    assert n_used == 2
    assert n_dropped == 1
    # [1, 2] misses 3.0; [2.5, 3.5] covers it.
    assert coverage == 0.5


def test_relative_error_basic():
    estimates = np.array([9.0, 11.0, np.nan])
    rel = relative_error(estimates, truth=10.0)
    np.testing.assert_allclose(rel[:2], [-0.1, 0.1])
    assert np.isnan(rel[2])
