import numpy as np
import pytest

from tail_estimator_audit.dgp import get_dgp, pareto_quantile
from tail_estimator_audit.estimators import (
    gaussian_quantile,
    gpd_pot_fit,
    gpd_quantile,
    hill_estimator,
    hill_quantile,
)


@pytest.fixture
def pareto_sample_large():
    rng = np.random.default_rng(7)
    dgp = get_dgp("pareto")
    return dgp.sample(rng, 20_000, 3.0)


def test_hill_recovers_pareto_alpha_at_large_n(pareto_sample_large):
    fit = hill_estimator(pareto_sample_large, k=1000)
    assert fit.alpha_hat == pytest.approx(3.0, abs=0.3)
    assert fit.ci_low < 3.0 < fit.ci_high


def test_hill_rejects_out_of_range_k():
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        hill_estimator(x, k=0)
    with pytest.raises(ValueError):
        hill_estimator(x, k=3)


def test_hill_rejects_nonpositive_threshold():
    # A sample where the (k+1)-th largest value is non-positive breaks the
    # log-ratio construction; this must raise rather than emit a silent NaN.
    x = np.array([-5.0, -3.0, -1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        hill_estimator(x, k=4)


def test_gpd_recovers_pareto_shape_at_large_n(pareto_sample_large):
    u = float(np.quantile(pareto_sample_large, 0.90))
    fit = gpd_pot_fit(pareto_sample_large, u)
    assert fit.converged
    assert fit.alpha_hat == pytest.approx(3.0, abs=0.5)
    assert fit.ci_low < 3.0 < fit.ci_high


def test_gpd_reports_nonconvergence_on_tiny_exceedance_count():
    rng = np.random.default_rng(0)
    dgp = get_dgp("pareto")
    x = dgp.sample(rng, 50, 3.0)
    u = float(np.max(x)) - 1e-9  # threshold above nearly everything: too few exceedances
    fit = gpd_pot_fit(x, u)
    assert not fit.converged
    assert np.isnan(fit.alpha_hat)


def test_hill_quantile_close_to_true_at_large_n(pareto_sample_large):
    q_hat = hill_quantile(pareto_sample_large, k=1000, q=0.999)
    q_true = pareto_quantile(0.999, alpha=3.0)
    assert q_hat == pytest.approx(q_true, rel=0.25)


def test_gpd_quantile_close_to_true_at_large_n(pareto_sample_large):
    u = float(np.quantile(pareto_sample_large, 0.90))
    q_hat = gpd_quantile(pareto_sample_large, u, q=0.999)
    q_true = pareto_quantile(0.999, alpha=3.0)
    assert q_hat == pytest.approx(q_true, rel=0.35)


def test_gaussian_quantile_understates_heavy_tail(pareto_sample_large):
    q_true = pareto_quantile(0.999, alpha=3.0)
    q_gauss = gaussian_quantile(pareto_sample_large, q=0.999)
    # The point of the baseline: it is not merely imprecise, it is wrong in
    # a specific, large, one-sided direction for a heavy right tail.
    assert q_gauss < 0.7 * q_true
