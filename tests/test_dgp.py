import numpy as np
import pytest

from tail_estimator_audit.dgp import (
    get_dgp,
    pareto_quantile,
    pareto_sample,
    student_t_quantile,
    student_t_sample,
)


def test_pareto_sample_domain():
    rng = np.random.default_rng(0)
    x = pareto_sample(rng, 10_000, alpha=3.0, x_m=1.0)
    assert np.all(x >= 1.0)


def test_pareto_sample_matches_closed_form_quantile():
    # Empirical quantile of a large Pareto sample should track the closed
    # form within Monte Carlo noise (not an estimator test: this checks the
    # generator itself, independent of Hill/GPD).
    rng = np.random.default_rng(1)
    x = pareto_sample(rng, 200_000, alpha=3.0, x_m=1.0)
    empirical = np.quantile(x, 0.99)
    closed_form = pareto_quantile(0.99, alpha=3.0, x_m=1.0)
    assert empirical == pytest.approx(closed_form, rel=0.05)


def test_pareto_rejects_nonpositive_alpha():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pareto_sample(rng, 10, alpha=0.0)


def test_student_t_sample_matches_closed_form_quantile():
    rng = np.random.default_rng(2)
    x = student_t_sample(rng, 200_000, nu=4.0)
    empirical = np.quantile(x, 0.99)
    closed_form = student_t_quantile(0.99, nu=4.0)
    assert empirical == pytest.approx(closed_form, rel=0.05)


def test_student_t_is_two_sided():
    rng = np.random.default_rng(3)
    x = student_t_sample(rng, 10_000, nu=5.0)
    assert (x < 0).sum() > 0
    assert (x > 0).sum() > 0


def test_get_dgp_registry_roundtrip():
    for name in ("pareto", "student_t"):
        dgp = get_dgp(name)
        assert dgp.name == name
        assert dgp.tail_index(3.0) == 3.0


def test_get_dgp_unknown_name_raises():
    with pytest.raises(ValueError):
        get_dgp("lognormal")


def test_seeded_reproducibility():
    dgp = get_dgp("pareto")
    x1 = dgp.sample(np.random.default_rng(123), 100, 3.0)
    x2 = dgp.sample(np.random.default_rng(123), 100, 3.0)
    np.testing.assert_array_equal(x1, x2)
