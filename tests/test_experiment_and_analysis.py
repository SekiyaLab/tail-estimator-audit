import numpy as np

from tail_estimator_audit.analysis import aggregate_quantile_error, aggregate_tail_index
from tail_estimator_audit.experiment import run_grid, run_replication


def test_run_replication_is_deterministic_given_seed():
    row1 = run_replication("pareto", 3.0, 1000, rep=0, root_seed=42)
    row2 = run_replication("pareto", 3.0, 1000, rep=0, root_seed=42)
    assert row1 == row2


def test_run_replication_differs_across_reps():
    row1 = run_replication("pareto", 3.0, 1000, rep=0, root_seed=42)
    row2 = run_replication("pareto", 3.0, 1000, rep=1, root_seed=42)
    assert row1["hill_alpha_hat"] != row2["hill_alpha_hat"]


def test_run_grid_shape_and_seed_independence_of_order():
    df1 = run_grid(
        dgp_names=("pareto",), tail_indices=(3.0,), sample_sizes=(500, 1000), n_reps=5, verbose=False
    )
    # Running just the n=1000 slice in isolation should reproduce the same
    # rows as pulling n=1000 out of the two-n grid: seeding is keyed by the
    # full cell identity, not by iteration order.
    df2 = run_grid(dgp_names=("pareto",), tail_indices=(3.0,), sample_sizes=(1000,), n_reps=5, verbose=False)
    a = df1[df1["n"] == 1000].reset_index(drop=True)
    b = df2.reset_index(drop=True)
    np.testing.assert_allclose(a["hill_alpha_hat"], b["hill_alpha_hat"])


def test_aggregate_tail_index_recovers_known_bias_sign():
    # Hand-built two-cell "grid" where Hill is unbiased and GPD is biased by
    # a known constant, to check the aggregation arithmetic itself rather
    # than real estimator behaviour.
    import pandas as pd

    df = pd.DataFrame(
        [
            dict(
                dgp="pareto", tail_index=3.0, n=1000,
                true_alpha=3.0,
                hill_alpha_hat=3.0, hill_ci_low=2.5, hill_ci_high=3.5,
                gpd_alpha_hat=4.0, gpd_ci_low=3.5, gpd_ci_high=4.5,
            ),
            dict(
                dgp="pareto", tail_index=3.0, n=1000,
                true_alpha=3.0,
                hill_alpha_hat=3.0, hill_ci_low=2.5, hill_ci_high=3.5,
                gpd_alpha_hat=4.0, gpd_ci_low=3.5, gpd_ci_high=4.5,
            ),
        ]
    )
    agg = aggregate_tail_index(df)
    hill_row = agg[agg["method"] == "hill"].iloc[0]
    gpd_row = agg[agg["method"] == "gpd"].iloc[0]
    assert hill_row["bias"] == 0.0
    assert hill_row["ci_coverage"] == 1.0
    assert gpd_row["bias"] == 1.0
    assert gpd_row["ci_coverage"] == 0.0


def test_aggregate_quantile_error_matches_hand_computation():
    import pandas as pd

    df = pd.DataFrame(
        [
            dict(dgp="pareto", tail_index=3.0, n=1000, true_q_0999=10.0, hill_q_0999=11.0, gpd_q_0999=9.0, gaussian_q_0999=5.0),
            dict(dgp="pareto", tail_index=3.0, n=1000, true_q_0999=10.0, hill_q_0999=11.0, gpd_q_0999=9.0, gaussian_q_0999=5.0),
        ]
    )
    agg = aggregate_quantile_error(df, ("0999",))
    hill_row = agg[agg["method"] == "hill"].iloc[0]
    assert hill_row["median_rel_err"] == 0.1
    gaussian_row = agg[agg["method"] == "gaussian"].iloc[0]
    assert gaussian_row["median_rel_err"] == -0.5
