"""Main Monte Carlo grid: tail-index and extreme-quantile recovery.

Deterministic and seeded end to end: every (DGP, tail index, n, replication)
cell draws from an independently seeded ``numpy.random.Generator`` derived
from one root seed via ``numpy.random.SeedSequence``, so the entire grid is
reproducible from a single integer with no dependence on iteration order.

The threshold rule used here (``k = floor(sqrt(n))`` for Hill, the empirical
90th percentile for GPD-POT) is the frozen rule from
``docs/research_contract.md``, fixed before any result was inspected. Its
sensitivity to that choice is measured separately in
``threshold_diagnostics.py``, not tuned away here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tail_estimator_audit.dgp import get_dgp
from tail_estimator_audit.estimators import (
    gaussian_quantile,
    gpd_pot_fit,
    gpd_quantile,
    hill_estimator,
    hill_quantile,
)

ROOT_SEED = 1729  # frozen; reused by threshold_diagnostics.py for a shared lineage

DGP_NAMES = ("pareto", "student_t")
TAIL_INDICES = (2.0, 3.0, 4.0, 6.0)
SAMPLE_SIZES = (500, 1000, 2000, 5000, 10000)
N_REPS = 400
QUANTILE_LEVELS = (0.99, 0.999, 0.9999)

def hill_k(n: int) -> int:
    """Frozen Hill threshold rule: top ``floor(sqrt(n))`` order statistics."""
    return int(np.floor(np.sqrt(n)))


GPD_THRESHOLD_PCT = 0.90


def _child_rng(root_seed: int, dgp_name: str, tail_index: float, n: int, rep: int) -> np.random.Generator:
    """Deterministic, order-independent RNG for one grid cell.

    Encodes every axis of the grid cell into the seed sequence entropy so
    that any single cell can be regenerated in isolation and always matches
    a full-grid run.
    """
    dgp_id = DGP_NAMES.index(dgp_name)
    tail_id = int(round(tail_index * 1000))
    seq = np.random.SeedSequence([root_seed, dgp_id, tail_id, n, rep])
    return np.random.default_rng(seq)


def run_replication(
    dgp_name: str,
    tail_index: float,
    n: int,
    rep: int,
    root_seed: int = ROOT_SEED,
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> dict:
    dgp = get_dgp(dgp_name)
    rng = _child_rng(root_seed, dgp_name, tail_index, n, rep)
    x = dgp.sample(rng, n, tail_index)
    true_alpha = dgp.tail_index(tail_index)

    k = hill_k(n)
    u = float(np.quantile(x, GPD_THRESHOLD_PCT))

    row: dict = dict(
        dgp=dgp_name,
        tail_index=tail_index,
        n=n,
        rep=rep,
        true_alpha=true_alpha,
    )

    try:
        hf = hill_estimator(x, k)
        row.update(
            hill_k=hf.k,
            hill_threshold=hf.threshold,
            hill_alpha_hat=hf.alpha_hat,
            hill_se=hf.se,
            hill_ci_low=hf.ci_low,
            hill_ci_high=hf.ci_high,
        )
    except ValueError:
        row.update(
            hill_k=k,
            hill_threshold=np.nan,
            hill_alpha_hat=np.nan,
            hill_se=np.nan,
            hill_ci_low=np.nan,
            hill_ci_high=np.nan,
        )

    gf = gpd_pot_fit(x, u)
    row.update(
        gpd_threshold=gf.threshold,
        gpd_n_exceed=gf.n_exceed,
        gpd_xi_hat=gf.xi_hat,
        gpd_sigma_hat=gf.sigma_hat,
        gpd_alpha_hat=gf.alpha_hat,
        gpd_se_alpha=gf.se_alpha,
        gpd_ci_low=gf.ci_low,
        gpd_ci_high=gf.ci_high,
        gpd_converged=gf.converged,
    )

    for q in quantile_levels:
        tag = str(q).replace(".", "")
        row[f"true_q_{tag}"] = dgp.true_quantile(q, tail_index)
        try:
            row[f"hill_q_{tag}"] = hill_quantile(x, k, q)
        except ValueError:
            row[f"hill_q_{tag}"] = np.nan
        row[f"gpd_q_{tag}"] = gpd_quantile(x, u, q)
        row[f"gaussian_q_{tag}"] = gaussian_quantile(x, q)

    return row


def run_grid(
    dgp_names: tuple[str, ...] = DGP_NAMES,
    tail_indices: tuple[float, ...] = TAIL_INDICES,
    sample_sizes: tuple[int, ...] = SAMPLE_SIZES,
    n_reps: int = N_REPS,
    root_seed: int = ROOT_SEED,
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    total_cells = len(dgp_names) * len(tail_indices) * len(sample_sizes)
    cell = 0
    t0 = time.time()
    for dgp_name in dgp_names:
        for tail_index in tail_indices:
            for n in sample_sizes:
                cell += 1
                for rep in range(n_reps):
                    rows.append(
                        run_replication(dgp_name, tail_index, n, rep, root_seed, quantile_levels)
                    )
                if verbose:
                    elapsed = time.time() - t0
                    print(
                        f"[{cell}/{total_cells}] {dgp_name} tail_index={tail_index} n={n} "
                        f"reps={n_reps} done ({elapsed:.1f}s elapsed)",
                        flush=True,
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/main_grid.csv"))
    parser.add_argument("--n-reps", type=int, default=N_REPS)
    parser.add_argument("--seed", type=int, default=ROOT_SEED)
    args = parser.parse_args()

    df = run_grid(n_reps=args.n_reps, root_seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_gpd_fail = int((~df["gpd_converged"]).sum())
    n_hill_fail = int(df["hill_alpha_hat"].isna().sum())
    print(f"wrote {len(df)} rows to {args.out}")
    print(f"GPD-POT non-convergence: {n_gpd_fail}/{len(df)} ({n_gpd_fail / len(df):.2%})")
    print(f"Hill failures: {n_hill_fail}/{len(df)} ({n_hill_fail / len(df):.2%})")


if __name__ == "__main__":
    main()
