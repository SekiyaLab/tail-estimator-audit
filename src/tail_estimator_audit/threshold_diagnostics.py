"""Threshold-sensitivity diagnostic: the instability the frozen rule hides.

``experiment.py`` reports headline numbers at one predeclared threshold per
estimator. This module answers a different, honest question: how much does
the tail-index estimate move if that single threshold choice had been
slightly different? For one fixed, representative (DGP, tail index, n) cell
per DGP, both estimators are swept across a wide range of threshold choices,
reusing the same underlying samples across the sweep so the only thing that
changes is the threshold.

This is also the source data for the classic Hill-plot instability figure
(replication 0 of the sweep, plotted directly rather than averaged).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tail_estimator_audit.dgp import get_dgp
from tail_estimator_audit.estimators import gpd_pot_fit, hill_estimator
from tail_estimator_audit.experiment import DGP_NAMES, ROOT_SEED

REPRESENTATIVE_TAIL_INDEX = 3.0
REPRESENTATIVE_N = 2000
N_REPS_DIAG = 200

# k values swept for Hill, log-spaced across the usable range (k must be < n).
K_GRID = np.unique(np.geomspace(10, REPRESENTATIVE_N // 2, 40).astype(int))
# Threshold percentiles swept for GPD-POT.
U_PCT_GRID = np.round(np.linspace(0.50, 0.99, 40), 4)


DIAG_TAG = 9999  # distinct lineage from the main grid's cell seeding


def _diag_rng(dgp_name: str, rep: int) -> np.random.Generator:
    # Deterministic, independent of PYTHONHASHSEED (unlike Python's builtin
    # hash() for strings, which is randomized per process by default).
    dgp_id = DGP_NAMES.index(dgp_name)
    seq = np.random.SeedSequence([ROOT_SEED, DIAG_TAG, dgp_id, rep])
    return np.random.default_rng(seq)


def sweep_hill(dgp_name: str, n_reps: int = N_REPS_DIAG) -> pd.DataFrame:
    dgp = get_dgp(dgp_name)
    rows = []
    for rep in range(n_reps):
        rng = _diag_rng(dgp_name, rep)
        x = dgp.sample(rng, REPRESENTATIVE_N, REPRESENTATIVE_TAIL_INDEX)
        for k in K_GRID:
            try:
                fit = hill_estimator(x, int(k))
                alpha_hat = fit.alpha_hat
            except ValueError:
                alpha_hat = np.nan
            rows.append(dict(dgp=dgp_name, rep=rep, k=int(k), alpha_hat=alpha_hat))
    return pd.DataFrame(rows)


def sweep_gpd(dgp_name: str, n_reps: int = N_REPS_DIAG) -> pd.DataFrame:
    dgp = get_dgp(dgp_name)
    rows = []
    for rep in range(n_reps):
        rng = _diag_rng(dgp_name, rep)
        x = dgp.sample(rng, REPRESENTATIVE_N, REPRESENTATIVE_TAIL_INDEX)
        for pct in U_PCT_GRID:
            u = float(np.quantile(x, pct))
            fit = gpd_pot_fit(x, u)
            rows.append(
                dict(
                    dgp=dgp_name,
                    rep=rep,
                    u_pct=float(pct),
                    n_exceed=fit.n_exceed,
                    alpha_hat=fit.alpha_hat,
                    converged=fit.converged,
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-reps", type=int, default=N_REPS_DIAG)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for dgp_name in ("pareto", "student_t"):
        hill_df = sweep_hill(dgp_name, args.n_reps)
        gpd_df = sweep_gpd(dgp_name, args.n_reps)
        hill_path = args.out_dir / f"threshold_sensitivity_hill_{dgp_name}.csv"
        gpd_path = args.out_dir / f"threshold_sensitivity_gpd_{dgp_name}.csv"
        hill_df.to_csv(hill_path, index=False)
        gpd_df.to_csv(gpd_path, index=False)
        gpd_fail = int((~gpd_df["converged"]).sum())
        print(f"{dgp_name}: wrote {hill_path} ({len(hill_df)} rows), {gpd_path} ({len(gpd_df)} rows)")
        print(f"{dgp_name}: GPD non-convergence across sweep: {gpd_fail}/{len(gpd_df)} ({gpd_fail / len(gpd_df):.2%})")


if __name__ == "__main__":
    main()
