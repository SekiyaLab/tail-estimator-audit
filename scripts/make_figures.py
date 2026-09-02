#!/usr/bin/env python
"""Render the four report figures from the raw results CSVs.

Presentation only: no estimation logic lives here. Reads
``results/main_grid.csv`` and ``results/threshold_sensitivity_*.csv``
(produced by ``run_experiments.py`` / ``run_threshold_diagnostics.py``) and
writes PNGs to ``figures/``.

Palette: fixed categorical order (blue = slot 1, orange = slot 2) reused
consistently across figures for the same identity (Pareto = blue,
Student-t = orange throughout; Hill = blue, GPD = orange in the quantile
figure). The Gaussian baseline is drawn as a dashed muted-gray line rather
than a fourth categorical color, on purpose: it is a failure reference, not
a peer competitor, and its styling should not suggest otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tail_estimator_audit.analysis import aggregate_quantile_error, aggregate_tail_index  # noqa: E402
from tail_estimator_audit.threshold_diagnostics import REPRESENTATIVE_TAIL_INDEX  # noqa: E402

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_FILL = "#cde2fb"
ORANGE_FILL = "#f8d3c2"

DGP_COLOR = {"pareto": BLUE, "student_t": ORANGE}
DGP_FILL = {"pareto": BLUE_FILL, "student_t": ORANGE_FILL}
DGP_LABEL = {"pareto": "Pareto", "student_t": "Student-t"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK_PRIMARY,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.titlecolor": INK_PRIMARY,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def fig_hill_plot(results_dir: Path, out_path: Path) -> None:
    """Figure 1: classic Hill-plot instability, one sample and the reps around it."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, dgp in zip(axes, ("pareto", "student_t")):
        df = pd.read_csv(results_dir / f"threshold_sensitivity_hill_{dgp}.csv")
        color, fill = DGP_COLOR[dgp], DGP_FILL[dgp]
        pivot = df.pivot(index="rep", columns="k", values="alpha_hat")
        ks = pivot.columns.to_numpy()
        for rep in pivot.index[:15]:
            ax.plot(ks, pivot.loc[rep], color=color, alpha=0.15, linewidth=0.9)
        mean_line = pivot.mean(axis=0)
        q25 = pivot.quantile(0.25, axis=0)
        q75 = pivot.quantile(0.75, axis=0)
        ax.fill_between(ks, q25, q75, color=fill, alpha=0.6, linewidth=0)
        ax.plot(ks, mean_line, color=color, linewidth=2.2, label=f"{DGP_LABEL[dgp]} (mean of 200 reps)")
        ax.axhline(REPRESENTATIVE_TAIL_INDEX, color=INK_MUTED, linestyle="--", linewidth=1.2, label="true tail index")
        ax.set_xscale("log")
        ax.set_xlabel("k (top order statistics used)")
        ax.set_title(f"{DGP_LABEL[dgp]}, n=2000, true tail index={REPRESENTATIVE_TAIL_INDEX:.0f}")
        ax.grid(True, axis="y", linewidth=0.8)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel(r"Hill estimate $\hat{\alpha}$")
    fig.suptitle("Figure 1. Hill-plot instability: the estimate depends visibly on k", y=1.02)
    fig.text(
        0.5, -0.03,
        "Faint lines: 15 individual replications. Bold line: mean across 200 replications, shaded band: IQR.",
        ha="center", color=INK_MUTED, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_bias_rmse_heatmaps(results_dir: Path, out_path: Path) -> None:
    """Figure 2: bias and RMSE of tail-index recovery across the full grid."""
    df = pd.read_csv(results_dir / "main_grid.csv")
    agg = aggregate_tail_index(df)

    dgps = ["pareto", "student_t"]
    methods = ["hill", "gpd"]
    method_label = {"hill": "Hill", "gpd": "GPD-POT"}
    tail_indices = sorted(agg["tail_index"].unique())
    sample_sizes = sorted(agg["n"].unique())

    # Color scale capped at the 90th percentile, not the max: a handful of
    # GPD-POT MLE blow-ups at small n / heavy tail (a genuine finding, see
    # README) would otherwise saturate the whole colormap toward one extreme
    # cell and make every other cell look uniformly near-zero. The exact
    # value is still printed as text in every cell regardless of capping.
    bias_max = float(agg["bias"].abs().quantile(0.90))
    rmse_max = float(agg["rmse"].quantile(0.90))

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    bias_im = rmse_im = None
    for row, dgp in enumerate(dgps):
        for col, (method, metric) in enumerate(
            [("hill", "bias"), ("hill", "rmse"), ("gpd", "bias"), ("gpd", "rmse")]
        ):
            ax = axes[row, col]
            sub = agg[(agg["dgp"] == dgp) & (agg["method"] == method)]
            mat = sub.pivot(index="tail_index", columns="n", values=metric).reindex(
                index=tail_indices, columns=sample_sizes
            )
            if metric == "bias":
                im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, aspect="auto")
                bias_im = im
            else:
                im = ax.imshow(mat.to_numpy(), cmap="Blues", vmin=0, vmax=rmse_max, aspect="auto")
                rmse_im = im
            ax.set_xticks(range(len(sample_sizes)), [str(n) for n in sample_sizes], fontsize=8)
            ax.set_yticks(range(len(tail_indices)), [str(a) for a in tail_indices], fontsize=8)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat.to_numpy()[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=INK_PRIMARY)
            title = f"{DGP_LABEL[dgp]} | {method_label[method]} {metric.upper()}"
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel("tail index")
            if row == 1:
                ax.set_xlabel("n")
    fig.subplots_adjust(right=0.90, wspace=0.4, hspace=0.35)
    # Two colorbars pinned to the far right edge in their own reserved axes,
    # rather than interleaved between subplot columns: with non-contiguous
    # source columns (bias in 0/2, RMSE in 1/3), fig.colorbar(ax=...) placed
    # the colorbar in the gap between columns, overlapping neighboring cells.
    cax_bias = fig.add_axes((0.92, 0.55, 0.015, 0.32))
    cax_rmse = fig.add_axes((0.92, 0.13, 0.015, 0.32))
    fig.colorbar(bias_im, cax=cax_bias, label="bias")
    fig.colorbar(rmse_im, cax=cax_rmse, label="RMSE")
    fig.suptitle("Figure 2. Tail-index bias and RMSE across tail-index x sample-size regimes", y=1.03)
    fig.text(
        0.5, -0.02,
        "Color scale capped at the 90th percentile of |bias| / RMSE so a few GPD-POT MLE blow-ups at small n "
        "and heavy tails (see README) do not saturate the whole colormap; exact values are printed in every cell.",
        ha="center", color=INK_MUTED, fontsize=8,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_quantile_error(results_dir: Path, out_path: Path, q_tag: str = "0999") -> None:
    """Figure 3: extreme-quantile relative error, with IQR, vs sample size."""
    df = pd.read_csv(results_dir / "main_grid.csv")
    df = df[df["tail_index"] == REPRESENTATIVE_TAIL_INDEX]
    agg = aggregate_quantile_error(df, (q_tag,))
    agg = agg[agg["q_tag"] == q_tag]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    method_style = {
        "hill": dict(color=BLUE, linestyle="-", label="Hill-Weissman"),
        "gpd": dict(color=ORANGE, linestyle="-", label="GPD-POT"),
        "gaussian": dict(color=INK_MUTED, linestyle="--", label="Gaussian (failure reference)"),
    }
    for ax, dgp in zip(axes, ("pareto", "student_t")):
        sub = agg[agg["dgp"] == dgp].sort_values("n")
        for method, style in method_style.items():
            m = sub[sub["method"] == method]
            ax.plot(m["n"], m["median_rel_err"], marker="o", markersize=4, **style)
            ax.fill_between(m["n"], m["q25_rel_err"], m["q75_rel_err"], color=style["color"], alpha=0.15, linewidth=0)
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_title(f"{DGP_LABEL[dgp]}, tail index={REPRESENTATIVE_TAIL_INDEX:.0f}")
        ax.grid(True, axis="y", linewidth=0.8)
    axes[0].set_ylabel("relative error, (q_hat - q_true) / q_true")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Figure 3. Extreme-quantile error at q=0.999, median and IQR over 400 replications", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_threshold_sensitivity(results_dir: Path, out_path: Path) -> None:
    """Figure 4: robustness diagnostic — alpha_hat vs threshold choice, both estimators."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    for dgp in ("pareto", "student_t"):
        df = pd.read_csv(results_dir / f"threshold_sensitivity_hill_{dgp}.csv")
        stats = df.groupby("k")["alpha_hat"].agg(mean="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        color, fill = DGP_COLOR[dgp], DGP_FILL[dgp]
        ax.plot(stats.index, stats["mean"], color=color, linewidth=2, label=DGP_LABEL[dgp])
        ax.fill_between(stats.index, stats["q25"], stats["q75"], color=fill, alpha=0.6, linewidth=0)
    ax.axhline(REPRESENTATIVE_TAIL_INDEX, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("k (top order statistics)")
    ax.set_ylabel(r"$\hat{\alpha}$")
    ax.set_title("Hill: small k -> high variance,\nlarge k -> bias toward bulk", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", linewidth=0.8)

    ax = axes[1]
    for dgp in ("pareto", "student_t"):
        df = pd.read_csv(results_dir / f"threshold_sensitivity_gpd_{dgp}.csv")
        df = df[df["converged"]]
        stats = df.groupby("u_pct")["alpha_hat"].agg(mean="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        color, fill = DGP_COLOR[dgp], DGP_FILL[dgp]
        ax.plot(stats.index, stats["mean"], color=color, linewidth=2, label=DGP_LABEL[dgp])
        ax.fill_between(stats.index, stats["q25"], stats["q75"], color=fill, alpha=0.6, linewidth=0)
    ax.axhline(REPRESENTATIVE_TAIL_INDEX, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.set_xlabel("threshold percentile u")
    ax.set_title("GPD-POT: low u -> more bias (assumption strained),\nhigh u -> high variance (fewer exceedances)", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", linewidth=0.8)

    fig.suptitle(
        f"Figure 4. Threshold-sensitivity diagnostic (n={2000}, true tail index={REPRESENTATIVE_TAIL_INDEX:.0f}); "
        "dashed line: frozen headline rule is one slice through this surface",
        y=1.06, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    fig_hill_plot(args.results_dir, args.figures_dir / "hill_plot.png")
    print("wrote figures/hill_plot.png")
    fig_bias_rmse_heatmaps(args.results_dir, args.figures_dir / "bias_rmse_heatmaps.png")
    print("wrote figures/bias_rmse_heatmaps.png")
    fig_quantile_error(args.results_dir, args.figures_dir / "quantile_error.png")
    print("wrote figures/quantile_error.png")
    fig_threshold_sensitivity(args.results_dir, args.figures_dir / "threshold_sensitivity.png")
    print("wrote figures/threshold_sensitivity.png")


if __name__ == "__main__":
    main()
