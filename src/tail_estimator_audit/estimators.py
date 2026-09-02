"""Tail-index and extreme-quantile estimators under audit.

Two tail-aware estimators (Hill, GPD peaks-over-threshold MLE) and one
deliberately naive baseline (Gaussian) are implemented here. The Gaussian
baseline is a failure reference: it has no notion of a tail index at all and
is included only so its extreme-quantile error can be compared against the
tail-aware estimators', never as a serious competitor for tail-index
recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

Z_975 = float(stats.norm.ppf(0.975))  # nominal 95% Wald / delta-method CI


@dataclass(frozen=True)
class HillFit:
    k: int
    threshold: float
    alpha_hat: float
    se: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class GPDFit:
    threshold: float
    n_exceed: int
    xi_hat: float
    sigma_hat: float
    alpha_hat: float
    se_alpha: float
    ci_low: float
    ci_high: float
    converged: bool


def hill_estimator(x: np.ndarray, k: int) -> HillFit:
    """Hill (1975) tail-index estimator using the top ``k`` order statistics.

    Assumes only regular variation of the right tail. ``k`` must satisfy
    ``1 <= k < n``. Standard error is the classical asymptotic
    ``alpha_hat / sqrt(k)``, valid as k, n -> infinity with k/n -> 0.
    """
    n = x.size
    if not (1 <= k < n):
        raise ValueError(f"k must be in [1, n), got k={k}, n={n}")
    x_asc = np.sort(x)
    top_k = x_asc[n - k :]  # k largest values, ascending
    threshold = x_asc[n - k - 1]  # the (k+1)-th largest: X_(n-k)
    if threshold <= 0:
        raise ValueError("Hill estimator requires strictly positive threshold")
    log_ratios = np.log(top_k) - np.log(threshold)
    gamma_hat = float(np.mean(log_ratios))
    alpha_hat = 1.0 / gamma_hat
    se = alpha_hat / np.sqrt(k)
    return HillFit(
        k=k,
        threshold=float(threshold),
        alpha_hat=alpha_hat,
        se=se,
        ci_low=alpha_hat - Z_975 * se,
        ci_high=alpha_hat + Z_975 * se,
    )


def hill_quantile(x: np.ndarray, k: int, q: float) -> float:
    """Weissman (1978) extreme-quantile extrapolation built on the Hill fit.

    ``q`` is the target quantile level (e.g. 0.999). Extrapolates beyond the
    observed sample using the Hill tail-index estimate; only reliable when
    ``1 - q`` is not vastly smaller than ``k / n``.
    """
    n = x.size
    fit = hill_estimator(x, k)
    p = 1.0 - q
    return fit.threshold * (k / (n * p)) ** (1.0 / fit.alpha_hat)


def gpd_pot_fit(x: np.ndarray, threshold: float) -> GPDFit:
    """Peaks-over-threshold GPD fit by maximum likelihood.

    Assumes the excess distribution above ``threshold`` is exactly
    generalized Pareto — a theorem (Pickands; Balkema-de Haan) that holds
    only asymptotically as the threshold grows, so at any finite threshold
    this is an approximation whose accuracy depends on the DGP.

    Standard errors use the analytic asymptotic covariance of the GPD MLE
    (Smith, 1987), valid for shape ``xi > -1/2``; not the delta method
    applied to a numerically differentiated likelihood.
    """
    exceedances = x[x > threshold] - threshold
    n_u = exceedances.size
    if n_u < 10:
        return GPDFit(threshold, n_u, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)
    try:
        xi_hat, _loc, sigma_hat = stats.genpareto.fit(exceedances, floc=0)
    except Exception:
        return GPDFit(threshold, n_u, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)

    if not np.isfinite(xi_hat) or not np.isfinite(sigma_hat) or sigma_hat <= 0:
        return GPDFit(threshold, n_u, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)

    if xi_hat <= 0:
        # A non-positive shape means this fit did not recover a heavy
        # (regularly varying) tail at all; alpha = 1/xi is undefined or
        # non-positive. Reported as a non-converged heavy-tail recovery,
        # not silently coerced into a number.
        return GPDFit(threshold, n_u, xi_hat, sigma_hat, np.nan, np.nan, np.nan, np.nan, False)

    alpha_hat = 1.0 / xi_hat
    if xi_hat > -0.5:
        se_xi = np.sqrt((1.0 + xi_hat) ** 2 / n_u)
        se_alpha = se_xi / xi_hat**2  # delta method: d(1/xi)/dxi = -1/xi^2
    else:
        se_alpha = np.nan

    return GPDFit(
        threshold=threshold,
        n_exceed=n_u,
        xi_hat=float(xi_hat),
        sigma_hat=float(sigma_hat),
        alpha_hat=alpha_hat,
        se_alpha=float(se_alpha),
        ci_low=alpha_hat - Z_975 * se_alpha if np.isfinite(se_alpha) else np.nan,
        ci_high=alpha_hat + Z_975 * se_alpha if np.isfinite(se_alpha) else np.nan,
        converged=True,
    )


def gpd_quantile(x: np.ndarray, threshold: float, q: float) -> float:
    """Extreme quantile from the GPD-POT fit at level ``q``.

    ``x_q = u + (sigma / xi) * ((p / zeta_u) ** (-xi) - 1)`` where
    ``zeta_u`` is the empirical exceedance rate over ``threshold`` and
    ``p = 1 - q`` (Coles, 2001, eq. 4.13). Returns ``nan`` if the underlying
    fit did not converge to a heavy-tailed shape.
    """
    fit = gpd_pot_fit(x, threshold)
    if not fit.converged:
        return np.nan
    n = x.size
    zeta_u = fit.n_exceed / n
    p = 1.0 - q
    return threshold + (fit.sigma_hat / fit.xi_hat) * ((p / zeta_u) ** (-fit.xi_hat) - 1.0)


def gaussian_quantile(x: np.ndarray, q: float) -> float:
    """Naive Gaussian extreme-quantile baseline: ``mean + z_q * sd``.

    Deliberately the wrong tool for a heavy-tailed target: included only to
    show how badly a normal approximation misjudges extreme quantiles, never
    as a tail-index competitor (a Gaussian has no tail index in this sense).
    """
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    z = float(stats.norm.ppf(q))
    return mean + z * sd
