"""Parametric heavy-tailed data-generating processes with exactly known tail index.

Both DGPs are drawn from ``numpy.random.Generator`` under explicit seeds. No
external or real-world data is read anywhere in this module. Each DGP exposes
its true tail index and a closed-form quantile function, which is what makes
recovery error measurable at all: without ground truth there is nothing to
score an estimator against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DGP:
    """A named heavy-tailed generator with a known tail index.

    ``tail_param`` is the distribution's own parameter (Pareto's ``alpha``,
    Student-t's ``nu``). ``tail_index`` is that same value expressed as the
    regularly-varying tail index, which is the quantity Hill and GPD-POT
    estimate. For both DGPs used here the two coincide numerically, but they
    are kept as separate concepts because that coincidence is a property of
    these specific families, not a general law.
    """

    name: str
    sample: Callable[[np.random.Generator, int, float], np.ndarray]
    true_quantile: Callable[[float, float], float]
    tail_index: Callable[[float], float]


def pareto_sample(rng: np.random.Generator, n: int, alpha: float, x_m: float = 1.0) -> np.ndarray:
    """Draw ``n`` i.i.d. Pareto(alpha, x_m) variates.

    Density ``alpha * x_m**alpha / x**(alpha + 1)`` for ``x >= x_m``. Exact
    tail index is ``alpha`` at every threshold, since Pareto is itself a
    generalized Pareto tail everywhere above ``x_m``, not just asymptotically.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    u = rng.uniform(size=n)
    return x_m * (1.0 - u) ** (-1.0 / alpha)


def pareto_quantile(p: float, alpha: float, x_m: float = 1.0) -> float:
    """Closed-form Pareto quantile: ``x_m * (1 - p) ** (-1 / alpha)``."""
    return x_m * (1.0 - p) ** (-1.0 / alpha)


def student_t_sample(rng: np.random.Generator, n: int, nu: float) -> np.ndarray:
    """Draw ``n`` i.i.d. Student-t(nu) variates.

    Only the right tail is used downstream for tail-index / quantile
    recovery (the sample is not folded to ``|X|``), to avoid conflating two
    tails that happen to be identical by symmetry into a single effective
    sample size.
    """
    if nu <= 0:
        raise ValueError("nu must be positive")
    return stats.t.rvs(df=nu, size=n, random_state=rng)


def student_t_quantile(p: float, nu: float) -> float:
    """Closed-form Student-t quantile via ``scipy.stats.t.ppf``."""
    return float(stats.t.ppf(p, df=nu))


def _identity(param: float) -> float:
    return param


PARETO = DGP(
    name="pareto",
    sample=pareto_sample,
    true_quantile=pareto_quantile,
    tail_index=_identity,
)

STUDENT_T = DGP(
    name="student_t",
    sample=student_t_sample,
    true_quantile=student_t_quantile,
    tail_index=_identity,
)

DGPS: dict[str, DGP] = {PARETO.name: PARETO, STUDENT_T.name: STUDENT_T}


def get_dgp(name: str) -> DGP:
    try:
        return DGPS[name]
    except KeyError as exc:
        raise ValueError(f"unknown DGP {name!r}; choices are {sorted(DGPS)}") from exc
