"""Scipy-based optimization algorithm wrappers.

ScipyMinimize  — wraps scipy.optimize.minimize (local methods)
ScipyDE        — wraps scipy.optimize.differential_evolution (global)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import scipy.optimize as opt

from tdoa_loc.algorithms.base import TDOAAlgorithm, TDOAResult


class ScipyMinimize(TDOAAlgorithm):
    """Wrapper for ``scipy.optimize.minimize`` (local optimizer).

    Args:
        method: Scipy method string (e.g. ``"Nelder-Mead"``, ``"BFGS"``).
        options: Passed directly to ``scipy.optimize.minimize`` as ``options``.
        kwargs: Any other keyword args for ``minimize``.
    """

    def __init__(
        self,
        method: str = "Nelder-Mead",
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.method = method
        self.options = options or {"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8}
        self.kwargs = kwargs
        self.name = f"Scipy-{method}"
        self.short_name = method[:8]

    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> TDOAResult:
        if x0 is None:
            x0 = np.zeros(len(lb))

        t0 = time.perf_counter()
        res = opt.minimize(
            objective,
            x0,
            method=self.method,
            options=self.options,
            **self.kwargs,
        )
        elapsed = time.perf_counter() - t0

        return TDOAResult(
            position=np.array(res.x, dtype=float),
            success=bool(res.success),
            wall_time=elapsed,
            extra={"message": res.message, "n_evals": res.nfev},
        )


class ScipyDE(TDOAAlgorithm):
    """Wrapper for ``scipy.optimize.differential_evolution`` (global optimizer).

    Args:
        popsize: Population multiplier (total pop = popsize * dim).
        maxiter: Maximum number of generations.
        tol: Relative tolerance for convergence.
        kwargs: Any other keyword args for ``differential_evolution``.
    """

    name = "Differential Evolution"
    short_name = "DE"

    def __init__(
        self,
        popsize: int = 15,
        maxiter: int = 1000,
        tol: float = 1e-6,
        **kwargs: Any,
    ):
        self.popsize = popsize
        self.maxiter = maxiter
        self.tol = tol
        self.kwargs = kwargs

    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> TDOAResult:
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        bounds = list(zip(lb, ub))

        t0 = time.perf_counter()
        res = opt.differential_evolution(
            objective,
            bounds,
            popsize=self.popsize,
            maxiter=self.maxiter,
            tol=self.tol,
            seed=seed,
            disp=False,
            **self.kwargs,
        )
        elapsed = time.perf_counter() - t0

        return TDOAResult(
            position=np.array(res.x, dtype=float),
            success=bool(res.success),
            wall_time=elapsed,
            extra={"message": res.message, "n_evals": res.nfev},
        )
