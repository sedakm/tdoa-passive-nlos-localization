"""Constrained / Classical Weighted Least Squares (CWLS) baseline.

CWLS linearises the nonlinear bistatic range equations around the current
position estimate and iterates a WLS update step until convergence.

At each iteration:
    d  = all_bistatic_distances(x_est, geometry)
    e  = r - d                           # residuals
    H  = compute_jacobian(x_est, geometry)
    dx = (H^T W H)^{-1} H^T W e         # WLS update
    x_est += dx

The weight matrix W is uniform (identity scaling) so the algorithm reduces
to standard iterative least squares. This is a classical closed-form baseline
that performs well in LOS conditions but degrades in heavy NLOS.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from tdoa_loc.algorithms.base import TDOAAlgorithm, TDOAResult
from tdoa_loc.geometry import Geometry
from tdoa_loc.physics import all_bistatic_distances
from tdoa_loc.likelihood import compute_jacobian


class CWLS(TDOAAlgorithm):
    """Iterative Weighted Least Squares linearization baseline.

    Note: This algorithm needs access to the geometry and measurements
    directly — it does not use the ``objective`` callable. The geometry and
    measurements are injected via ``set_context`` before each ``solve`` call,
    which is handled automatically by the Monte Carlo runner.

    Args:
        max_iter: Maximum number of linearization iterations.
        tol: Convergence threshold (norm of update step in metres).
        weight: ``"uniform"`` (identity W) or ``"noise"`` (W ∝ 1/sigma_eq^2,
                uniform if sigma parameters are unavailable).
    """

    name = "CWLS"
    short_name = "CWLS"

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-6,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self._geometry: Optional[Geometry] = None
        self._r: Optional[np.ndarray] = None

    def set_context(self, geometry: Geometry, r: np.ndarray) -> None:
        """Inject geometry and measurements before calling ``solve``."""
        self._geometry = geometry
        self._r = r

    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> TDOAResult:
        if self._geometry is None or self._r is None:
            raise RuntimeError(
                "CWLS requires geometry and measurements. "
                "Call set_context(geometry, r) before solve()."
            )

        geometry = self._geometry
        r = self._r

        if x0 is None:
            x0 = np.zeros(2)
        x_est = np.array(x0, dtype=float)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        t0 = time.perf_counter()
        converged = False
        n_iter = 0

        for n_iter in range(self.max_iter):
            d = all_bistatic_distances(x_est, geometry)
            e = r - d
            H = compute_jacobian(x_est, geometry)

            HtH = H.T @ H
            Hte = H.T @ e

            try:
                dx = np.linalg.solve(HtH, Hte)
            except np.linalg.LinAlgError:
                break

            x_est = x_est + dx
            # Clip to bounds to avoid divergence
            x_est = np.clip(x_est, lb, ub)

            if np.linalg.norm(dx) < self.tol:
                converged = True
                break

        elapsed = time.perf_counter() - t0

        return TDOAResult(
            position=x_est.copy(),
            success=converged,
            wall_time=elapsed,
            extra={"n_iter": n_iter + 1, "converged": converged},
        )
