"""Abstract base class and adapter for TDOA localization algorithms.

Every algorithm must implement ``solve(objective, lb, ub, x0, seed) -> TDOAResult``.

OptbenchAdapter
---------------
Wraps any ``optbench.algorithms.Algorithm`` (from the sedakm/optim-alg-testing repo)
into the ``TDOAAlgorithm`` interface without a hard package dependency.

Usage::

    # Requires: pip install -e /path/to/optim-alg-testing
    from optbench.algorithms.boa import ButterflyOptimization
    from tdoa_loc.algorithms.base import OptbenchAdapter

    boa = OptbenchAdapter(ButterflyOptimization(pop_size=100), max_evals=10_000)
    result = boa.solve(objective, lb, ub)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class TDOAResult:
    """Result from one algorithm run on one MC trial.

    Attributes:
        position:  Estimated target position [x, y], shape (2,).
        success:   Whether the algorithm converged / reported success.
        wall_time: Elapsed wall-clock time in seconds.
        extra:     Algorithm-specific extras (convergence curve, n_evals, …).
    """

    position: np.ndarray
    success: bool
    wall_time: float
    extra: Dict[str, Any] = field(default_factory=dict)


class TDOAAlgorithm(ABC):
    """Abstract base class for all TDOA localization algorithms.

    Subclasses must set ``name`` / ``short_name`` and implement ``solve``.
    """

    name: str = "BaseAlgorithm"
    short_name: str = "BASE"

    @abstractmethod
    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> TDOAResult:
        """Minimize ``objective`` over the box [lb, ub].

        Args:
            objective: Scalar function to minimise (negative log-likelihood).
            lb: Lower bounds, shape (2,).
            ub: Upper bounds, shape (2,).
            x0: Optional initial guess, shape (2,).
            seed: Random seed for reproducibility.

        Returns:
            TDOAResult with estimated position and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(short_name={self.short_name!r})"


# ---------------------------------------------------------------------------
# Adapter for optbench algorithms
# ---------------------------------------------------------------------------

class OptbenchAdapter(TDOAAlgorithm):
    """Wraps an ``optbench.algorithms.Algorithm`` into ``TDOAAlgorithm``.

    The wrapped algorithm must conform to the optbench interface::

        algo.solve(problem: optbench.algorithms.base.Problem,
                   seed: Optional[int]) -> OptimizationResult

    Requires ``optbench`` installed in the active Python environment::

        pip install -e /path/to/optim-alg-testing

    Args:
        optbench_algo: An instantiated optbench Algorithm object.
        max_evals: Maximum function evaluations budget passed to the Problem.
    """

    def __init__(self, optbench_algo: Any, max_evals: int = 10_000):
        self._algo = optbench_algo
        self.max_evals = max_evals
        self.name = getattr(optbench_algo, "name", type(optbench_algo).__name__)
        self.short_name = getattr(optbench_algo, "short_name", self.name[:8])

    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> TDOAResult:
        try:
            from optbench.algorithms.base import Problem  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "optbench is not installed. "
                "Run: pip install -e /path/to/optim-alg-testing"
            ) from exc

        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        problem = Problem(
            func=objective,
            lb=lb,
            ub=ub,
            dim=len(lb),
            max_evals=self.max_evals,
        )

        t0 = time.perf_counter()
        result = self._algo.solve(problem, seed=seed)
        elapsed = time.perf_counter() - t0

        convergence = getattr(result, "convergence_curve", np.array([]))
        n_evals = getattr(result, "n_evals", self.max_evals)
        wall_time = result.wall_time if result.wall_time > 0 else elapsed

        return TDOAResult(
            position=np.array(result.best_position, dtype=float),
            success=True,
            wall_time=wall_time,
            extra={"convergence": convergence, "n_evals": n_evals},
        )
