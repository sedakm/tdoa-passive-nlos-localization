"""Pufferfish Optimization Algorithm (POA) for TDOA localization.

Implements the nature-inspired metaheuristic from:
    Al-Baik et al., "Pufferfish Optimization Algorithm: A New Bio-Inspired
    Metaheuristic Algorithm for Solving Optimization Problems",
    Biomimetics 2024, 9, 65. https://doi.org/10.3390/biomimetics9020065

The algorithm models the defensive behavior of a pufferfish against a predator
in two alternating phases per iteration:

  Phase 1 – Exploration (predator attack):
      v_i = X_best - X_rand * r1 - (X_i - X_rand) * r2
      where r1, r2 ~ U(0,1) and X_rand is a randomly selected population member.
      Models the pufferfish fleeing toward the best-known region while gaining
      spatial diversity from a random individual.

  Phase 2 – Exploitation (predator retreats, pufferfish inflates):
      u_i = X_i * cos(2π * r) + X_best
      where r ~ U(0,1).
      Models local refinement around the best-known position.

After each phase, boundary reflection is applied and a greedy acceptance rule
retains the new position only if it improves the objective.

Complexity: O(NP * n * T)  — linear in population size, dimension, and iterations.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from tdoa_loc.algorithms.base import TDOAAlgorithm, TDOAResult


class POA(TDOAAlgorithm):
    """Pufferfish Optimization Algorithm.

    Args:
        pop_size:  Number of candidate solutions (NP). Default 30.
        max_iter:  Maximum number of iterations (T). Default 500.
    """

    name = "Pufferfish Optimization Algorithm"
    short_name = "POA"

    def __init__(self, pop_size: int = 30, max_iter: int = 500) -> None:
        self.pop_size = pop_size
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Reflect out-of-bounds components back into [lb, ub]."""
        x = np.where(x < lb, 2 * lb - x, x)
        x = np.where(x > ub, 2 * ub - x, x)
        return np.clip(x, lb, ub)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> TDOAResult:
        rng = np.random.default_rng(seed)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        n = len(lb)

        # --- Initialization (Eq. 33) ---
        pop = lb + rng.random((self.pop_size, n)) * (ub - lb)
        fitness = np.array([objective(ind) for ind in pop])

        best_idx = int(np.argmin(fitness))
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        n_evals = self.pop_size
        t0 = time.perf_counter()

        for _ in range(self.max_iter):
            for i in range(self.pop_size):

                # --- Phase 1: Exploration (predator attack) ---
                rand_idx = rng.integers(0, self.pop_size)
                x_rand = pop[rand_idx]
                r1, r2 = rng.random(), rng.random()

                v = best_pos - x_rand * r1 - (pop[i] - x_rand) * r2
                v = self._clip(v, lb, ub)

                f_v = objective(v)
                n_evals += 1
                if f_v < fitness[i]:
                    pop[i] = v
                    fitness[i] = f_v
                    if f_v < best_fit:
                        best_pos = v.copy()
                        best_fit = f_v

                # --- Phase 2: Exploitation (predator retreats) ---
                r = rng.random()
                u = pop[i] * np.cos(2 * np.pi * r) + best_pos
                u = self._clip(u, lb, ub)

                f_u = objective(u)
                n_evals += 1
                if f_u < fitness[i]:
                    pop[i] = u
                    fitness[i] = f_u
                    if f_u < best_fit:
                        best_pos = u.copy()
                        best_fit = f_u

        elapsed = time.perf_counter() - t0
        return TDOAResult(
            position=best_pos,
            success=True,
            wall_time=elapsed,
            extra={"n_evals": n_evals, "best_fit": best_fit},
        )
