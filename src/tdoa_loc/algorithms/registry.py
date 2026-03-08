"""Algorithm registry for config-driven instantiation.

Usage::

    from tdoa_loc.algorithms.registry import registry

    # Built-ins are pre-registered
    algo = registry.create("DE", popsize=15, maxiter=500)

    # Register a custom algorithm
    registry.register("MyAlgo", MyAlgoClass)
    algo = registry.create("MyAlgo", pop_size=100)

    print(registry.list())  # ['CWLS', 'DE', 'NM', ...]
"""

from __future__ import annotations

from typing import Any, Dict, Type

from tdoa_loc.algorithms.base import TDOAAlgorithm


class AlgorithmRegistry:
    """Maps algorithm short names to their classes."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[TDOAAlgorithm]] = {}

    def register(self, name: str, cls: Type[TDOAAlgorithm]) -> None:
        self._registry[name] = cls

    def get(self, name: str) -> Type[TDOAAlgorithm]:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Algorithm '{name}' not found. Available: {available}")
        return self._registry[name]

    def create(self, name: str, **params: Any) -> TDOAAlgorithm:
        return self.get(name)(**params)

    def list(self) -> list[str]:
        return sorted(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Global registry instance
registry = AlgorithmRegistry()

# ---------------------------------------------------------------------------
# Pre-register built-in algorithms
# ---------------------------------------------------------------------------
from tdoa_loc.algorithms.scipy_wrappers import ScipyMinimize, ScipyDE  # noqa: E402
from tdoa_loc.algorithms.cwls import CWLS  # noqa: E402

registry.register("DE", ScipyDE)
registry.register("NM", lambda **kw: ScipyMinimize(method="Nelder-Mead", **kw))
registry.register("BFGS", lambda **kw: ScipyMinimize(method="BFGS", **kw))
registry.register("Powell", lambda **kw: ScipyMinimize(method="Powell", **kw))
registry.register("CWLS", CWLS)
