"""Algorithm interface and built-in implementations for TDOA localization."""

from tdoa_loc.algorithms.base import TDOAAlgorithm, TDOAResult, OptbenchAdapter
from tdoa_loc.algorithms.scipy_wrappers import ScipyMinimize, ScipyDE
from tdoa_loc.algorithms.cwls import CWLS
from tdoa_loc.algorithms.poa import POA
from tdoa_loc.algorithms.registry import registry

__all__ = [
    "TDOAAlgorithm",
    "TDOAResult",
    "OptbenchAdapter",
    "ScipyMinimize",
    "ScipyDE",
    "CWLS",
    "POA",
    "registry",
]
