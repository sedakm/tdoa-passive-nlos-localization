"""Geometry definitions for passive TDOA localization scenarios.

Three geometry types are supported:
  - CircularGeometry : N receivers uniformly placed on a circle around the TX.
  - RandomGeometry   : N receivers randomly placed within a rectangular area.
  - CustomGeometry   : Explicit TX and receiver positions.

Two target types are supported:
  - FixedTarget  : The target stays at a fixed position across all MC runs.
  - RandomTarget : The target is resampled uniformly within a rectangle each run.

Use ``get_geometry_from_config`` / ``get_target_from_config`` to build objects
from the YAML config dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geometry classes
# ---------------------------------------------------------------------------

@dataclass
class CircularGeometry:
    """Transmitter at ``tx``, N receivers uniformly on a circle of ``radius``."""

    tx: np.ndarray
    n_receivers: int
    radius: float

    def __post_init__(self):
        self.tx = np.asarray(self.tx, dtype=float)

    @property
    def receivers(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, self.n_receivers, endpoint=False)
        return self.radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


@dataclass
class RandomGeometry:
    """Transmitter at ``tx``, N receivers randomly placed in ``area_bounds``."""

    tx: np.ndarray
    n_receivers: int
    area_bounds: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    seed: Optional[int] = None
    _receivers: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self.tx = np.asarray(self.tx, dtype=float)
        if self._receivers is None:
            self._receivers = self._sample(self.seed)

    def _sample(self, seed: Optional[int]) -> np.ndarray:
        rng = np.random.default_rng(seed)
        xmin, xmax, ymin, ymax = self.area_bounds
        xs = rng.uniform(xmin, xmax, self.n_receivers)
        ys = rng.uniform(ymin, ymax, self.n_receivers)
        return np.stack([xs, ys], axis=1)

    def resample(self, seed: Optional[int] = None) -> "RandomGeometry":
        """Return a new RandomGeometry with freshly sampled receiver positions."""
        return RandomGeometry(
            tx=self.tx.copy(),
            n_receivers=self.n_receivers,
            area_bounds=self.area_bounds,
            seed=seed,
        )

    @property
    def receivers(self) -> np.ndarray:
        return self._receivers


@dataclass
class CustomGeometry:
    """Explicit transmitter and receiver positions."""

    tx: np.ndarray
    receivers: np.ndarray

    def __post_init__(self):
        self.tx = np.asarray(self.tx, dtype=float)
        self.receivers = np.asarray(self.receivers, dtype=float)
        if self.receivers.ndim != 2 or self.receivers.shape[1] != 2:
            raise ValueError("receivers must be shape (N, 2)")

    @property
    def n_receivers(self) -> int:
        return self.receivers.shape[0]


Geometry = CircularGeometry | RandomGeometry | CustomGeometry


# ---------------------------------------------------------------------------
# Target classes
# ---------------------------------------------------------------------------

@dataclass
class FixedTarget:
    """A target at a fixed known position."""

    position: np.ndarray

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)

    def get_position(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return self.position.copy()


@dataclass
class RandomTarget:
    """A target resampled uniformly within ``area_bounds`` on each call."""

    area_bounds: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)

    def get_position(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        xmin, xmax, ymin, ymax = self.area_bounds
        return np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])


Target = FixedTarget | RandomTarget


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_geometry_from_config(cfg: dict) -> Geometry:
    """Build a geometry object from a YAML config dict."""
    gtype = cfg["type"].lower()
    tx = np.asarray(cfg.get("tx", [0.0, 0.0]), dtype=float)

    if gtype == "circular":
        return CircularGeometry(
            tx=tx,
            n_receivers=int(cfg["n_receivers"]),
            radius=float(cfg["radius"]),
        )
    elif gtype == "random":
        return RandomGeometry(
            tx=tx,
            n_receivers=int(cfg["n_receivers"]),
            area_bounds=tuple(cfg["area_bounds"]),
            seed=cfg.get("seed"),
        )
    elif gtype == "custom":
        return CustomGeometry(
            tx=tx,
            receivers=np.asarray(cfg["receivers"], dtype=float),
        )
    else:
        raise ValueError(f"Unknown geometry type: '{gtype}'. Use circular/random/custom.")


def get_target_from_config(cfg: dict) -> Target:
    """Build a target object from a YAML config dict."""
    ttype = cfg["type"].lower()
    if ttype == "fixed":
        return FixedTarget(position=np.asarray(cfg["position"], dtype=float))
    elif ttype == "random":
        return RandomTarget(area_bounds=tuple(cfg["area_bounds"]))
    else:
        raise ValueError(f"Unknown target type: '{ttype}'. Use fixed/random.")
