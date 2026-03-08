"""SimulationResults: storage, loading, and summary statistics.

Data is persisted as:
  - ``<name>.h5``   — HDF5 file with two datasets: ``data`` and ``crlb``
  - ``<name>.json`` — JSON sidecar with experiment metadata / config snapshot

Loading::

    results = SimulationResults.load("results/inside_geometry_20260308.h5")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tdoa_loc.simulation.monte_carlo import SimulationConfig


@dataclass
class SimulationResults:
    """Container for Monte Carlo simulation results.

    Attributes:
        config:  The :class:`SimulationConfig` used to produce these results.
        data:    DataFrame with columns:
                 ``sigma_b``, ``sigma_b_db``, ``run_id``, ``algorithm``,
                 ``error``, ``wall_time``, ``success``,
                 ``est_x``, ``est_y``, ``true_x``, ``true_y``.
        crlb:    DataFrame with columns: ``sigma_b``, ``sigma_b_db``, ``peb``.
    """

    config: "SimulationConfig"
    data: pd.DataFrame
    crlb: pd.DataFrame

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def rmse_table(self) -> pd.DataFrame:
        """Return RMSE per (algorithm, sigma_b_db) pivot table."""
        valid = self.data.dropna(subset=["error"])
        return (
            valid.groupby(["algorithm", "sigma_b_db"])["error"]
            .apply(lambda x: float(np.sqrt(np.mean(x ** 2))))
            .unstack(level="sigma_b_db")
        )

    def median_error_table(self) -> pd.DataFrame:
        valid = self.data.dropna(subset=["error"])
        return (
            valid.groupby(["algorithm", "sigma_b_db"])["error"]
            .median()
            .unstack(level="sigma_b_db")
        )

    def success_rate_table(self) -> pd.DataFrame:
        return (
            self.data.groupby(["algorithm", "sigma_b_db"])["success"]
            .mean()
            .mul(100)
            .unstack(level="sigma_b_db")
        )

    def timing_summary(self) -> pd.DataFrame:
        return (
            self.data.dropna(subset=["wall_time"])
            .groupby("algorithm")["wall_time"]
            .agg(["mean", "median", "std", "min", "max"])
            .rename(columns={"mean": "mean_s", "median": "median_s",
                              "std": "std_s", "min": "min_s", "max": "max_s"})
        )

    def errors_at_sigma_b(self, sigma_b: float) -> pd.DataFrame:
        """Return all error values at the noise level closest to ``sigma_b``."""
        closest = float(self.data["sigma_b"].sub(sigma_b).abs().min())
        mask = np.isclose(self.data["sigma_b"], sigma_b + closest - closest, atol=closest + 1e-9)
        # simpler: find closest unique sigma_b
        unique = self.data["sigma_b"].unique()
        best = unique[np.argmin(np.abs(unique - sigma_b))]
        return self.data[np.isclose(self.data["sigma_b"], best)][
            ["algorithm", "error", "wall_time", "est_x", "est_y"]
        ].copy()

    def algorithm_names(self) -> list[str]:
        return sorted(self.data["algorithm"].unique().tolist())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save results to HDF5 + JSON sidecar.

        Args:
            path: Output path (with or without ``.h5`` extension).

        Returns:
            Path to the saved ``.h5`` file.
        """
        path = Path(path)
        if path.suffix not in (".h5", ".hdf5"):
            path = path.with_suffix(".h5")
        path.parent.mkdir(parents=True, exist_ok=True)

        self.data.to_hdf(str(path), key="data", mode="w", complevel=5)
        self.crlb.to_hdf(str(path), key="crlb", mode="a", complevel=5)

        # JSON sidecar with config snapshot
        meta = _config_to_dict(self.config)
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(meta, indent=2, default=_json_default))

        return path

    @classmethod
    def load(cls, path: str | Path) -> "SimulationResults":
        """Load results from a previously saved ``.h5`` file.

        Args:
            path: Path to the ``.h5`` file.

        Returns:
            Reconstructed :class:`SimulationResults`.  The ``config`` field
            contains a lightweight :class:`_StubConfig` with the raw dict.
        """
        path = Path(path)
        data = pd.read_hdf(str(path), key="data")
        crlb = pd.read_hdf(str(path), key="crlb")

        sidecar = path.with_suffix(".json")
        meta: Dict[str, Any] = {}
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())

        return cls(config=_StubConfig(meta), data=data, crlb=crlb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubConfig:
    """Minimal config proxy used when loading results without the full config."""

    def __init__(self, meta: Dict[str, Any]) -> None:
        self._meta = meta
        self.experiment_name = meta.get("experiment_name", "experiment")

    def __getattr__(self, name: str) -> Any:
        return self._meta.get(name)


def _config_to_dict(config: "SimulationConfig") -> Dict[str, Any]:
    """Convert a SimulationConfig to a JSON-serialisable dict."""
    d: Dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "geometry_cfg": config.geometry_cfg,
        "target_cfg": config.target_cfg,
        "sigma_los": config.sigma_los,
        "p_los": config.p_los,
        "mu_b_ratio": config.mu_b_ratio,
        "sigma_b_db_range": list(config.sigma_b_db_range),
        "n_sigma_levels": config.n_sigma_levels,
        "n_runs": config.n_runs,
        "bounds": list(config.bounds),
        "n_jobs": config.n_jobs,
        "seed": config.seed,
        "algorithms": [
            {"short_name": a.short_name, "name": a.name}
            for a in config.algorithms
        ],
    }
    return d


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")
