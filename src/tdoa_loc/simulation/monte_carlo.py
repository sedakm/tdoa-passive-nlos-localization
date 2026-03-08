"""Monte Carlo simulation runner for TDOA localization experiments.

Noise sweep
-----------
The primary noise parameter swept on the x-axis is ``sigma_b`` (NLOS scatter
std). The sweep is defined in dB as:

    10 * log10(sigma_b^2) ∈ [sigma_b_db_min, sigma_b_db_max]

i.e.,  sigma_b = 10^(sigma_b_db / 20).

For each sigma_b level:
  - mu_b = mu_b_ratio * sigma_b   (proportional bias)
  - CRLB PEB is computed at the true target position
  - n_runs independent MC trials are run (parallelised with joblib)
  - Every trial generates fresh measurements and applies all algorithms

RandomGeometry
--------------
If the geometry type is RandomGeometry, receiver positions are re-sampled for
each run using a seed derived from the global seed + run index.

RandomTarget
------------
Similarly, target positions are re-sampled per run when RandomTarget is used.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from tdoa_loc.algorithms.base import TDOAAlgorithm, TDOAResult
from tdoa_loc.algorithms.cwls import CWLS
from tdoa_loc.geometry import (
    CircularGeometry, RandomGeometry, CustomGeometry,
    FixedTarget, RandomTarget,
    get_geometry_from_config, get_target_from_config,
    Geometry, Target,
)
from tdoa_loc.likelihood import compute_crlb, negative_log_likelihood
from tdoa_loc.physics import generate_measurements


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Full specification of a Monte Carlo experiment.

    Fields loaded from YAML via ``SimulationConfig.from_dict``.
    """

    geometry_cfg: Dict[str, Any]
    target_cfg: Dict[str, Any]
    sigma_los: float
    p_los: float
    mu_b_ratio: float                   # mu_b = mu_b_ratio * sigma_b
    sigma_b_db_range: Tuple[float, float]  # (min_dB, max_dB) of 10log10(σ_b²)
    n_sigma_levels: int
    n_runs: int
    algorithms: List[TDOAAlgorithm]
    bounds: Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    n_jobs: int = -1
    seed: int = 42
    experiment_name: str = "experiment"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], algorithms: List[TDOAAlgorithm]) -> "SimulationConfig":
        noise = cfg["noise"]
        sim = cfg["simulation"]
        b = sim["bounds"]
        return cls(
            geometry_cfg=cfg["geometry"],
            target_cfg=cfg["target"],
            sigma_los=float(noise["sigma_los"]),
            p_los=float(noise["p_los"]),
            mu_b_ratio=float(noise.get("mu_b_ratio", 2.5)),
            sigma_b_db_range=(
                float(noise["sigma_b_db_range"][0]),
                float(noise["sigma_b_db_range"][1]),
            ),
            n_sigma_levels=int(noise.get("n_sigma_levels", 21)),
            n_runs=int(sim.get("n_runs", 500)),
            algorithms=algorithms,
            bounds=tuple(b),
            n_jobs=int(sim.get("n_jobs", -1)),
            seed=int(sim.get("seed", 42)),
            experiment_name=cfg.get("experiment_name", "experiment"),
        )


# ---------------------------------------------------------------------------
# Single-run worker (must be top-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _run_single(
    run_idx: int,
    base_seed: int,
    geometry_cfg: Dict[str, Any],
    target_cfg: Dict[str, Any],
    sigma_los: float,
    sigma_b: float,
    mu_b: float,
    p_los: float,
    bounds: Tuple[float, float, float, float],
    algo_configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Execute one MC run; returns list of result dicts (one per algorithm)."""
    rng = np.random.default_rng(base_seed + run_idx)

    # Resolve geometry (resample if random)
    geometry = get_geometry_from_config(geometry_cfg)
    if isinstance(geometry, RandomGeometry):
        geometry = geometry.resample(seed=int(base_seed + run_idx * 1000))

    # Resolve target position
    target = get_target_from_config(target_cfg)
    true_pos = target.get_position(rng)

    # Generate measurements
    r = generate_measurements(true_pos, geometry, sigma_los, sigma_b, mu_b, p_los, rng)

    lb = np.array([bounds[0], bounds[2]], dtype=float)
    ub = np.array([bounds[1], bounds[3]], dtype=float)

    records = []
    for ac in algo_configs:
        algo = _instantiate_algo(ac)

        # Inject context for CWLS
        if isinstance(algo, CWLS):
            algo.set_context(geometry, r)

        obj = partial(
            negative_log_likelihood,
            r=r,
            geometry=geometry,
            sigma_los=sigma_los,
            sigma_b=sigma_b,
            mu_b=mu_b,
            p_los=p_los,
        )

        try:
            result: TDOAResult = algo.solve(obj, lb, ub, x0=np.zeros(2), seed=run_idx)
            error = float(np.linalg.norm(result.position - true_pos))
            records.append({
                "run_id": run_idx,
                "algorithm": ac["short_name"],
                "error": error,
                "wall_time": result.wall_time,
                "success": int(result.success),
                "est_x": result.position[0],
                "est_y": result.position[1],
                "true_x": true_pos[0],
                "true_y": true_pos[1],
            })
        except Exception as exc:  # noqa: BLE001
            records.append({
                "run_id": run_idx,
                "algorithm": ac["short_name"],
                "error": np.nan,
                "wall_time": np.nan,
                "success": 0,
                "est_x": np.nan,
                "est_y": np.nan,
                "true_x": true_pos[0],
                "true_y": true_pos[1],
            })
    return records


def _instantiate_algo(ac: Dict[str, Any]) -> TDOAAlgorithm:
    """Re-instantiate an algorithm from its config dict (needed for parallel workers)."""
    atype = ac["type"]
    params = ac.get("params", {})

    if atype == "scipy_de":
        from tdoa_loc.algorithms.scipy_wrappers import ScipyDE
        return ScipyDE(**params)
    elif atype == "scipy_minimize":
        from tdoa_loc.algorithms.scipy_wrappers import ScipyMinimize
        method = params.pop("method", "Nelder-Mead")
        return ScipyMinimize(method=method, **params)
    elif atype == "cwls":
        from tdoa_loc.algorithms.cwls import CWLS
        return CWLS(**params)
    elif atype == "optbench":
        cls_path = ac["class"]
        module_name, class_name = cls_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        optbench_algo = cls(**params)
        from tdoa_loc.algorithms.base import OptbenchAdapter
        return OptbenchAdapter(optbench_algo, max_evals=ac.get("max_evals", 10_000))
    else:
        raise ValueError(f"Unknown algorithm type: '{atype}'")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_monte_carlo(config: SimulationConfig) -> "SimulationResults":
    """Run the full Monte Carlo experiment defined by ``config``.

    Returns a :class:`~tdoa_loc.simulation.results.SimulationResults` object
    containing all raw trial data and per-level CRLB values.
    """
    from tdoa_loc.simulation.results import SimulationResults
    import pandas as pd

    # Build sigma_b sweep from dB range
    db_min, db_max = config.sigma_b_db_range
    sigma_b_db_vals = np.linspace(db_min, db_max, config.n_sigma_levels)
    sigma_b_vals = 10.0 ** (sigma_b_db_vals / 20.0)

    # Build algorithm config list (serialisable for parallel workers)
    algo_configs = _build_algo_configs(config.algorithms)

    # Compute CRLB for each noise level (needs a reference position)
    ref_geometry = get_geometry_from_config(config.geometry_cfg)
    ref_target = get_target_from_config(config.target_cfg)
    if isinstance(ref_target, FixedTarget):
        crlb_pos = ref_target.position
    else:
        # For random targets, use centre of bounds as CRLB reference
        b = config.bounds
        crlb_pos = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2])

    crlb_rows = []
    for sigma_b in sigma_b_vals:
        _, _, peb = compute_crlb(crlb_pos, ref_geometry, config.sigma_los, sigma_b, config.p_los)
        crlb_rows.append({"sigma_b": sigma_b, "sigma_b_db": 10 * np.log10(sigma_b ** 2), "peb": peb})
    crlb_df = pd.DataFrame(crlb_rows)

    # Run MC across all noise levels
    all_rows: List[Dict] = []

    for sigma_b in tqdm(sigma_b_vals, desc="Noise levels", unit="level"):
        mu_b = config.mu_b_ratio * sigma_b

        rows = Parallel(n_jobs=config.n_jobs)(
            delayed(_run_single)(
                run_idx=i,
                base_seed=config.seed,
                geometry_cfg=config.geometry_cfg,
                target_cfg=config.target_cfg,
                sigma_los=config.sigma_los,
                sigma_b=float(sigma_b),
                mu_b=float(mu_b),
                p_los=config.p_los,
                bounds=config.bounds,
                algo_configs=algo_configs,
            )
            for i in range(config.n_runs)
        )

        for run_rows in rows:
            for rec in run_rows:
                rec["sigma_b"] = float(sigma_b)
                rec["sigma_b_db"] = float(10 * np.log10(sigma_b ** 2))
                all_rows.append(rec)

    data_df = pd.DataFrame(all_rows)
    return SimulationResults(config=config, data=data_df, crlb=crlb_df)


def _build_algo_configs(algorithms: List[TDOAAlgorithm]) -> List[Dict[str, Any]]:
    """Convert algorithm instances to serialisable config dicts for workers."""
    configs = []
    for algo in algorithms:
        if isinstance(algo, CWLS):
            configs.append({"type": "cwls", "short_name": algo.short_name,
                             "params": {"max_iter": algo.max_iter, "tol": algo.tol}})
        elif hasattr(algo, "_algo"):  # OptbenchAdapter
            configs.append({
                "type": "optbench",
                "short_name": algo.short_name,
                "class": f"{type(algo._algo).__module__}.{type(algo._algo).__name__}",
                "params": getattr(algo._algo, "params", {}),
                "max_evals": algo.max_evals,
            })
        else:
            # ScipyDE / ScipyMinimize
            atype = "scipy_de" if hasattr(algo, "popsize") else "scipy_minimize"
            params: Dict[str, Any] = {}
            if atype == "scipy_de":
                params = {"popsize": algo.popsize, "maxiter": algo.maxiter, "tol": algo.tol}
            else:
                params = {"method": algo.method, "options": algo.options}
            configs.append({"type": atype, "short_name": algo.short_name, "params": params})
    return configs
