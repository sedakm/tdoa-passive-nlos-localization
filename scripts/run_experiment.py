#!/usr/bin/env python3
"""Run a Monte Carlo TDOA localization experiment from a YAML config file.

Usage
-----
    python scripts/run_experiment.py --config configs/inside.yaml
    python scripts/run_experiment.py --config configs/outside.yaml --n-runs 200 --n-jobs 4
    python scripts/run_experiment.py --config configs/random.yaml --output-dir results/

The results are saved to ``<output_dir>/<experiment_name>_<timestamp>.h5``
with a JSON sidecar at the same path.

After the run a summary table is printed showing RMSE per algorithm per noise level.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Make sure the src package is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tdoa_loc.algorithms.cwls import CWLS
from tdoa_loc.algorithms.scipy_wrappers import ScipyDE, ScipyMinimize
from tdoa_loc.algorithms.poa import POA
from tdoa_loc.algorithms.base import TDOAAlgorithm, OptbenchAdapter
from tdoa_loc.simulation.monte_carlo import SimulationConfig, run_monte_carlo


def build_algorithms(cfg: dict) -> list[TDOAAlgorithm]:
    """Instantiate algorithm objects from the config list."""
    import importlib
    algos: list[TDOAAlgorithm] = []

    for entry in cfg.get("algorithms", []):
        atype = entry["type"]
        params = entry.get("params", {}) or {}
        short = entry.get("short_name", atype)

        if atype == "scipy_de":
            a = ScipyDE(**params)
            a.short_name = short
            algos.append(a)

        elif atype == "scipy_minimize":
            method = params.pop("method", "Nelder-Mead")
            a = ScipyMinimize(method=method, **params)
            a.short_name = short
            algos.append(a)

        elif atype == "cwls":
            a = CWLS(**params)
            a.short_name = short
            algos.append(a)

        elif atype == "poa":
            a = POA(**params)
            a.short_name = short
            algos.append(a)

        elif atype == "optbench":
            cls_path = entry["class"]
            module_name, class_name = cls_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                optbench_algo = cls(**params)
                a = OptbenchAdapter(optbench_algo, max_evals=entry.get("max_evals", 10_000))
                a.short_name = short
                algos.append(a)
            except ImportError as exc:
                print(f"[WARNING] Cannot import {cls_path}: {exc}. Skipping.")

        else:
            print(f"[WARNING] Unknown algorithm type '{atype}'. Skipping.")

    if not algos:
        raise ValueError("No valid algorithms found in config. Check the 'algorithms' section.")
    return algos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo TDOA localization experiment."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Override n_runs from config.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Override n_jobs (parallelism) from config.")
    parser.add_argument("--output-dir", default=None,
                        help="Override output results directory.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.n_runs is not None:
        raw_cfg["simulation"]["n_runs"] = args.n_runs
    if args.n_jobs is not None:
        raw_cfg["simulation"]["n_jobs"] = args.n_jobs

    algorithms = build_algorithms(raw_cfg)
    sim_config = SimulationConfig.from_dict(raw_cfg, algorithms)

    print(f"Experiment : {sim_config.experiment_name}")
    print(f"Algorithms : {[a.short_name for a in algorithms]}")
    print(f"Noise levels : {sim_config.n_sigma_levels} "
          f"({sim_config.sigma_b_db_range[0]} … {sim_config.sigma_b_db_range[1]} dB)")
    print(f"Runs / level : {sim_config.n_runs}")
    print(f"Parallelism  : n_jobs={sim_config.n_jobs}")
    print()

    t0 = time.perf_counter()
    results = run_monte_carlo(sim_config)
    elapsed = time.perf_counter() - t0

    print(f"\nSimulation finished in {elapsed:.1f} s")

    # Save results
    out_dir = Path(args.output_dir or raw_cfg.get("output", {}).get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{sim_config.experiment_name}_{timestamp}"
    saved = results.save(out_path)
    print(f"Results saved → {saved}")
    print(f"Sidecar JSON  → {saved.with_suffix('.json')}")

    # Print summary RMSE table
    print("\n--- RMSE Summary (m) ---")
    rmse = results.rmse_table()
    import pandas as pd
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(rmse.to_string())


if __name__ == "__main__":
    main()
