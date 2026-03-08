#!/usr/bin/env python3
"""Standalone ML objective function surface visualization.

Generates the 3D surface + 2D contour/quiver figure (FIG_00 style)
WITHOUT needing to run a full Monte Carlo simulation.

Usage
-----
    # Basic: use geometry and noise params from a config
    python scripts/plot_ml_function.py --config configs/inside.yaml

    # Custom noise level
    python scripts/plot_ml_function.py --config configs/inside.yaml --sigma-b 10.0

    # Higher resolution grid
    python scripts/plot_ml_function.py --config configs/inside.yaml --n-points 200

    # Save to file
    python scripts/plot_ml_function.py --config configs/inside.yaml --save figures/ml_inside.png

    # Use an existing measurements file (JSON list of floats)
    python scripts/plot_ml_function.py --config configs/inside.yaml \\
        --measurements-file results/sample_measurements.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tdoa_loc.geometry import get_geometry_from_config, get_target_from_config
from tdoa_loc.visualization.ml_surface import plot_ml_function


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ML objective function surface from a config file."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--sigma-b", type=float, default=None,
                        help="NLOS scatter std σ_b (m). Overrides config noise level. "
                             "Default: mid-range from config sweep.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for measurement generation. Default: 0.")
    parser.add_argument("--n-points", type=int, default=150,
                        help="Grid resolution (n_points × n_points). Default: 150.")
    parser.add_argument("--grid-range", type=float, default=None,
                        help="Symmetric grid limit in metres (e.g. 600). "
                             "Default: auto from receiver positions.")
    parser.add_argument("--measurements-file", default=None,
                        help="JSON file with pre-computed measurements (list of floats).")
    parser.add_argument("--save", default=None,
                        help="Output file path (e.g. figures/ml.png). "
                             "If not set, the figure is shown interactively.")
    parser.add_argument("--title", default="",
                        help="Custom figure title.")
    args = parser.parse_args()

    # ---- load config ---------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    geometry = get_geometry_from_config(raw_cfg["geometry"])
    target = get_target_from_config(raw_cfg["target"])

    noise = raw_cfg["noise"]
    sigma_los = float(noise["sigma_los"])
    p_los = float(noise["p_los"])
    mu_b_ratio = float(noise.get("mu_b_ratio", 2.5))

    # Determine sigma_b
    if args.sigma_b is not None:
        sigma_b = float(args.sigma_b)
    else:
        # Use mid-range of the sweep
        db_range = noise.get("sigma_b_db_range", [-50, 50])
        mid_db = (float(db_range[0]) + float(db_range[1])) / 2.0
        sigma_b = float(10 ** (mid_db / 20.0))

    mu_b = mu_b_ratio * sigma_b

    print(f"Geometry    : {raw_cfg['geometry']['type']}")
    print(f"Target      : {raw_cfg['target']['type']}")
    print(f"sigma_b     : {sigma_b:.4f} m  "
          f"({10 * np.log10(sigma_b**2):.1f} dB)")
    print(f"mu_b        : {mu_b:.4f} m")
    print(f"sigma_los   : {sigma_los} m,  p_los = {p_los}")
    print(f"Grid        : {args.n_points} × {args.n_points}")

    # ---- optional pre-loaded measurements ------------------------------
    measurements = None
    if args.measurements_file:
        with open(args.measurements_file) as f:
            measurements = np.array(json.load(f), dtype=float)
        print(f"Measurements loaded from: {args.measurements_file}")

    # ---- plot ----------------------------------------------------------
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_range = (-args.grid_range, args.grid_range) if args.grid_range else None

    fig = plot_ml_function(
        geometry=geometry,
        target=target,
        sigma_los=sigma_los,
        sigma_b=sigma_b,
        mu_b=mu_b,
        p_los=p_los,
        measurements=measurements,
        grid_range=grid_range,
        n_points=args.n_points,
        seed=args.seed,
        title=args.title,
    )

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
