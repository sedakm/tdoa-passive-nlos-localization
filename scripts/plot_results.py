#!/usr/bin/env python3
"""Generate all publication plots from saved simulation results.

Usage
-----
    # All plots for one result file
    python scripts/plot_results.py --results results/inside_geometry_20260308.h5

    # Choose which plots to generate
    python scripts/plot_results.py --results results/*.h5 --plots rmse,cdf,stats

    # Select noise level for CDF / box plots
    python scripts/plot_results.py --results results/inside_geometry_*.h5 \\
        --plots cdf,box --cdf-sigma-b 5.0

Available plot types
--------------------
    rmse      — RMSE vs 10log(σ²) + CRLB (one figure per result file)
    cdf       — CDF of position error at selected noise level
    box       — Box plots of errors and timing
    wilcoxon  — Wilcoxon signed-rank p-value heatmap
    scatter   — Scatter of position estimates
    geometry  — Scene geometry visualisation
    all       — Generate all of the above
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tdoa_loc.simulation.results import SimulationResults
from tdoa_loc.visualization.rmse_plot import plot_rmse_vs_noise
from tdoa_loc.visualization.cdf_plot import plot_cdf
from tdoa_loc.visualization import stats_plots


def process_file(result_path: Path, args: argparse.Namespace) -> None:
    print(f"\nLoading: {result_path}")
    results = SimulationResults.load(result_path)
    exp_name = results.config.experiment_name

    fig_dir = Path(args.save_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plots = set(p.strip().lower() for p in args.plots.split(","))
    if "all" in plots:
        plots = {"rmse", "cdf", "box", "wilcoxon", "scatter", "geometry"}

    stem = result_path.stem  # filename without extension

    # ---- RMSE vs noise --------------------------------------------------
    if "rmse" in plots:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_rmse_vs_noise(results, ax=ax)
        out = fig_dir / f"{stem}_rmse.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- CDF ------------------------------------------------------------
    if "cdf" in plots:
        # Find a "medium" noise level if not specified
        import numpy as np
        unique_sb = results.data["sigma_b"].unique()
        if args.cdf_sigma_b is not None:
            sb_val = float(args.cdf_sigma_b)
        else:
            # Use median noise level
            sb_val = float(np.median(unique_sb))

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_cdf(results, sigma_b=sb_val, ax=ax)
        out = fig_dir / f"{stem}_cdf.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- Box plots (errors + timing) ------------------------------------
    if "box" in plots:
        import numpy as np
        unique_sb = results.data["sigma_b"].unique()
        sb_val = float(args.cdf_sigma_b) if args.cdf_sigma_b else float(np.median(unique_sb))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        stats_plots.plot_error_box(results, sigma_b=sb_val, ax=axes[0])
        stats_plots.plot_timing_box(results, ax=axes[1])
        fig.suptitle(exp_name, fontsize=12)
        out = fig_dir / f"{stem}_box.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

        # Efficiency scatter
        fig, ax = plt.subplots(figsize=(7, 5))
        stats_plots.plot_rmse_vs_time(results, sigma_b=sb_val, ax=ax)
        out = fig_dir / f"{stem}_efficiency.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- Wilcoxon heatmap -----------------------------------------------
    if "wilcoxon" in plots:
        import numpy as np
        unique_sb = results.data["sigma_b"].unique()
        sb_val = float(args.cdf_sigma_b) if args.cdf_sigma_b else float(np.median(unique_sb))

        fig, ax = plt.subplots(figsize=(7, 6))
        stats_plots.plot_wilcoxon_heatmap(results, sigma_b=sb_val, ax=ax)
        out = fig_dir / f"{stem}_wilcoxon.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- Scatter of estimates -------------------------------------------
    if "scatter" in plots:
        import numpy as np
        unique_sb = results.data["sigma_b"].unique()
        sb_val = float(args.cdf_sigma_b) if args.cdf_sigma_b else float(np.median(unique_sb))

        fig, ax = plt.subplots(figsize=(7, 7))
        stats_plots.plot_scatter_estimates(results, sigma_b=sb_val, ax=ax)
        out = fig_dir / f"{stem}_scatter.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- Geometry -------------------------------------------------------
    if "geometry" in plots:
        from tdoa_loc.geometry import get_geometry_from_config, get_target_from_config
        meta = results.config._meta if hasattr(results.config, "_meta") else {}
        if "geometry_cfg" in meta and "target_cfg" in meta:
            geom = get_geometry_from_config(meta["geometry_cfg"])
            target = get_target_from_config(meta["target_cfg"])
            fig, ax = plt.subplots(figsize=(6, 6))
            stats_plots.plot_geometry(geom, target, ax=ax)
            out = fig_dir / f"{stem}_geometry.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out}")
        else:
            print("  [SKIP] geometry: config not available in loaded results.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from saved Monte Carlo results."
    )
    parser.add_argument(
        "--results", required=True, nargs="+",
        help="Path(s) to .h5 result files. Supports globs, e.g. results/*.h5"
    )
    parser.add_argument(
        "--plots", default="rmse,cdf",
        help="Comma-separated list of plot types: rmse,cdf,box,wilcoxon,scatter,geometry,all"
    )
    parser.add_argument("--cdf-sigma-b", type=float, default=None,
                        help="sigma_b value (m) for CDF / box plots. Default: median level.")
    parser.add_argument("--save-dir", default="figures/",
                        help="Directory to save figures. Default: figures/")
    args = parser.parse_args()

    # Expand globs
    paths: list[Path] = []
    for pattern in args.results:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern))

    if not paths:
        print("[ERROR] No result files found.")
        sys.exit(1)

    for p in paths:
        if not p.exists():
            print(f"[WARNING] File not found: {p} — skipping.")
            continue
        process_file(p, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
