"""Additional statistical visualizations for TDOA localization experiments.

Functions
---------
plot_error_box         — box plots of position error per algorithm at a given noise level
plot_timing_box        — box plots of wall time per algorithm
plot_rmse_vs_time      — RMSE vs median wall time (efficiency scatter)
plot_wilcoxon_heatmap  — pairwise Wilcoxon signed-rank p-value matrix
plot_scatter_estimates — scatter of estimated positions across MC runs
plot_geometry          — visualize TX, receiver, and target positions
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from tdoa_loc.simulation.results import SimulationResults
from tdoa_loc.visualization.rmse_plot import _MARKERS, _COLORS


# ---------------------------------------------------------------------------
# Box plot: errors at a fixed noise level
# ---------------------------------------------------------------------------

def plot_error_box(
    results: SimulationResults,
    sigma_b: float,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Box plot of position error for each algorithm at ``sigma_b``."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    unique_sb = results.data["sigma_b"].unique()
    closest_sb = unique_sb[np.argmin(np.abs(unique_sb - sigma_b))]
    subset = results.data[
        np.isclose(results.data["sigma_b"], closest_sb) &
        results.data["error"].notna()
    ]

    algo_names = results.algorithm_names()
    data_for_box = [
        subset[subset["algorithm"] == a]["error"].values for a in algo_names
    ]
    bplot = ax.boxplot(data_for_box, patch_artist=True, showfliers=True,
                       medianprops={"color": "black", "linewidth": 2})

    for patch, color in zip(bplot["boxes"], _COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(algo_names) + 1))
    ax.set_xticklabels(algo_names, rotation=30, ha="right")
    ax.set_ylabel("Position Error (m)")
    ax.set_title(
        f"Error Distribution at σ_b = {closest_sb:.2f} m"
        f"  ({10 * np.log10(closest_sb**2):.1f} dB)"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Box plot: wall time
# ---------------------------------------------------------------------------

def plot_timing_box(
    results: SimulationResults,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Box plot of wall time (seconds) per algorithm across all runs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    algo_names = results.algorithm_names()
    valid = results.data.dropna(subset=["wall_time"])
    data_for_box = [
        valid[valid["algorithm"] == a]["wall_time"].values * 1e3  # ms
        for a in algo_names
    ]
    bplot = ax.boxplot(data_for_box, patch_artist=True, showfliers=False,
                       medianprops={"color": "black", "linewidth": 2})

    for patch, color in zip(bplot["boxes"], _COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(algo_names) + 1))
    ax.set_xticklabels(algo_names, rotation=30, ha="right")
    ax.set_ylabel("Wall Time (ms)")
    ax.set_title("Computation Time per Algorithm")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Efficiency scatter: RMSE vs median time
# ---------------------------------------------------------------------------

def plot_rmse_vs_time(
    results: SimulationResults,
    sigma_b: float,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Scatter of median RMSE vs median wall time at a fixed noise level."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    unique_sb = results.data["sigma_b"].unique()
    closest_sb = unique_sb[np.argmin(np.abs(unique_sb - sigma_b))]
    subset = results.data[
        np.isclose(results.data["sigma_b"], closest_sb) &
        results.data["error"].notna() &
        results.data["wall_time"].notna()
    ]

    algo_names = results.algorithm_names()
    for idx, algo in enumerate(algo_names):
        d = subset[subset["algorithm"] == algo]
        if d.empty:
            continue
        rmse = float(np.sqrt(np.mean(d["error"].values ** 2)))
        t_ms = float(d["wall_time"].median() * 1e3)
        ax.scatter(t_ms, rmse, s=120, marker=_MARKERS[idx % len(_MARKERS)],
                   color=_COLORS[idx % len(_COLORS)], label=algo, zorder=4,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(algo, (t_ms, rmse), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Median Wall Time (ms)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"Efficiency  (σ_b = {closest_sb:.2f} m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank p-value heatmap
# ---------------------------------------------------------------------------

def plot_wilcoxon_heatmap(
    results: SimulationResults,
    sigma_b: float,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Pairwise Wilcoxon signed-rank test p-value heatmap at a noise level.

    Cells below ``alpha`` are highlighted. The test compares whether the
    error distributions of two algorithms differ significantly.
    """
    from scipy.stats import wilcoxon

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    unique_sb = results.data["sigma_b"].unique()
    closest_sb = unique_sb[np.argmin(np.abs(unique_sb - sigma_b))]
    subset = results.data[
        np.isclose(results.data["sigma_b"], closest_sb) &
        results.data["error"].notna()
    ]

    algo_names = results.algorithm_names()
    n = len(algo_names)
    pmat = np.ones((n, n))

    errors_by_algo = {}
    for algo in algo_names:
        d = subset[subset["algorithm"] == algo]
        errors_by_algo[algo] = d.sort_values("run_id")["error"].values

    for i, a in enumerate(algo_names):
        for j, b in enumerate(algo_names):
            if i == j:
                continue
            ea = errors_by_algo[a]
            eb = errors_by_algo[b]
            min_len = min(len(ea), len(eb))
            if min_len < 20:
                continue
            try:
                _, p = wilcoxon(ea[:min_len], eb[:min_len], alternative="two-sided")
                pmat[i, j] = p
            except Exception:
                pass

    im = ax.imshow(pmat, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="p-value")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(algo_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(algo_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = pmat[i, j]
            text = f"{val:.3f}" if val < 1 else "—"
            color = "white" if val < alpha else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_title(
        f"Wilcoxon p-values  (σ_b = {closest_sb:.2f} m, α = {alpha})\n"
        "Green = significant difference"
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Scatter of position estimates
# ---------------------------------------------------------------------------

def plot_scatter_estimates(
    results: SimulationResults,
    sigma_b: float,
    algorithm: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    max_points: int = 500,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Scatter plot of estimated positions at a fixed noise level.

    Args:
        algorithm: If None, plots all algorithms. Otherwise plots only the named one.
        max_points: Maximum scatter points per algorithm (random subsample if needed).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    unique_sb = results.data["sigma_b"].unique()
    closest_sb = unique_sb[np.argmin(np.abs(unique_sb - sigma_b))]
    subset = results.data[
        np.isclose(results.data["sigma_b"], closest_sb) &
        results.data["est_x"].notna()
    ]

    algo_names = [algorithm] if algorithm else results.algorithm_names()

    for idx, algo in enumerate(algo_names):
        d = subset[subset["algorithm"] == algo][["est_x", "est_y"]].dropna()
        if len(d) > max_points:
            d = d.sample(max_points, random_state=0)
        ax.scatter(d["est_x"], d["est_y"], s=12, alpha=0.4,
                   color=_COLORS[idx % len(_COLORS)], label=algo)

    # True position
    true_x = subset["true_x"].iloc[0] if "true_x" in subset.columns else None
    if true_x is not None:
        true_y = subset["true_y"].iloc[0]
        ax.plot(true_x, true_y, "r*", markersize=14, label="True Target", zorder=5)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Position Estimates  (σ_b = {closest_sb:.2f} m)")
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Geometry visualizer
# ---------------------------------------------------------------------------

def plot_geometry(
    geometry,
    target=None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Visualize the scene geometry: TX, receivers, and target."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    rx = geometry.receivers
    ax.plot(*geometry.tx, "b^", markersize=12, label="TX", zorder=5)
    ax.scatter(rx[:, 0], rx[:, 1], s=80, marker="s", color="steelblue",
               label="Receivers", zorder=4)
    for i, r in enumerate(rx):
        ax.annotate(f"Rx{i+1}", r, textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    if target is not None:
        pos = target.get_position()
        ax.plot(*pos, "r*", markersize=14, label="Target", zorder=5)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Localization Geometry")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
