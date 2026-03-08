"""CDF of localization error plot (FIG_04 style).

For a selected noise level (sigma_b), shows the empirical CDF of position
error (LE in metres) for each algorithm. Markers are placed at evenly-spaced
quantiles along each curve, matching the reference figure style.

Usage::

    from tdoa_loc.visualization.cdf_plot import plot_cdf
    fig, ax = plt.subplots()
    plot_cdf(results, sigma_b=5.0, ax=ax)
    fig.savefig("cdf.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from tdoa_loc.simulation.results import SimulationResults
from tdoa_loc.visualization.rmse_plot import _MARKERS, _COLORS


def plot_cdf(
    results: SimulationResults,
    sigma_b: float,
    ax: Optional[plt.Axes] = None,
    n_marker_points: int = 12,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Axes:
    """Plot empirical CDF of position error for each algorithm at ``sigma_b``.

    Args:
        results:          Simulation results object.
        sigma_b:          The noise level (in metres) to select. The closest
                          available level in the results will be used.
        ax:               Existing axes. Created if None.
        n_marker_points:  Number of markers placed along each CDF curve.
        title:            Plot title.
        save_path:        Save figure to this path if provided.
        figsize:          Figure size for a new figure.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Find closest available sigma_b
    unique_sb = results.data["sigma_b"].unique()
    closest_sb = unique_sb[np.argmin(np.abs(unique_sb - sigma_b))]
    sigma_b_db = float(10 * np.log10(closest_sb ** 2))

    algo_names = results.algorithm_names()
    subset = results.data[
        np.isclose(results.data["sigma_b"], closest_sb) &
        results.data["error"].notna()
    ]

    for idx, algo in enumerate(algo_names):
        errors = np.sort(subset[subset["algorithm"] == algo]["error"].values)
        if len(errors) == 0:
            continue

        n = len(errors)
        cdf = np.arange(1, n + 1) / n

        marker = _MARKERS[idx % len(_MARKERS)]
        color = _COLORS[idx % len(_COLORS)]

        # Plot smooth line
        ax.plot(errors, cdf, color=color, linewidth=1.8, label=algo, zorder=3)

        # Overlay markers at evenly-spaced quantile positions
        marker_indices = np.linspace(0, n - 1, n_marker_points, dtype=int)
        ax.plot(
            errors[marker_indices], cdf[marker_indices],
            marker=marker, color=color, markersize=7,
            linestyle="none", markerfacecolor="none",
            markeredgewidth=1.5, zorder=4,
        )

    ax.set_xlabel("LE (m)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(
            f"CDF — {results.config.experiment_name}"
            f"  (σ_b = {closest_sb:.2f} m, {sigma_b_db:.1f} dB)",
            fontsize=11,
        )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
