"""RMSE vs noise plot (FIG_01/02/03 style).

X axis: 10·log₁₀(σ_b²)  [dB]
Y axis: 10·log₁₀(RMSE)  [dB·m]

Each algorithm gets a unique color + marker. CRLB (PEB) is shown as a
solid green line without markers, matching the reference figures.

Usage::

    from tdoa_loc.visualization.rmse_plot import plot_rmse_vs_noise
    fig, ax = plt.subplots()
    plot_rmse_vs_noise(results, ax=ax, title="Inside geometry")
    fig.savefig("rmse_inside.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from tdoa_loc.simulation.results import SimulationResults

# Marker / color cycle that matches the reference paper style
_MARKERS = ["D", "o", "s", "v", "^", "p", "h", "*", "X"]
_COLORS = ["tab:red", "tab:cyan", "tab:purple", "tab:blue",
           "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]


def plot_rmse_vs_noise(
    results: SimulationResults,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Axes:
    """Plot 10·log₁₀(RMSE) vs 10·log₁₀(σ_b²) for all algorithms + CRLB.

    Args:
        results:   Simulation results object.
        ax:        Existing axes to draw into. Created if None.
        title:     Plot title.
        save_path: If provided, save figure to this path.
        figsize:   Figure size when creating a new figure.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    algo_names = results.algorithm_names()
    valid = results.data.dropna(subset=["error"])

    # ---- per-algorithm RMSE curves ------------------------------------
    for idx, algo in enumerate(algo_names):
        subset = valid[valid["algorithm"] == algo]
        grouped = (
            subset.groupby("sigma_b_db")["error"]
            .apply(lambda x: float(np.sqrt(np.mean(x ** 2))))
            .reset_index()
        )
        grouped.columns = ["sigma_b_db", "rmse"]

        x_db = grouped["sigma_b_db"].values
        y_db = 10 * np.log10(np.maximum(grouped["rmse"].values, 1e-12))

        marker = _MARKERS[idx % len(_MARKERS)]
        color = _COLORS[idx % len(_COLORS)]

        ax.plot(
            x_db, y_db,
            marker=marker, markersize=6, linewidth=1.5,
            color=color, label=algo,
            markerfacecolor="none", markeredgewidth=1.5,
        )

    # ---- CRLB line ----------------------------------------------------
    crlb = results.crlb.dropna(subset=["peb"])
    crlb_x = crlb["sigma_b_db"].values
    crlb_y = 10 * np.log10(np.maximum(crlb["peb"].values, 1e-12))
    ax.plot(crlb_x, crlb_y, color="green", linewidth=2.0,
            linestyle="-", label="CRLB", zorder=5)

    # ---- formatting ---------------------------------------------------
    ax.set_xlabel(r"$10\log(\sigma^2\ [\mathrm{m}^2])$", fontsize=12)
    ax.set_ylabel(r"$10\log(\mathrm{RMSE}\ [\mathrm{m}])$", fontsize=12)
    ax.set_title(title or results.config.experiment_name, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
