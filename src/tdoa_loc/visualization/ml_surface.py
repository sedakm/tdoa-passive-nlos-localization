"""ML objective function visualization (FIG_00 style).

Produces a figure with two subplots:
  Left  — 3D surface of J_ML(x) = -log_likelihood(x), jet colormap
  Right — 2D filled contour + gradient arrows (quiver), receiver/TX/target markers

Can be used standalone without running simulations::

    from tdoa_loc.visualization.ml_surface import plot_ml_function
    from tdoa_loc.geometry import CircularGeometry, FixedTarget

    geometry = CircularGeometry(tx=[0,0], n_receivers=4, radius=500)
    target   = FixedTarget(position=[200, 150])
    fig = plot_ml_function(geometry, target,
                           sigma_los=2, sigma_b=5, mu_b=12.5, p_los=0.7)
    fig.savefig("ml_surface.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

from tdoa_loc.geometry import Geometry, Target, FixedTarget
from tdoa_loc.likelihood import negative_log_likelihood
from tdoa_loc.physics import generate_measurements


def plot_ml_function(
    geometry: Geometry,
    target: Target,
    sigma_los: float,
    sigma_b: float,
    mu_b: float,
    p_los: float,
    measurements: Optional[np.ndarray] = None,
    grid_range: Optional[Tuple[float, float]] = None,
    n_points: int = 150,
    seed: int = 0,
    fig: Optional[plt.Figure] = None,
    title: str = "",
) -> plt.Figure:
    """Plot the ML objective function surface and contour.

    Args:
        geometry:     Scene geometry (tx + receivers).
        target:       Target (used for true position marker and measurement gen).
        sigma_los:    LOS noise std (m).
        sigma_b:      NLOS scatter std (m).
        mu_b:         NLOS mean bias (m).
        p_los:        LOS probability.
        measurements: Pre-generated measurements. If None, one realisation is
                      generated from the target position.
        grid_range:   (x_min_max, y_min_max) symmetric range for the grid.
                      If None, inferred from receiver positions + 20 %.
        n_points:     Grid resolution per axis.
        seed:         RNG seed for measurement generation.
        fig:          Existing figure to draw into (new one created if None).
        title:        Figure suptitle.

    Returns:
        Matplotlib Figure.
    """
    # ---- resolve true position and measurements -------------------------
    rng = np.random.default_rng(seed)
    true_pos = target.get_position(rng if not isinstance(target, FixedTarget) else None)

    if measurements is None:
        measurements = generate_measurements(
            true_pos, geometry, sigma_los, sigma_b, mu_b, p_los, rng
        )

    # ---- compute grid ---------------------------------------------------
    if grid_range is None:
        rx_extent = np.max(np.abs(geometry.receivers)) * 1.2
        lim = max(rx_extent, 50.0)
    else:
        lim = grid_range[1]

    x_vals = np.linspace(-lim, lim, n_points)
    y_vals = np.linspace(-lim, lim, n_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(n_points):
        for j in range(n_points):
            Z[j, i] = negative_log_likelihood(
                np.array([x_vals[i], y_vals[j]]),
                measurements, geometry, sigma_los, sigma_b, mu_b, p_los,
            )

    # ---- figure ---------------------------------------------------------
    if fig is None:
        fig = plt.figure(figsize=(14, 6))
    fig.clf()

    # -- left: 3D surface --
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(X, Y, Z, cmap="jet", rstride=2, cstride=2,
                              linewidth=0, antialiased=False, alpha=0.9)
    ax3d.set_xlabel("x (m)", labelpad=6)
    ax3d.set_ylabel("y (m)", labelpad=6)
    ax3d.set_zlabel(r"$J_{ML}(\mathbf{x})$", labelpad=6)
    ax3d.set_title("3D ML Objective Surface")
    ax3d.view_init(elev=30, azim=-60)

    # -- right: 2D contour + quiver --
    ax2d = fig.add_subplot(1, 2, 2)

    # Filled contour
    levels = np.linspace(Z.min(), np.percentile(Z, 95), 40)
    cf = ax2d.contourf(X, Y, Z, levels=levels, cmap="jet_r")
    plt.colorbar(cf, ax=ax2d, shrink=0.85, label=r"$J_{ML}$")

    # Contour lines
    ax2d.contour(X, Y, Z, levels=20, colors="k", linewidths=0.5, alpha=0.6)

    # Gradient arrows (quiver)
    skip = max(1, n_points // 20)
    dZdy, dZdx = np.gradient(Z, y_vals, x_vals)
    ax2d.quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        -dZdx[::skip, ::skip], -dZdy[::skip, ::skip],
        alpha=0.35, color="white", scale=None, scale_units="xy",
        width=0.002,
    )

    # Markers: TX, receivers, true target
    ax2d.plot(*geometry.tx, "b^", markersize=9, label="TX", zorder=5)
    rx = geometry.receivers
    ax2d.plot(rx[:, 0], rx[:, 1], "bs", markersize=7, label="Receivers", zorder=5)
    ax2d.plot(*true_pos, "rx", markersize=12, markeredgewidth=2.5,
              label="True Target", zorder=6)

    ax2d.set_xlabel("x (m)")
    ax2d.set_ylabel("y (m)")
    ax2d.set_title("2D Contour + Gradient")
    ax2d.legend(loc="upper right", fontsize=8)
    ax2d.set_xlim(-lim, lim)
    ax2d.set_ylim(-lim, lim)
    ax2d.set_aspect("equal")
    ax2d.grid(True, linestyle="--", alpha=0.3)

    suptitle = title or f"ML Objective Function  (σ_b={sigma_b:.2f} m, μ_b={mu_b:.2f} m)"
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    return fig
