"""Physical measurement model for bistatic passive TDOA localization.

Bistatic range for receiver i:
    d_i(x) = ||x - tx|| + ||x - rx_i||

Noisy measurement model (LOS/NLOS mixture):
    r_i = d_i(x_true) + n_i
    where n_i ~ N(0, sigma_los^2)          with probability p_los
          n_i ~ N(mu_b, sigma_b^2)         with probability p_nlos = 1 - p_los
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tdoa_loc.geometry import Geometry


def bistatic_distance(x: np.ndarray, tx: np.ndarray, rx: np.ndarray) -> float:
    """Compute bistatic (two-way) range: ||x - tx|| + ||x - rx||."""
    return float(np.linalg.norm(x - tx) + np.linalg.norm(x - rx))


def all_bistatic_distances(x: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Return bistatic distances from ``x`` to all receivers.

    Returns:
        Array of shape (N,) with d_i = ||x - tx|| + ||x - rx_i||.
    """
    x = np.asarray(x, dtype=float)
    d_tx = np.linalg.norm(x - geometry.tx)
    diff = geometry.receivers - x  # (N, 2)
    d_rxs = np.linalg.norm(diff, axis=1)  # (N,)
    return d_tx + d_rxs


def generate_measurements(
    true_target: np.ndarray,
    geometry: Geometry,
    sigma_los: float,
    sigma_b: float,
    mu_b: float,
    p_los: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate noisy bistatic range measurements for one Monte Carlo trial.

    Each measurement is independently drawn from a LOS/NLOS mixture:
        r_i = d_i + n_i
        n_i ~ N(0, sigma_los^2)   with prob p_los
        n_i ~ N(mu_b, sigma_b^2)  with prob 1 - p_los

    Args:
        true_target: True target position, shape (2,).
        geometry: Scene geometry (tx + receivers).
        sigma_los: Standard deviation of LOS noise (m).
        sigma_b: Standard deviation of NLOS scatter noise (m).
        mu_b: Mean of NLOS bias (m). Typically positive.
        p_los: Probability that a measurement is LOS.
        rng: NumPy random generator. Created if None.

    Returns:
        Noisy measurements array, shape (N,).
    """
    if rng is None:
        rng = np.random.default_rng()

    true_target = np.asarray(true_target, dtype=float)
    distances = all_bistatic_distances(true_target, geometry)
    n = len(distances)

    los_mask = rng.random(n) < p_los
    noise = np.where(
        los_mask,
        rng.normal(0.0, sigma_los, n),
        rng.normal(mu_b, sigma_b, n),
    )
    return distances + noise
