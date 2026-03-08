"""Maximum likelihood objective and Cramér-Rao Lower Bound for bistatic TDOA.

Conditional measurement density (Gaussian mixture in residual form):
    p(r_i | x) = p_los  * N(e_i; 0,    sigma_los^2)
               + p_nlos * N(e_i; mu_b,  sigma_b^2)
    where e_i(x) = r_i - d_i(x)

Joint log-likelihood (conditional independence):
    ell(x) = sum_i log[ p_los * N(...) + p_nlos * N(...) ]

MLE:  x_hat = argmax_x  ell(x)
           = argmin_x  -ell(x)

CRLB (approximate, mixture variance):
    sigma_eq^2 = p_los * sigma_los^2 + p_nlos * (sigma_los^2 + sigma_b^2)
    FIM  = (1 / sigma_eq^2) * H^T H
    CRLB = FIM^{-1}
    PEB  = sqrt(trace(CRLB))
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tdoa_loc.geometry import Geometry
from tdoa_loc.physics import all_bistatic_distances


# ---------------------------------------------------------------------------
# Gaussian PDF (scalar)
# ---------------------------------------------------------------------------

def _gaussian(x: float, mu: float, sigma: float) -> float:
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _gaussian_vec(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def log_likelihood(
    x: np.ndarray,
    r: np.ndarray,
    geometry: Geometry,
    sigma_los: float,
    sigma_b: float,
    mu_b: float,
    p_los: float,
) -> float:
    """Compute the joint log-likelihood ell(x) for measurements ``r``.

    Args:
        x: Candidate target position, shape (2,).
        r: Observed bistatic range measurements, shape (N,).
        geometry: Scene geometry.
        sigma_los: LOS noise std (m).
        sigma_b: NLOS scatter std (m).
        mu_b: NLOS mean bias (m).
        p_los: LOS probability.

    Returns:
        Scalar log-likelihood value (higher = more likely).
    """
    x = np.asarray(x, dtype=float)
    p_nlos = 1.0 - p_los
    d = all_bistatic_distances(x, geometry)
    e = r - d  # residuals, shape (N,)

    los_pdf = _gaussian_vec(e, 0.0, sigma_los)
    nlos_pdf = _gaussian_vec(e, mu_b, sigma_b)
    mixture = p_los * los_pdf + p_nlos * nlos_pdf

    return float(np.sum(np.log(np.maximum(mixture, 1e-300))))


def negative_log_likelihood(
    x: np.ndarray,
    r: np.ndarray,
    geometry: Geometry,
    sigma_los: float,
    sigma_b: float,
    mu_b: float,
    p_los: float,
) -> float:
    """Negative log-likelihood — for use with minimization algorithms."""
    return -log_likelihood(x, r, geometry, sigma_los, sigma_b, mu_b, p_los)


# ---------------------------------------------------------------------------
# Jacobian of bistatic distances w.r.t. target position
# ---------------------------------------------------------------------------

def compute_jacobian(x: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Compute the N×2 Jacobian matrix of bistatic distances at position ``x``.

    Row i:  grad_x d_i(x) = (x - tx)/||x - tx||  +  (x - rx_i)/||x - rx_i||

    Args:
        x: Target position, shape (2,).
        geometry: Scene geometry.

    Returns:
        H of shape (N, 2).
    """
    x = np.asarray(x, dtype=float)
    eps = 1e-9

    d_tx = np.linalg.norm(x - geometry.tx) + eps
    grad_tx = (x - geometry.tx) / d_tx  # shape (2,)

    diff_rx = x - geometry.receivers   # (N, 2)
    d_rxs = np.linalg.norm(diff_rx, axis=1, keepdims=True) + eps  # (N, 1)
    grad_rx = diff_rx / d_rxs  # (N, 2)

    H = grad_tx[np.newaxis, :] + grad_rx  # (N, 2)
    return H


# ---------------------------------------------------------------------------
# CRLB
# ---------------------------------------------------------------------------

def compute_crlb(
    x: np.ndarray,
    geometry: Geometry,
    sigma_los: float,
    sigma_b: float,
    p_los: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the approximate CRLB at position ``x``.

    The equivalent variance of the LOS/NLOS mixture is:
        sigma_eq^2 = p_los * sigma_los^2 + p_nlos * (sigma_los^2 + sigma_b^2)

    Args:
        x: Target position, shape (2,).
        geometry: Scene geometry.
        sigma_los: LOS noise std (m).
        sigma_b: NLOS scatter std (m).
        p_los: LOS probability.

    Returns:
        (FIM, CRLB, PEB) where FIM and CRLB are 2×2 arrays and PEB is scalar.
    """
    p_nlos = 1.0 - p_los
    sigma_eq2 = p_los * sigma_los ** 2 + p_nlos * (sigma_los ** 2 + sigma_b ** 2)

    H = compute_jacobian(x, geometry)  # (N, 2)
    FIM = (1.0 / sigma_eq2) * (H.T @ H)

    try:
        CRLB = np.linalg.inv(FIM)
        PEB = float(np.sqrt(np.trace(CRLB)))
    except np.linalg.LinAlgError:
        CRLB = np.full((2, 2), np.nan)
        PEB = np.nan

    return FIM, CRLB, PEB
