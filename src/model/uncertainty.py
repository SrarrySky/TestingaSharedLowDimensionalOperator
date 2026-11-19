"""Uncertainty propagation and forecast-interval calibration helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm


def _validate_gaussian_posterior(
    mean: np.ndarray, cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
     validate gaussian posterior.
    :param mean: Input argument used by this operation.
    :param cov: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if mean.ndim != 1 or cov.shape != (mean.size, mean.size):
        raise ValueError("mean and cov have incompatible shapes")
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError("mean and cov must be finite")
    if not np.allclose(cov, cov.T, atol=1e-10, rtol=1e-10):
        raise ValueError("cov must be symmetric")
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("cov must be positive semidefinite")
    return mean, cov


def edge_sign_probability(a0: float, b: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """P(a0 + b^T z > 0) for Gaussian filtered state posterior."""
    if not np.isfinite(a0):
        raise ValueError("a0 must be finite")
    b = np.asarray(b, dtype=float)
    mean, cov = _validate_gaussian_posterior(mean, cov)
    if b.shape != mean.shape or not np.isfinite(b).all():
        raise ValueError("b must be a finite vector with the same dimension as mean")
    m = float(a0 + b @ mean)
    var = float(b @ cov @ b)
    if var <= 1e-18:
        return float(m > 0.0)
    return float(norm.cdf(m / np.sqrt(var)))


def gaussian_state_draws(
    mean: np.ndarray,
    cov: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Gaussian state draws.
    :param mean: Input argument used by this operation.
    :param cov: Input argument used by this operation.
    :param draws: Input argument used by this operation.
    :param rng: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    mean, cov = _validate_gaussian_posterior(mean, cov)
    if isinstance(draws, bool | np.bool_) or not isinstance(draws, int | np.integer) or draws < 1:
        raise ValueError("draws must be a positive integer")
    return rng.multivariate_normal(mean, cov, size=draws)


def edge_sign_probability_draws(
    a0: float,
    b: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Return analytic positive-sign probabilities for many filtered posteriors."""
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if means.ndim != 2 or covariances.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError("means/covariances must have shapes (T,d)/(T,d,d)")
    return np.asarray(
        [
            edge_sign_probability(a0, b, mean, covariance)
            for mean, covariance in zip(means, covariances, strict=True)
        ]
    )


def transmitter_ranking_probability(
    transmitter_draws: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Estimate pairwise ranking probabilities from posterior metric draws.

    "transmitter_draws" has shape "(draws, n)" or "(..., draws, n)";
    the returned matrix has "n x n" entries and contains
    "P(transmitter_i > transmitter_j)".  Ties contribute one half.
    """
    values = np.asarray(transmitter_draws, dtype=float)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("transmitter_draws must be a finite array with a draw axis")
    if axis < -values.ndim or axis >= values.ndim:
        raise ValueError("axis is out of bounds for transmitter_draws")
    values = np.moveaxis(values, axis, 0)
    draws = values.shape[0]
    n = values.shape[-1]
    if n < 1:
        raise ValueError("transmitter_draws must contain at least one sector")
    # After moving the draw axis to the front, all middle axes are independent
    # origins/times and the final axis is the sector dimension.
    flat = values.reshape(draws, -1, n).transpose(1, 0, 2)
    out = np.zeros((flat.shape[0], n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            out[:, i, j] = np.mean(flat[:, :, i] > flat[:, :, j], axis=1)
            out[:, i, j] += 0.5 * np.mean(flat[:, :, i] == flat[:, :, j], axis=1)
    context_shape = values.shape[1:-1]
    return out[0] if not context_shape else out.reshape(context_shape + (n, n))


def predictive_state_paths(
    mean: np.ndarray,
    cov: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    u_future: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    *,
    return_initial: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Draw z_t then simulate z_{t+1:t+H} from the declared transition model."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)
    Q = np.asarray(Q, dtype=float)
    u_future = np.asarray(u_future, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")
    d = F.shape[0]
    if G.ndim != 2 or G.shape[0] != d:
        raise ValueError("G must have shape (state_dim, covariate_dim)")
    if Q.shape != (d, d) or not np.allclose(Q, Q.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Q must be a symmetric state_dim x state_dim matrix")
    if np.linalg.eigvalsh(Q).min() < -1e-10:
        raise ValueError("Q must be positive semidefinite")
    if u_future.ndim != 2 or u_future.shape[1] != G.shape[1]:
        raise ValueError("u_future must have shape (horizon, covariate_dim)")
    if (
        not np.isfinite(F).all()
        or not np.isfinite(G).all()
        or not np.isfinite(Q).all()
        or not np.isfinite(u_future).all()
    ):
        raise ValueError("transition inputs must be finite")
    H = u_future.shape[0]
    mean, cov = _validate_gaussian_posterior(mean, cov)
    if mean.shape != (d,) or cov.shape != (d, d):
        raise ValueError("mean/covariance dimension does not match F")
    if isinstance(draws, bool | np.bool_) or not isinstance(draws, int | np.integer) or draws < 1:
        raise ValueError("draws must be a positive integer")
    paths = np.zeros((draws, H, d))
    current = gaussian_state_draws(mean, cov, draws, rng)
    initial = current.copy()
    for h in range(H):
        noise = rng.multivariate_normal(np.zeros(d), Q, size=draws)
        current = current @ F.T + u_future[h] @ G.T + noise
        paths[:, h] = current
    return (initial, paths) if return_initial else paths


def posterior_metric_summary(
    state_draws: np.ndarray,
    metric: Callable[[np.ndarray], float],
    credible: float = 0.95,
) -> dict[str, float]:
    """
    Posterior metric summary.
    :param state_draws: Input argument used by this operation.
    :param metric: Input argument used by this operation.
    :param credible: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if not 0.0 < credible < 1.0:
        raise ValueError("credible must lie strictly between zero and one")
    if len(state_draws) < 2:
        raise ValueError("at least two state draws are required")
    values = np.asarray([metric(z) for z in state_draws], dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("metric must return finite scalar values")
    alpha = (1.0 - credible) / 2.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }
