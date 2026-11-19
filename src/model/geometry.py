"""Spectral and subspace diagnostics for latent transition operators.

The diagnostics are descriptive rather than model-fitting procedures: they
quantify stability, non-normality, and geometric alignment while preserving
the matrix dimensions and conventions used by the state-space models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import subspace_angles

from model.propagation import companion_matrix, propagation_matrix, selection_matrix


def spectral_radius(C: np.ndarray) -> float:
    """Return the largest eigenvalue modulus of a square transition matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def nonnormality(C: np.ndarray, epsilon: float = 1e-12) -> float:
    """Measure departure from normality using the Frobenius commutator norm."""
    comm = C.T @ C - C @ C.T
    return float(np.linalg.norm(comm, "fro") / (np.linalg.norm(C, "fro") ** 2 + epsilon))


def companion_nonnormality(
    C: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """
    Companion nonnormality.
    :param C: Input argument used by this operation.
    :param epsilon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return nonnormality(
        C,
        epsilon=epsilon,
    )


def shift_adjusted_nonnormality(
    C: np.ndarray,
    n: int,
    lag: int,
    epsilon: float = 1e-12,
) -> float:
    """Measure non-normality after removing the deterministic companion shift."""
    expected = n * lag

    if C.shape != (expected, expected):
        raise ValueError("C shape is inconsistent with n and lag")

    zero_lags = np.zeros(
        (lag, n, n),
        dtype=float,
    )
    shift_only = companion_matrix(zero_lags)
    dynamic = C - shift_only
    comm = dynamic.T @ dynamic - dynamic @ dynamic.T
    denom = (
        np.linalg.norm(
            dynamic,
            "fro",
        )
        ** 2
        + epsilon
    )

    return float(
        np.linalg.norm(
            comm,
            "fro",
        )
        / denom
    )


def dynamic_nonnormality(
    C: np.ndarray,
    n: int,
    lag: int,
    epsilon: float = 1e-12,
) -> float:
    """
    Dynamic nonnormality.
    :param C: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param epsilon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return shift_adjusted_nonnormality(
        C=C,
        n=n,
        lag=lag,
        epsilon=epsilon,
    )


@dataclass(frozen=True)
class TransientGainProfile:
    """Finite-horizon transient-gain diagnostics."""

    gains: np.ndarray
    peak_gain: float
    peak_horizon: int


def transient_gain_profile(
    C: np.ndarray,
    horizon: int,
) -> TransientGainProfile:
    """Return ||C^h||_2 for every h and the earliest peak horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    power = np.eye(
        C.shape[0],
        dtype=float,
    )
    gains = np.empty(
        horizon,
        dtype=float,
    )

    for idx in range(horizon):
        power = C @ power
        gains[idx] = float(
            np.linalg.norm(
                power,
                2,
            )
        )

    # np.argmax returns the first occurrence,
    # giving a deterministic earliest-peak convention.
    peak_idx = int(np.argmax(gains))

    return TransientGainProfile(
        gains=gains,
        peak_gain=float(gains[peak_idx]),
        peak_horizon=peak_idx + 1,
    )


def transient_gain(
    C: np.ndarray,
    horizon: int,
) -> float:
    """Maximum finite-horizon transient gain."""
    return transient_gain_profile(
        C,
        horizon,
    ).peak_gain


@dataclass(frozen=True)
class ObservableGainProfile:
    """Shock-to-observed-variable finite-horizon gain."""

    gains: np.ndarray
    peak_gain: float
    peak_horizon: int


def observable_transient_gain_profile(
    C: np.ndarray,
    n: int,
    lag: int,
    horizon: int,
) -> ObservableGainProfile:
    """
    Observable transient gain profile.
    :param C: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    expected = n * lag

    if C.shape != (expected, expected):
        raise ValueError("C shape is inconsistent with n and lag")

    J = np.zeros(
        (n, n * lag),
        dtype=float,
    )
    J[:, :n] = np.eye(n)
    power = np.eye(
        C.shape[0],
        dtype=float,
    )
    gains = np.empty(
        horizon,
        dtype=float,
    )

    for idx in range(horizon):
        power = C @ power
        response_operator = J @ power @ J.T
        gains[idx] = float(
            np.linalg.norm(
                response_operator,
                2,
            )
        )

    peak_idx = int(np.argmax(gains))

    return ObservableGainProfile(
        gains=gains,
        peak_gain=float(gains[peak_idx]),
        peak_horizon=peak_idx + 1,
    )


def observable_transient_gain(
    C: np.ndarray,
    n: int,
    lag: int,
    horizon: int,
) -> float:
    """Maximum observed-variable propagation gain."""
    return observable_transient_gain_profile(
        C=C,
        n=n,
        lag=lag,
        horizon=horizon,
    ).peak_gain


def companion_basis(B: np.ndarray) -> np.ndarray:
    """Convert B shape (d, L, N, N) to companion derivative matrices C_r."""
    d, L, n, _ = B.shape
    bases = np.zeros((d, n * L, n * L))
    for r in range(d):
        bases[r, :n, : n * L] = np.concatenate(list(B[r]), axis=1)
    return bases


def matrix_power_derivative(C: np.ndarray, C_r: np.ndarray, h: int) -> np.ndarray:
    """
    Matrix power derivative.
    :param C: Input argument used by this operation.
    :param C_r: Input argument used by this operation.
    :param h: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    out = np.zeros_like(C)
    powers = [np.eye(C.shape[0])]
    for _ in range(h):
        powers.append(C @ powers[-1])
    for m in range(h):
        out += powers[m] @ C_r @ powers[h - 1 - m]
    return out


def pair_response_derivative(
    C: np.ndarray,
    C_r: np.ndarray,
    n: int,
    lag: int,
    i: int,
    j: int,
    h: int,
) -> float:
    """
    Pair response derivative.
    :param C: Input argument used by this operation.
    :param C_r: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param i: Input argument used by this operation.
    :param j: Input argument used by this operation.
    :param h: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    J = selection_matrix(n, lag)
    D = matrix_power_derivative(C, C_r, h)
    return float(np.eye(n)[:, j] @ J @ D @ J.T @ np.eye(n)[:, i])


def energy_derivative(
    C: np.ndarray,
    C_r: np.ndarray,
    n: int,
    lag: int,
    i: int,
    j: int,
    horizon: int,
    discount: float = 1.0,
) -> float:
    """
    Energy derivative.
    :param C: Input argument used by this operation.
    :param C_r: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param i: Input argument used by this operation.
    :param j: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :param discount: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    J = selection_matrix(n, lag)
    ei, ej = np.eye(n)[:, i], np.eye(n)[:, j]
    total = 0.0
    for h in range(1, horizon + 1):
        power = np.linalg.matrix_power(C, h)
        resp = float(ej @ J @ power @ J.T @ ei)
        dresp = float(ej @ J @ matrix_power_derivative(C, C_r, h) @ J.T @ ei)
        total += 2.0 * (discount ** (h - 1)) * resp * dresp
    return float(total)


def state_susceptibility(
    a_lags: np.ndarray,
    B: np.ndarray,
    horizon: int,
    discount: float = 1.0,
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Return per-state-dimension and aggregate propagation elasticity norms."""
    C = companion_matrix(a_lags)
    bases = companion_basis(B)
    L, n, _ = a_lags.shape
    P = propagation_matrix(C, n=n, lag=L, horizon=horizon, discount=discount)
    d = B.shape[0]
    per_r = np.zeros(d)
    for r in range(d):
        sq = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dp = energy_derivative(C, bases[r], n, L, i, j, horizon, discount)
                elasticity = dp / (epsilon + P[i, j])
                sq += elasticity * elasticity
        per_r[r] = np.sqrt(sq)
    return per_r, float(np.linalg.norm(per_r))


def reversal_hyperplane(a0: float, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Return normalized normal vector and offset for b^T z + a0 = 0."""
    norm = float(np.linalg.norm(b))
    if norm <= 0:
        raise ValueError("b must be nonzero")
    return b / norm, float(a0 / norm)


def reversal_distance(a0: float, b: np.ndarray, z: np.ndarray) -> float:
    """
    Reversal distance.
    :param a0: Input argument used by this operation.
    :param b: Input argument used by this operation.
    :param z: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    norm = float(np.linalg.norm(b))
    if norm <= 0:
        return float("inf")
    return float(abs(a0 + b @ z) / norm)


@dataclass(frozen=True)
class LowRankDiagnostics:
    singular_values: np.ndarray
    explained_ratio_d: float
    effective_rank: float
    eigengap_d: float


def low_rank_diagnostics(a_path: np.ndarray, d: int) -> LowRankDiagnostics:
    """Diagnostics on Delta a_t; eigengap uses sample covariance (ddof=1).

    Singular values remain on the centered-increment matrix scale. Unobserved
    covariance directions are zero eigenvalues, including when p exceeds T-2.
    """
    a_path = np.asarray(a_path, dtype=float)
    if a_path.ndim != 2 or not np.isfinite(a_path).all():
        raise ValueError("a_path must be a finite matrix with shape (T, p)")
    if (
        isinstance(d, bool | np.bool_)
        or not isinstance(d, int | np.integer)
        or not 1 <= d <= a_path.shape[1]
    ):
        raise ValueError("d must be an integer between 1 and the coefficient dimension")
    delta = np.diff(a_path, axis=0)
    if delta.shape[0] < 2:
        raise ValueError("Need at least three coefficient-path observations")
    _, s, _ = np.linalg.svd(delta - delta.mean(axis=0), full_matrices=False)
    energy = s * s
    total = float(energy.sum())
    ratio = float(energy[:d].sum() / total) if total > 0 else 1.0
    p = energy / total if total > 0 else np.zeros_like(energy)
    p = p[p > 0]
    erank = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
    eigenvalues = np.zeros(a_path.shape[1] + 1)
    eigenvalues[: len(energy)] = energy / (delta.shape[0] - 1)
    eigengap = float(eigenvalues[d - 1] - eigenvalues[d])
    return LowRankDiagnostics(s, ratio, erank, eigengap)


def operator_subspace_angle(B_true: np.ndarray, B_est: np.ndarray) -> np.ndarray:
    """
    Operator subspace angle.
    :param B_true: Input argument used by this operation.
    :param B_est: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return subspace_angles(B_true, B_est)


def davis_kahan_sin_theta_bound(perturbation_norm: float, eigengap: float) -> float:
    """Return a conservative Davis--Kahan sin-theta bound.

    For the symmetric covariance perturbation used by the operator-subspace
    diagnostic, we use the standard conservative form
    "min(1, 2 ||E||_2 / delta)".  The factor two is intentional: it keeps
    the implementation aligned with the finite-sample subspace guarantee used
    in the paper rather than silently reporting the sharper-looking but
    unsupported one-factor heuristic.
    """
    if not np.isfinite(perturbation_norm) or perturbation_norm < 0:
        raise ValueError("perturbation_norm must be finite and nonnegative")
    if not np.isfinite(eigengap):
        raise ValueError("eigengap must be finite")
    if eigengap <= 0:
        return float("inf")
    return float(min(1.0, 2.0 * perturbation_norm / eigengap))
