"""Linear transition propagation and block-companion matrix construction."""

from __future__ import annotations

import numpy as np


def companion_matrix(a_lags: np.ndarray) -> np.ndarray:
    """VAR(L) companion matrix for A_lags shape (L, N, N)."""
    L, n, n2 = a_lags.shape
    if n != n2:
        raise ValueError("A_lags must contain square matrices")
    C = np.zeros((n * L, n * L), dtype=float)
    C[:n, : n * L] = np.concatenate(list(a_lags), axis=1)
    if L > 1:
        C[n:, :-n] = np.eye(n * (L - 1))
    return C


def selection_matrix(n: int, lag: int) -> np.ndarray:
    """
    Selection matrix.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    J = np.zeros((n, n * lag))
    J[:, :n] = np.eye(n)
    return J


def pair_response(C: np.ndarray, n: int, lag: int, i: int, j: int, h: int) -> float:
    """
    Pair response.
    :param C: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param i: Input argument used by this operation.
    :param j: Input argument used by this operation.
    :param h: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    J = selection_matrix(n, lag)
    ei = np.eye(n)[:, i]
    ej = np.eye(n)[:, j]
    return float(ej @ J @ np.linalg.matrix_power(C, h) @ J.T @ ei)


def propagation_energy(
    C: np.ndarray,
    n: int,
    lag: int,
    i: int,
    j: int,
    horizon: int,
    discount: float = 1.0,
) -> float:
    """
    Propagation energy.
    :param C: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param i: Input argument used by this operation.
    :param j: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :param discount: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return float(
        sum(
            (discount ** (h - 1)) * pair_response(C, n, lag, i, j, h) ** 2
            for h in range(1, horizon + 1)
        )
    )


def propagation_matrix(
    C: np.ndarray,
    n: int,
    lag: int,
    horizon: int,
    discount: float = 1.0,
) -> np.ndarray:
    """
    Propagation matrix.
    :param C: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :param discount: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    P = np.zeros((n, n), dtype=float)
    J = selection_matrix(n, lag)
    power = np.eye(C.shape[0])
    for h in range(1, horizon + 1):
        power = C @ power
        # R[target, source] = r_{source -> target}(h)
        R = J @ power @ J.T
        # Store as P[source, target].
        P += (discount ** (h - 1)) * (R * R).T

    return P


def directionality_ratio(P: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Directionality ratio.
    :param P: Input argument used by this operation.
    :param epsilon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return np.log((epsilon + P) / (epsilon + P.T))


def transmitter_scores(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transmitter scores.
    :param P: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    q = P.copy()
    np.fill_diagonal(q, 0.0)
    transmitter = q.sum(axis=1)
    receiver = q.sum(axis=0)
    return transmitter, receiver, transmitter - receiver


def reachability_gramian(C: np.ndarray, n: int, lag: int, i: int, horizon: int) -> np.ndarray:
    """
    Reachability gramian.
    :param C: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param i: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    J = selection_matrix(n, lag)
    b = J.T @ np.eye(n)[:, i]
    W = np.zeros_like(C)
    power = np.eye(C.shape[0])
    for _ in range(horizon):
        v = power @ b
        W += np.outer(v, v)
        power = C @ power
    return W


def pathwise_product(C_path: list[np.ndarray], horizon: int | None = None) -> np.ndarray:
    """Return C_h ... C_1 using the first `horizon` matrices."""
    if not C_path:
        raise ValueError("C_path must not be empty")
    h = len(C_path) if horizon is None else horizon
    if h < 1 or h > len(C_path):
        raise ValueError("Invalid horizon")
    Phi = np.eye(C_path[0].shape[0])
    for C in C_path[:h]:
        Phi = C @ Phi
    return Phi


def frozen_pathwise_decomposition(C0: np.ndarray, C_path: list[np.ndarray]) -> np.ndarray:
    """Exact telescoping RHS for Phi_h - C0^h.

    C_path = [C_1, ..., C_h].
    """
    h = len(C_path)
    d = C0.shape[0]
    total = np.zeros((d, d))
    for m in range(1, h + 1):
        left = np.eye(d)
        # C_h ... C_{m+1}
        for idx in range(h, m, -1):
            left = left @ C_path[idx - 1]
        total += left @ (C_path[m - 1] - C0) @ np.linalg.matrix_power(C0, m - 1)
    return total


def frozen_pathwise_residual(C0: np.ndarray, C_path: list[np.ndarray]) -> float:
    """
    Frozen pathwise residual.
    :param C0: Input argument used by this operation.
    :param C_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    phi = pathwise_product(C_path)
    rhs = frozen_pathwise_decomposition(C0, C_path)
    lhs = phi - np.linalg.matrix_power(C0, len(C_path))
    return float(np.linalg.norm(lhs - rhs, ord="fro"))


def state_migration_gap(C0: np.ndarray, C_path: list[np.ndarray]) -> float:
    """
    State migration gap.
    :param C0: Input argument used by this operation.
    :param C_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    max_gap = 0.0
    for h in range(1, len(C_path) + 1):
        phi = pathwise_product(C_path, h)
        frozen = np.linalg.matrix_power(C0, h)
        max_gap = max(max_gap, float(np.linalg.norm(phi - frozen, 2)))
    return max_gap


def pathwise_gain(C_path: list[np.ndarray]) -> float:
    """
    Pathwise gain.
    :param C_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    gain = 0.0
    for h in range(1, len(C_path) + 1):
        gain = max(gain, float(np.linalg.norm(pathwise_product(C_path, h), 2)))
    return gain
