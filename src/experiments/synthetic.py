"""Competing synthetic DGPs used to falsify the DLC-SSM hypothesis.

The generators share one observation equation and differ only in how the VAR
operator path is produced.  Every dataset carries the full ground truth
operator path, support, latent factors/states, and a small metadata record so
that structural and geometry metrics can be computed without re-fitting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles
from scipy.special import ndtr

from model.dlcssm import DLCSSMParameters
from model.geometry import matrix_power_derivative, spectral_radius
from model.propagation import (
    companion_matrix,
    frozen_pathwise_residual,
)


@dataclass(frozen=True)
class SyntheticDataset:
    x: np.ndarray
    u: np.ndarray
    z: np.ndarray
    a_path: np.ndarray
    support: np.ndarray
    params: DLCSSMParameters | None
    scenario: str
    # Full truths are optional to preserve the original seven-argument API.
    operators: np.ndarray | None = None  # (T, lag, n, n)
    factors: np.ndarray | None = None  # latent common factor f_t
    regimes: np.ndarray | None = None
    observed_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _stability_lag_factors(operators: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Common lag scaling that bounds every supplied companion spectral radius.

    A_l -> s**l A_l scales all companion eigenvalues by s, unlike uniform
    coefficient scaling when L > 1. For a path, one common transform preserves
    an affine operator basis. This is simulation normalization, not a guarantee
    of stability for every possible future latent state or matrix product.
    """
    operators = np.asarray(operators, dtype=float)
    if (
        operators.ndim not in (3, 4)
        or min(operators.shape) < 1
        or operators.shape[-1] != operators.shape[-2]
        or not np.isfinite(operators).all()
    ):
        raise ValueError("operators must have finite shape (L,N,N) or (T,L,N,N)")
    if not np.isfinite(target) or not 0 < target < 1:
        raise ValueError("target must lie strictly between zero and one")
    path = operators[None] if operators.ndim == 3 else operators
    radius = max(spectral_radius(companion_matrix(a)) for a in path)
    scale = min(1.0, target / radius) if radius else 1.0
    return scale ** np.arange(1, path.shape[1] + 1, dtype=float)


def _stable_random_var(
    n: int, lag: int, rng: np.random.Generator, sparsity: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """
     stable random var.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param rng: Input argument used by this operation.
    :param sparsity: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    support = rng.random((lag, n, n)) < sparsity
    for ell in range(lag):
        np.fill_diagonal(support[ell], True)
    A = rng.normal(0.0, 0.04, size=(lag, n, n)) * support
    A[0] += 0.18 * np.eye(n)
    A *= _stability_lag_factors(A, 0.85)[:, None, None]
    return A, support


def _edge_path(operators: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
     edge path.
    :param operators: Input argument used by this operation.
    :param support: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return np.asarray([A[support] for A in operators], dtype=float)


def _simulate(
    operators: np.ndarray,
    support: np.ndarray,
    rng: np.random.Generator,
    *,
    u: np.ndarray | None = None,
    factors: np.ndarray | None = None,
    Lambda: np.ndarray | None = None,
    D: np.ndarray | None = None,
    noise_scale: float = 0.25,
    noise_df: float | None = None,
    heteroskedastic: bool = False,
    strong_noise_ar: float = 0.0,
    missing_rate: float = 0.0,
    asynchronous: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     simulate.
    :param operators: Input argument used by this operation.
    :param support: Input argument used by this operation.
    :param rng: Input argument used by this operation.
    :param u: Input argument used by this operation.
    :param factors: Input argument used by this operation.
    :param Lambda: Input argument used by this operation.
    :param D: Input argument used by this operation.
    :param noise_scale: Input argument used by this operation.
    :param noise_df: Input argument used by this operation.
    :param heteroskedastic: Input argument used by this operation.
    :param strong_noise_ar: Input argument used by this operation.
    :param missing_rate: Input argument used by this operation.
    :param asynchronous: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    T, lag, n, _ = operators.shape
    q = 0 if u is None else u.shape[1]
    k = 0 if factors is None else factors.shape[1]
    u = np.zeros((T, 0)) if u is None else u
    factors = np.zeros((T, 0)) if factors is None else factors
    Lambda = np.zeros((n, k)) if Lambda is None else Lambda
    D = np.zeros((n, q)) if D is None else D
    x = np.zeros((T, n), dtype=float)
    innovations = np.zeros((T, n), dtype=float)
    for t in range(T):
        if noise_df is None:
            eps = rng.normal(size=n)
        else:
            eps = rng.standard_t(noise_df, size=n) / np.sqrt(noise_df / (noise_df - 2.0))
        scale = (
            noise_scale * (0.5 + 0.8 * abs(factors[t, 0])) if heteroskedastic and k else noise_scale
        )
        eps = scale * eps + strong_noise_ar * (innovations[t - 1] if t else 0.0)
        innovations[t] = eps
        mean = Lambda @ factors[t] + D @ u[t]
        if t >= lag:
            mean = mean + sum(operators[t, ell] @ x[t - ell - 1] for ell in range(lag))
        x[t] = mean + eps
    observed = np.ones_like(x, dtype=bool)
    if missing_rate > 0:
        observed &= rng.random(x.shape) >= missing_rate
        if asynchronous:
            # Each series has its own observation clock.
            for j in range(n):
                observed[:, j] &= rng.random(T) >= missing_rate
        x = x.copy()
        x[~observed] = np.nan
    return x, observed, innovations


def _dataset(
    operators: np.ndarray,
    support: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    z: np.ndarray,
    scenario: str,
    *,
    factors: np.ndarray | None = None,
    regimes: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
    params: DLCSSMParameters | None = None,
    **metadata: Any,
) -> SyntheticDataset:
    """
     dataset.
    :param operators: Input argument used by this operation.
    :param support: Input argument used by this operation.
    :param x: Input argument used by this operation.
    :param u: Input argument used by this operation.
    :param z: Input argument used by this operation.
    :param scenario: Input argument used by this operation.
    :param factors: Input argument used by this operation.
    :param regimes: Input argument used by this operation.
    :param observed_mask: Input argument used by this operation.
    :param params: Input argument used by this operation.
    :param **metadata: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return SyntheticDataset(
        x=x,
        u=u,
        z=z,
        a_path=_edge_path(operators, support),
        support=support,
        params=params,
        scenario=scenario,
        operators=operators,
        factors=factors,
        regimes=regimes,
        observed_mask=observed_mask,
        metadata=metadata,
    )


def generate_static_var(T: int = 1200, n: int = 8, lag: int = 2, seed: int = 0) -> SyntheticDataset:
    """
    Generate static var.
    :param T: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)
    A, support = _stable_random_var(n, lag, rng)
    operators = np.repeat(A[None, ...], T, axis=0)
    u = rng.normal(size=(T, 2))
    f = rng.normal(size=(T, 1))
    x, mask, _ = _simulate(
        operators,
        support,
        rng,
        u=u,
        factors=f,
        Lambda=rng.normal(0, 0.08, (n, 1)),
        D=rng.normal(0, 0.03, (n, 2)),
    )
    return _dataset(
        operators,
        support,
        x,
        u,
        np.zeros((T, 1)),
        "static_var",
        factors=f,
        observed_mask=mask,
        base_operator=A.copy(),
    )


def generate_shared_low_rank(
    T: int = 1200, n: int = 8, lag: int = 2, d: int = 2, seed: int = 0
) -> SyntheticDataset:
    """
    Generate shared low rank.
    :param T: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param d: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)
    A0, support = _stable_random_var(n, lag, rng)
    B = rng.normal(0, 0.012, size=(d, lag, n, n)) * support[None, ...]
    F = np.diag(np.linspace(0.92, 0.75, d))
    G = rng.normal(0, 0.05, size=(d, 2))
    Q = 0.03 * np.eye(d)
    Lambda = rng.normal(0, 0.08, size=(n, d))
    D = rng.normal(0, 0.03, size=(n, 2))
    u = rng.normal(size=(T, 2))
    z = np.zeros((T, d))
    for t in range(1, T):
        z[t] = F @ z[t - 1] + G @ u[t] + rng.multivariate_normal(np.zeros(d), Q)
    operators = np.asarray([A0 + np.einsum("r,rlij->lij", z[t], B) for t in range(T)])
    # A common lag transform preserves the exact A0 + sum_r z_r B_r truth.
    lag_factors = _stability_lag_factors(operators)
    A0 *= lag_factors[:, None, None]
    B *= lag_factors[None, :, None, None]
    operators *= lag_factors[None, :, None, None]
    # The common factor is the same latent state used by the DLC-SSM truth;
    # this keeps the generated observation equation exactly consistent with
    # "params.Lambda" while still exposing it explicitly as "factors".
    factors = z.copy()
    noise_scale = 0.25
    x, mask, _ = _simulate(
        operators,
        support,
        rng,
        u=u,
        factors=factors,
        Lambda=Lambda,
        D=D,
        noise_scale=noise_scale,
    )
    params = DLCSSMParameters(
        c=np.zeros(n),
        A0=A0,
        B=B,
        Lambda=Lambda,
        D=D,
        F=F,
        G=G[:, :2],
        Q=Q,
        R=noise_scale**2 * np.eye(n),
        m0=z[lag].copy(),
        P0=np.eye(d),
    )
    return _dataset(
        operators,
        support,
        x,
        u,
        z,
        "shared_low_rank",
        factors=factors,
        observed_mask=mask,
        params=params,
        base_operator=A0.copy(),
        basis=B.copy(),
        state_transition=F.copy(),
        observation_noise_scale=noise_scale,
        stability_lag_factors=lag_factors,
        stability_scope="realized simulation path; not all latent states or switched products",
    )


def generate_common_factor_only(
    T: int = 1200, n: int = 8, lag: int = 2, d: int = 2, seed: int = 0
) -> SyntheticDataset:
    """Null cross-sectional structure: own-lag persistence plus a latent common factor.

    The DGP has *no* off-diagonal causal edges ("support" is diagonal-only), so any
    cross-sectional comovement is carried exclusively by the shared latent factor via
    "Lambda".  A method that hallucinates off-diagonal edges on this scenario is
    over-fitting spurious structure; a method that recovers only the diagonal and the
    factor loading is behaving correctly (honest refusal of absent causality).
    """
    rng = np.random.default_rng(seed)
    # Diagonal-only support: own-lag persistence, zero cross-sectional spillover.
    A0, support = _stable_random_var(n, lag, rng, sparsity=0.0)
    B = np.zeros((d, lag, n, n))
    # Common-factor dynamics identical in spirit to shared_low_rank.
    F = np.diag(np.linspace(0.92, 0.75, d))
    G = rng.normal(0, 0.05, size=(d, 2))
    Q = 0.03 * np.eye(d)
    Lambda = rng.normal(0, 0.08, size=(n, d))
    D = rng.normal(0, 0.03, size=(n, 2))
    u = rng.normal(size=(T, 2))
    z = np.zeros((T, d))
    for t in range(1, T):
        z[t] = F @ z[t - 1] + G @ u[t] + rng.multivariate_normal(np.zeros(d), Q)
    operators = np.repeat(A0[None], T, axis=0)
    factors = z.copy()
    noise_scale = 0.25
    x, mask, _ = _simulate(
        operators, support, rng, u=u, factors=factors, Lambda=Lambda, D=D, noise_scale=noise_scale
    )
    params = DLCSSMParameters(
        c=np.zeros(n),
        A0=A0,
        B=B,
        Lambda=Lambda,
        D=D,
        F=F,
        G=G[:, :2],
        Q=Q,
        R=noise_scale**2 * np.eye(n),
        m0=z[lag].copy(),
        P0=np.eye(d),
    )
    return _dataset(
        operators,
        support,
        x,
        u,
        z,
        "common_factor_only",
        factors=factors,
        observed_mask=mask,
        params=params,
        base_operator=A0.copy(),
        basis=B.copy(),
        state_transition=F.copy(),
        observation_noise_scale=noise_scale,
        stability_scope="diagonal-only VAR plus latent common factor; no cross-sectional edges",
    )


def generate_discrete_regime(
    T: int = 1200, n: int = 8, lag: int = 2, seed: int = 1
) -> SyntheticDataset:
    """
    Generate discrete regime.
    :param T: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)
    A0, support = _stable_random_var(n, lag, rng)
    delta = rng.normal(0, 0.03, size=A0.shape) * support
    A1, A2 = A0 - delta, A0 + delta
    trans = np.array([[0.97, 0.03], [0.04, 0.96]])
    regimes = np.zeros(T, dtype=int)
    for t in range(1, T):
        regimes[t] = rng.choice(2, p=trans[regimes[t - 1]])
    operators = np.asarray([A1 if s == 0 else A2 for s in regimes])
    u = rng.normal(size=(T, 2))
    x, mask, _ = _simulate(operators, support, rng, u=u, noise_scale=0.25)
    return _dataset(
        operators,
        support,
        x,
        u,
        np.where(regimes[:, None] == 0, -1.0, 1.0),
        "discrete_regime",
        regimes=regimes,
        observed_mask=mask,
        transition_matrix=trans,
        regime_operators=np.stack([A1, A2]),
    )


def generate_smooth_calendar(
    T: int = 1200, n: int = 8, lag: int = 2, seed: int = 2
) -> SyntheticDataset:
    """
    Generate smooth calendar.
    :param T: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)
    A0, support = _stable_random_var(n, lag, rng)
    B1 = rng.normal(0, 0.02, A0.shape) * support
    phase = np.sin(np.linspace(0, 6 * np.pi, T))
    operators = np.asarray([A0 + phase[t] * B1 for t in range(T)])
    u = rng.normal(size=(T, 2))
    x, mask, _ = _simulate(operators, support, rng, u=u)
    return _dataset(
        operators,
        support,
        x,
        u,
        phase[:, None],
        "smooth_calendar",
        observed_mask=mask,
        base_operator=A0.copy(),
        calendar_basis=B1.copy(),
    )


def generate_unrestricted_tvp(
    T: int = 1200, n: int = 8, lag: int = 2, seed: int = 3
) -> SyntheticDataset:
    """
    Generate unrestricted tvp.
    :param T: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)
    A, support = _stable_random_var(n, lag, rng)
    p = int(support.sum())
    state = A[support].copy()
    states = np.zeros((T, p))
    operators = np.zeros((T, lag, n, n))
    innovation_cov = (0.0015**2) * np.eye(p)
    for t in range(T):
        if t:
            state = state + rng.multivariate_normal(np.zeros(p), innovation_cov)
        op = np.zeros_like(A)
        op[support] = state
        op *= _stability_lag_factors(op)[:, None, None]
        state = op[support]
        operators[t], states[t] = op, state
    u = rng.normal(size=(T, 2))
    x, mask, _ = _simulate(operators, support, rng, u=u)
    return _dataset(
        operators,
        support,
        x,
        u,
        states,
        "unrestricted_tvp",
        observed_mask=mask,
        innovation_cov=innovation_cov,
        coefficient_state_transition=np.eye(p),
        stability_scope="lag-scaled random-walk proposals; not an unmodified Gaussian random walk",
    )


def generate_network_operator(
    T: int = 1200, n: int = 8, lag: int = 2, K: int = 3, seed: int = 4
) -> SyntheticDataset:
    """
    Generate network operator.
    :param T: Input argument used by this operation.
    :param n: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param K: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)
    A0, support = _stable_random_var(n, lag, rng)
    W = rng.normal(0, 0.025, size=(K, lag, n, n)) * support
    theta = np.zeros((T, K))
    F = np.diag(np.linspace(0.93, 0.80, K))
    for t in range(1, T):
        theta[t] = F @ theta[t - 1] + rng.normal(0, 0.04, K)
    operators = np.asarray([A0 + np.einsum("k,klij->lij", theta[t], W) for t in range(T)])
    lag_factors = _stability_lag_factors(operators)
    A0 *= lag_factors[:, None, None]
    W *= lag_factors[None, :, None, None]
    operators *= lag_factors[None, :, None, None]
    u = rng.normal(size=(T, 2))
    x, mask, _ = _simulate(operators, support, rng, u=u)
    return _dataset(
        operators,
        support,
        x,
        u,
        theta,
        "network_operator",
        observed_mask=mask,
        base_operator=A0.copy(),
        known_basis=W.copy(),
        state_transition=F.copy(),
    )


def matched_spectrum_nonnormal_pair(
    alpha: float = 0.8, coupling: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Two 2x2 matrices with the same eigenvalues but different non-normality."""
    low = np.diag([alpha, alpha * 0.9])
    high = np.array([[alpha, coupling], [0.0, alpha * 0.9]])
    return low, high


def generate_matched_nonnormal(
    T: int = 600,
    n: int = 2,
    lag: int = 1,
    horizon: int = 5,
    seed: int = 5,
    alpha: float = 0.8,
    coupling: float = 2.0,
    **_: int,
) -> SyntheticDataset:
    """Generate matched-spectrum low/high non-normal systems.

    The original implementation silently ignored "n" and "lag".  That made
    the scenario dispatcher return a 2-variable lag-1 process even when the
    experiment requested a different dimensionality, and could silently
    invalidate a method-by-DGP comparison.  The 2x2 construction is embedded
    in the requested observation dimension; additional lags are zero so that
    both systems retain exactly the same companion spectrum.
    """
    if isinstance(T, bool | np.bool_) or T < 1:
        raise ValueError("T must be a positive integer")
    if isinstance(n, bool | np.bool_) or n < 2:
        raise ValueError("n must be at least 2")
    if isinstance(lag, bool | np.bool_) or lag < 1:
        raise ValueError("lag must be a positive integer")
    if isinstance(horizon, bool | np.bool_) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    rng = np.random.default_rng(seed)
    low_block, high_block = matched_spectrum_nonnormal_pair(alpha, coupling)
    low = np.zeros((n, n), dtype=float)
    high = np.zeros((n, n), dtype=float)
    # Use the same diagonal/eigenvalues for both systems.  The upper
    # triangular coupling changes transient geometry without changing them.
    low[:2, :2] = low_block
    high[:2, :2] = high_block
    if n > 2:
        extra = np.linspace(alpha * 0.7, alpha * 0.3, n - 2)
        low[2:, 2:] = np.diag(extra)
        high[2:, 2:] = np.diag(extra)

    support = np.zeros((lag, n, n), dtype=bool)
    support[0] = True
    low_lags = np.zeros((lag, n, n), dtype=float)
    high_lags = np.zeros((lag, n, n), dtype=float)
    low_lags[0] = low
    high_lags[0] = high
    operators = np.repeat(low_lags[None, ...], T, axis=0)
    x, mask, _ = _simulate(operators, support, rng, noise_scale=0.08)
    high_path = np.repeat(high_lags[None, ...], T, axis=0)
    high_x, _, _ = _simulate(
        high_path,
        support,
        np.random.default_rng(seed + 1),
        noise_scale=0.08,
    )
    gains = np.array(
        [np.linalg.norm(np.linalg.matrix_power(low, h), 2) for h in range(1, horizon + 1)]
    )
    high_gains = np.array(
        [np.linalg.norm(np.linalg.matrix_power(high, h), 2) for h in range(1, horizon + 1)]
    )
    return _dataset(
        operators,
        support,
        x,
        np.zeros((T, 0)),
        np.zeros((T, 1)),
        "matched_nonnormal",
        observed_mask=mask,
        low_operator=low,
        high_operator=high,
        low_lags=low_lags,
        high_lags=high_lags,
        high_x=high_x,
        low_gain=gains,
        high_gain=high_gains,
        horizon=horizon,
    )


def generate_financial_stress(
    stress: str, T: int = 1200, n: int = 8, lag: int = 2, seed: int = 10
) -> SyntheticDataset:
    """Generate one S7 stress factor at a time, with a reproducible label."""
    key = stress.lower().replace("-", "_")
    rng = np.random.default_rng(seed)
    base = generate_static_var(T=T, n=n, lag=lag, seed=seed)
    operators = np.asarray(base.operators).copy()
    factors = np.asarray(base.factors).copy() if base.factors is not None else np.zeros((T, 1))
    u = base.u.copy()
    kwargs: dict[str, Any] = {}
    if key in {"latent_confounder", "confounder"}:
        factors = rng.normal(size=(T, 2))
        kwargs["Lambda"] = rng.normal(0, 0.15, (n, 2))
    elif key in {"student_t", "heavy_tail", "heavy_tailed"}:
        kwargs["noise_df"] = 4.0
    elif key in {"heteroskedastic", "heteroskedasticity"}:
        kwargs["heteroskedastic"] = True
    elif key in {"strong_autocorrelation", "autocorrelation"}:
        kwargs["strong_noise_ar"] = 0.65
    elif key in {"missing", "missing_async", "asynchronous"}:
        kwargs["missing_rate"] = 0.15
        kwargs["asynchronous"] = True
    elif key in {"structural_break", "break"}:
        operators[T // 2 :] = (
            operators[T // 2 :] + rng.normal(0, 0.035, operators[T // 2 :].shape) * base.support
        )
        for t in range(T // 2, T):
            operators[t] *= _stability_lag_factors(operators[t])[:, None, None]
    elif key in {"support_misspecification", "misspecified_support"}:
        reported = base.support.copy()
        edges = np.argwhere(reported)
        for ell, i, j in edges[::3]:
            reported[ell, i, j] = False
        # Preserve the true support in metadata while exposing the misspecified one.
        x, mask, _ = _simulate(
            operators,
            base.support,
            rng,
            u=u,
            factors=factors,
            Lambda=np.zeros((n, 1)),
            D=np.zeros((n, 2)),
        )
        return _dataset(
            operators,
            reported,
            x,
            u,
            np.zeros((T, 1)),
            "stress_support_misspecification",
            factors=factors,
            observed_mask=mask,
            true_support=base.support.copy(),
            true_a_path=_edge_path(operators, base.support),
            stress=key,
        )
    else:
        raise ValueError(f"Unknown stress factor: {stress}")
    x, mask, _ = _simulate(
        operators,
        base.support,
        rng,
        u=u,
        factors=factors,
        Lambda=kwargs.pop("Lambda", np.zeros((n, 1))),
        D=np.zeros((n, 2)),
        **kwargs,
    )
    return _dataset(
        operators,
        base.support,
        x,
        u,
        np.zeros((T, 1)),
        f"stress_{key}",
        factors=factors,
        observed_mask=mask,
        stress=key,
    )


def generate_scenario(scenario: str, **kwargs: int) -> SyntheticDataset:
    """
    Generate scenario.
    :param scenario: Input argument used by this operation.
    :param **kwargs: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    key = scenario.lower().replace("-", "_")
    if key in {"static", "static_var", "s0"}:
        return generate_static_var(**kwargs)
    if key in {"shared_low_rank", "low_rank", "s5"}:
        return generate_shared_low_rank(**kwargs)
    if key in {"common_factor_only", "null_structure", "no_causal_edges", "s8"}:
        return generate_common_factor_only(**kwargs)
    if key in {"discrete_regime", "regime", "s1"}:
        return generate_discrete_regime(**kwargs)
    if key in {"smooth_calendar", "smooth_tvp", "s2"}:
        return generate_smooth_calendar(**kwargs)
    if key in {"unrestricted_tvp", "high_dimensional", "s3"}:
        return generate_unrestricted_tvp(**kwargs)
    if key in {"network_operator", "prespecified_network", "s4"}:
        return generate_network_operator(**kwargs)
    if key in {"matched_nonnormal", "matched_spectrum", "s6"}:
        return generate_matched_nonnormal(**kwargs)
    if key == "s7":
        return generate_financial_stress(str(kwargs.pop("stress", "student_t")), **kwargs)
    if key.startswith("stress_"):
        return generate_financial_stress(key[7:], **kwargs)
    if key in {
        "latent_confounder",
        "confounder",
        "student_t",
        "heavy_tail",
        "heteroskedastic",
        "strong_autocorrelation",
        "missing_async",
        "structural_break",
        "support_misspecification",
    }:
        return generate_financial_stress(key, **kwargs)
    raise ValueError(f"Unknown synthetic scenario: {scenario}")


def _f1(precision: float, recall: float) -> float:
    """
     f1.
    :param precision: Input argument used by this operation.
    :param recall: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def synthetic_metrics(
    true_support: np.ndarray,
    estimated_support: np.ndarray,
    true_a_path: np.ndarray | None = None,
    estimated_a_path: np.ndarray | None = None,
) -> dict[str, float]:
    """Ground-truth metrics used by the method × DGP table/heatmap."""
    y = np.asarray(true_support, bool).ravel()
    yhat = np.asarray(estimated_support, bool).ravel()
    tp = float(np.sum(y & yhat))
    fp = float(np.sum(~y & yhat))
    fn = float(np.sum(y & ~yhat))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {"edge_precision": precision, "edge_recall": recall, "edge_f1": _f1(precision, recall)}
    # Orientation ignores lag multiplicity and evaluates only off-diagonal
    # directed pairs, as required by the synthetic protocol.
    if y.ndim == 1 and np.asarray(true_support).ndim == 3:
        ts, es = np.asarray(true_support, bool), np.asarray(estimated_support, bool)
        to = np.any(ts, axis=0)
        eo = np.any(es, axis=0)
        np.fill_diagonal(to, False)
        np.fill_diagonal(eo, False)
        otp = float(np.sum(to & eo))
        ofp = float(np.sum(~to & eo))
        ofn = float(np.sum(to & ~eo))
        out["orientation_precision"] = otp / (otp + ofp) if otp + ofp else 0.0
        out["orientation_recall"] = otp / (otp + ofn) if otp + ofn else 0.0
    if true_a_path is not None and estimated_a_path is not None:
        out["dynamic_coefficient_rmse"] = float(
            np.sqrt(np.mean((np.asarray(true_a_path) - np.asarray(estimated_a_path)) ** 2))
        )
    return out


def _basis_columns(basis: np.ndarray) -> np.ndarray:
    """Convert common basis layouts to "(coefficient_dim, rank)".

    DLC-SSM stores bases as "(d, lag, target, source)" while estimated
    coefficient subspaces are often already "(p, d)".  Treating every input
    as if its first axis were the rank was the source of the previous reshape
    bug.  The convention below is explicit for the model layout and preserves
    ordinary two-dimensional "(p, d)" matrices.
    """
    array = np.asarray(basis, dtype=float)
    if array.ndim == 2:
        return array
    if array.ndim == 4:
        return array.reshape(array.shape[0], -1).T
    if array.ndim == 3:
        # A lag-major tensor (lag, target, source) represents one column.
        return array.reshape(-1, 1)
    raise ValueError("basis must have shape (p, d), (d, L, n, n), or (L, n, n)")


def operator_subspace_metrics(
    true_basis: np.ndarray,
    estimated_basis: np.ndarray,
) -> dict[str, float]:
    """Compare operator subspaces using principal angles and projections."""
    true_columns = _basis_columns(true_basis)
    estimated_columns = _basis_columns(estimated_basis)
    if true_columns.shape[0] != estimated_columns.shape[0]:
        raise ValueError("true_basis and estimated_basis have different coefficient dimensions")
    if true_columns.shape[1] == 0 or estimated_columns.shape[1] == 0:
        raise ValueError("basis must contain at least one column")
    true_u, true_s, _ = np.linalg.svd(true_columns, full_matrices=False)
    estimated_u, estimated_s, _ = np.linalg.svd(estimated_columns, full_matrices=False)
    true_rank = int(np.sum(true_s > np.finfo(float).eps * true_s[0])) if true_s.size else 0
    estimated_rank = (
        int(np.sum(estimated_s > np.finfo(float).eps * estimated_s[0])) if estimated_s.size else 0
    )
    if true_rank != estimated_rank:
        raise ValueError("true_basis and estimated_basis have different effective ranks")
    rank = true_rank
    true_u = true_u[:, :rank]
    estimated_u = estimated_u[:, :rank]
    angles = subspace_angles(true_u, estimated_u)
    true_projection = true_u @ true_u.T
    estimated_projection = estimated_u @ estimated_u.T
    return {
        "principal_angle_max": float(np.max(angles, initial=0.0)),
        "projection_distance": (
            float(np.linalg.norm(true_projection - estimated_projection, 2))
            if true_projection.size
            else 0.0
        ),
        "effective_rank": float(true_rank),
    }


def theory_checks(seed: int = 11, horizon: int = 5) -> dict[str, float]:
    """Numerically validate T1–T6 from the paper on a small analytic system."""
    rng = np.random.default_rng(seed)
    ds = generate_shared_low_rank(T=250, n=3, lag=2, d=2, seed=seed)
    p = ds.params
    assert p is not None
    Rg = np.array([[1.2, 0.3], [-0.2, 0.9]])
    z = rng.normal(size=2)
    pg = p.apply_gauge(Rg)
    err_gauge = float(np.linalg.norm(p.a_lags(z) - pg.a_lags(Rg @ z)))
    a = ds.a_path
    centered = np.diff(a, axis=0)
    s = np.linalg.svd(np.cov(centered, rowvar=False), compute_uv=False)
    rank_err = float(np.sum(s[2:] ** 2) / (np.sum(s**2) + 1e-12))
    C = p.companion(z)
    Cr = np.zeros_like(C)
    Cr[:3, :6] = np.concatenate([p.B[0, 0], p.B[0, 1]], axis=1)
    eps = 1e-6
    analytic = matrix_power_derivative(C, Cr, horizon)
    numeric = (
        np.linalg.matrix_power(p.companion(z + eps * np.array([1.0, 0.0])), horizon)
        - np.linalg.matrix_power(p.companion(z - eps * np.array([1.0, 0.0])), horizon)
    ) / (2 * eps)
    C0 = companion_matrix(np.asarray(ds.operators[0]))
    path = [companion_matrix(ds.operators[t]) for t in range(1, horizon + 1)]
    decomp = frozen_pathwise_residual(C0, path)
    # affine Gaussian sign probability versus Monte Carlo frequency (T6)
    m, sd = 0.2, 0.7
    analytic_prob = float(ndtr(m / sd))
    draws = rng.normal(m, sd, 20000)
    mc_prob = float(np.mean(draws > 0))
    return {
        "gauge_invariance_error": err_gauge,
        "low_rank_tail_energy": rank_err,
        "sensitivity_finite_difference_error": float(np.linalg.norm(analytic - numeric)),
        "frozen_pathwise_residual": float(decomp),
        "sign_probability_analytic": analytic_prob,
        "sign_probability_mc_error": abs(analytic_prob - mc_prob),
    }


def method_dgp_heatmap(
    results: Mapping[str, Mapping[str, float]], metric: str = "edge_f1"
) -> np.ndarray:
    """Return a deterministic method × DGP matrix for plotting."""
    methods = list(results)
    scenarios = sorted({s for row in results.values() for s in row})
    return np.asarray(
        [
            [
                float(results[m].get(s, {}).get(metric, np.nan))
                if isinstance(results[m].get(s), Mapping)
                else float(results[m].get(s, np.nan))
                for s in scenarios
            ]
            for m in methods
        ]
    ), scenarios
