"""Dynamic latent conditional state-space model primitives and fitting logic.

This module stores the parameterisation used in the paper experiments: a
latent regime/covariate vector changes autoregressive coefficient matrices,
while the observation model remains conditionally Gaussian.  The comments
below distinguish algebraic reparameterisations (which preserve the observed
distribution) from statistical fitting operations (which change the model).
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from causal.causal_support import SupportResult
from model.geometry import spectral_radius
from model.kalman_em import (
    FilterResult,
    SmoothResult,
    kalman_filter,
    rts_smoother,
    transition_m_step,
)
from model.propagation import companion_matrix


@dataclass
class DLCSSMParameters:
    """Complete parameter bundle for the dynamic latent conditional SSM."""
    c: np.ndarray
    A0: np.ndarray
    B: np.ndarray
    Lambda: np.ndarray
    D: np.ndarray
    F: np.ndarray
    G: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray

    @property
    def state_dim(self) -> int:
        """
        State dim.
        :return: Result produced by this operation.
        """
        return self.B.shape[0]

    @property
    def lag(self) -> int:
        """
        Lag.
        :return: Result produced by this operation.
        """
        return self.A0.shape[0]

    @property
    def n(self) -> int:
        """
        N.
        :return: Result produced by this operation.
        """
        return self.A0.shape[1]

    def a_lags(self, z: np.ndarray) -> np.ndarray:
        """Evaluate all lag matrices at latent coordinate ``z``."""
        return self.A0 + np.einsum("r,rlij->lij", z, self.B)

    def companion(self, z: np.ndarray) -> np.ndarray:
        """Return the block companion transition induced by ``z``."""
        return companion_matrix(self.a_lags(z))

    def apply_gauge(self, Rg: np.ndarray) -> DLCSSMParameters:
        """Apply ``z* = Rg z`` while preserving the induced observation law.

        This transformation changes the latent coordinate system, not the
        observable process.  Every affected mean, loading, transition, and
        covariance term is transformed consistently so downstream diagnostics
        can compare gauge-equivalent fits.
        """
        Rinv = np.linalg.inv(Rg)
        B_new = np.einsum("sr,slij->rlij", Rinv, self.B)
        return DLCSSMParameters(
            c=self.c.copy(),
            A0=self.A0.copy(),
            B=B_new,
            Lambda=self.Lambda @ Rinv,
            D=self.D.copy(),
            F=Rg @ self.F @ Rinv,
            G=Rg @ self.G,
            Q=Rg @ self.Q @ Rg.T,
            R=self.R.copy(),
            m0=Rg @ self.m0,
            P0=Rg @ self.P0 @ Rg.T,
        )

    def coefficient_subspace(
        self, support: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a0 in R^p and B in R^(p x d) for active lag coefficients."""
        L, n, _ = self.A0.shape
        if support is None:
            support = np.ones((L, n, n), dtype=bool)
        rows_a: list[float] = []
        rows_b: list[np.ndarray] = []
        for lag in range(L):
            for j in range(n):
                for i in range(n):
                    if support[lag, j, i]:
                        rows_a.append(float(self.A0[lag, j, i]))
                        rows_b.append(self.B[:, lag, j, i].copy())
        return np.asarray(rows_a), np.asarray(rows_b).reshape(len(rows_a), self.state_dim)

    def coefficient_path(self, states: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
        """
        Coefficient path.
        :param states: Input argument used by this operation.
        :param support: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        a0, Bmat = self.coefficient_subspace(support)
        return a0[None, :] + states @ Bmat.T


@dataclass
class FitResult:
    params: DLCSSMParameters
    filtered: FilterResult
    smoothed: SmoothResult
    loglik_history: list[float]
    max_spectral_radius: float
    penalized_objective_history: list[float] = field(default_factory=list)
    penalty_history: list[float] = field(default_factory=list)
    optimization_to_output_gauge: np.ndarray | None = None
    termination_reason: str = "unknown"


def _lag_vector(x: np.ndarray, t: int, lag: int) -> np.ndarray:
    # [x_{t-1}, x_{t-2}, ...] with source variable changing fastest.
    """
     lag vector.
    :param x: Input argument used by this operation.
    :param t: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return np.concatenate([x[t - k] for k in range(1, lag + 1)])


def build_observation_terms(
    x: np.ndarray,
    u: np.ndarray,
    params: DLCSSMParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build observation terms.
    :param x: Input argument used by this operation.
    :param u: Input argument used by this operation.
    :param params: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    L, n, _ = params.A0.shape
    T = x.shape[0]
    if T <= L:
        raise ValueError("Not enough observations for lag order")
    y = x[L:].copy()
    u_seg = u[L:].copy()
    offsets = np.zeros_like(y)
    H = np.zeros((T - L, n, params.state_dim))

    for k, t in enumerate(range(L, T)):
        offset = params.c + params.D @ u[t]
        for lag in range(L):
            xlag = x[t - lag - 1]
            offset = offset + params.A0[lag] @ xlag
        offsets[k] = offset
        for r in range(params.state_dim):
            col = params.Lambda[:, r].copy()
            for lag in range(L):
                col += params.B[r, lag] @ x[t - lag - 1]
            H[k, :, r] = col
    return y, offsets, H, u_seg


def _static_initialization(
    x: np.ndarray,
    u: np.ndarray,
    lag: int,
    d: int,
    support: np.ndarray,
    ridge: float,
    jitter: float,
) -> tuple[DLCSSMParameters, np.ndarray]:
    """
     static initialization.
    :param x: Input argument used by this operation.
    :param u: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param d: Input argument used by this operation.
    :param support: Input argument used by this operation.
    :param ridge: Input argument used by this operation.
    :param jitter: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    T, n = x.shape
    q = u.shape[1]
    rows = []
    y = []
    for t in range(lag, T):
        rows.append(np.concatenate([[1.0], u[t], _lag_vector(x, t, lag)]))
        y.append(x[t])
    X = np.asarray(rows)
    Y = np.asarray(y)
    gram = X.T @ X + ridge * np.eye(X.shape[1])
    theta = np.linalg.solve(gram, X.T @ Y).T  # N x K

    c = theta[:, 0]
    D = theta[:, 1 : 1 + q]
    A0 = np.zeros((lag, n, n))
    start = 1 + q
    for ell in range(lag):
        A0[ell] = theta[:, start + ell * n : start + (ell + 1) * n]
        A0[ell] *= support[ell]

    base = np.zeros_like(Y)
    for k, t in enumerate(range(lag, T)):
        base[k] = c + D @ u[t]
        for ell in range(lag):
            base[k] += A0[ell] @ x[t - ell - 1]
    residual = Y - base
    U, s, Vt = np.linalg.svd(residual - residual.mean(axis=0), full_matrices=False)
    d_eff = min(d, Vt.shape[0])
    z = np.zeros((len(Y), d))
    z[:, :d_eff] = U[:, :d_eff] * s[:d_eff]
    z_scale = z.std(axis=0, ddof=1)
    z_scale = np.where(z_scale > 1e-8, z_scale, 1.0)
    z /= z_scale

    Lambda = np.linalg.lstsq(z, residual, rcond=None)[0].T
    B = np.zeros((d, lag, n, n))

    if len(z) > 1:
        Zprev = np.column_stack([z[:-1], u[lag + 1 :]])
        W = np.linalg.lstsq(Zprev, z[1:], rcond=None)[0].T
        F = W[:, :d]
        G = W[:, d:]
        state_resid = z[1:] - Zprev @ W.T
        Q = np.cov(state_resid.T) if len(state_resid) > 1 else np.eye(d)
    else:
        F = 0.8 * np.eye(d)
        G = np.zeros((d, q))
        Q = 0.1 * np.eye(d)
    Q = np.atleast_2d(Q) + jitter * np.eye(d)

    obs_resid = residual - z @ Lambda.T
    R = np.cov(obs_resid.T) if len(obs_resid) > 1 else np.eye(n)
    R = np.atleast_2d(R) + jitter * np.eye(n)

    params = DLCSSMParameters(
        c=c,
        A0=A0,
        B=B,
        Lambda=Lambda,
        D=D,
        F=F,
        G=G,
        Q=Q,
        R=R,
        m0=z[0].copy(),
        P0=np.eye(d),
    )
    return params, z


def _pack_theta(params: DLCSSMParameters) -> np.ndarray:
    """
     pack theta.
    :param params: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    n, q = params.n, params.D.shape[1]
    L, d = params.lag, params.state_dim
    k_det = 1 + q + L * n
    k_s = 1 + L * n
    theta = np.zeros((n, k_det + d * k_s))
    theta[:, 0] = params.c
    theta[:, 1 : 1 + q] = params.D
    for ell in range(L):
        theta[:, 1 + q + ell * n : 1 + q + (ell + 1) * n] = params.A0[ell]
    for r in range(d):
        base = k_det + r * k_s
        theta[:, base] = params.Lambda[:, r]
        for ell in range(L):
            theta[:, base + 1 + ell * n : base + 1 + (ell + 1) * n] = params.B[r, ell]
    return theta


def _unpack_theta(theta: np.ndarray, params: DLCSSMParameters) -> None:
    """
     unpack theta.
    :param theta: Input argument used by this operation.
    :param params: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    n, q = params.n, params.D.shape[1]
    L, d = params.lag, params.state_dim
    k_det = 1 + q + L * n
    k_s = 1 + L * n
    params.c = theta[:, 0].copy()
    params.D = theta[:, 1 : 1 + q].copy()
    for ell in range(L):
        params.A0[ell] = theta[:, 1 + q + ell * n : 1 + q + (ell + 1) * n]
    for r in range(d):
        base = k_det + r * k_s
        params.Lambda[:, r] = theta[:, base]
        for ell in range(L):
            params.B[r, ell] = theta[:, base + 1 + ell * n : base + 1 + (ell + 1) * n]


def _observation_m_step(
    x: np.ndarray,
    u: np.ndarray,
    smooth: SmoothResult,
    params: DLCSSMParameters,
    support: np.ndarray,
    lambda_group: float,
    lambda_B: float,
    lambda_ridge: float,
    prox_iterations: int,
    jitter: float,
    diagonal_R: bool,
    edge_weights: np.ndarray | None = None,
) -> None:
    """
     observation m step.
    :param x: Input argument used by this operation.
    :param u: Input argument used by this operation.
    :param smooth: Input argument used by this operation.
    :param params: Input argument used by this operation.
    :param support: Input argument used by this operation.
    :param lambda_group: Input argument used by this operation.
    :param lambda_B: Input argument used by this operation.
    :param lambda_ridge: Input argument used by this operation.
    :param prox_iterations: Input argument used by this operation.
    :param jitter: Input argument used by this operation.
    :param diagonal_R: Input argument used by this operation.
    :param edge_weights: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    L, n, _ = params.A0.shape
    d, q = params.state_dim, u.shape[1]
    Tobs = x.shape[0] - L
    weights = np.ones_like(params.A0) if edge_weights is None else edge_weights
    k_det = 1 + q + L * n
    k_s = 1 + L * n
    K = k_det + d * k_s

    Gram = np.zeros((K, K))
    Cross = np.zeros((n, K))
    YY = np.zeros((n, n))

    for k, t in enumerate(range(L, x.shape[0])):
        lagvec = _lag_vector(x, t, L)
        det = np.concatenate([[1.0], u[t], lagvec])
        svec = np.concatenate([[1.0], lagvec])
        mz = smooth.mean[k]
        Ezz = smooth.cov[k] + np.outer(mz, mz)
        ew = np.kron(mz, svec)
        Eww = np.kron(Ezz, np.outer(svec, svec))

        phi_mean = np.concatenate([det, ew])
        Gram[:k_det, :k_det] += np.outer(det, det)
        Gram[:k_det, k_det:] += np.outer(det, ew)
        Gram[k_det:, :k_det] += np.outer(ew, det)
        Gram[k_det:, k_det:] += Eww
        Cross += np.outer(x[t], phi_mean)
        YY += np.outer(x[t], x[t])

    Gram /= Tobs
    Cross /= Tobs
    YY /= Tobs

    theta = _pack_theta(params)
    # Negative expected Gaussian log likelihood per observation, with R fixed.
    Rinv = np.linalg.solve(params.R, np.eye(n))
    lipschitz = float(np.linalg.norm(Rinv, 2) * np.linalg.norm(Gram, 2) + lambda_ridge + 1e-12)
    step = 1.0 / lipschitz

    def soft(v: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft.
        :param v: Input argument used by this operation.
        :param threshold: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    for _ in range(prox_iterations):
        grad = Rinv @ (theta @ Gram - Cross) + lambda_ridge * theta
        candidate = theta - step * grad

        for j in range(n):
            for ell in range(L):
                for i in range(n):
                    idx_a = 1 + q + ell * n + i
                    idx_b = [k_det + r * k_s + 1 + ell * n + i for r in range(d)]
                    if not support[ell, j, i]:
                        candidate[j, idx_a] = 0.0
                        candidate[j, idx_b] = 0.0
                        continue
                    bsoft = soft(candidate[j, idx_b], step * lambda_B * weights[ell, j, i])
                    group = np.concatenate([[candidate[j, idx_a]], bsoft])
                    norm = float(np.linalg.norm(group))
                    shrink = max(
                        0.0, 1.0 - step * lambda_group * weights[ell, j, i] / max(norm, 1e-18)
                    )
                    group *= shrink
                    candidate[j, idx_a] = group[0]
                    candidate[j, idx_b] = group[1:]

        if np.linalg.norm(candidate - theta) <= 1e-8 * (1.0 + np.linalg.norm(theta)):
            theta = candidate
            break
        theta = candidate

    _unpack_theta(theta, params)

    # Exact expected residual covariance under current latent moments.
    SSE = YY - theta @ Cross.T - Cross @ theta.T + theta @ Gram @ theta.T
    SSE = (SSE + SSE.T) * 0.5
    if diagonal_R:
        diag = np.maximum(np.diag(SSE), jitter)
        params.R = np.diag(diag)
    else:
        values, vectors = np.linalg.eigh(SSE)
        params.R = (vectors * np.maximum(values, jitter)) @ vectors.T


def canonical_gauge(
    params: DLCSSMParameters,
    smooth: SmoothResult,
    support: np.ndarray,
    *,
    return_transform: bool = False,
) -> tuple[DLCSSMParameters, SmoothResult] | tuple[DLCSSMParameters, SmoothResult, np.ndarray]:
    """Whiten the posterior mixture covariance, rotate B, and fix signs.

    This is an output representation only, not a penalty-preserving M-step.
    We do not center the states: that would require a transition intercept,
    which this model does not contain. Repeated eigenvalues leave rotations
    unidentified; the sign convention cannot resolve that degeneracy.
    """
    z = smooth.mean
    d = z.shape[1]
    centered = z - z.mean(axis=0)
    second = centered.T @ centered / len(z) + smooth.cov.mean(axis=0)
    second = (second + second.T) * 0.5
    evals, evecs = np.linalg.eigh(second)
    W = np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T

    p1 = params.apply_gauge(W)
    _, Bmat = p1.coefficient_subspace(support)
    Bmat = Bmat.reshape(-1, d)
    gram = Bmat.T @ Bmat
    evals_b, U = np.linalg.eigh(gram)
    order = np.argsort(evals_b)[::-1]
    U = U[:, order]
    Rrot = U.T
    p2 = p1.apply_gauge(Rrot)

    # deterministic sign convention from the largest absolute basis coefficient
    _, B2 = p2.coefficient_subspace(support)
    signs = np.ones(d)
    for r in range(d):
        idx = int(np.argmax(np.abs(B2[:, r]))) if B2.shape[0] else 0
        if B2.shape[0] and B2[idx, r] < 0:
            signs[r] = -1.0
    S = np.diag(signs)
    p3 = p2.apply_gauge(S)
    Rtotal = S @ Rrot @ W

    mean = smooth.mean @ Rtotal.T
    cov = np.einsum("ab,tbc,dc->tad", Rtotal, smooth.cov, Rtotal)
    lag_cov = np.einsum("ab,tbc,dc->tad", Rtotal, smooth.lag_cov, Rtotal)
    gain = Rtotal @ smooth.gain @ np.linalg.inv(Rtotal)
    result = SmoothResult(mean, cov, lag_cov, gain)
    if return_transform:
        return p3, result, Rtotal
    return p3, result


def _penalty(
    params: DLCSSMParameters,
    lambda_group: float,
    lambda_B: float,
    lambda_ridge: float,
    edge_weights: np.ndarray | None = None,
) -> float:
    """Penalty in the fixed optimization coordinates (Q = I)."""
    weights = np.ones_like(params.A0) if edge_weights is None else edge_weights
    groups = np.sqrt(params.A0**2 + np.sum(params.B**2, axis=0))
    return float(
        lambda_group * np.sum(weights * groups)
        + lambda_B * np.sum(weights[None] * np.abs(params.B))
        + 0.5 * lambda_ridge * np.sum(_pack_theta(params) ** 2)
    )


def _expected_objective(
    x: np.ndarray, u: np.ndarray, smooth: SmoothResult, params: DLCSSMParameters, penalty: float
) -> float:
    """Negative complete-data Q / Tobs + penalty, omitting constants.

    Used as a GEM acceptance check with the OLD posterior held fixed.
    Includes the initial state and transitions, not just observation residuals.
    """
    y, offset, H, u_seg = build_observation_terms(x, u, params)
    mean, cov = smooth.mean, smooth.cov
    residual = y - offset - np.einsum("tnd,td->tn", H, mean)
    obs_sse = residual.T @ residual + np.einsum("tni,tij,tmj->nm", H, cov, H)

    def gaussian(covariance: np.ndarray, sse: np.ndarray, count: int) -> float:
        """
        Gaussian.
        :param covariance: Input argument used by this operation.
        :param sse: Input argument used by this operation.
        :param count: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return float("inf")
        return 0.5 * (count * logdet + np.trace(np.linalg.solve(covariance, sse)))

    value = gaussian(params.R, obs_sse, len(y))
    delta0 = mean[0] - params.m0
    value += gaussian(params.P0, cov[0] + np.outer(delta0, delta0), 1)
    residual_z = mean[1:] - mean[:-1] @ params.F.T - u_seg[1:] @ params.G.T
    state_sse = residual_z.T @ residual_z
    for t in range(1, len(y)):
        cross = smooth.lag_cov[t]
        state_sse += (
            cov[t] + params.F @ cov[t - 1] @ params.F.T - cross @ params.F.T - params.F @ cross.T
        )
    value += gaussian(params.Q, state_sse, len(y) - 1)
    return float(value / len(y) + penalty)


def _transform_filter(filtered: FilterResult, transform: np.ndarray) -> FilterResult:
    """
     transform filter.
    :param filtered: Input argument used by this operation.
    :param transform: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return FilterResult(
        filtered.mean @ transform.T,
        transform @ filtered.cov @ transform.T,
        filtered.pred_mean @ transform.T,
        transform @ filtered.pred_cov @ transform.T,
        filtered.loglik,
        filtered.innovation.copy(),
    )


class DLCSSM:
    def __init__(
        self,
        lag: int = 3,
        state_dim: int = 2,
        lambda_group: float = 1e-3,
        lambda_B: float = 5e-4,
        lambda_ridge: float = 1e-6,
        max_iter: int = 80,
        min_iter: int = 5,
        tol: float = 1e-5,
        prox_iterations: int = 200,
        jitter: float = 1e-6,
        diagonal_R: bool = True,
        canonicalize: bool = True,
        seed: int = 0,
        tier_b_multiplier: float = 2.0,
    ) -> None:
        """
          init  .
        :param lag: Input argument used by this operation.
        :param state_dim: Input argument used by this operation.
        :param lambda_group: Input argument used by this operation.
        :param lambda_B: Input argument used by this operation.
        :param lambda_ridge: Input argument used by this operation.
        :param max_iter: Input argument used by this operation.
        :param min_iter: Input argument used by this operation.
        :param tol: Input argument used by this operation.
        :param prox_iterations: Input argument used by this operation.
        :param jitter: Input argument used by this operation.
        :param diagonal_R: Input argument used by this operation.
        :param canonicalize: Input argument used by this operation.
        :param seed: Input argument used by this operation.
        :param tier_b_multiplier: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        if (
            not np.isfinite([lambda_group, lambda_B, lambda_ridge]).all()
            or min(lambda_group, lambda_B, lambda_ridge) < 0
        ):
            raise ValueError("penalty weights must be nonnegative")
        if jitter <= 0 or max_iter < 1 or prox_iterations < 1:
            raise ValueError("jitter, max_iter and prox_iterations must be positive")
        if not np.isfinite(tier_b_multiplier) or tier_b_multiplier <= 1:
            raise ValueError("tier_b_multiplier must be finite and greater than 1")
        self.tier_b_multiplier = float(tier_b_multiplier)
        self.edge_weights_: np.ndarray | None = None
        self.support_tier_: np.ndarray | None = None
        self.penalty_gauge_: np.ndarray | None = None
        self.lag = lag
        self.state_dim = state_dim
        self.lambda_group = lambda_group
        self.lambda_B = lambda_B
        self.lambda_ridge = lambda_ridge
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.prox_iterations = prox_iterations
        self.jitter = jitter
        self.diagonal_R = diagonal_R
        self.canonicalize = canonicalize
        self.seed = seed
        self.params_: DLCSSMParameters | None = None
        self.support_: np.ndarray | None = None
        self.fit_result_: FitResult | None = None

    def fit(
        self, x: np.ndarray, u: np.ndarray, support: np.ndarray | SupportResult | None = None
    ) -> FitResult:
        """Fit on Train; pass SupportResult to retain orientation-tier penalties.

        A bare boolean mask remains supported and has uniform edge weights.
        """
        np.random.default_rng(self.seed)  # explicit deterministic initialization path
        T, n = x.shape
        tier = None
        if isinstance(support, SupportResult):
            tier = np.asarray(support.tier).copy()
            support = support.support
        if support is None:
            support = np.ones((self.lag, n, n), dtype=bool)
        support = np.asarray(support, dtype=bool)
        if support.shape != (self.lag, n, n):
            raise ValueError("support shape must be (lag, n, n)")

        if tier is None:
            tier = np.where(support, "untyped", "none")
        if tier.shape != support.shape:
            raise ValueError("tier shape must match support")
        if np.any(support & ~np.isin(tier, ["A", "B", "predictive", "untyped"])):
            raise ValueError("Active support must have A/B or predictive/untyped tiers")
        self.support_tier_ = tier.copy()
        self.edge_weights_ = np.where(support & (tier == "B"), self.tier_b_multiplier, 1.0)

        params, _ = _static_initialization(
            x,
            u,
            self.lag,
            self.state_dim,
            support,
            ridge=self.lambda_ridge,
            jitter=self.jitter,
        )
        # Fix the innovation scale once. With free Q, B -> B/a and z -> a*z
        # preserves the likelihood while driving the coefficient penalty to zero.
        # Q=I selects a scale; F/G remain free. Orthogonal orientation is learned
        # in these coordinates, and the L1 penalty is explicitly basis-dependent.
        chol = np.linalg.cholesky(params.Q)
        params = params.apply_gauge(np.linalg.solve(chol, np.eye(self.state_dim)))
        params.Q = np.eye(self.state_dim)
        if self.diagonal_R:
            params.R = np.diag(np.maximum(np.diag(params.R), self.jitter))

        def evaluate(p: DLCSSMParameters) -> FilterResult:
            """
            Evaluate.
            :param p: Input argument used by this operation.
            :return: Result produced by this operation.
            """
            y, offset, H, u_seg = build_observation_terms(x, u, p)
            # Covariances are floored by the M-step; no coordinate-dependent
            # additive solve jitter is needed for these positive-definite systems.
            return kalman_filter(y, offset, H, u_seg, p.F, p.G, p.Q, p.R, p.m0, p.P0, jitter=0.0)

        def penalty(p: DLCSSMParameters) -> float:
            """
            Penalty.
            :param p: Input argument used by this operation.
            :return: Result produced by this operation.
            """
            return _penalty(
                p, self.lambda_group, self.lambda_B, self.lambda_ridge, self.edge_weights_
            )

        count = T - self.lag
        filt = evaluate(params)
        history = [filt.loglik]
        penalties = [penalty(params)]
        objectives = [filt.loglik / count - penalties[-1]]
        termination = "max_iter"
        for iteration in range(self.max_iter):
            smooth = rts_smoother(filt, params.F, jitter=0.0)
            old_q = _expected_objective(x, u, smooth, params, penalties[-1])
            candidate = deepcopy(params)
            # Exact conditional transition update with Q held at I. Ridge is
            # solely the observation-theta penalty declared in _penalty.
            candidate.F, candidate.G, _ = transition_m_step(
                smooth, u[self.lag :], ridge=0.0, jitter=self.jitter
            )
            candidate.m0 = smooth.mean[0].copy()
            values, vectors = np.linalg.eigh(smooth.cov[0])
            candidate.P0 = (vectors * np.maximum(values, self.jitter)) @ vectors.T
            _observation_m_step(
                x,
                u,
                smooth,
                candidate,
                support,
                self.lambda_group,
                self.lambda_B,
                self.lambda_ridge,
                self.prox_iterations,
                self.jitter,
                self.diagonal_R,
                self.edge_weights_,
            )

            # Finite proximal solves and floating point arithmetic need a GEM
            # guard. Backtrack the whole proposed update, preserving SPD/masks.
            accepted = False
            for backtrack in range(25):
                alpha = 0.5**backtrack
                trial = DLCSSMParameters(
                    **{
                        name: getattr(params, name)
                        + alpha * (getattr(candidate, name) - getattr(params, name))
                        for name in DLCSSMParameters.__dataclass_fields__
                    }
                )
                trial_penalty = penalty(trial)
                trial_q = _expected_objective(x, u, smooth, trial, trial_penalty)
                if not np.isfinite(trial_q) or trial_q > old_q:
                    continue
                trial_filter = evaluate(trial)
                objective = trial_filter.loglik / count - trial_penalty
                if np.isfinite(objective) and objective >= objectives[-1]:
                    accepted = True
                    break
            if not accepted:
                termination = "no_acceptable_gem_step"
                break
            improvement = (objective - objectives[-1]) / (1.0 + abs(objectives[-1]))
            params, filt = trial, trial_filter
            history.append(filt.loglik)
            penalties.append(trial_penalty)
            objectives.append(objective)
            if iteration + 1 >= self.min_iter and improvement < self.tol:
                termination = "converged"
                break

        smooth = rts_smoother(filt, params.F, jitter=0.0)
        transform = np.eye(self.state_dim)
        if self.canonicalize:
            params, smooth, transform = canonical_gauge(
                params, smooth, support, return_transform=True
            )
            filt = _transform_filter(filt, transform)
        self.penalty_gauge_ = transform

        sample_idx = np.linspace(0, len(smooth.mean) - 1, min(50, len(smooth.mean)), dtype=int)
        max_rho = max(spectral_radius(params.companion(smooth.mean[i])) for i in sample_idx)

        self.params_ = params
        self.support_ = support
        self.fit_result_ = FitResult(
            params,
            filt,
            smooth,
            history,
            float(max_rho),
            objectives,
            penalties,
            transform.copy(),
            termination,
        )
        return self.fit_result_

    def filter(self, x: np.ndarray, u: np.ndarray) -> FilterResult:
        """Forward-only state inference. This method never invokes the smoother."""
        if self.params_ is None:
            raise RuntimeError("Model is not fit")
        y, offset, H, u_seg = build_observation_terms(x, u, self.params_)
        return kalman_filter(
            y,
            offset,
            H,
            u_seg,
            self.params_.F,
            self.params_.G,
            self.params_.Q,
            self.params_.R,
            self.params_.m0,
            self.params_.P0,
            jitter=0.0,
        )

    def penalty_value(self) -> float:
        """Evaluate the fitted penalty in optimization coordinates, even after gauge-fixing."""
        if self.params_ is None or self.penalty_gauge_ is None:
            raise RuntimeError("Penalty coordinates unavailable; fit or load a new checkpoint")
        training_params = self.params_.apply_gauge(np.linalg.inv(self.penalty_gauge_))
        return _penalty(
            training_params, self.lambda_group, self.lambda_B, self.lambda_ridge, self.edge_weights_
        )

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """
        Save.
        :param path: Input argument used by this operation.
        :param metadata: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        if self.params_ is None or self.support_ is None:
            raise RuntimeError("Model is not fit")
        p = self.params_
        payload = {
            "c": p.c,
            "A0": p.A0,
            "B": p.B,
            "Lambda": p.Lambda,
            "D": p.D,
            "F": p.F,
            "G": p.G,
            "Q": p.Q,
            "R": p.R,
            "m0": p.m0,
            "P0": p.P0,
            "support": self.support_.astype(np.uint8),
            "metadata_json": np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        }
        if self.edge_weights_ is not None:
            payload["edge_penalty_weights"] = self.edge_weights_
            payload["support_tier"] = self.support_tier_
            payload["tier_b_multiplier"] = np.asarray(self.tier_b_multiplier)
        payload["estimation_convention"] = np.asarray("fixed_Q_identity_weighted_GEM_v2")
        payload["output_gauge"] = np.asarray(
            "posterior_mixture_covariance_no_centering" if self.canonicalize else "optimization"
        )
        if self.penalty_gauge_ is not None:
            payload["penalty_gauge"] = self.penalty_gauge_
            payload["penalty_weights"] = np.asarray(
                [self.lambda_group, self.lambda_B, self.lambda_ridge]
            )
            payload["optimization_gauge"] = np.asarray("Q=I")
        if self.fit_result_ is not None:
            payload["loglik_history"] = np.asarray(self.fit_result_.loglik_history)
            payload["penalized_objective_history"] = np.asarray(
                self.fit_result_.penalized_objective_history
            )
            payload["penalty_history"] = np.asarray(self.fit_result_.penalty_history)
            payload["termination_reason"] = np.asarray(self.fit_result_.termination_reason)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> DLCSSM:
        """
        Load.
        :param path: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        with np.load(path, allow_pickle=False) as z:
            A0 = z["A0"]
            B = z["B"]
            model = cls(lag=A0.shape[0], state_dim=B.shape[0])
            model.params_ = DLCSSMParameters(
                c=z["c"],
                A0=A0,
                B=B,
                Lambda=z["Lambda"],
                D=z["D"],
                F=z["F"],
                G=z["G"],
                Q=z["Q"],
                R=z["R"],
                m0=z["m0"],
                P0=z["P0"],
            )
            model.support_ = z["support"].astype(bool)
            model.edge_weights_ = (
                z["edge_penalty_weights"].copy()
                if "edge_penalty_weights" in z
                else np.ones_like(A0)
            )
            model.support_tier_ = (
                z["support_tier"].copy()
                if "support_tier" in z
                else np.where(model.support_, "untyped", "none")
            )
            if "tier_b_multiplier" in z:
                model.tier_b_multiplier = float(z["tier_b_multiplier"])
            if "output_gauge" in z:
                model.canonicalize = str(z["output_gauge"]) != "optimization"
            if "penalty_gauge" in z and "penalty_weights" in z:
                model.penalty_gauge_ = z["penalty_gauge"].copy()
                model.lambda_group, model.lambda_B, model.lambda_ridge = map(
                    float, z["penalty_weights"]
                )

        return model
