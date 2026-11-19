"""Numerically robust Kalman filtering, smoothing, and Gaussian EM updates.

The routines in this module implement the latent linear-Gaussian state-space
machinery used by the experimental models.  Arrays follow the convention
that the first axis indexes time, while covariance matrices are represented
explicitly rather than through square-root factors.  Symmetrisation and
adaptive jitter are therefore part of the numerical contract of the module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _sym(a: np.ndarray) -> np.ndarray:
    """Project a numerically perturbed covariance matrix onto the symmetric set."""
    return (a + a.T) * 0.5


def _solve_psd(a: np.ndarray, b: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    """Solve a positive-semidefinite system, increasing diagonal jitter on failure."""
    eye = np.eye(a.shape[0])
    last: Exception | None = None
    for k in range(6):
        try:
            cf = cho_factor(_sym(a) + (10**k) * jitter * eye, lower=True, check_finite=False)
            return cho_solve(cf, b, check_finite=False)
        except Exception as exc:
            last = exc
    raise np.linalg.LinAlgError("PSD solve failed") from last


@dataclass
class FilterResult:
    """Container for filtered and one-step-predicted Gaussian moments."""
    mean: np.ndarray
    cov: np.ndarray
    pred_mean: np.ndarray
    pred_cov: np.ndarray
    loglik: float
    innovation: np.ndarray


@dataclass
class SmoothResult:
    """Container for smoothed moments and lag-one posterior covariances."""
    mean: np.ndarray
    cov: np.ndarray
    lag_cov: np.ndarray
    gain: np.ndarray


def kalman_filter(
    y: np.ndarray,
    obs_offset: np.ndarray,
    obs_matrix: np.ndarray,
    state_input: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    jitter: float = 1e-9,
) -> FilterResult:
    """Forward-only Kalman filter with a time-varying observation matrix.

    Model:
        z_t = F z_{t-1} + G u_t + eta_t
        y_t = offset_t + H_t z_t + eps_t

    The returned likelihood is the observed-data Gaussian log-likelihood.
    It is accumulated only after the innovation covariance has been made
    numerically positive definite, so the value remains comparable across
    EM iterations even when the model is close to singular.
    """
    t_steps, n = y.shape
    d = F.shape[0]
    if obs_matrix.shape != (t_steps, n, d):
        raise ValueError("obs_matrix shape mismatch")

    mf = np.zeros((t_steps, d))
    Pf = np.zeros((t_steps, d, d))
    mp = np.zeros_like(mf)
    Pp = np.zeros_like(Pf)
    innovations = np.zeros((t_steps, n))
    loglik = 0.0

    prev_m = m0.copy()
    prev_P = P0.copy()
    eye_d = np.eye(d)

    for t in range(t_steps):
        if t == 0:
            m_pred = prev_m
            P_pred = prev_P
        else:
            m_pred = F @ prev_m + G @ state_input[t]
            P_pred = F @ prev_P @ F.T + Q
        P_pred = _sym(P_pred)

        # The observation operator may change with time; this is what permits
        # the same filter to represent masks, interventions, and dynamic views.
        H = obs_matrix[t]
        v = y[t] - obs_offset[t] - H @ m_pred
        S = _sym(H @ P_pred @ H.T + R)
        PHt = P_pred @ H.T
        K = _solve_psd(S, PHt.T, jitter=jitter).T

        m_new = m_pred + K @ v
        # Joseph form is numerically safer than (I-KH)P.
        IKH = eye_d - K @ H
        P_new = _sym(IKH @ P_pred @ IKH.T + K @ R @ K.T)

        # The log-determinant and Mahalanobis term are the two contributions to
        # the conditional Gaussian likelihood at the current time step.
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            S = S + jitter * np.eye(n)
            sign, logdet = np.linalg.slogdet(S)
        mahal = float(v @ _solve_psd(S, v, jitter=jitter))
        loglik += -0.5 * (n * np.log(2.0 * np.pi) + logdet + mahal)

        mp[t] = m_pred
        Pp[t] = P_pred
        mf[t] = m_new
        Pf[t] = P_new
        innovations[t] = v
        prev_m, prev_P = m_new, P_new

    return FilterResult(mf, Pf, mp, Pp, float(loglik), innovations)


def rts_smoother(
    filtered: FilterResult,
    F: np.ndarray,
    jitter: float = 1e-9,
) -> SmoothResult:
    """Run the Rauch--Tung--Striebel backward recursion on the training window.

    Smoothing conditions every historical state on all observations in the
    window, whereas filtering conditions it only on observations available up
    to that time.  The lag covariance is retained because it is required by
    the transition M-step.
    """
    mf, Pf, mp, Pp = filtered.mean, filtered.cov, filtered.pred_mean, filtered.pred_cov
    T, d = mf.shape
    ms = mf.copy()
    Ps = Pf.copy()
    gains = np.zeros((max(T - 1, 0), d, d))

    for t in range(T - 2, -1, -1):
        # J_t maps the discrepancy at t+1 back to the preceding state.  The
        # PSD solve is preferable to an explicit inverse near weak excitation.
        J = _solve_psd(Pp[t + 1], F @ Pf[t], jitter=jitter).T
        gains[t] = J
        ms[t] = mf[t] + J @ (ms[t + 1] - mp[t + 1])
        Ps[t] = _sym(Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T)

    lag_cov = np.zeros((T, d, d))
    for t in range(1, T):
        # Cov(z_t, z_{t-1} | Y) = P^s_t J_{t-1}^T.
        lag_cov[t] = Ps[t] @ gains[t - 1].T

    return SmoothResult(ms, Ps, lag_cov, gains)


def transition_m_step(
    smooth: SmoothResult,
    state_input: np.ndarray,
    ridge: float = 1e-8,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ``F``, ``G``, and ``Q`` from smoothed Gaussian sufficient statistics.

    The transition and exogenous input are estimated jointly by solving one
    regularised normal equation.  This keeps correlated latent uncertainty in
    the cross-moment terms instead of replacing the smoothed states by point
    estimates.
    """
    ms, Ps, P_lag = smooth.mean, smooth.cov, smooth.lag_cov
    T, d = ms.shape
    q = state_input.shape[1]
    if T < 2:
        raise ValueError("Need at least two latent states for transition M-step")

    k = d + q
    S_xx = np.zeros((k, k))
    S_yx = np.zeros((d, k))
    S_yy = np.zeros((d, d))

    for t in range(1, T):
        mu_prev = ms[t - 1]
        mu_t = ms[t]
        u_t = state_input[t]

        E_prevprev = Ps[t - 1] + np.outer(mu_prev, mu_prev)
        E_tt = Ps[t] + np.outer(mu_t, mu_t)
        E_tprev = P_lag[t] + np.outer(mu_t, mu_prev)

        # Assemble E[[z_(t-1), u_t] [z_(t-1), u_t]^T] and its response
        # cross-moment.  The input is observed, so its moment is deterministic.
        Exx = np.block(
            [
                [E_prevprev, np.outer(mu_prev, u_t)],
                [np.outer(u_t, mu_prev), np.outer(u_t, u_t)],
            ]
        )
        Eyx = np.concatenate([E_tprev, np.outer(mu_t, u_t)], axis=1)

        S_xx += Exx
        S_yx += Eyx
        S_yy += E_tt

    W = S_yx @ np.linalg.pinv(S_xx + ridge * np.eye(k))
    F = W[:, :d]
    G = W[:, d:]

    count = T - 1
    Q = (S_yy - W @ S_yx.T - S_yx @ W.T + W @ S_xx @ W.T) / count
    Q = _sym(Q) + jitter * np.eye(d)
    return F, G, Q
