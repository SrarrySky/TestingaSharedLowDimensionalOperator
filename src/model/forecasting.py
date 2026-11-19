"""Shared one-step DLC forecasts with an explicit covariate information policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.kalman_em import FilterResult


@dataclass(frozen=True)
class OneStepForecast:
    mean: np.ndarray
    covariance: np.ndarray
    state_mean: np.ndarray
    nll: np.ndarray
    filtered: FilterResult


def one_step_forecasts(model, x, u, *, covariate_policy="persistence") -> OneStepForecast:
    """Row t predicts x_t from the posterior at t-1 (no smoother).

    Persistence plugs u_{t-1} into BOTH state and observation equations. It is
    an ex-ante plug-in forecast, not a marginal density integrating uncertainty
    in u_t. Conditional mode instead scores x_t given contemporaneous u_t.
    Observed u_t is used for the posterior update only after the forecast.
    Rows through lag are unavailable: they lack a preceding filtered posterior.
    Parameters must have been fitted using the permitted historical sample.
    """
    if covariate_policy not in ("persistence", "conditional"):
        raise ValueError("covariate_policy must be persistence or conditional")
    p = model.params_
    if p is None:
        raise RuntimeError("Model is not fitted")
    x, u = np.asarray(x, dtype=float), np.asarray(u, dtype=float)
    if (
        x.ndim != 2
        or u.ndim != 2
        or len(x) != len(u)
        or not np.isfinite(x).all()
        or not np.isfinite(u).all()
    ):
        raise ValueError("x and u must be finite aligned matrices")
    if x.shape[1] != len(p.c) or u.shape[1] != p.G.shape[1]:
        raise ValueError("x/u dimensions do not match model parameters")
    filtered = model.filter(x, u)
    rows, n = x.shape
    mean = np.full_like(x, np.nan)
    covariance = np.full((rows, n, n), np.nan)
    state_mean = np.full((rows, p.state_dim), np.nan)
    nll = np.full(rows, np.nan)
    for t in range(p.lag + 1, rows):
        input_t = u[t - 1] if covariate_policy == "persistence" else u[t]
        index = t - p.lag - 1
        m = p.F @ filtered.mean[index] + p.G @ input_t
        P = p.F @ filtered.cov[index] @ p.F.T + p.Q
        H = p.Lambda.copy()
        offset = p.c + p.D @ input_t
        for ell in range(p.lag):
            history = x[t - ell - 1]
            offset = offset + p.A0[ell] @ history
            H += np.einsum("dij,j->id", p.B[:, ell], history)
        mean[t] = offset + H @ m
        S = p.R + H @ P @ H.T
        covariance[t] = (S + S.T) / 2
        state_mean[t] = m
        chol = np.linalg.cholesky(covariance[t])
        residual = np.linalg.solve(chol, x[t] - mean[t])
        nll[t] = 0.5 * (
            n * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + residual @ residual
        )
    return OneStepForecast(mean, covariance, state_mean, nll, filtered)
