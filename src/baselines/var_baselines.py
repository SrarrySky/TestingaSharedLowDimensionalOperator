"""Conventional VAR baselines; coefficient convention A[lag, target, source].

All predict methods return target-date forecasts using observations strictly
before that date. TVP states are unrestricted matrix-normal random walks, not
a low-rank factor model, stochastic-volatility TVP-SVAR or rolling regression.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


def positive_int(value, name):
    """
    Positive int.
    :param value: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def matrix(x):
    """
    Matrix.
    :param x: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite T x N matrix")
    return x


def design(x, lag):
    """
    Design.
    :param x: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    x = matrix(x)
    lag = positive_int(lag, "lag")
    if len(x) <= lag:
        raise ValueError("Not enough rows for lag")
    return np.column_stack(
        [np.ones(len(x) - lag)] + [x[lag - ell : len(x) - ell] for ell in range(1, lag + 1)]
    ), x[lag:]


def lag_matrices(beta, lag):
    """
    Lag matrices.
    :param beta: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    n = beta.shape[1]
    return beta[1:].reshape(lag, n, n).transpose(0, 2, 1)


def covariance(residual, floor=1e-6):
    """Train residual second moment with a declared absolute eigenvalue floor."""
    residual = matrix(residual)
    s = residual.T @ residual / len(residual)
    w, v = np.linalg.eigh((s + s.T) / 2)
    return (v * np.maximum(w, floor)) @ v.T


class StaticVAR:
    """OLS (penalty=0), ridge, or equation-wise L1 sparse VAR with intercept.

    Ridge objective: ||Y-XB||_F^2 + penalty*||B_lags||_F^2.
    Lasso objective per target: ||y-Xb||^2/(2m) + penalty*||b_lags||_1.
    """

    def __init__(self, lag=1, penalty=0.01, sparse=False):
        """
          init  .
        :param lag: Input argument used by this operation.
        :param penalty: Input argument used by this operation.
        :param sparse: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        self.lag = positive_int(lag, "lag")
        if not np.isfinite(penalty) or penalty < 0 or (sparse and penalty == 0):
            raise ValueError("Invalid penalty")
        self.penalty, self.sparse = float(penalty), sparse

    def fit(self, x):
        """
        Fit.
        :param x: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        X, Y = design(x, self.lag)
        if self.sparse:
            self.beta_ = np.empty((X.shape[1], Y.shape[1]))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                for j in range(Y.shape[1]):
                    m = Lasso(alpha=self.penalty, max_iter=20000, tol=1e-7)
                    m.fit(X[:, 1:], Y[:, j])
                    self.beta_[:, j] = np.r_[m.intercept_, m.coef_]
        elif self.penalty == 0:
            self.beta_ = np.linalg.lstsq(X, Y, rcond=None)[0]
        else:
            ridge = np.diag(np.r_[0.0, np.full(X.shape[1] - 1, self.penalty)])
            self.beta_ = solve(X.T @ X + ridge, X.T @ Y, assume_a="pos")
        self.sigma_ = covariance(Y - X @ self.beta_)
        self.a_ = lag_matrices(self.beta_, self.lag)
        return self

    def predict(self, x):
        """
        Predict.
        :param x: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        X, _ = design(x, self.lag)
        if X.shape[1] != len(self.beta_):
            raise ValueError("VAR dimension mismatch")
        return np.vstack([np.full((self.lag, self.beta_.shape[1]), np.nan), X @ self.beta_])


@dataclass
class TVPPath:
    prediction: np.ndarray
    beta_filtered: np.ndarray
    variance_multiplier: np.ndarray


class TVPVAR:
    """Exact separable Kalman filter for an unrestricted coefficient random walk.

    vec(B_t)=vec(B_{t-1})+eta_t, Cov(eta)=Sigma kron (q I),
    x_t=B_t' h_t+eps_t, Cov(eps)=Sigma. Sigma is Train-fitted;
    q and p0 are frozen specifications. Cov(vec(B)|past)=Sigma kron P.
    Zero-mean prior at the first usable date avoids seeding with future OLS
    coefficients and re-assimilating those observations. Train Sigma makes
    Train paths retrospective nuisance-fit paths, not prequential estimates.
    """

    def __init__(self, lag=1, process_variance=1e-5, p0=1.0, ridge=0.01):
        """
          init  .
        :param lag: Input argument used by this operation.
        :param process_variance: Input argument used by this operation.
        :param p0: Input argument used by this operation.
        :param ridge: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        self.lag = positive_int(lag, "lag")
        if (
            not np.isfinite([process_variance, p0, ridge]).all()
            or process_variance < 0
            or p0 <= 0
            or ridge <= 0
        ):
            raise ValueError("Invalid TVP specification")
        self.q, self.p0, self.ridge = process_variance, p0, ridge

    def fit(self, x):
        """
        Fit.
        :param x: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        base = StaticVAR(self.lag, self.ridge).fit(x)
        self.sigma_ = base.sigma_.copy()
        self.n_ = self.sigma_.shape[0]
        return self

    def filter(self, x):
        """
        Filter.
        :param x: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        x = matrix(x)
        if x.shape[1] != self.n_:
            raise ValueError("TVP dimension mismatch")
        X, Y = design(x, self.lag)
        k, n = X.shape[1], Y.shape[1]
        B, P = np.zeros((k, n)), self.p0 * np.eye(k)
        path = np.full((len(x), k, n), np.nan)
        pred = np.full_like(x, np.nan)
        multiplier = np.full(len(x), np.nan)
        for t, (h, y) in enumerate(zip(X, Y, strict=True), start=self.lag):
            P.flat[:: k + 1] += self.q
            ph = P @ h
            s = 1.0 + h @ ph
            pred[t] = h @ B  # Before seeing x_t.
            multiplier[t] = s
            B = B + np.outer(ph / s, y - pred[t])
            P -= np.outer(ph, ph) / s
            P = (P + P.T) / 2
            path[t] = B  # Information through close t; features only.
        if not np.isfinite(path[self.lag :]).all() or np.any(multiplier[self.lag :] <= 0):
            raise FloatingPointError("Nonfinite TVP filter")
        return TVPPath(pred, path, multiplier)


@dataclass(frozen=True)
class SmoothPath:
    prediction: np.ndarray
    # Target row t: local coefficients fitted using responses through t-1.
    operators: np.ndarray


class SmoothVAR:
    """One-sided Epanechnikov local-linear VAR, re-estimated at each origin.

    Uses the preceding `bandwidth` response dates, never the target response.
    A causal forecasting adaptation of the smooth TV-VAR family, not a
    replication of Gao/Tan's two-sided estimator or structural TV-GIRF inference.
    """

    def __init__(self, lag=1, bandwidth=252, ridge=1.0):
        """
          init  .
        :param lag: Input argument used by this operation.
        :param bandwidth: Input argument used by this operation.
        :param ridge: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        self.lag = positive_int(lag, "lag")
        self.bandwidth = positive_int(bandwidth, "bandwidth")
        if self.bandwidth < 3 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Invalid smooth VAR specification")
        self.ridge = ridge

    def predict(self, x, start=None):
        """
        Predict.
        :param x: Input argument used by this operation.
        :param start: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        return self.forecast_path(x, start).prediction

    def forecast_path(self, x, start=None):
        """Forecasts and their exact past-only local operator estimates."""
        x = matrix(x)
        X, Y = design(x, self.lag)
        out = np.full_like(x, np.nan)
        operators = np.full((len(x), self.lag, x.shape[1], x.shape[1]), np.nan)
        first = (
            self.lag + self.bandwidth if start is None else max(start, self.lag + self.bandwidth)
        )
        k = X.shape[1]
        # Offsets are relative to the forecast target; every weight is past-only.
        d = np.arange(-self.bandwidth, 0, dtype=float) / (self.bandwidth + 1)
        w = 0.75 * (1 - d * d)
        penalty = self.ridge * np.eye(2 * k)
        penalty[0, 0] = 0
        for t in range(first, len(x)):
            idx = t - self.lag
            xx, yy = X[idx - self.bandwidth : idx], Y[idx - self.bandwidth : idx]
            z = np.column_stack([xx, xx * d[:, None]])
            b = solve(z.T @ (w[:, None] * z) + penalty, z.T @ (w[:, None] * yy), assume_a="pos")
            out[t] = X[idx] @ b[:k]
            operators[t] = lag_matrices(b[:k], self.lag)
        return SmoothPath(out, operators)


def gaussian_nll(actual, prediction, sigma, multiplier=None):
    """Joint one-step negative log density; average log density is -mean(NLL)."""
    error = np.asarray(actual) - prediction
    sign, logdet = np.linalg.slogdet(sigma)
    if sign != 1:
        raise ValueError("Predictive covariance must be positive definite")
    s = np.ones(len(error)) if multiplier is None else np.asarray(multiplier)
    inv = solve(sigma, np.eye(len(sigma)), assume_a="pos")
    quad = np.einsum("ti,ij,tj->t", error, inv, error)
    return 0.5 * (
        error.shape[1] * np.log(2 * np.pi) + logdet + error.shape[1] * np.log(s) + quad / s
    )
