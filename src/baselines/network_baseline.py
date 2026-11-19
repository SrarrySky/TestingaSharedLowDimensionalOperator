"""Prespecified network-operator Kalman baseline, not a learned edge basis.

A_t = sum_k theta_{k,t} W_k with a Gaussian random-walk coefficient state.
Default W_k are I and a Train correlation adjacency at each lag. The adjacency,
intercept and observation covariance are frozen after Train; q is selected on
Validation. This is a declared network-operator model-class comparison, not an
exact reproduction of a particular paper's estimation procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.var_baselines import covariance, matrix, positive_int
from model.kalman_em import FilterResult, kalman_filter


@dataclass(frozen=True)
class NetworkPath:
    prediction: np.ndarray
    predictive_covariance: np.ndarray
    operators: np.ndarray
    nll: np.ndarray
    filtered: FilterResult


class FixedNetworkSSM:
    def __init__(self, lag=1, process_variance=1e-5, p0=1.0, ridge=1.0):
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
            raise ValueError("Invalid fixed-network specification")
        self.q, self.p0, self.ridge = process_variance, p0, ridge

    def _loadings(self, x):
        """
         loadings.
        :param x: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        n = self.basis_.shape[-1]
        if x.shape[1] != n or len(x) <= self.lag:
            raise ValueError("Insufficient history or network dimension mismatch")
        return np.array(
            [
                sum(
                    (
                        np.einsum("kij,j->ik", self.basis_[:, ell], x[t - ell - 1])
                        for ell in range(self.lag)
                    ),
                    start=np.zeros((n, len(self.basis_))),
                )
                for t in range(self.lag, len(x))
            ]
        )

    def fit(self, x, basis=None):
        """Estimate Train nuisance parameters; optional basis must be prespecified.

        A supplied basis has shape (K,L,N,N) in the units of x. No future data
        or oracle DGP basis may be supplied in a non-oracle comparison.
        """
        x = matrix(x)
        n = x.shape[1]
        if len(x) <= self.lag + 2:
            raise ValueError("Not enough Train observations")
        if basis is None:
            if np.any(x.std(axis=0) <= 1e-12):
                raise ValueError("Train correlation adjacency needs nonconstant variables")
            adjacency = np.atleast_2d(np.corrcoef(x, rowvar=False))
            np.fill_diagonal(adjacency, 0)
            adjacency /= max(float(np.linalg.norm(adjacency, 2)), 1e-12)
            basis = np.zeros((2 * self.lag, self.lag, n, n))
            for ell in range(self.lag):
                basis[2 * ell, ell] = np.eye(n)
                basis[2 * ell + 1, ell] = adjacency
            self.basis_source_ = "train_correlation_adjacency_and_identity_per_lag"
        else:
            self.basis_source_ = "externally_prespecified"
        basis = np.asarray(basis, dtype=float)
        if (
            basis.ndim != 4
            or basis.shape[0] < 1
            or basis.shape[1:] != (self.lag, n, n)
            or not np.isfinite(basis).all()
        ):
            raise ValueError("basis must have finite shape (K,L,N,N)")
        self.basis_ = basis.copy()
        H = self._loadings(x)
        rows = len(H)
        # Joint pooled static fit supplies c/R only, not a future-seeded state.
        design = np.column_stack((np.tile(np.eye(n), (rows, 1)), H.reshape(rows * n, -1)))
        penalty = np.diag(np.r_[np.zeros(n), np.full(len(basis), self.ridge)])
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ x[self.lag :].ravel())
        self.intercept_ = beta[:n]
        residual = x[self.lag :] - (design @ beta).reshape(rows, n)
        self.sigma_ = covariance(residual)
        return self

    def filter(self, x):
        """
        Filter.
        :param x: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        x = matrix(x)
        H = self._loadings(x)
        d, n = len(self.basis_), x.shape[1]
        offsets = np.broadcast_to(self.intercept_, (len(H), n))
        result = kalman_filter(
            x[self.lag :],
            offsets,
            H,
            np.empty((len(H), 0)),
            np.eye(d),
            np.empty((d, 0)),
            self.q * np.eye(d),
            self.sigma_,
            np.zeros(d),
            self.p0 * np.eye(d),
            jitter=0.0,
        )
        prediction = np.full_like(x, np.nan)
        prediction[self.lag :] = offsets + np.einsum("tnd,td->tn", H, result.pred_mean)
        pred_cov = np.full((len(x), n, n), np.nan)
        pred_cov[self.lag :] = H @ result.pred_cov @ H.transpose(0, 2, 1) + self.sigma_
        operators = np.full((len(x), self.lag, n, n), np.nan)
        operators[self.lag :] = np.einsum("tk,klij->tlij", result.mean, self.basis_)
        nll = np.full(len(x), np.nan)
        for t in range(self.lag, len(x)):
            chol = np.linalg.cholesky(pred_cov[t])
            error = np.linalg.solve(chol, x[t] - prediction[t])
            nll[t] = 0.5 * (n * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + error @ error)
        return NetworkPath(prediction, pred_cov, operators, nll, result)
