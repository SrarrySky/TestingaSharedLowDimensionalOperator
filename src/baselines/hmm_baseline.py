"""Hidden-Markov baseline with explicit state and emission bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


@dataclass
class HMMFit:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray


class GaussianHMMBaseline:
    """Gaussian HMM baseline with forward-only posterior probabilities."""

    def __init__(
        self,
        n_states: int = 3,
        seed: int = 0,
        n_iter: int = 200,
        n_init: int = 5,
        tol: float = 1e-3,
    ) -> None:
        """
          init  .
        :param n_states: Input argument used by this operation.
        :param seed: Input argument used by this operation.
        :param n_iter: Input argument used by this operation.
        :param n_init: Input argument used by this operation.
        :param tol: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        if min(n_states, n_iter, n_init) < 1 or tol <= 0:
            raise ValueError("States, iterations, restarts and tolerance must be positive")
        self.n_init = n_init
        self.tol = tol
        self.diagnostics_: list[dict] = []
        self.n_states = n_states
        self.seed = seed
        self.n_iter = n_iter
        self.fit_: HMMFit | None = None

    def fit(self, v: np.ndarray) -> GaussianHMMBaseline:
        """
        Fit.
        :param v: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install HMM extra: `pip install -e '.[hmm]'`.") from exc

        v = _matrix(v, "v")
        if len(v) < self.n_states * 2:
            raise ValueError("Too few training rows for HMM")
        self.fit_ = None
        self.diagnostics_ = []
        best_score = -np.inf
        for restart in range(self.n_init):
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.seed + restart,
            )
            try:
                model.fit(v)
                score = float(model.score(v))
                history = list(model.monitor_.history)
                converged = len(history) >= 2 and 0 <= history[-1] - history[-2] < self.tol
                if not np.isfinite(score):
                    raise ValueError("Non-finite training likelihood")
                self.diagnostics_.append(
                    dict(
                        seed=self.seed + restart,
                        train_loglik=score,
                        converged=converged,
                        iterations=int(model.monitor_.iter),
                    )
                )
                if score > best_score:
                    best_score = score
                    self.fit_ = HMMFit(
                        model.startprob_.copy(),
                        model.transmat_.copy(),
                        model.means_.copy(),
                        model.covars_.copy(),
                    )
            except (ValueError, np.linalg.LinAlgError) as exc:
                self.diagnostics_.append(dict(seed=self.seed + restart, error=str(exc)))
        if self.fit_ is None:
            raise RuntimeError(f"All HMM restarts failed: {self.diagnostics_}")
        self.train_loglik_ = best_score
        return self

    def filtered_probabilities(
        self, v: np.ndarray, previous: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Filtered probabilities.
        :param v: Input argument used by this operation.
        :param previous: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        if self.fit_ is None:
            raise RuntimeError("HMM is not fit")
        v = _matrix(v, "v", allow_empty=True)
        f = self.fit_
        if v.shape[1] != f.means.shape[1]:
            raise ValueError("HMM feature dimension mismatch")
        if len(v) == 0:
            return np.empty((0, self.n_states))
        T = len(v)
        K = self.n_states
        log_emit = np.zeros((T, K))
        for k in range(K):
            log_emit[:, k] = multivariate_normal.logpdf(
                v, mean=f.means[k], cov=f.covars[k], allow_singular=True
            )

        out = np.zeros((T, K))
        prior = (
            f.startprob
            if previous is None
            else _probabilities(np.asarray(previous)[None, :], 1, K)[0] @ f.transmat
        )
        with np.errstate(divide="ignore"):
            log_prior = np.log(prior)
        log_alpha = log_prior + log_emit[0]
        log_alpha -= logsumexp(log_alpha)
        out[0] = np.exp(log_alpha)
        with np.errstate(divide="ignore"):
            log_trans = np.log(f.transmat)

        for t in range(1, T):
            predicted = logsumexp(log_alpha[:, None] + log_trans, axis=0)
            log_alpha = predicted + log_emit[t]
            log_alpha -= logsumexp(log_alpha)
            out[t] = np.exp(log_alpha)
        return out


class RegimeVARBaseline:
    """Soft regime-specific ridge VAR using filtered HMM probabilities."""

    def __init__(self, lag: int = 3, ridge: float = 1e-4) -> None:
        """
          init  .
        :param lag: Input argument used by this operation.
        :param ridge: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        if lag < 1 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("lag and ridge must be positive")
        self.lag = lag
        self.ridge = ridge
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, x: np.ndarray, regime_prob: np.ndarray) -> RegimeVARBaseline:
        """
        Fit.
        :param x: Input argument used by this operation.
        :param regime_prob: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        x = _matrix(x, "x")
        T, n = x.shape
        if self.lag >= T:
            raise ValueError("Not enough rows for VAR lag")
        regime_prob = _probabilities(regime_prob, T)
        K = regime_prob.shape[1]
        X = []
        Y = []
        W = []
        for t in range(self.lag, T):
            X.append(np.concatenate([x[t - ell] for ell in range(1, self.lag + 1)]))
            Y.append(x[t])
            W.append(regime_prob[t])
        X, Y, W = np.asarray(X), np.asarray(Y), np.asarray(W)
        X1 = np.column_stack([np.ones(len(X)), X])

        coef = np.zeros((K, n, 1 + self.lag * n))
        for k in range(K):
            w = np.sqrt(np.maximum(W[:, k], 1e-12))
            Xw = X1 * w[:, None]
            Yw = Y * w[:, None]
            gram = Xw.T @ Xw + self.ridge * np.eye(X1.shape[1])
            coef[k] = np.linalg.solve(gram, Xw.T @ Yw).T
        self.intercept_ = coef[:, :, 0]
        self.coef_ = coef[:, :, 1:].reshape(K, n, self.lag, n).transpose(0, 2, 1, 3)
        return self

    def predict(self, x: np.ndarray, regime_prob: np.ndarray) -> np.ndarray:
        """Predict x[t] using x[:t] and caller-supplied probabilities for S_t.

        For genuine next-day forecasts pass p(S_t | v[:t]), not the filtered
        probability that has already incorporated the target-day observation.
        Initial lag rows are NaN; rows after that use observed lagged returns.
        """
        if self.coef_ is None:
            raise RuntimeError("Regime VAR is not fit")
        x = _matrix(x, "x")
        if x.shape[1] != self.coef_.shape[-1]:
            raise ValueError("VAR feature dimension mismatch")
        prob = _probabilities(regime_prob, len(x), len(self.coef_))
        out = np.full_like(x, np.nan)
        for t in range(self.lag, len(x)):
            means = self.intercept_.copy()
            for ell in range(self.lag):
                means += np.einsum("kij,j->ki", self.coef_[:, ell], x[t - ell - 1])
            out[t] = prob[t] @ means
        return out


def _matrix(values, name, allow_empty=False):
    """
     matrix.
    :param values: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :param allow_empty: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] == 0
        or (len(values) == 0 and not allow_empty)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must be a finite 2D matrix")
    return values


def _probabilities(values, rows, states=None):
    """
     probabilities.
    :param values: Input argument used by this operation.
    :param rows: Input argument used by this operation.
    :param states: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    values = _matrix(values, "probabilities")
    if (
        len(values) != rows
        or (states is not None and values.shape[1] != states)
        or (values < 0).any()
        or not np.allclose(values.sum(axis=1), 1)
    ):
        raise ValueError("Invalid probability simplex or shape")
    return values
