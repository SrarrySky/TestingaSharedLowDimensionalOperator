"""Leakage-aware forecast scoring and statistical evaluation utilities.

Functions in this module operate on realised out-of-sample observations and
forecast objects.  They intentionally keep target construction separate from
model fitting so that uncertainty and economic metrics can be audited using
the same timestamp alignment as the forecasting experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge


def max_drawdown(returns: np.ndarray) -> float:
    """Return the worst fractional drawdown from an initial unit of wealth."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or not r.size or not np.isfinite(r).all() or np.any(r < -1):
        raise ValueError("returns must be a nonempty finite vector of simple returns >= -1")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    if not np.isfinite(wealth).all():
        raise ValueError("compounded wealth is nonfinite")
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Return the mean of the empirical lower-tail returns at level ``alpha``."""
    r = np.sort(np.asarray(returns, dtype=float))
    k = max(1, int(np.ceil(alpha * len(r))))
    return float(r[:k].mean())


def downside_semivariance(returns: np.ndarray) -> float:
    """Return mean squared negative return, with positive returns clipped to zero."""
    r = np.asarray(returns, dtype=float)
    neg = np.minimum(r, 0.0)
    return float(np.mean(neg * neg))


def future_downside_target(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Finite-horizon downside semivariance target using future observations."""
    T = len(returns)
    out = np.full(T, np.nan)
    for t in range(T - horizon):
        out[t] = downside_semivariance(returns[t + 1 : t + horizon + 1])
    return out


def is_degenerate_feature(
    x: np.ndarray,
    atol: float = 1e-12,
) -> bool:
    """Return True if a feature has essentially no finite variation.
    This function must be applied to the estimation sample only(e.g. Train), never to the full Train/Validation/Test sample.
    """
    x = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)
    finite = x[np.isfinite(x)]

    if finite.size < 2:
        return True

    # Range check catches exactly/approximately constant features.
    if np.ptp(finite) <= atol:
        return True

    # Extra protection before z-scoring.
    return np.std(finite, ddof=0) <= atol


@dataclass(frozen=True)
class OOSFeatureTransform:
    """Train-fitted feature screening and standardization.
    keep_mask is defined relative to the ORIGINAL feature matrix.
    The same mask, mean, and scale must be reused onValidation/Test to prevent look-ahead leakage.
    """

    keep_mask: np.ndarray
    original_names: tuple[str, ...]
    kept_names: tuple[str, ...]
    dropped_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    standardize: bool
    atol: float

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Apply the Train-fitted feature transform."""
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != len(self.original_names):
            raise ValueError("Feature dimension does not match " "the fitted OOSFeatureTransform")

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or inf. "
                "Align/drop unavailable observations "
                "before OOS feature transformation."
            )

        out = X[
            :,
            self.keep_mask,
        ]

        if self.standardize:
            out = (out - self.mean) / self.scale

        return out


def fit_oos_feature_transform(
    X_train: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    standardize: bool = True,
    atol: float = 1e-12,
    verbose: bool = True,
) -> OOSFeatureTransform:
    """Fit feature screening/scaling using the estimation sample only.
    Degenerate features are removed using X_train only.
    Validation/Test must subsequently be transformed with the returnedobject and must never participate in deciding which features survive.
    """
    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")

    names = tuple(str(name) for name in feature_names)

    if X_train.shape[1] != len(names):
        raise ValueError("feature_names length must equal " "X_train.shape[1]")

    if not np.all(np.isfinite(X_train)):
        raise ValueError(
            "X_train contains NaN or inf. " "Align/drop unavailable observations first."
        )

    degenerate = np.array(
        [
            is_degenerate_feature(
                X_train[:, j],
                atol=atol,
            )
            for j in range(X_train.shape[1])
        ],
        dtype=bool,
    )

    keep_mask = ~degenerate

    if not np.any(keep_mask):
        raise ValueError("All candidate features are degenerate " "on the estimation sample.")

    kept_names = tuple(names[j] for j in range(len(names)) if keep_mask[j])

    dropped_names = tuple(names[j] for j in range(len(names)) if not keep_mask[j])

    X_kept = X_train[
        :,
        keep_mask,
    ]
    mean = X_kept.mean(axis=0)
    scale = X_kept.std(
        axis=0,
        ddof=0,
    )

    # This should already be prevented by the degeneracy screen,
    # but keep a final numerical guard.
    if np.any(scale <= atol):
        raise RuntimeError(
            "A retained feature has near-zero scale. "
            "Increase atol or inspect the feature builder."
        )

    if not standardize:
        # Keep arrays shape-compatible, but transform() will not use them.
        mean = np.zeros_like(mean)
        scale = np.ones_like(scale)

    if verbose:
        print(
            "[oos] kept features:",
            ", ".join(kept_names),
        )

        if dropped_names:
            print(
                "[oos] dropped degenerate features:",
                ", ".join(dropped_names),
            )

    return OOSFeatureTransform(
        keep_mask=keep_mask,
        original_names=names,
        kept_names=kept_names,
        dropped_names=dropped_names,
        mean=mean,
        scale=scale,
        standardize=standardize,
        atol=atol,
    )


@dataclass(frozen=True)
class OOSRegressionResult:
    prediction: np.ndarray
    loss: np.ndarray
    coef: np.ndarray


def ridge_oos_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> OOSRegressionResult:
    """
    Ridge oos fit predict.
    :param X_train: Input argument used by this operation.
    :param y_train: Input argument used by this operation.
    :param X_test: Input argument used by this operation.
    :param y_test: Input argument used by this operation.
    :param alpha: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    _validate_regression_matrix(
        X_train,
        "X_train",
    )
    _validate_regression_matrix(
        X_test,
        "X_test",
    )
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return OOSRegressionResult(pred, (y_test - pred) ** 2, np.asarray(model.coef_))


def logistic_oos_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
) -> OOSRegressionResult:
    """
    Logistic oos fit predict.
    :param X_train: Input argument used by this operation.
    :param y_train: Input argument used by this operation.
    :param X_test: Input argument used by this operation.
    :param y_test: Input argument used by this operation.
    :param C: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    eps = 1e-12
    loss = -(y_test * np.log(pred + eps) + (1 - y_test) * np.log(1 - pred + eps))
    return OOSRegressionResult(pred, loss, np.asarray(model.coef_).ravel())


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1
) -> tuple[float, float]:
    """Two-sided DM test with a simple Newey-West variance through horizon-1."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return float("nan"), float("nan")
    mean = float(d.mean())
    centered = d - mean
    gamma0 = float(centered @ centered / T)
    var = gamma0
    for k in range(1, min(horizon, T)):
        gamma = float(centered[k:] @ centered[:-k] / T)
        var += 2.0 * (1.0 - k / horizon) * gamma
    stat = mean / np.sqrt(max(var / T, 1e-18))
    p = 2.0 * stats.norm.sf(abs(stat))
    return float(stat), float(p)


def moving_block_bootstrap_ci(
    values: np.ndarray,
    block_length: int,
    draws: int,
    seed: int,
    statistic=np.mean,
    level: float = 0.95,
) -> tuple[float, float]:
    """
    Moving block bootstrap ci.
    :param values: Input argument used by this operation.
    :param block_length: Input argument used by this operation.
    :param draws: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :param statistic: Input argument used by this operation.
    :param level: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    T = len(v)
    stats_out = np.zeros(draws)
    blocks = int(np.ceil(T / block_length))
    for b in range(draws):
        sample = []
        for _ in range(blocks):
            start = int(rng.integers(0, max(1, T - block_length + 1)))
            sample.extend(v[start : start + block_length])
        stats_out[b] = statistic(np.asarray(sample[:T]))
    alpha = (1 - level) / 2
    return float(np.quantile(stats_out, alpha)), float(np.quantile(stats_out, 1 - alpha))


def _validate_regression_matrix(
    X: np.ndarray,
    name: str,
) -> None:
    """
     validate regression matrix.
    :param X: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    X = np.asarray(
        X,
        dtype=float,
    )
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or inf")


def save_oos_feature_transform(
    transform: OOSFeatureTransform,
    path: str | Path,
) -> None:
    """Persist a Train-fitted OOS feature transform."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        path,
        keep_mask=transform.keep_mask.astype(np.uint8),
        original_names=np.asarray(
            transform.original_names,
            dtype="U128",
        ),
        kept_names=np.asarray(
            transform.kept_names,
            dtype="U128",
        ),
        dropped_names=np.asarray(
            transform.dropped_names,
            dtype="U128",
        ),
        mean=transform.mean,
        scale=transform.scale,
        standardize=np.asarray(
            [int(transform.standardize)],
            dtype=np.uint8,
        ),
        atol=np.asarray(
            [transform.atol],
            dtype=float,
        ),
    )


def load_oos_feature_transform(
    path: str | Path,
) -> OOSFeatureTransform:
    """Load a previously Train-fitted OOS feature transform."""
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        return OOSFeatureTransform(
            keep_mask=z["keep_mask"].astype(bool),
            original_names=tuple(str(x) for x in z["original_names"]),
            kept_names=tuple(str(x) for x in z["kept_names"]),
            dropped_names=tuple(str(x) for x in z["dropped_names"]),
            mean=z["mean"],
            scale=z["scale"],
            standardize=bool(z["standardize"][0]),
            atol=float(z["atol"][0]),
        )
