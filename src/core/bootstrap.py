"""Dependence-preserving bootstrap utilities for time-indexed experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def _positive_integer(value: int, name: str) -> int:
    """
     positive integer.
    :param value: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_values(values: np.ndarray) -> np.ndarray:
    """
     validate values.
    :param values: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("values must have a nonempty time axis")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("values must be finite numeric data")
    return values


def _validate_bootstrap(n: int, block_length: int, draws: int) -> tuple[int, int, int]:
    """
     validate bootstrap.
    :param n: Input argument used by this operation.
    :param block_length: Input argument used by this operation.
    :param draws: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    block_length = _positive_integer(block_length, "block_length")
    draws = _positive_integer(draws, "draws")
    if block_length > n:
        raise ValueError("block_length cannot exceed the number of observations")
    return n, block_length, draws


def moving_block_indices(
    n: int,
    block_length: int,
    draws: int,
    seed: int | np.integer,
) -> np.ndarray:
    """Generate circular moving-block indices with shape "(draws, n)"."""
    n, block_length, draws = _validate_bootstrap(_positive_integer(n, "n"), block_length, draws)
    if isinstance(seed, bool | np.bool_) or not isinstance(seed, int | np.integer):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    blocks = int(np.ceil(n / block_length))
    output = np.empty((draws, n), dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    for draw in range(draws):
        starts = rng.integers(0, n, size=blocks)
        output[draw] = np.concatenate([(start + offsets) % n for start in starts])[:n]
    return output


def stationary_bootstrap_indices(
    n: int,
    mean_block_length: int | float,
    draws: int,
    seed: int | np.integer,
) -> np.ndarray:
    """Generate Politis--Romano stationary-bootstrap indices.

    The block length is geometrically distributed with the requested expected
    value.  Circular indexing is used, which preserves the requested sample
    length without dropping observations at the end of a block.
    """
    n = _positive_integer(n, "n")
    draws = _positive_integer(draws, "draws")
    if isinstance(mean_block_length, bool | np.bool_) or not np.isfinite(mean_block_length):
        raise ValueError("mean_block_length must be finite and positive")
    mean_block_length = float(mean_block_length)
    if mean_block_length < 1:
        raise ValueError("mean_block_length must be at least one")
    if isinstance(seed, bool | np.bool_) or not isinstance(seed, int | np.integer):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    probability = min(1.0, 1.0 / mean_block_length)
    output = np.empty((draws, n), dtype=np.int64)
    for draw in range(draws):
        output[draw, 0] = rng.integers(0, n)
        for t in range(1, n):
            if rng.random() < probability:
                output[draw, t] = rng.integers(0, n)
            else:
                output[draw, t] = (output[draw, t - 1] + 1) % n
    return output


def _resample(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
     resample.
    :param values: Input argument used by this operation.
    :param indices: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return values[indices]


def bootstrap_samples(
    values: np.ndarray,
    *,
    method: str,
    block_length: int | float,
    draws: int,
    seed: int | np.integer,
) -> np.ndarray:
    """Return dependence-preserving resamples with the original shape plus draws."""
    values = _validate_values(values)
    if method == "moving_block":
        indices = moving_block_indices(values.shape[0], block_length, draws, seed)
    elif method == "stationary":
        indices = stationary_bootstrap_indices(values.shape[0], block_length, draws, seed)
    else:
        raise ValueError("method must be 'moving_block' or 'stationary'")
    return _resample(values, indices)


def bootstrap_statistics(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    method: str = "moving_block",
    block_length: int | float = 20,
    draws: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Evaluate a scalar statistic on each dependence-preserving resample."""
    samples = bootstrap_samples(
        values,
        method=method,
        block_length=block_length,
        draws=draws,
        seed=seed,
    )
    result = np.asarray([statistic(sample) for sample in samples], dtype=float)
    if result.shape != (draws,) or not np.isfinite(result).all():
        raise ValueError("statistic must return one finite scalar per resample")
    return result


def percentile_interval(
    statistics: np.ndarray,
    *,
    level: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile confidence interval from bootstrap statistics."""
    statistics = np.asarray(statistics, dtype=float)
    if statistics.ndim != 1 or not statistics.size or not np.isfinite(statistics).all():
        raise ValueError("statistics must be a nonempty finite vector")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie strictly between zero and one")
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(statistics, alpha)), float(np.quantile(statistics, 1.0 - alpha))


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    method: str = "moving_block",
    block_length: int | float = 20,
    draws: int = 2000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    """Evaluate a statistic and return its dependence-preserving percentile CI."""
    return percentile_interval(
        bootstrap_statistics(
            values,
            statistic,
            method=method,
            block_length=block_length,
            draws=draws,
            seed=seed,
        ),
        level=level,
    )


def paired_bootstrap_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    method: str = "moving_block",
    block_length: int | float = 20,
    draws: int = 2000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return paired bootstrap differences and their percentile interval."""
    values_a = _validate_values(values_a)
    values_b = _validate_values(values_b)
    if values_a.shape != values_b.shape:
        raise ValueError("paired bootstrap inputs must have the same shape")
    if method == "moving_block":
        indices = moving_block_indices(values_a.shape[0], block_length, draws, seed)
    elif method == "stationary":
        indices = stationary_bootstrap_indices(values_a.shape[0], block_length, draws, seed)
    else:
        raise ValueError("method must be 'moving_block' or 'stationary'")
    differences = np.asarray(
        [
            float(statistic(values_a[index])) - float(statistic(values_b[index]))
            for index in indices
        ],
        dtype=float,
    )
    if not np.isfinite(differences).all():
        raise ValueError("statistic must return finite values")
    return differences, percentile_interval(differences, level=level)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    hac_lags: int = 0,
) -> tuple[float, float]:
    """Return a two-sided DM statistic and normal-reference p-value.

    "loss_a - loss_b" is the differential, so a positive statistic favors
    "b".  This is the asymptotic HAC test used for descriptive OOS forecast
    comparisons; it is not a nested-model significance correction.  The
    caller must provide losses already aligned to the same consecutive dates.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("loss arrays must be aligned one-dimensional vectors of length >= 2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("loss arrays must be finite")
    hac_lags = _positive_integer(hac_lags + 1, "hac_lags + 1") - 1
    if hac_lags >= len(a):
        raise ValueError("hac_lags must be smaller than the number of observations")
    differential = a - b
    centered = differential - differential.mean()
    long_run = float(centered @ centered / len(differential))
    for lag in range(1, hac_lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / len(differential))
        long_run += 2.0 * (1.0 - lag / (hac_lags + 1.0)) * covariance
    if long_run <= 0:
        raise ValueError("loss differential has nonpositive HAC long-run variance")
    statistic = float(differential.mean() / np.sqrt(long_run / len(differential)))
    return statistic, float(2.0 * stats.norm.sf(abs(statistic)))


def holm_adjust(pvalues: np.ndarray | list[float | None]) -> list[float | None]:
    """Apply Holm's step-down correction to one predeclared family."""
    values = list(pvalues)
    for pvalue in values:
        if pvalue is not None and (not np.isfinite(pvalue) or not 0 <= pvalue <= 1):
            raise ValueError("p-values must be finite values in [0, 1] or None")
    result: list[float | None] = [None] * len(values)
    order = sorted(
        (index for index, pvalue in enumerate(values) if pvalue is not None),
        key=lambda index: values[index],
    )
    previous = 0.0
    # The multiplicity family is the complete predeclared family, including
    # comparisons whose p-value is unavailable because a cell was unsupported.
    family_size = len(values)
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (family_size - rank) * values[index]))
        result[index] = previous
    return result
