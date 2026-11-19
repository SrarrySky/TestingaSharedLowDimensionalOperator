from __future__ import annotations

import numpy as np
import pytest

from core.bootstrap import (
    bootstrap_ci,
    bootstrap_samples,
    bootstrap_statistics,
    diebold_mariano,
    holm_adjust,
    moving_block_indices,
    paired_bootstrap_difference,
    stationary_bootstrap_indices,
)


def test_moving_block_bootstrap_is_deterministic_and_preserves_shape():
    values = np.arange(30.0).reshape(15, 2)
    first = bootstrap_samples(values, method="moving_block", block_length=3, draws=20, seed=11)
    second = bootstrap_samples(values, method="moving_block", block_length=3, draws=20, seed=11)
    assert first.shape == (20, 15, 2)
    np.testing.assert_array_equal(first, second)
    indices = moving_block_indices(15, 3, 20, 11)
    assert np.all((indices >= 0) & (indices < 15))


def test_stationary_bootstrap_is_deterministic_and_supports_multivariate_values():
    values = np.arange(24.0).reshape(12, 2)
    first = bootstrap_samples(values, method="stationary", block_length=4, draws=10, seed=4)
    second = bootstrap_samples(values, method="stationary", block_length=4, draws=10, seed=4)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (10, 12, 2)
    indices = stationary_bootstrap_indices(12, 4, 10, 4)
    assert np.all((indices >= 0) & (indices < 12))


def test_statistics_and_paired_difference_use_shared_blocks():
    values = np.arange(40.0)
    stats_a = bootstrap_statistics(values, np.mean, block_length=5, draws=30, seed=9)
    stats_b = bootstrap_statistics(values, np.mean, block_length=5, draws=30, seed=9)
    np.testing.assert_array_equal(stats_a, stats_b)
    difference, interval = paired_bootstrap_difference(
        values + 1.0,
        values,
        np.mean,
        block_length=5,
        draws=30,
        seed=9,
    )
    assert np.allclose(difference, 1.0)
    assert interval == pytest.approx((1.0, 1.0))
    assert bootstrap_ci(values, block_length=5, draws=30, seed=9)[0] <= np.mean(values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "unknown", "block_length": 2},
        {"method": "moving_block", "block_length": 0},
        {"method": "stationary", "block_length": 0},
    ],
)
def test_bootstrap_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        bootstrap_samples(np.arange(5.0), draws=2, seed=1, **kwargs)


def test_bootstrap_rejects_nonfinite_values_and_shape_mismatch():
    with pytest.raises(ValueError, match="finite"):
        bootstrap_samples(
            np.array([1.0, np.nan]), method="moving_block", block_length=1, draws=2, seed=1
        )
    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap_difference(np.ones(5), np.ones(4), block_length=1, draws=2, seed=1)
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_samples(np.ones(5), method="moving_block", block_length=1.5, draws=2, seed=1)


def test_dm_and_holm_are_validated_and_deterministic():
    first = diebold_mariano(np.arange(20.0), np.arange(20.0) + 0.1, hac_lags=3)
    second = diebold_mariano(np.arange(20.0), np.arange(20.0) + 0.1, hac_lags=3)
    assert first == second
    assert first[0] < 0
    assert holm_adjust([0.01, 0.04, None, 0.5]) == [0.04, 0.12, None, 1.0]
    with pytest.raises(ValueError, match="aligned"):
        diebold_mariano(np.ones(5), np.ones(4))
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust([1.1])
