from __future__ import annotations

import numpy as np
import pytest

from core.bootstrap import moving_block_indices
from experiments.geometry_experiments import (
    _moving_block_index_stream,
    _projection_distance,
    coefficient_path_from_states,
    low_rank_path_report,
    reversal_map,
    stable_amplification_report,
)


def test_coefficient_path_and_low_rank_bootstrap_are_reproducible():
    rng = np.random.default_rng(8)
    states = rng.normal(size=(80, 2))
    a0 = np.zeros(5)
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
        ]
    )
    path = coefficient_path_from_states(a0, basis, states)
    assert path.shape == (80, 5)
    first = low_rank_path_report(path, 2, block_length=8, bootstrap_count=20, seed=3)
    second = low_rank_path_report(path, 2, block_length=8, bootstrap_count=20, seed=3)
    np.testing.assert_array_equal(first.bootstrap_singular_values, second.bootstrap_singular_values)
    np.testing.assert_array_equal(first.bootstrap_max_sin_theta, second.bootstrap_max_sin_theta)
    np.testing.assert_array_equal(
        first.bootstrap_projection_distance, second.bootstrap_projection_distance
    )
    assert first.diagnostics.explained_ratio_d > 0.9
    assert first.reference_basis.shape == (5, 2)
    assert first.summary()["bootstrap_count"] == 20


def test_subspace_distance_is_invariant_to_basis_orientation():
    rng = np.random.default_rng(1)
    states = rng.normal(size=(60, 2))
    path = coefficient_path_from_states(np.zeros(4), np.eye(4, 2), states)
    report = low_rank_path_report(path, 2, block_length=6, bootstrap_count=10, seed=4)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = coefficient_path_from_states(
        np.zeros(4), np.eye(4, 2) @ rotation, states @ rotation.T
    )
    rotated_report = low_rank_path_report(rotated, 2, block_length=6, bootstrap_count=10, seed=4)
    np.testing.assert_allclose(
        report.bootstrap_projection_distance,
        rotated_report.bootstrap_projection_distance,
    )


def test_streamed_indices_match_original_generator():
    expected = moving_block_indices(37, 8, 5, 19)
    stream = _moving_block_index_stream(37, 8, 19)
    actual = np.asarray([next(stream) for _ in range(5)])
    np.testing.assert_array_equal(actual, expected)


def test_projection_distance_matches_explicit_projectors():
    rng = np.random.default_rng(4)
    left, _ = np.linalg.qr(rng.normal(size=(13, 3)))
    right, _ = np.linalg.qr(rng.normal(size=(13, 3)))
    explicit = np.linalg.norm(left @ left.T - right @ right.T, 2)
    assert _projection_distance(left, right) == pytest.approx(explicit, rel=1e-12, abs=1e-12)


def test_reversal_map_reports_coverage_and_constant_boundary():
    states = np.array([[-2.0], [-0.5], [0.5], [2.0]])
    result = reversal_map(0.0, np.array([1.0]), states)
    assert result.status == "reversal-capable"
    assert result.positive_fraction == pytest.approx(0.5)
    assert result.negative_fraction == pytest.approx(0.5)
    assert np.allclose(result.distance, np.abs(states[:, 0]))

    constant = reversal_map(2.0, np.zeros(1), states)
    assert constant.status == "constant"
    assert constant.positive_fraction == 1.0
    assert np.all(np.isinf(constant.distance))


def test_stable_but_amplifying_report_uses_companion_geometry():
    # A stable, nonnormal Jordan block has a transient gain greater than one.
    lagged = np.zeros((3, 1, 2, 2))
    lagged[:, 0, 0, 0] = [0.8, 0.0, 0.0]
    lagged[:, 0, 0, 1] = [2.0, 0.0, 0.0]
    lagged[:, 0, 1, 1] = [0.8, 0.0, 0.0]
    report = stable_amplification_report(
        lagged,
        horizon=5,
        amplification_threshold=1.0,
        stability_threshold=1.0,
    )
    assert report.spectral_radii[0] == pytest.approx(0.8)
    assert report.peak_gains[0] > 1.0
    assert report.stable_but_amplifying[0]
    assert report.stable_but_amplifying_fraction == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "call",
    [
        lambda: coefficient_path_from_states(np.zeros(2), np.ones((3, 1)), np.ones((5, 1))),
        lambda: low_rank_path_report(np.ones((2, 2)), 1, block_length=2, bootstrap_count=2),
        lambda: reversal_map(0.0, np.ones(2), np.ones((3, 1))),
        lambda: stable_amplification_report(np.ones((3, 1, 2, 3))),
    ],
)
def test_geometry_producers_reject_incompatible_inputs(call):
    with pytest.raises(ValueError):
        call()
