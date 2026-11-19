from __future__ import annotations

import numpy as np
import pytest

from experiments.synthetic import matched_spectrum_nonnormal_pair
from model.geometry import (
    nonnormality,
    observable_transient_gain,
    observable_transient_gain_profile,
    shift_adjusted_nonnormality,
    transient_gain,
    transient_gain_profile,
)
from model.propagation import (
    companion_matrix,
    pair_response,
    propagation_matrix,
    transmitter_scores,
)


def test_companion_shape_and_transmitter_scores() -> None:
    A = np.zeros((2, 3, 3))
    A[0] = 0.2 * np.eye(3)
    A[0, 1, 0] = 0.1
    C = companion_matrix(A)
    assert C.shape == (6, 6)
    P = propagation_matrix(C, n=3, lag=2, horizon=5, discount=0.95)
    trans, recv, net = transmitter_scores(P)
    assert trans.shape == recv.shape == net.shape == (3,)
    assert np.all(np.isfinite(P))


def test_matched_spectrum_can_have_different_transient_geometry() -> None:
    low, high = matched_spectrum_nonnormal_pair()
    assert np.allclose(np.sort(np.linalg.eigvals(low)), np.sort(np.linalg.eigvals(high)))
    assert nonnormality(high) > nonnormality(low)
    assert transient_gain(high, 8) > transient_gain(low, 8)


def test_propagation_matrix_source_target_orientation() -> None:
    A = np.zeros((1, 2, 2))
    A[0, 1, 0] = 0.3
    C = companion_matrix(A)
    response = pair_response(
        C,
        n=2,
        lag=1,
        i=0,
        j=1,
        h=1,
    )
    assert response == pytest.approx(0.3)
    P = propagation_matrix(
        C,
        n=2,
        lag=1,
        horizon=1,
        discount=1.0,
    )
    # Source-first convention.
    assert P[0, 1] == pytest.approx(0.09)
    assert P[1, 0] == pytest.approx(0.0)
    trans, recv, net = transmitter_scores(P)
    assert trans[0] == pytest.approx(0.09)
    assert trans[1] == pytest.approx(0.0)
    assert recv[0] == pytest.approx(0.0)
    assert recv[1] == pytest.approx(0.09)
    assert net[0] > 0
    assert net[1] < 0


def test_transient_gain_profile_reports_peak_horizon() -> None:
    C = np.array(
        [
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    profile = transient_gain_profile(
        C,
        horizon=4,
    )
    assert np.allclose(
        profile.gains,
        [2.0, 0.0, 0.0, 0.0],
    )
    assert np.isclose(
        profile.peak_gain,
        2.0,
    )
    assert profile.peak_horizon == 1
    # Old API must remain consistent.
    assert np.isclose(
        transient_gain(C, 4),
        profile.peak_gain,
    )


def test_shift_adjusted_nonnormality_removes_pure_shift() -> None:
    n = 2
    lag = 3
    # No estimated propagation at all.
    A = np.zeros(
        (lag, n, n),
        dtype=float,
    )
    C_shift_only = companion_matrix(A)
    adjusted_zero = shift_adjusted_nonnormality(
        C_shift_only,
        n=n,
        lag=lag,
    )
    assert np.isclose(
        adjusted_zero,
        0.0,
        atol=1e-14,
    )
    # Add one genuinely directional dynamic edge:
    # source 0 -> target 1
    A[0, 1, 0] = 0.4
    C_dynamic = companion_matrix(A)
    adjusted_dynamic = shift_adjusted_nonnormality(
        C_dynamic,
        n=n,
        lag=lag,
    )
    assert adjusted_dynamic > 0.0


def test_observable_gain_removes_companion_shift_baseline() -> None:
    n = 2
    lag = 3

    # Pure companion shift: no actual propagation.
    A = np.zeros(
        (lag, n, n),
        dtype=float,
    )

    C = companion_matrix(A)

    profile = observable_transient_gain_profile(
        C=C,
        n=n,
        lag=lag,
        horizon=5,
    )

    assert np.allclose(
        profile.gains,
        0.0,
        atol=1e-14,
    )

    assert np.isclose(
        profile.peak_gain,
        0.0,
        atol=1e-14,
    )


def test_observable_gain_detects_directional_propagation() -> None:
    n = 2
    lag = 1
    A = np.zeros(
        (lag, n, n),
        dtype=float,
    )
    # source 0 -> target 1
    A[0, 1, 0] = 0.5
    C = companion_matrix(A)
    profile = observable_transient_gain_profile(
        C=C,
        n=n,
        lag=lag,
        horizon=3,
    )
    assert np.isclose(
        profile.gains[0],
        0.5,
    )
    assert np.isclose(
        observable_transient_gain(
            C,
            n=n,
            lag=lag,
            horizon=3,
        ),
        0.5,
    )
