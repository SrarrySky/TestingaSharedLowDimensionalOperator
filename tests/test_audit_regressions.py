"""Independent numerical counterexamples from the implementation audit."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from core.evaluation import max_drawdown
from experiments.synthetic import (
    _stability_lag_factors,
    generate_shared_low_rank,
    operator_subspace_metrics,
)
from model.geometry import low_rank_diagnostics, spectral_radius
from model.propagation import companion_matrix


@pytest.mark.parametrize(
    "returns,expected",
    [
        ([-0.1], -0.1),
        ([-0.1, -0.2], -0.28),
        ([0.1, -0.2, 0.05], -0.2),
        ([0.1, 0.2], 0.0),
        ([-1.0, 0.2], -1.0),
    ],
)
def test_drawdown_includes_initial_wealth(returns, expected):
    assert max_drawdown(np.array(returns)) == pytest.approx(expected)


@pytest.mark.parametrize("returns", [[], [[0.1]], [np.nan], [np.inf], [-1.01]])
def test_drawdown_rejects_invalid_returns(returns):
    with pytest.raises(ValueError):
        max_drawdown(np.array(returns))


def test_eigengap_matches_independent_sample_covariance():
    path = np.random.default_rng(217).normal(size=(27, 5)).cumsum(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(np.diff(path, axis=0), rowvar=False))[::-1]
    for d in range(1, 6):
        expected = eigenvalues[d - 1] - (eigenvalues[d] if d < 5 else 0)
        assert low_rank_diagnostics(path, d).eigengap_d == pytest.approx(expected)
    # Short, wide paths have unobserved zero covariance eigenvalues.
    assert low_rank_diagnostics(path[:3], 5).eigengap_d == 0
    assert low_rank_diagnostics(np.ones((5, 2)), 2).effective_rank == 0


@pytest.mark.parametrize("d", [0, -1, 3, True, 1.5])
def test_low_rank_rejects_invalid_dimension(d):
    with pytest.raises(ValueError, match="d must"):
        low_rank_diagnostics(np.ones((5, 2)), d)


def test_projection_distance_uses_spectral_not_frobenius_norm():
    basis = np.eye(4)
    metrics = operator_subspace_metrics(basis[:, :2], basis[:, 2:])
    assert metrics["projection_distance"] == pytest.approx(1.0)
    assert metrics["principal_angle_max"] == pytest.approx(np.pi / 2)
    assert operator_subspace_metrics(basis[:, :2], basis[:, :2])["projection_distance"] == 0


@pytest.mark.parametrize("lag", [1, 2, 3, 5])
def test_lag_scaling_bounds_full_companion_radius(lag):
    rng = np.random.default_rng(127)
    a = rng.normal(size=(lag, 3, 3))
    before = a.copy()
    factors = _stability_lag_factors(a, 0.8)
    scaled = a * factors[:, None, None]
    assert spectral_radius(companion_matrix(scaled)) == pytest.approx(0.8)
    assert_allclose(a, before)
    # Independent companion similarity identity: C_new = s D C_old D^-1.
    s = factors[0]
    diagonal = np.repeat(s ** -np.arange(lag, dtype=float), 3)
    expected = s * diagonal[:, None] * companion_matrix(a) / diagonal[None, :]
    assert_allclose(companion_matrix(scaled), expected)


def test_var2_counterexample():
    a = np.array([[[0.0]], [[4.0]]])
    scaled = a * _stability_lag_factors(a)[:, None, None]
    assert spectral_radius(companion_matrix(scaled)) == pytest.approx(0.95)


def test_forced_stabilization_preserves_shared_basis_and_noise_truth(monkeypatch):
    import experiments.synthetic as synthetic

    def unstable_base(n, lag, rng):
        a = np.zeros((lag, n, n))
        a[-1] = np.eye(n) * 4
        return a, np.ones_like(a, dtype=bool)

    original = synthetic._simulate
    scales = []

    def capture(*args, **kwargs):
        scales.append(kwargs["noise_scale"])
        return original(*args, **kwargs)

    monkeypatch.setattr(synthetic, "_stable_random_var", unstable_base)
    monkeypatch.setattr(synthetic, "_simulate", capture)
    ds = generate_shared_low_rank(T=80, n=2, lag=2, d=2, seed=51)
    assert np.any(ds.metadata["stability_lag_factors"] < 1)
    assert_allclose(ds.operators, np.array([ds.params.a_lags(z) for z in ds.z]))
    assert_allclose(ds.params.R, scales[0] ** 2 * np.eye(2))
    assert max(spectral_radius(companion_matrix(a)) for a in ds.operators) <= 0.95 + 1e-12


def test_generated_residual_covariance_matches_truth():
    ds = generate_shared_low_rank(T=3000, n=2, lag=2, d=1, seed=912)
    p = ds.params
    residual = []
    for t in range(p.lag, len(ds.x)):
        mean = p.c + p.Lambda @ ds.z[t] + p.D @ ds.u[t]
        for ell, a in enumerate(p.a_lags(ds.z[t])):
            mean += a @ ds.x[t - ell - 1]
        residual.append(ds.x[t] - mean)
    assert_allclose(np.cov(residual, rowvar=False), p.R, atol=0.005)


def test_empty_support_keeps_basis_and_path_dimensions():
    ds = generate_shared_low_rank(T=15, n=2, lag=2, d=2, seed=4)
    mask = np.zeros_like(ds.support)
    a, b = ds.params.coefficient_subspace(mask)
    assert a.shape == (0,)
    assert b.shape == (0, 2)
    assert ds.params.coefficient_path(ds.z, mask).shape == (15, 0)
