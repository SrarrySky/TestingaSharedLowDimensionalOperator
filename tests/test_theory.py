from __future__ import annotations

import numpy as np

from model.dlcssm import DLCSSMParameters
from model.geometry import low_rank_diagnostics, matrix_power_derivative
from model.propagation import frozen_pathwise_residual


def _params(seed: int = 0) -> DLCSSMParameters:
    rng = np.random.default_rng(seed)
    n, L, d, q = 4, 2, 2, 2
    return DLCSSMParameters(
        c=rng.normal(size=n),
        A0=rng.normal(scale=0.05, size=(L, n, n)),
        B=rng.normal(scale=0.03, size=(d, L, n, n)),
        Lambda=rng.normal(size=(n, d)),
        D=rng.normal(size=(n, q)),
        F=np.array([[0.8, 0.1], [0.0, 0.7]]),
        G=rng.normal(scale=0.1, size=(d, q)),
        Q=0.1 * np.eye(d),
        R=0.2 * np.eye(n),
        m0=np.zeros(d),
        P0=np.eye(d),
    )


def test_gauge_invariance() -> None:
    rng = np.random.default_rng(4)
    p = _params()
    R = np.array([[1.4, 0.25], [-0.15, 0.8]])
    p2 = p.apply_gauge(R)
    for _ in range(10):
        z = rng.normal(size=2)
        assert np.allclose(p.a_lags(z), p2.a_lags(R @ z), atol=1e-12)
        assert np.allclose(p.companion(z), p2.companion(R @ z), atol=1e-12)


def test_low_rank_operator_variation() -> None:
    rng = np.random.default_rng(5)
    T, p, d = 1000, 30, 2
    B = rng.normal(size=(p, d))
    z = np.cumsum(rng.normal(size=(T, d)), axis=0)
    a = rng.normal(size=p)[None, :] + z @ B.T
    diag = low_rank_diagnostics(a, d)
    assert diag.explained_ratio_d > 1 - 1e-12
    assert diag.effective_rank <= d + 1e-10


def test_matrix_power_sensitivity_matches_finite_difference() -> None:
    rng = np.random.default_rng(6)
    C = rng.normal(scale=0.15, size=(6, 6))
    Cr = rng.normal(scale=0.05, size=(6, 6))
    h = 5
    eps = 1e-6
    analytic = matrix_power_derivative(C, Cr, h)
    numeric = (
        np.linalg.matrix_power(C + eps * Cr, h) - np.linalg.matrix_power(C - eps * Cr, h)
    ) / (2 * eps)
    assert np.linalg.norm(analytic - numeric) < 1e-8


def test_frozen_pathwise_exact_decomposition() -> None:
    rng = np.random.default_rng(7)
    C0 = rng.normal(scale=0.1, size=(5, 5))
    path = [C0 + rng.normal(scale=0.02, size=(5, 5)) for _ in range(7)]
    assert frozen_pathwise_residual(C0, path) < 1e-12
