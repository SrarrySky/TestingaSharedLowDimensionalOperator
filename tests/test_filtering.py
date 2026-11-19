from __future__ import annotations

import numpy as np

from model.kalman_em import kalman_filter


def test_filtered_state_does_not_depend_on_future_observations() -> None:
    rng = np.random.default_rng(10)
    T, n, d, q = 60, 3, 2, 1
    y = rng.normal(size=(T, n))
    offset = np.zeros_like(y)
    H = rng.normal(scale=0.2, size=(T, n, d))
    u = rng.normal(size=(T, q))
    F = np.array([[0.85, 0.05], [0.0, 0.75]])
    G = rng.normal(scale=0.05, size=(d, q))
    Q = 0.05 * np.eye(d)
    R = 0.2 * np.eye(n)
    m0 = np.zeros(d)
    P0 = np.eye(d)

    full = kalman_filter(y, offset, H, u, F, G, Q, R, m0, P0)
    cutoff = 31
    trunc = kalman_filter(y[:cutoff], offset[:cutoff], H[:cutoff], u[:cutoff], F, G, Q, R, m0, P0)
    assert np.allclose(full.mean[:cutoff], trunc.mean, atol=1e-12)
    assert np.allclose(full.cov[:cutoff], trunc.cov, atol=1e-12)
