from __future__ import annotations

import numpy as np

from model.uncertainty import (
    edge_sign_probability,
    edge_sign_probability_draws,
    transmitter_ranking_probability,
)


def test_edge_sign_probability_matches_monte_carlo() -> None:
    rng = np.random.default_rng(12)
    mean = np.array([0.2, -0.1])
    cov = np.array([[0.3, 0.04], [0.04, 0.2]])
    b = np.array([0.7, -0.4])
    a0 = -0.05
    analytic = edge_sign_probability(a0, b, mean, cov)
    z = rng.multivariate_normal(mean, cov, size=200_000)
    mc = np.mean(a0 + z @ b > 0)
    assert abs(analytic - mc) < 0.006


def test_vectorized_sign_probability_and_ranking_probability() -> None:
    means = np.array([[0.2, -0.1], [0.0, 0.3]])
    covariances = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    probabilities = edge_sign_probability_draws(0.0, np.array([1.0, 0.0]), means, covariances)
    np.testing.assert_allclose(probabilities, [0.5792597, 0.5], atol=1e-6)

    draws = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    ranking = transmitter_ranking_probability(draws)
    assert ranking.shape == (2, 2)
    assert ranking[0, 1] == 2 / 3
    assert ranking[1, 0] == 1 / 3
