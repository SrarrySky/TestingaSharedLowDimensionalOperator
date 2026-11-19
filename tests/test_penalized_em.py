"""Focused regression checks; run with unittest or pytest."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from model.dlcssm import (
    DLCSSM,
    DLCSSMParameters,
    _expected_objective,
    _observation_m_step,
    _pack_theta,
    _penalty,
    _unpack_theta,
    build_observation_terms,
    canonical_gauge,
)
from model.kalman_em import kalman_filter, rts_smoother


def fixture(d=2):
    rng = np.random.default_rng(341)
    n, count = 3, 65
    p = DLCSSMParameters(
        c=np.zeros(n),
        A0=np.eye(n)[None] * 0.15,
        B=rng.normal(0, 0.08, (d, 1, n, n)),
        Lambda=rng.normal(0, 0.3, (n, d)),
        D=np.zeros((n, 1)),
        F=0.6 * np.eye(d),
        G=np.ones((d, 1)) * 0.1,
        Q=np.eye(d) * 0.3,
        R=np.array([[0.8, 0.2, 0.1], [0.2, 0.5, 0.05], [0.1, 0.05, 0.3]]),
        m0=np.ones(d) * 0.2,
        P0=np.eye(d) * 0.6,
    )
    u = rng.normal(size=(count, 1))
    x = np.zeros((count, n))
    z = p.m0.copy()
    for t in range(1, count):
        z = p.F @ z + p.G @ u[t] + rng.multivariate_normal(np.zeros(d), p.Q)
        x[t] = p.a_lags(z)[0] @ x[t - 1] + p.Lambda @ z + rng.multivariate_normal(np.zeros(n), p.R)
    return p, x, u


def infer(p, x, u):
    y, offset, H, useg = build_observation_terms(x, u, p)
    f = kalman_filter(y, offset, H, useg, p.F, p.G, p.Q, p.R, p.m0, p.P0, jitter=0.0)
    return f, rts_smoother(f, p.F, jitter=0.0)


class PenalizedEMTests(unittest.TestCase):
    def test_full_R_observation_update_matches_independent_optimizer(self):
        p, x, u = fixture()
        _, smooth = infer(p, x, u)
        shape = _pack_theta(p).shape

        def loss(flat):
            trial = deepcopy(p)
            _unpack_theta(flat.reshape(shape), trial)
            return _expected_objective(x, u, smooth, trial, _penalty(trial, 0, 0, 0.05))

        result = minimize(
            loss, _pack_theta(p).ravel(), method="BFGS", options={"gtol": 1e-7, "maxiter": 500}
        )
        candidate = deepcopy(p)
        _observation_m_step(
            x, u, smooth, candidate, np.ones_like(p.A0, bool), 0, 0, 0.05, 10000, 1e-6, False
        )
        self.assertLess(abs(loss(_pack_theta(candidate).ravel()) - result.fun), 1e-7)
        assert_allclose(_pack_theta(candidate).ravel(), result.x, atol=2e-5)

    def test_canonical_posterior_and_prediction_invariance(self):
        for d in (1, 2):
            p, x, u = fixture(d)
            f, smooth = infer(p, x, u)
            p2, s2, transform = canonical_gauge(
                p, smooth, np.ones_like(p.A0, bool), return_transform=True
            )
            f2, recomputed = infer(p2, x, u)
            assert_allclose(f2.loglik, f.loglik, atol=1e-10)
            assert_allclose(recomputed.mean, s2.mean, atol=1e-10)
            assert_allclose(recomputed.cov, s2.cov, atol=1e-10)
            assert_allclose(recomputed.gain, s2.gain, atol=1e-10)
            assert_allclose(recomputed.lag_cov, s2.lag_cov, atol=1e-10)
            centered = s2.mean - s2.mean.mean(axis=0)
            assert_allclose(
                centered.T @ centered / len(centered) + s2.cov.mean(axis=0), np.eye(d), atol=1e-10
            )
            for z, z2 in zip(smooth.mean, s2.mean, strict=True):
                assert_allclose(p.a_lags(z), p2.a_lags(z2), atol=1e-11)
            self.assertGreater(np.linalg.norm(s2.mean.mean(axis=0)), 0)

    def test_fit_monotonic_mask_and_gauge(self):
        _, x, u = fixture()
        mask = np.ones((1, 3, 3), bool)
        mask[0, 0, 2] = False
        for diagonal in (False, True):
            for canonical in (False, True):
                model = DLCSSM(
                    lag=1,
                    max_iter=8,
                    min_iter=3,
                    prox_iterations=250,
                    diagonal_R=diagonal,
                    canonicalize=canonical,
                    lambda_group=0.03,
                    lambda_B=0.015,
                    lambda_ridge=0.001,
                )
                result = model.fit(x, u, mask)
                self.assertGreater(len(result.penalized_objective_history), 2)
                self.assertTrue(np.all(np.diff(result.penalized_objective_history) >= 0))
                assert_allclose(
                    np.array(result.loglik_history) / (len(x) - 1) - result.penalty_history,
                    result.penalized_objective_history,
                )
                assert_allclose(result.filtered.loglik, result.loglik_history[-1])
                assert_allclose(model.penalty_value(), result.penalty_history[-1], atol=1e-10)
                p = result.params
                assert_allclose(p.A0[~mask], 0, atol=0)
                assert_allclose(p.B[:, ~mask], 0, atol=0)
                training = p.apply_gauge(np.linalg.inv(model.penalty_gauge_))
                assert_allclose(training.Q, np.eye(2), atol=1e-10)
                f = model.filter(x, u)
                assert_allclose(f.mean, result.filtered.mean, atol=1e-8)
                assert_allclose(f.loglik, result.filtered.loglik, atol=1e-8)

    def test_penalty_survives_checkpoint_roundtrip(self):
        _, x, u = fixture()
        model = DLCSSM(lag=1, max_iter=3, lambda_group=0.07, lambda_B=0.02)
        result = model.fit(x, u)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            model.save(path)
            restored = DLCSSM.load(path)
            assert_allclose(restored.penalty_value(), model.penalty_value(), atol=1e-10)
            assert_allclose(restored.filter(x, u).mean, model.filter(x, u).mean)
            with np.load(path) as saved:
                assert_allclose(
                    saved["penalized_objective_history"], result.penalized_objective_history
                )

    def test_legacy_checkpoint_still_loads(self):
        p, x, u = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.npz"
            np.savez(path, **vars(p), support=np.ones_like(p.A0, bool))
            model = DLCSSM.load(path)
            self.assertTrue(np.isfinite(model.filter(x, u).loglik))
            with self.assertRaisesRegex(RuntimeError, "coordinates unavailable"):
                model.penalty_value()

    def test_bad_mstep_is_rejected_without_corrupting_history(self):
        _, x, u = fixture()

        def corrupt(*args, **kwargs):
            args[3].c[:] = np.nan

        model = DLCSSM(lag=1, max_iter=3)
        with patch("model.dlcssm._observation_m_step", side_effect=corrupt):
            result = model.fit(x, u)
        self.assertEqual(result.termination_reason, "no_acceptable_gem_step")
        self.assertEqual(len(result.penalized_objective_history), 1)
        self.assertTrue(np.isfinite(result.filtered.loglik))

    def test_empty_support_and_one_dimensional_state(self):
        _, x, u = fixture(1)
        model = DLCSSM(lag=1, state_dim=1, max_iter=3)
        result = model.fit(x, u, np.zeros((1, 3, 3), bool))
        assert_allclose(result.params.A0, 0)
        assert_allclose(result.params.B, 0)
        self.assertTrue(np.isfinite(result.filtered.loglik))

    def test_unconstrained_scale_would_evade_penalty(self):
        p, x, u = fixture()
        scaled = p.apply_gauge(10 * np.eye(2))
        self.assertLess(_penalty(scaled, 0.1, 0.1, 0.01), _penalty(p, 0.1, 0.1, 0.01))
        assert_allclose(infer(p, x, u)[0].loglik, infer(scaled, x, u)[0].loglik, atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
