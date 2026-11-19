"""Run with python -m unittest discover -s tests -p test_pathwise_oos.py."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from core.data import PreparedDataset
from model.dlcssm import DLCSSM, DLCSSMParameters
from model.propagation import pathwise_gain, state_migration_gap
from model.uncertainty import predictive_state_paths
from oos.oos_final import (
    FinalOOSConfig,
    build_final_oos_feature_matrix,
    prepare_designs,
    structural_features,
)
from oos.oos_specs import build_default_oos_specs, select_feature_spec
from oos.pathwise_oos import predictive_pathwise_summary


def params():
    return DLCSSMParameters(
        c=np.zeros(2),
        A0=np.array([[[0.3, 0.2], [0, 0.4]]]),
        B=np.array([[[[0.1, 0.15], [0.05, -0.1]]]]),
        Lambda=np.array([[0.3], [0.2]]),
        D=np.zeros((2, 3)),
        F=np.array([[0.8]]),
        G=np.array([[0.1, 0, 0]]),
        Q=np.array([[0.05]]),
        R=np.eye(2) * 0.1,
        m0=np.zeros(1),
        P0=np.eye(1) * 0.2,
    )


class PathwiseTests(unittest.TestCase):
    def test_deterministic_limit_and_paired_frozen_state(self):
        p = params()
        p.Q[:] = 0
        mean = np.array([0.4])
        u = np.array([0.2, 0, 0])
        point = structural_features(p, mean, u, FinalOOSConfig(horizon=3))
        s = predictive_pathwise_summary(p, mean, np.zeros((1, 1)), u, horizon=3, draws=12)
        self.assertAlmostEqual(s["migration_gap"]["mean"], point[1])
        self.assertAlmostEqual(s["path_gain"]["mean"], point[3])
        self.assertEqual(s["migration_gap"]["lower"], s["migration_gap"]["upper"])
        p.F[:] = 1
        p.G[:] = 0
        s = predictive_pathwise_summary(p, mean, np.eye(1) * 0.5, u, draws=100)
        self.assertEqual(s["migration_gap"]["mean"], 0)

    def test_independent_path_metrics_and_legacy_api(self):
        p = params()
        mean = np.array([0.2])
        cov = np.array([[0.3]])
        u = np.zeros(3)
        initial, paths = predictive_state_paths(
            mean,
            cov,
            p.F,
            p.G,
            p.Q,
            np.zeros((3, 3)),
            40,
            np.random.default_rng(np.random.SeedSequence([7, 9])),
            return_initial=True,
        )
        gap = []
        gain = []
        for z, path in zip(initial, paths, strict=True):
            matrices = [p.companion(v) for v in path]
            gap.append(state_migration_gap(p.companion(z), matrices))
            gain.append(pathwise_gain(matrices))
        s = predictive_pathwise_summary(p, mean, cov, u, horizon=3, draws=40, seed=7, row_index=9)
        for name, values in [("migration_gap", gap), ("path_gain", gain)]:
            self.assertAlmostEqual(s[name]["mean"], np.mean(values))
            self.assertAlmostEqual(s[name]["lower"], np.quantile(values, 0.025))
            self.assertAlmostEqual(s[name]["upper"], np.quantile(values, 0.975))
            self.assertEqual(s[name]["exceedance_probability"], np.mean(np.array(values) > 1))
        old = predictive_state_paths(
            mean, cov, p.F, p.G, p.Q, np.zeros((3, 3)), 40, np.random.default_rng(0)
        )
        self.assertEqual(old.shape, (40, 3, 1))

    def test_process_noise(self):
        p = params()
        p.F[:] = 1
        p.G[:] = 0
        s = predictive_pathwise_summary(p, np.zeros(1), np.zeros((1, 1)), np.zeros(3), draws=100)
        self.assertGreater(s["migration_gap"]["mean"], 0)
        self.assertGreater(s["migration_gap"]["std"], 0)

    def test_pipeline_future_invariance_and_m4(self):
        rng = np.random.default_rng(21)
        T = 72
        ds = PreparedDataset(
            x=rng.normal(size=(T, 2)),
            u=rng.normal(size=(T, 3)),
            dates=pd.date_range("2020-01-01", periods=T).astype(str).to_numpy(),
            sector_codes=np.array(["a", "b"]),
            sector_names=np.array(["A", "B"]),
            covariate_names=np.array(["market_return", "realized_volatility", "turnover_proxy"]),
            split=np.array(["train"] * 30 + ["validation"] * 21 + ["test"] * 21),
            x_mean=np.zeros(2),
            x_scale=np.ones(2),
            u_mean=np.zeros(3),
            u_scale=np.ones(3),
        )
        frame = pd.DataFrame(
            dict(
                date=ds.dates,
                information_date=ds.dates,
                parameter_fit_end=ds.dates[29],
                baseline__hmm_probability_0=rng.uniform(0.1, 0.3, T),
                baseline__hmm_probability_1=rng.uniform(0.1, 0.3, T),
                baseline__dy_connectedness_h2=rng.uniform(size=T),
                baseline__tvp_risk_h2=rng.uniform(size=T),
            )
        )
        model = DLCSSM()
        model.params_ = params()
        cfg = FinalOOSConfig(horizon=2, correlation_window=4, predictive_draws=24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.csv"
            frame.to_csv(path, index=False)

            def build(data):
                return build_final_oos_feature_matrix(
                    data, model, baseline_csv=path, model_fit_end=ds.dates[29], config=cfg
                )

            bundle = build(ds)
            np.testing.assert_equal(bundle.X, build(ds).X)
            changed = replace(ds, x=ds.x.copy(), u=ds.u.copy())
            changed.x[55:] += 100
            changed.u[55:] -= 50
            later = build(changed)
            np.testing.assert_equal(
                bundle.X[bundle.row_indices < 55], later.X[later.row_indices < 55]
            )
            prefix = replace(ds, x=ds.x[:60], u=ds.u[:60], dates=ds.dates[:60], split=ds.split[:60])
            short = build(prefix)
            np.testing.assert_equal(bundle.X[: len(short.X)], short.X)
        j = bundle.feature_names.index("geom__state_migration_gap_h2")
        k = bundle.feature_names.index("predictive__migration_gap_mean_h2")
        np.testing.assert_equal(bundle.X[:, j], bundle.X[:, k])
        specs = build_default_oos_specs(2)
        np.testing.assert_equal(select_feature_spec(bundle, specs["M4"])[:, -1], bundle.X[:, j])
        designs, effective, y, masks, transform = prepare_designs(bundle, ds, cfg)
        self.assertIn("geom__state_migration_gap_h2", effective["M4"])
        self.assertFalse(any("predictive__" in n or "diag__" in n for n in effective["M5"]))
        for split, mask in masks.items():
            self.assertTrue(mask.any())
            self.assertTrue(np.all(ds.split[bundle.row_indices[mask] + 2] == split))
            self.assertTrue(np.isfinite(designs["M5"][mask]).all())

    def test_invalid_configuration_and_covariance(self):
        for kwargs in [
            dict(predictive_draws=0),
            dict(predictive_draws=True),
            dict(predictive_seed=-1),
            dict(predictive_credible=1),
            dict(migration_gap_threshold=np.nan),
        ]:
            with self.assertRaises(ValueError):
                FinalOOSConfig(**kwargs)
        with self.assertRaises(ValueError):
            predictive_pathwise_summary(params(), np.zeros(1), -np.eye(1), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
