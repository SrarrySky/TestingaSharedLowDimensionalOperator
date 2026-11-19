"""Statistical, leakage and actual CLI integration checks; unittest compatible."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from oos.oos_experiment import (
    PredictiveConfig,
    _dm,
    _validate_export_metadata,
    evaluate_designs,
)
from oos.oos_specs import build_default_oos_specs


def fixture():
    rng = np.random.default_rng(2026)
    n = 120
    x = rng.normal(size=(n, 7))
    y = np.maximum(0.01 + 0.003 * x[:, 0] + 0.002 * x[:, 5] + 0.001 * rng.normal(size=n), 0)
    rows = np.arange(n)
    masks = dict(
        train=rows < 38, validation=(rows >= 40) & (rows < 78), test=(rows >= 80) & (rows < 118)
    )
    effective = {f"M{i}": tuple(f"x{j}" for j in range(i + 2)) for i in range(6)}
    designs = {name: x[:, : len(features)].copy() for name, features in effective.items()}
    dates = pd.date_range("2020-01-01", periods=n).astype(str).to_numpy().astype(str)
    return designs, y, masks, rows, dates, effective


class PredictiveTests(unittest.TestCase):
    def test_current_label_requires_the_current_feature_contract(self):
        specs = build_default_oos_specs(5, n_regimes=3, include_bk=False)
        metadata = {
            "spec_version": "main-m0-m5-pathwise-v2",
            "config": {"horizon": 5, "n_regimes": 3, "include_bk": False},
            "declared_specs": {
                name: list(spec.feature_names) for name, spec in specs.items()
            },
        }
        _validate_export_metadata(metadata)
        metadata["declared_specs"]["M5"] = metadata["declared_specs"]["M5"][:-1]
        with self.assertRaisesRegex(ValueError, "contract differs"):
            _validate_export_metadata(metadata)

    def run_fixture(self, data=None):
        return evaluate_designs(
            *(fixture() if data is None else data),
            horizon=2,
            config=PredictiveConfig(alphas=(0.1, 1.0), block_length=4, bootstrap_draws=99),
        )

    def test_test_labels_never_change_selection_or_predictions(self):
        data = list(fixture())
        original = self.run_fixture(data)
        data[1] = data[1].copy()
        data[1][data[2]["test"]] *= 100
        changed = self.run_fixture(data)
        self.assertEqual(original["selected"], changed["selected"])
        self.assertEqual(original["fitted"], changed["fitted"])
        np.testing.assert_array_equal(
            original["predictions"].prediction, changed["predictions"].prediction
        )
        self.assertFalse(original["metrics"].equals(changed["metrics"]))

    def test_metrics_and_reproducibility(self):
        r = self.run_fixture()
        pd.testing.assert_frame_equal(r["comparisons"], self.run_fixture()["comparisons"])
        data = fixture()
        mean = data[1][data[2]["train"]].mean()
        for _, metric in r["metrics"].iterrows():
            p = r["predictions"]
            p = p[(p.model == metric.model) & (p.split == metric.split)]
            mse = np.mean((p.actual - p.prediction) ** 2)
            self.assertAlmostEqual(metric.mse, mse)
            self.assertAlmostEqual(
                metric.r2_oos_train_mean, 1 - mse / np.mean((p.actual - mean) ** 2)
            )
        self.assertEqual(len(r["comparisons"]), 8)

    def test_identical_and_intercept_only_specs(self):
        data = list(fixture())
        data[0] = {f"M{i}": np.empty((120, 0)) for i in range(6)}
        data[5] = {f"M{i}": () for i in range(6)}
        r = self.run_fixture(data)
        self.assertTrue(r["comparisons"].identical_predictions.all())
        self.assertTrue((r["comparisons"].mean_loss_improvement == 0).all())
        self.assertTrue(r["comparisons"].dm_p_two_sided.isna().all())
        self.assertTrue(all(v["intercept_only"] for v in r["selected"].values()))

    def test_tiny_financial_losses_preserve_dm_scale(self):
        d = np.random.default_rng(2).normal(0.1, 1, 100)
        np.testing.assert_allclose(_dm(d, 4)[:2], _dm(d * 1e-14, 4)[:2], rtol=1e-12)

    def test_reject_gap_and_boundary_leak(self):
        data = list(fixture())
        data[2]["test"][90] = False
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.run_fixture(data)
        data = list(fixture())
        data[2]["train"][38] = True
        with self.assertRaisesRegex(ValueError, "purge"):
            self.run_fixture(data)

    def test_invalid_config(self):
        for kwargs in (
            dict(alphas=[0]),
            dict(block_length=0),
            dict(seed=True),
            dict(confidence_level=1),
            dict(hac_lags=-1),
        ):
            with self.assertRaises(ValueError):
                PredictiveConfig(**kwargs)

    def test_real_exporter_and_predictive_cli(self):
        # Synthetic integration input is not financial performance evidence.
        from test_pathwise_oos import params

        from core.data import PreparedDataset
        from model.dlcssm import DLCSSM
        from oos.oos_final import FinalOOSConfig

        rng = np.random.default_rng(7)
        n = 90
        dates = pd.date_range("2020-01-01", periods=n).astype(str).to_numpy().astype(str)
        ds = PreparedDataset(
            x=rng.normal(size=(n, 2)),
            u=rng.normal(size=(n, 3)),
            dates=dates,
            sector_codes=np.array(["a", "b"]),
            sector_names=np.array(["A", "B"]),
            covariate_names=np.array(["market_return", "realized_volatility", "turnover_proxy"]),
            split=np.array(["train"] * 40 + ["validation"] * 25 + ["test"] * 25),
            x_mean=np.zeros(2),
            x_scale=np.ones(2),
            u_mean=np.zeros(3),
            u_scale=np.ones(3) * 0.01,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez_compressed(root / "data.npz", **asdict(ds))
            model = DLCSSM()
            model.params_ = params()
            model.support_ = np.ones((1, 2, 2), dtype=bool)
            model.save(root / "model.npz", metadata={"fit_split": "train"})
            pd.DataFrame(
                dict(
                    date=dates,
                    information_date=dates,
                    parameter_fit_end=dates[39],
                    baseline__hmm_probability_0=rng.uniform(0.1, 0.3, n),
                    baseline__hmm_probability_1=rng.uniform(0.1, 0.3, n),
                    baseline__dy_connectedness_h2=rng.uniform(size=n),
                    baseline__tvp_risk_h2=rng.uniform(size=n),
                )
            ).to_csv(root / "baseline.csv", index=False)
            (root / "feature.json").write_text(
                json.dumps(
                    asdict(FinalOOSConfig(horizon=2, correlation_window=4, predictive_draws=8))
                )
            )
            (root / "predictive.json").write_text(
                json.dumps(
                    asdict(PredictiveConfig(alphas=(0.1, 1.0), block_length=4, bootstrap_draws=49))
                )
            )
            command = [
                sys.executable,
                "-m",
                "oos.oos_experiment",
                "--dataset",
                str(root / "data.npz"),
                "--model",
                str(root / "model.npz"),
                "--model-fit-end",
                dates[39],
                "--baselines",
                str(root / "baseline.csv"),
                "--feature-config",
                str(root / "feature.json"),
                "--config",
                str(root / "predictive.json"),
                "--output",
                str(root / "out"),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((root / "out/report.json").read_text())
            self.assertEqual(report["sample_counts"]["test"], 23)
            self.assertEqual(len(report["selected"]), 6)
            metrics = pd.read_csv(root / "out/metrics.csv")
            self.assertEqual(len(metrics), 12)
            self.assertTrue(np.isfinite(metrics.mse).all())
            second = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("not empty", second.stderr)


if __name__ == "__main__":
    unittest.main()
