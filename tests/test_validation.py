"""Run with python -m unittest discover -s tests -p test_validation.py."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.config import load_yaml
from experiments.validation import (
    candidate_configs,
    make_model,
    score_validation,
    tune_validation,
    validation_window,
)
from model.dlcssm import DLCSSM

ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72)
        self.ds = SimpleNamespace(
            x=rng.normal(scale=0.15, size=(60, 2)),
            u=rng.normal(scale=0.1, size=(60, 1)),
            dates=np.arange("2020-01-01", "2020-03-01", dtype="datetime64[D]"),
            split=np.array(["train"] * 40 + ["validation"] * 12 + ["test"] * 8),
        )
        self.cfg = load_yaml(ROOT / "config/dlcssm.yaml")
        self.cfg["fit"].update(max_em_iterations=2, min_em_iterations=1, prox_iterations=3)
        self.cfg["hyperparameter_grid"] = {
            "lag": [1, 2],
            "state_dimension": [1],
            "group_penalty": [0.001],
            "state_l1_penalty": [0.0005],
        }
        self.cfg["stability"]["validation_reject_if_spectral_radius_above"] = 100.0
        self.calls = []

    def support(self, lag, train):
        self.assertEqual(train.shape, (40, 2))
        np.testing.assert_array_equal(train, self.ds.x[:40])
        self.calls.append(lag)
        return np.ones((lag, 2, 2), dtype=bool), "synthetic_test"

    def test_real_fit_selection_save_and_test_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            report = tune_validation(self.ds, self.cfg, self.support, a)
            self.assertEqual(self.calls, [1, 2])
            self.assertEqual(report["validation_count"], 12)
            self.assertEqual(report["covariate_policy"], "persistence")
            self.assertEqual(report["information_set"], "through t-1")
            best = max(report["trials"], key=lambda t: t["validation_score"])
            self.assertEqual(report["selected_trial"], best["trial"])
            loaded = DLCSSM.load(a / "selected_model.npz")
            selected = load_yaml(a / "selected_model.yaml")
            self.assertEqual(loaded.lag, selected["lag"])
            self.assertNotIn("hyperparameter_grid", selected)
            self.assertEqual(len(candidate_configs(selected)), 1)
            changed = copy.deepcopy(self.ds)
            changed.x[52:] = np.nan
            changed.u[52:] = np.inf
            b = tune_validation(changed, self.cfg, self.support, Path(tmp) / "b")
            self.assertEqual(report, b)
            with self.assertRaises(FileExistsError):
                tune_validation(self.ds, self.cfg, self.support, a)

    def test_validation_boundary_matches_full_forward_likelihood(self):
        x, u, count = validation_window(self.ds)
        for cfg in candidate_configs(self.cfg):
            model = make_model(cfg)
            model.fit(x[:count], u[:count])
            score, _ = score_validation(model, x, u, count, [0.05, 0.5, 0.95])
            expected = (model.filter(x, u).loglik - model.filter(x[:count], u[:count]).loglik) / (
                len(x) - count
            )
            self.assertAlmostEqual(score, expected, places=10)

    def test_legacy_config_keeps_explicitly_reported_conditional_score(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pop("selection")
        with tempfile.TemporaryDirectory() as tmp:
            report = tune_validation(self.ds, cfg, self.support, tmp)
        self.assertEqual(report["covariate_policy"], "conditional")
        self.assertEqual(report["information_set"], "through t-1 plus u_t")

    def test_invalid_covariate_policy_fails_before_output_creation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["selection"]["covariate_policy"] = "future"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "invalid"
            with self.assertRaisesRegex(ValueError, "covariate_policy"):
                tune_validation(self.ds, cfg, self.support, output)
            self.assertFalse(output.exists())

    def test_unstable_candidates_cannot_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("experiments.validation.score_validation", side_effect=[(20.0, 101.0), (10.0, 0.8)]):
                report = tune_validation(self.ds, self.cfg, self.support, tmp)
            self.assertEqual(report["selected_trial"], 1)
            self.assertEqual(report["trials"][0]["status"], "rejected_unstable")

    def test_all_rejected_writes_report_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("experiments.validation.score_validation", return_value=(20.0, 101.0)),
                self.assertRaisesRegex(RuntimeError, "No stable"),
            ):
                tune_validation(self.ds, self.cfg, self.support, tmp)
            self.assertFalse((Path(tmp) / "selected_model.npz").exists())
            report = json.loads((Path(tmp) / "validation_trials.json").read_text())
            self.assertIsNone(report["selected_trial"])

    def test_numeric_failure_is_recorded_and_next_candidate_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "experiments.validation.score_validation",
                side_effect=[np.linalg.LinAlgError("bad covariance"), (10.0, 0.8)],
            ):
                report = tune_validation(self.ds, self.cfg, self.support, tmp)
            self.assertEqual(report["selected_trial"], 1)
            self.assertEqual(report["trials"][0]["status"], "failed_numerically")

    def test_entrypoint_discovers_and_saves_support_per_lag(self):
        import yaml

        from experiments.validation import run_tuning

        cfg = load_yaml(ROOT / "config/causal.yaml")
        cfg.update(method="linear_lasso", bootstrap_count=2, block_length=5, support_mode="stable")
        self.ds.sector_names = np.array(["A", "B"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.yaml").write_text(yaml.safe_dump(self.cfg))
            (root / "causal.yaml").write_text(yaml.safe_dump(cfg))
            with patch("experiments.validation.load_prepared_dataset", return_value=self.ds):
                report = run_tuning(
                    root / "unused.npz",
                    root / "model.yaml",
                    root / "causal.yaml",
                    root / "out",
                    jobs=1,
                )
            self.assertEqual(len(report["trials"]), 2)
            for lag in (1, 2):
                self.assertTrue((root / "out" / f"causal_support_lag{lag}.npz").exists())

    def test_invalid_split_grid_and_support_fail(self):
        bad = copy.deepcopy(self.ds)
        bad.split[10] = "test"
        with self.assertRaises(ValueError):
            validation_window(bad)
        bad = copy.deepcopy(self.cfg)
        bad["hyperparameter_grid"]["lag"] = []
        with self.assertRaises(ValueError):
            candidate_configs(bad)
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "Support shape"),
        ):
            tune_validation(self.ds, self.cfg, lambda lag, x: (np.ones((3, 2, 2)), "wrong"), tmp)


if __name__ == "__main__":
    unittest.main()
