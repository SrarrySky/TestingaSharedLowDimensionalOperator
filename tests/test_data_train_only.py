"""Offline pipeline regressions; runnable with unittest or pytest."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.data import load_prepared_dataset, prepare_dataset


class TrainOnlyCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(
        self, name, policy="drop_dates", missing_split=None, threshold=0.8, empty_train=False
    ):
        root = self.root / name
        raw = root / "raw"
        raw.mkdir(parents=True)
        dates = pd.date_range("2020-01-01", periods=31)
        codes = ["A", "B", "C", "D"]
        pd.DataFrame({"ts_code": codes, "name": codes}).to_csv(
            raw / "sw_level1_classification.csv.gz", index=False
        )
        frames = []
        for j, code in enumerate(codes):
            close = 100 * np.cumprod(1 + 0.001 * (j + 1) + 0.002 * np.sin(np.arange(31) + j))
            # C has 7/10 valid Train returns; D has none, but both recover later.
            if code == "C":
                close[2:4] = np.nan
            if code == "D":
                close[:11] = np.nan
            if code == "B" and missing_split == "validation":
                close[12:17] = np.nan
            if code == "B" and missing_split == "test":
                close[23:28] = np.nan
            frames.append(
                pd.DataFrame(
                    {
                        "ts_code": code,
                        "trade_date": dates.strftime("%Y%m%d"),
                        "close": close,
                        "amount": 1000 + np.arange(31) * (j + 1),
                        "float_mv": 100000,
                    }
                )
            )
        sectors = pd.concat(frames, ignore_index=True)
        if empty_train:
            sectors = sectors[~sectors.trade_date.between("20200102", "20200111")]
        sectors.to_csv(raw / "sw_level1_daily.csv.gz", index=False)
        pd.DataFrame(
            {
                "trade_date": dates.strftime("%Y%m%d"),
                "close": 100 * np.cumprod(1 + 0.002 + 0.001 * np.cos(np.arange(31))),
            }
        ).to_csv(raw / "broad_market_daily.csv.gz", index=False)
        cfg = {
            "seed": 42,
            "splits": {
                "train": {"start": "2020-01-02", "end": "2020-01-11"},
                "validation": {"start": "2020-01-12", "end": "2020-01-21"},
                "test": {"start": "2020-01-22", "end": "2020-01-31"},
            },
            "market": {"include_fx": False},
            "preprocess": {
                "missing_policy": policy,
                "min_sector_history_fraction": threshold,
                "realized_vol_window": 2,
                "standardize_returns": True,
                "standardize_covariates": True,
                "covariates": [
                    "market_return",
                    "realized_volatility",
                    "turnover_proxy",
                    "cross_sectional_dispersion",
                ],
            },
            "storage": {
                "raw_dir": str(raw),
                "processed_dir": str(root / "processed"),
                "dataset_file": str(root / "processed/dataset.npz"),
                "metadata_file": str(root / "processed/metadata.json"),
            },
        }
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        ds = prepare_dataset(path)
        meta = json.loads(Path(cfg["storage"]["metadata_file"]).read_text())
        loaded = load_prepared_dataset(cfg["storage"]["dataset_file"])
        np.testing.assert_array_equal(loaded.sector_codes, ds.sector_codes)
        np.testing.assert_allclose(loaded.x, ds.x)
        return ds, meta

    def assert_future_invariant(self, policy, split):
        baseline, meta = self.prepare("base", policy)
        changed, changed_meta = self.prepare("changed", policy, split)
        self.assertEqual(baseline.sector_codes.tolist(), ["A", "B"])
        np.testing.assert_array_equal(baseline.sector_codes, changed.sector_codes)
        self.assertEqual(meta["sector_screening"], changed_meta["sector_screening"])
        self.assertEqual(meta["sector_screening"]["observations_before_missing_policy"], 10)
        self.assertAlmostEqual(meta["sector_screening"]["history_fraction"]["C"], 0.7)
        for name in ("x", "u", "dates"):
            a = getattr(baseline, name)[baseline.split == "train"]
            b = getattr(changed, name)[changed.split == "train"]
            np.testing.assert_array_equal(a, b)
        for name in ("x_mean", "x_scale", "u_mean", "u_scale"):
            np.testing.assert_array_equal(getattr(baseline, name), getattr(changed, name))

    def test_drop_dates_validation_invariance(self):
        self.assert_future_invariant("drop_dates", "validation")

    def test_drop_dates_test_invariance(self):
        self.assert_future_invariant("drop_dates", "test")

    def test_median_validation_invariance(self):
        self.assert_future_invariant("cross_sectional_median", "validation")

    def test_median_test_invariance(self):
        self.assert_future_invariant("cross_sectional_median", "test")

    def test_threshold_is_inclusive(self):
        ds, _ = self.prepare("inclusive", threshold=0.7)
        self.assertEqual(ds.sector_codes.tolist(), ["A", "B", "C"])

    def test_zero_threshold_excludes_no_train_history(self):
        ds, _ = self.prepare("zero", threshold=0)
        self.assertEqual(ds.sector_codes.tolist(), ["A", "B", "C"])

    def test_invalid_thresholds_fail(self):
        for i, threshold in enumerate([-0.1, 1.1, float("nan"), float("inf")]):
            with (
                self.subTest(threshold=threshold),
                self.assertRaisesRegex(ValueError, "min_sector_history_fraction"),
            ):
                self.prepare(str(i), threshold=threshold)

    def test_empty_train_fails(self):
        with self.assertRaisesRegex(RuntimeError, "No Train observations"):
            self.prepare("empty", empty_train=True)


if __name__ == "__main__":
    unittest.main()
