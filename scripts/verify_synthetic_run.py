"""Verify a synthetic benchmark archive without refitting any model.

The verifier checks grid coverage, terminal ledger records, portable NPZ files,
saved-prediction metrics, the Test-only seed summary, and optional source/artifact
hashes.  It never repairs or replaces failed cells.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.experiment_contract import safe_load_npz
from experiments.synthetic_benchmark import (
    PROJECT_ROOT,
    build_seed_summary,
    scenario_variants,
    sha256_file,
    source_sha256,
)

TERMINAL_PREFIXES = ("success", "failed", "unsupported")
RECOMPUTED_COLUMNS = ("mse_standardized", "rmse_raw", "mean_nll_raw")


def _read_json(path: Path) -> Any:
    """
    Read a UTF-8 JSON artifact from disk.
    :param path, pathlib.Path: JSON file path.
    :return: Any: Parsed JSON value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _close(actual: float, expected: float, label: str) -> None:
    """
    Assert that two scalar diagnostics agree within numerical tolerance.
    :param actual, float: Value read from the produced artifact.
    :param expected, float: Reference value.
    :param label, str: Diagnostic name used in failure messages.
    :return: None.
    """
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError(f"{label} mismatch: saved={actual!r}, recomputed={expected!r}")


def _expected_cells(config: dict[str, Any]) -> set[tuple[str, str, int, str]]:
    return {
        (scenario, variant, int(seed), method)
        for scenario in config["scenarios"]
        for variant in scenario_variants(scenario)
        for seed in config["seeds"]
        for method in config["methods"]
    }


def _verify_hashes(
    root: Path, manifest: dict[str, Any], *, require_current_source: bool
) -> tuple[bool, int]:
    config_hash = manifest.get("config_sha256")
    if config_hash is not None and config_hash != sha256_file(root / "config.json"):
        raise ValueError("config.json SHA-256 does not match manifest")

    recorded_artifacts = manifest.get("artifact_sha256")
    if recorded_artifacts is not None:
        for relative, expected in recorded_artifacts.items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Manifest artifact is missing: {relative}")
            if sha256_file(path) != expected:
                raise ValueError(f"Artifact SHA-256 mismatch: {relative}")

    recorded_source = manifest.get("source_sha256")
    source_matches = False
    if recorded_source:
        source_matches = recorded_source == source_sha256(PROJECT_ROOT)
    if require_current_source and not recorded_source:
        raise ValueError("Manifest has no source_sha256 inventory")
    if require_current_source and not source_matches:
        raise ValueError(
            "Run source inventory differs from the current src/ tree; rerun under a frozen source"
        )
    return source_matches, len(recorded_artifacts or {})


def _verify_prediction_metrics(
    root: Path, record: dict[str, Any], global_metrics: pd.DataFrame
) -> None:
    method_dir = (root / record["relative_path"]).resolve()
    if not method_dir.is_relative_to(root):
        raise ValueError(f"Cell path escapes the run directory: {record['relative_path']}")
    record_path = method_dir / "record.json"
    if not record_path.is_file() or _read_json(record_path) != record:
        raise ValueError(f"Cell record does not match ledger: {record['relative_path']}")
    if record["status"] != "success":
        return

    case_dir = method_dir.parent
    prepared = safe_load_npz(
        case_dir / "prepared.npz", required_keys=("x", "x_mean", "x_scale", "split")
    )
    prediction = safe_load_npz(
        method_dir / "predictions.npz",
        required_keys=("mean_raw", "nll_raw", "split", "dates"),
    )
    local = pd.read_csv(method_dir / "metrics.csv")
    if set(local["split"]) != {"validation", "test"} or len(local) != 2:
        raise ValueError(f"Successful cell must have one Validation and one Test row: {method_dir}")
    if not np.array_equal(prepared["split"].astype(str), prediction["split"].astype(str)):
        raise ValueError(f"Prepared/prediction split mismatch: {method_dir}")

    predicted_standardized = (prediction["mean_raw"] - prepared["x_mean"]) / prepared["x_scale"]
    actual_raw = prepared["x"] * prepared["x_scale"] + prepared["x_mean"]
    key_mask = (
        (global_metrics["scenario"] == record["scenario"])
        & (global_metrics["variant"] == record["variant"])
        & (global_metrics["seed"].astype(int) == int(record["seed"]))
        & (global_metrics["method"] == record["method"])
    )
    global_cell = global_metrics[key_mask]
    if len(global_cell) != 2 or set(global_cell["split"]) != {"validation", "test"}:
        raise ValueError(f"Global metrics coverage mismatch: {record['relative_path']}")

    for split in ("validation", "test"):
        mask = prepared["split"].astype(str) == split
        expected = {
            "mse_standardized": float(
                np.mean((prepared["x"][mask] - predicted_standardized[mask]) ** 2)
            ),
            "rmse_raw": float(
                np.sqrt(np.mean((actual_raw[mask] - prediction["mean_raw"][mask]) ** 2))
            ),
            "mean_nll_raw": float(np.mean(prediction["nll_raw"][mask])),
        }
        local_row = local.loc[local["split"] == split].iloc[0]
        global_row = global_cell.loc[global_cell["split"] == split].iloc[0]
        if int(local_row["n"]) != int(mask.sum()) or int(global_row["n"]) != int(mask.sum()):
            raise ValueError(f"Saved sample count mismatch: {record['relative_path']} {split}")
        for column in RECOMPUTED_COLUMNS:
            _close(float(local_row[column]), expected[column], f"local {column}")
            _close(float(global_row[column]), expected[column], f"global {column}")


def verify_run(
    run: str | Path,
    *,
    require_current_source: bool = False,
    expected_config: str | Path | None = None,
) -> dict[str, Any]:
    """
    Verify hashes, metrics, and coverage of one synthetic experiment run.
    :param root, pathlib.Path: Root directory containing run artifacts.
    :param config, dict: Expected runner configuration.
    :return: report, dict: Verification report suitable for JSON serialization.
    """
    """Validate one completed benchmark directory and return an audit report."""
    root = Path(run).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    required = ("config.json", "ledger.json", "manifest.json", "metrics.csv", "seed_summary.csv")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Synthetic run is missing required files: {missing}")

    config = _read_json(root / "config.json")
    if expected_config is not None and config != _read_json(Path(expected_config)):
        raise ValueError("Run config differs from --config")
    ledger = _read_json(root / "ledger.json")
    manifest = _read_json(root / "manifest.json")
    if not isinstance(ledger, list):
        raise TypeError("ledger.json must contain a list")
    if manifest.get("attempted_cells") not in (None, len(ledger)):
        raise ValueError("Manifest attempted_cells does not match ledger length")
    successful = sum(row.get("status") == "success" for row in ledger)
    if manifest.get("successful_cells") not in (None, successful):
        raise ValueError("Manifest successful_cells does not match ledger")

    keys = [(row["scenario"], row["variant"], int(row["seed"]), row["method"]) for row in ledger]
    if len(keys) != len(set(keys)):
        raise ValueError("Synthetic ledger contains duplicate method cells")
    expected = _expected_cells(config)
    if set(keys) != expected:
        raise ValueError(
            f"Synthetic grid mismatch: missing={sorted(expected - set(keys))}, "
            f"extra={sorted(set(keys) - expected)}"
        )
    for row in ledger:
        status = str(row.get("status", ""))
        if not status.startswith(TERMINAL_PREFIXES):
            raise ValueError(f"Nonterminal ledger status: {status!r}")

    metrics = pd.read_csv(root / "metrics.csv")
    for column in ("scenario", "variant", "seed", "method", "split", *RECOMPUTED_COLUMNS):
        if column not in metrics:
            raise ValueError(f"metrics.csv is missing column {column!r}")
    for record in ledger:
        _verify_prediction_metrics(root, record, metrics)

    expected_summary = build_seed_summary(config, ledger, metrics.to_dict("records"))
    # Match the persisted CSV representation, where missing numeric values are NaN
    # rather than the in-memory builder's Python None values.
    expected_summary = pd.read_csv(StringIO(expected_summary.to_csv(index=False)))
    saved_summary = pd.read_csv(root / "seed_summary.csv")
    pd.testing.assert_frame_equal(
        saved_summary.reset_index(drop=True),
        expected_summary.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )

    npz_files = sorted(root.rglob("*.npz"))
    for path in npz_files:
        safe_load_npz(path)
    source_matches, hashed_artifacts = _verify_hashes(
        root, manifest, require_current_source=require_current_source
    )
    status_counts = dict(sorted(Counter(str(row["status"]) for row in ledger).items()))
    return {
        "status": "verified",
        "run": str(root),
        "attempted_cells": len(ledger),
        "status_counts": status_counts,
        "metric_rows": int(len(metrics)),
        "npz_files_checked": len(npz_files),
        "source_matches_current": source_matches,
        "manifest_artifacts_checked": hashed_artifacts,
        "test_summary_recomputed": True,
    }


def main() -> None:
    """
    Parse command-line arguments and verify the selected synthetic run.
    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Require exact JSON configuration equality")
    parser.add_argument("--require-current-source", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = verify_run(
        args.run,
        require_current_source=args.require_current_source,
        expected_config=args.config,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
