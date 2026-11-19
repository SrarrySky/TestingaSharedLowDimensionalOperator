import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.synthetic_benchmark import artifact_sha256, build_seed_summary, source_sha256
from scripts.summarize_oos_block_sensitivity import combine
from scripts.summarize_s5_recovery import summarize
from scripts.verify_synthetic_run import verify_run


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_portable_source_and_artifact_hashes(tmp_path):
    source = tmp_path / "src" / "pkg"
    source.mkdir(parents=True)
    (source / "a.py").write_text("value = 1\n", encoding="utf-8")
    assert set(source_sha256(tmp_path)) == {"src/pkg/a.py"}
    (tmp_path / "data.txt").write_text("data", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert set(artifact_sha256(tmp_path)) == {"data.txt", "src/pkg/a.py"}


def test_verify_synthetic_run_recomputes_saved_predictions(tmp_path):
    run = tmp_path / "run"
    method = run / "s5_base" / "seed_7" / "static"
    method.mkdir(parents=True)
    config = {"scenarios": ["s5"], "seeds": [7], "methods": ["static"]}
    record = {
        "scenario": "s5",
        "variant": "base",
        "seed": 7,
        "method": "static",
        "status": "success",
        "relative_path": "s5_base/seed_7/static",
    }
    _write_json(run / "config.json", config)
    _write_json(run / "ledger.json", [record])
    _write_json(run / "manifest.json", {})
    _write_json(method / "record.json", record)
    split = np.array(["train", "train", "validation", "test"])
    x = np.arange(4, dtype=float)[:, None]
    np.savez_compressed(
        method.parent / "prepared.npz",
        x=x,
        x_mean=np.array([0.0]),
        x_scale=np.array([2.0]),
        split=split,
    )
    mean_raw = np.array([0.0, 2.0, 2.0, 4.0])[:, None]
    np.savez_compressed(
        method / "predictions.npz",
        mean_raw=mean_raw,
        nll_raw=np.array([0.0, 0.0, 0.5, 0.7]),
        split=split,
        dates=np.array(["1", "2", "3", "4"]),
    )
    local = pd.DataFrame(
        [
            {
                "split": "validation",
                "n": 1,
                "mse_standardized": 1.0,
                "rmse_raw": 2.0,
                "mean_nll_raw": 0.5,
            },
            {
                "split": "test",
                "n": 1,
                "mse_standardized": 1.0,
                "rmse_raw": 2.0,
                "mean_nll_raw": 0.7,
            },
        ]
    )
    local.to_csv(method / "metrics.csv", index=False)
    global_metrics = local.assign(
        scenario="s5",
        variant="base",
        seed=7,
        method="static",
        status="success",
        relative_path=record["relative_path"],
    )
    global_metrics.to_csv(run / "metrics.csv", index=False)
    build_seed_summary(config, [record], global_metrics.to_dict("records")).to_csv(
        run / "seed_summary.csv", index=False
    )
    report = verify_run(run)
    assert report["attempted_cells"] == 1
    assert report["test_summary_recomputed"] is True
    assert report["npz_files_checked"] == 2


def test_s5_recovery_summary_renames_history_and_counts_termination(tmp_path):
    source = tmp_path / "source.json"
    records = []
    for seed, termination in ((7, "max_iter"), (19, "converged"), (31, "max_iter")):
        records.append(
            {
                "T": 1200,
                "noise_scale": 0.25,
                "seed": seed,
                "support_edges": 50,
                "termination": termination,
                "em_iterations": 401,
                "principal_angles_deg": [80.0, 60.0],
                "projection_distance": 0.9,
                "dynamic_coefficient_rmse": 0.04,
                "transmitter_rank_spearman": 0.5,
            }
        )
    _write_json(source, records)
    manifest = summarize(source, tmp_path / "out")
    rows = pd.read_csv(tmp_path / "out" / "s5_recovery_rows.csv")
    summary = pd.read_csv(tmp_path / "out" / "s5_recovery_summary.csv")
    assert "objective_history_length" in rows
    assert "em_iterations" not in rows
    assert (rows["maximum_principal_angle_deg"] == 80.0).all()
    assert summary.loc[0, "max_iter_count"] == 2
    assert manifest["termination_counts"] == {"converged": 1, "max_iter": 2}


def _oos_run(root: Path, block: int) -> None:
    root.mkdir()
    report = {
        "config": {"alphas": [0.1, 1.0], "block_length": block, "seed": 42},
        "selected": {"M0": {"alpha": 1.0}, "M1": {"alpha": 1.0}},
        "fitted": {"M0": {"coef": [1.0]}, "M1": {"coef": [1.0, 2.0]}},
        "sample_counts": {"test": 2},
        "input_sha256": {"designs.npz": "abc"},
    }
    _write_json(root / "report.json", report)
    pd.DataFrame({"model": ["M0", "M1"], "alpha": [1.0, 1.0]}).to_csv(
        root / "candidates.csv", index=False
    )
    pd.DataFrame({"model": ["M0", "M1"], "mse": [2.0, 1.0]}).to_csv(
        root / "metrics.csv", index=False
    )
    pd.DataFrame({"model": ["M0", "M1"], "prediction": [0.2, 0.3]}).to_csv(
        root / "predictions.csv", index=False
    )
    pd.DataFrame(
        {
            "reference": ["M0"],
            "candidate": ["M1"],
            "mean_loss_improvement": [1.0],
            "ci_low": [1.0 - block / 100],
            "ci_high": [1.0 + block / 100],
            "relative_mse_improvement": [0.5],
            "relative_ci_low": [0.4 - block / 1000],
            "relative_ci_high": [0.6 + block / 1000],
            "dm_stat": [2.0],
        }
    ).to_csv(root / "comparisons.csv", index=False)


def test_oos_block_combiner_allows_only_interval_changes(tmp_path):
    runs = {}
    for block in (10, 20, 40):
        path = tmp_path / f"block_{block}"
        _oos_run(path, block)
        runs[block] = path
    report = combine(runs, tmp_path / "combined")
    table = pd.read_csv(tmp_path / "combined" / "block_sensitivity.csv")
    assert report["fixed_point_estimates"] is True
    assert table.loc[0, "block_20_ci_low"] == 0.8
    assert table.loc[0, "block_40_relative_ci_high"] == 0.64
