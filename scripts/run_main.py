#!/usr/bin/env python3
"""Publication-oriented main experiment orchestrator.

This file is intentionally a thin orchestration layer.  Numerical producers
remain in :mod:`dlcssm`; this command supplies a common configuration,
provenance, stage/cell ledger, resume protocol and publication exit policy.

The runner never replaces an unavailable method with another estimator.  A
missing input or optional dependency is a terminal ledger status and is also
included in "coverage.json".
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.experiment_contract import (  # noqa: E402
    ExperimentArtifactRun,
    read_json,
    validate_ledger,
)

SCHEMA_VERSION = "main-runner-v1"
STAGES = (
    "synthetic",
    "real_data",
    "causal_support",
    "model",
    "baselines",
    "oos",
    "geometry",
    "ablation_falsification",
    "coverage",
)

_DEFAULT_STAGE_CONFIG: dict[str, dict[str, Any]] = {
    "synthetic": {
        "enabled": True,
        "required": True,
        "config": "config/synthetic_main.json",
    },
    "real_data": {
        "enabled": True,
        "required": True,
        "dataset": "data/processed/dataset.npz",
        "data_config": "config/data.yaml",
        "prepare": False,
    },
    "causal_support": {
        "enabled": True,
        "required": True,
        "dataset": "data/processed/dataset.npz",
        "causal_config": "config/causal.yaml",
        "model_config": "config/dlcssm.yaml",
    },
    "model": {
        "enabled": True,
        "required": True,
        "dataset": "data/processed/dataset.npz",
        "model_config": "config/dlcssm.yaml",
    },
    "baselines": {
        "enabled": True,
        "required": True,
        "dataset": "data/processed/dataset.npz",
        "config": "config/baselines.json",
    },
    "oos": {
        "enabled": True,
        "required": True,
        "dataset": "data/processed/dataset.npz",
        "feature_config": "config/oos_final.json",
        "predictive_config": "config/oos_predictive.json",
    },
    "geometry": {"enabled": True, "required": True},
    "ablation_falsification": {"enabled": True, "required": True, "horizon": 5, "seed": 0},
    "coverage": {"enabled": True, "required": True},
}

_PATH_KEYS = {
    "config",
    "data_config",
    "dataset",
    "causal_config",
    "model_config",
    "feature_config",
    "predictive_config",
    "baselines",
    "model",
}


class StageUnsupported(RuntimeError):
    """A stage could not be computed without an illegitimate substitution."""

    def __init__(self, status: str, reason: str, **details: Any) -> None:
        if status not in {
            "unsupported",
            "unsupported_missing_input",
            "unsupported_missing_dependency",
        }:
            raise ValueError(f"Invalid unsupported status: {status}")
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.details = details


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write a strict JSON file using replace, including on interrupted runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return int(value)


def _normalize_stage(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        value = {"enabled": value}
    if not isinstance(value, dict):
        raise ValueError(f"stages.{name} must be a mapping or boolean")
    result = copy.deepcopy(_DEFAULT_STAGE_CONFIG[name])
    result.update(value)
    if not isinstance(result["enabled"], bool) or not isinstance(result["required"], bool):
        raise ValueError(f"stages.{name}.enabled and required must be boolean")
    for key in _PATH_KEYS:
        if key in result and result[key] is not None:
            if not isinstance(result[key], str | Path):
                raise ValueError(f"stages.{name}.{key} must be a path string")
            result[key] = str(_resolve_path(result[key]))
    if "jobs" in result:
        result["jobs"] = _positive_int(result["jobs"], f"stages.{name}.jobs")
    return result


def validate_runner_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the runner contract without touching outputs."""
    if not isinstance(config, dict):
        raise ValueError("main runner config must be a mapping")
    if config.get("version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"version must be {SCHEMA_VERSION!r}")
    result: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "experiment": config.get("experiment", "dlcssm_main"),
        "purpose": config.get("purpose", "publication"),
        "seed": config.get("seed", 0),
        "threads": config.get("threads", 1),
        "inner_threads": config.get("inner_threads", 1),
        "publication_mode": config.get("publication_mode", True),
        "stages": {},
        "metadata": copy.deepcopy(config.get("metadata", {})),
    }
    if not isinstance(result["experiment"], str) or not result["experiment"].strip():
        raise ValueError("experiment must be nonempty")
    if result["purpose"] not in {"publication", "engineering_smoke", "debug"}:
        raise ValueError("purpose must be publication, engineering_smoke or debug")
    result["seed"] = _positive_int(result["seed"], "seed", allow_zero=True)
    result["threads"] = _positive_int(result["threads"], "threads")
    result["inner_threads"] = _positive_int(result["inner_threads"], "inner_threads")
    if not isinstance(result["publication_mode"], bool):
        raise ValueError("publication_mode must be boolean")
    if not isinstance(result["metadata"], dict):
        raise ValueError("metadata must be a mapping")

    supplied_stages = config.get("stages", {})
    if not isinstance(supplied_stages, dict):
        raise ValueError("stages must be a mapping")
    unknown = sorted(set(supplied_stages) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    for name in STAGES:
        result["stages"][name] = _normalize_stage(name, supplied_stages.get(name, {}))

    required_stages = config.get("required_stages")
    if required_stages is not None:
        if not isinstance(required_stages, list) or any(
            not isinstance(item, str) or item not in STAGES for item in required_stages
        ):
            raise ValueError("required_stages must be a list of known stage names")
        required_set = set(required_stages)
        for name in STAGES:
            result["stages"][name]["required"] = name in required_set
    for name, stage in result["stages"].items():
        if stage["required"] and not stage["enabled"]:
            raise ValueError(f"Required stage {name!r} cannot be disabled")
    result["required_stages"] = [name for name in STAGES if result["stages"][name]["required"]]
    return result


def load_runner_config(path: str | Path) -> dict[str, Any]:
    return validate_runner_config(_load_mapping(_resolve_path(path)))


def _set_thread_environment(threads: int, inner_threads: int) -> dict[str, str]:
    """Set conservative BLAS/OpenMP defaults and return the applied values."""
    values = {
        "MAIN_THREADS": str(threads),
        "MAIN_INNER_THREADS": str(inner_threads),
        "OMP_NUM_THREADS": str(inner_threads),
        "MKL_NUM_THREADS": str(inner_threads),
        "OPENBLAS_NUM_THREADS": str(inner_threads),
        "NUMEXPR_NUM_THREADS": str(inner_threads),
        "VECLIB_MAXIMUM_THREADS": str(inner_threads),
        "BLIS_NUM_THREADS": str(inner_threads),
        "TBB_NUM_THREADS": str(inner_threads),
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


def _stage_attempt(root: Path, stage: str) -> Path:
    stage_root = root / "stages" / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in stage_root.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = stage_root / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _artifact_paths(root: Path, directory: Path) -> list[str]:
    return [
        path.relative_to(root).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _records(root: Path) -> list[dict[str, Any]]:
    path = root / "ledger.json"
    return read_json(path) if path.is_file() else []


def _record_for(records: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("cell_id") == stage), None)


def _successful_attempt_is_valid(root: Path, record: dict[str, Any]) -> bool:
    if record.get("status") != "success":
        return False
    attempt = root / str(record.get("attempt", ""))
    if not attempt.is_dir():
        raise ValueError(f"Missing cached stage attempt: {attempt}")
    for relative in record.get("artifacts", []):
        if not (root / relative).is_file():
            raise ValueError(f"Missing cached stage artifact: {root / relative}")
    return True


def _stage_dataset(stage: dict[str, Any], default: str = "data/processed/dataset.npz") -> Path:
    return _resolve_path(stage.get("dataset", default))


def _require_files(stage: dict[str, Any], pairs: list[tuple[str, Path]]) -> None:
    missing = [label for label, path in pairs if not path.is_file()]
    if missing:
        raise StageUnsupported(
            "unsupported_missing_input",
            "Required stage inputs are missing",
            missing_inputs=missing,
        )


def _real_data(attempt: Path, stage: dict[str, Any]) -> dict[str, Any]:
    dataset = _stage_dataset(stage)
    data_config = _resolve_path(stage.get("data_config", "config/data.yaml"))
    if not dataset.is_file() and stage.get("prepare", False):
        _require_files(stage, [("data_config", data_config)])
        from core.data import prepare_dataset

        prepared = prepare_dataset(data_config)
        if not dataset.is_file():
            raise StageUnsupported(
                "unsupported_missing_input",
                "prepare_dataset returned data but did not create the declared dataset path",
            )
        del prepared
    _require_files(stage, [("dataset", dataset)])
    from core.data import load_prepared_dataset
    from oos.oos_final import validate_dataset

    ds = load_prepared_dataset(dataset)
    dates = validate_dataset(ds)
    _write_json_atomic(
        attempt / "dataset_summary.json",
        {
            "dataset": str(dataset),
            "rows": int(len(ds.x)),
            "sectors": int(ds.x.shape[1]),
            "covariates": [str(value) for value in ds.covariate_names],
            "split_counts": dict(Counter(str(value) for value in ds.split)),
            "date_start": str(dates[0]),
            "date_end": str(dates[-1]),
            "information_policy": "prepared Train-scaled data; no network download by the orchestrator",
        },
    )
    return {"status": "success", "rows": int(len(ds.x))}


def _causal_support(attempt: Path, stage: dict[str, Any], threads: int) -> dict[str, Any]:
    dataset = _stage_dataset(stage)
    causal_config = _resolve_path(stage.get("causal_config", "config/causal.yaml"))
    model_config = _resolve_path(stage.get("model_config", "config/dlcssm.yaml"))
    _require_files(
        stage,
        [("dataset", dataset), ("causal_config", causal_config), ("model_config", model_config)],
    )
    from causal.causal_support import discover_support, save_support
    from core.config import load_yaml
    from core.data import load_prepared_dataset
    from experiments.validation import candidate_configs

    ds = load_prepared_dataset(dataset)
    model_cfg = load_yaml(model_config)
    causal_cfg = load_yaml(causal_config)
    lags = sorted({int(item["lag"]) for item in candidate_configs(model_cfg)})
    if not lags:
        lags = [int(causal_cfg["lag"])]
    train = ds.x[np.asarray(ds.split) == "train"]
    for lag in lags:
        local = copy.deepcopy(causal_cfg)
        local["lag"] = lag
        local.setdefault("parallel", {})["n_jobs"] = min(threads, int(stage.get("jobs", threads)))
        result = discover_support(train, local)
        save_support(result, attempt / f"lag{lag}.npz", ds.sector_names)
    return {"status": "success", "lags": lags, "method": str(causal_cfg.get("method", "unknown"))}


def _support_attempt(root: Path) -> Path | None:
    records = _records(root)
    record = _record_for(records, "causal_support")
    if record and record.get("status") == "success":
        return root / str(record["attempt"])
    return None


def _model(attempt: Path, stage: dict[str, Any], root: Path, threads: int) -> dict[str, Any]:
    dataset = _stage_dataset(stage)
    model_config = _resolve_path(stage.get("model_config", "config/dlcssm.yaml"))
    support_root = _support_attempt(root)
    _require_files(stage, [("dataset", dataset), ("model_config", model_config)])
    if support_root is None:
        raise StageUnsupported(
            "unsupported_missing_input", "Successful causal_support stage is required"
        )
    from causal.causal_support import load_support
    from core.config import load_yaml
    from core.data import load_prepared_dataset
    from experiments.validation import candidate_configs, tune_validation

    ds = load_prepared_dataset(dataset)
    model_cfg = load_yaml(model_config)
    lags = sorted({int(item["lag"]) for item in candidate_configs(model_cfg)})
    missing = [
        str(support_root / f"lag{lag}.npz")
        for lag in lags
        if not (support_root / f"lag{lag}.npz").is_file()
    ]
    if missing:
        raise StageUnsupported(
            "unsupported_missing_input",
            "Support artifacts for the model grid are missing",
            missing_inputs=missing,
        )

    def support_for_lag(lag: int, _x_train: np.ndarray):
        support = load_support(support_root / f"lag{lag}.npz")
        return support, support.method

    report = tune_validation(
        ds,
        model_cfg,
        support_for_lag,
        attempt,
        progress=lambda message: print(f"[main:model] {message}", flush=True),
        jobs=min(threads, int(stage.get("jobs", threads))),
    )
    return {"status": "success", "selected_trial": report.get("selected_trial"), "lags": lags}


def _model_attempt(root: Path) -> Path | None:
    record = _record_for(_records(root), "model")
    if record and record.get("status") == "success":
        return root / str(record["attempt"])
    return None


def _baseline(attempt: Path, stage: dict[str, Any], root: Path) -> dict[str, Any]:
    dataset = _stage_dataset(stage)
    baseline_config = _resolve_path(stage.get("config", "config/baselines.json"))
    model_root = _model_attempt(root)
    _require_files(stage, [("dataset", dataset), ("config", baseline_config)])
    if model_root is None or not (model_root / "selected_model.npz").is_file():
        raise StageUnsupported(
            "unsupported_missing_input", "Successful DLCSSM model checkpoint is required"
        )
    from core.data import load_prepared_dataset
    from experiments.baseline_experiment import BaselineConfig, run_experiment
    from model.dlcssm import DLCSSM
    from oos.oos_final import FinalOOSConfig, load_baselines

    ds = load_prepared_dataset(dataset)
    config = BaselineConfig(**_load_mapping(baseline_config))
    model = DLCSSM.load(model_root / "selected_model.npz")
    result = run_experiment(
        ds, config, dlc_model=model, progress=lambda m: print(f"[main:baseline] {m}", flush=True)
    )
    result["features"].to_csv(attempt / "baselines.csv", index=False)
    for name in ("summary", "candidates", "comparisons", "losses"):
        result[name].to_csv(attempt / f"{name}.csv", index=False)
    np.savez_compressed(attempt / "baseline_paths.npz", **result["arrays"])
    report = _json_safe(result["report"])
    report["runner_input_paths"] = {"dataset": str(dataset), "config": str(baseline_config)}
    report["dlc_fit_checkpoint"] = str(model_root / "selected_model.npz")
    _write_json_atomic(attempt / "report.json", report)
    # Validate the producer output using the unchanged downstream consumer.
    load_baselines(
        attempt / "baselines.csv",
        ds,
        FinalOOSConfig(horizon=config.horizon, n_regimes=config.n_regimes),
    )
    return {"status": "success", "models": list(result["summary"]["model"].astype(str))}


def _baseline_attempt(root: Path) -> Path | None:
    record = _record_for(_records(root), "baselines")
    if record and record.get("status") == "success":
        return root / str(record["attempt"])
    return None


def _oos(attempt: Path, stage: dict[str, Any], root: Path) -> dict[str, Any]:
    dataset = _stage_dataset(stage)
    feature_config = _resolve_path(stage.get("feature_config", "config/oos_final.json"))
    predictive_config = _resolve_path(stage.get("predictive_config", "config/oos_predictive.json"))
    model_root = _model_attempt(root)
    baseline_root = _baseline_attempt(root)
    _require_files(
        stage,
        [
            ("dataset", dataset),
            ("feature_config", feature_config),
            ("predictive_config", predictive_config),
        ],
    )
    if model_root is None or not (model_root / "selected_model.npz").is_file():
        raise StageUnsupported(
            "unsupported_missing_input", "Successful model checkpoint is required"
        )
    if baseline_root is None or not (baseline_root / "baselines.csv").is_file():
        raise StageUnsupported(
            "unsupported_missing_input", "Successful baselines artifact is required"
        )
    from dataclasses import asdict

    from core.data import load_prepared_dataset
    from model.dlcssm import DLCSSM
    from oos.oos_experiment import PredictiveConfig, score_export
    from oos.oos_features import save_oos_feature_matrix
    from oos.oos_final import FinalOOSConfig, build_final_oos_feature_matrix, prepare_designs
    from oos.oos_specs import SPEC_VERSION, build_default_oos_specs

    ds = load_prepared_dataset(dataset)
    feature = FinalOOSConfig(**_load_mapping(feature_config))
    predictive = PredictiveConfig(**_load_mapping(predictive_config))
    model = DLCSSM.load(model_root / "selected_model.npz")
    train_end = str(ds.dates[np.asarray(ds.split) == "train"][-1])
    design = attempt / "design"
    design.mkdir(parents=True, exist_ok=False)
    bundle = build_final_oos_feature_matrix(
        ds,
        model,
        baseline_csv=baseline_root / "baselines.csv",
        model_fit_end=train_end,
        config=feature,
    )
    designs, effective, target, masks, transform = prepare_designs(bundle, ds, feature)
    save_oos_feature_matrix(bundle, design / "features.npz")
    np.savez_compressed(
        design / "designs.npz",
        **designs,
        y=target,
        dates=bundle.dates.astype(str),
        row_indices=bundle.row_indices,
        **{f"mask_{name}": mask for name, mask in masks.items()},
        transform_mean=transform.mean,
        transform_scale=transform.scale,
        transform_keep_mask=transform.keep_mask,
    )
    specs = build_default_oos_specs(
        feature.horizon, n_regimes=feature.n_regimes, include_bk=feature.include_bk
    )
    _write_json_atomic(
        design / "specification.json",
        {
            "spec_version": SPEC_VERSION,
            "config": asdict(feature),
            "model_fit_end_declaration": train_end,
            "target": "mean(min(raw_market_return[t+1:t+H+1], 0)^2)",
            "declared_specs": {name: spec.feature_names for name, spec in specs.items()},
            "effective_specs": effective,
            "dropped_train_constant_features": transform.dropped_names,
            "transform_original_names": transform.original_names,
            "transform_kept_names": transform.kept_names,
            "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        },
    )
    result = score_export(design, attempt, predictive)
    return {
        "status": "success",
        "sample_counts": result["sample_counts"],
        "selected": result["selected"],
    }


def _geometry(attempt: Path, stage: dict[str, Any], root: Path) -> dict[str, Any]:
    dataset = _stage_dataset(stage)
    model_root = _model_attempt(root)
    if model_root is None or not (model_root / "selected_model.npz").is_file():
        raise StageUnsupported(
            "unsupported_missing_input", "Successful model checkpoint is required"
        )
    _require_files(stage, [("dataset", dataset)])
    from core.data import load_prepared_dataset
    from experiments.geometry_experiments import (
        coefficient_path_from_states,
        low_rank_path_report,
        reversal_map,
        stable_amplification_report,
    )
    from model.dlcssm import DLCSSM
    from model.propagation import companion_matrix

    ds = load_prepared_dataset(dataset)
    model = DLCSSM.load(model_root / "selected_model.npz")
    filt = model.filter(ds.x, ds.u)
    params = model.params_
    states = filt.mean
    a0, basis = params.coefficient_subspace(model.support_)
    path = coefficient_path_from_states(a0, basis, states)
    dimension = min(
        int(stage.get("dimension", max(1, min(params.state_dim, path.shape[1])))), path.shape[1]
    )
    block = min(int(stage.get("block_length", 20)), len(path))
    draws = int(stage.get("bootstrap_count", 100))
    if len(path) < 2 or block < 1:
        raise ValueError("Insufficient filtered coefficient path for geometry stage")
    low_rank = low_rank_path_report(
        path,
        dimension,
        block_length=block,
        bootstrap_count=draws,
        seed=int(stage.get("seed", 0)),
        checkpoint=attempt.parent.parent / "geometry_bootstrap.npz",
        progress=bool(stage.get("progress", True)),
        timeout_seconds=stage.get("timeout_seconds"),
        memory_gb=stage.get("memory_gb"),
    )
    lag_path = np.asarray([params.a_lags(state) for state in states])
    stable = stable_amplification_report(
        lag_path,
        horizon=int(stage.get("horizon", 10)),
        amplification_threshold=float(stage.get("amplification_threshold", 1.0)),
        stability_threshold=float(stage.get("stability_threshold", 1.0)),
    )
    reversal = reversal_map(float(a0[0]), basis[0], states) if len(a0) else None
    companions = np.asarray([companion_matrix(lags) for lags in lag_path])
    np.savez_compressed(
        attempt / "geometry_paths.npz",
        states=states,
        coefficient_path=path,
        lag_path=lag_path,
        spectral_radii=stable.spectral_radii,
        peak_gains=stable.peak_gains,
        stable_but_amplifying=stable.stable_but_amplifying,
        companions=companions,
    )
    summary = {
        "version": "main-geometry-stage-v1",
        "low_rank": low_rank.summary(),
        "stable_amplification": {
            "spectral_radii": stable.spectral_radii.tolist(),
            "nonnormalities": stable.nonnormalities.tolist(),
            "peak_gains": stable.peak_gains.tolist(),
            "peak_horizons": stable.peak_horizons.tolist(),
            "stable_but_amplifying": stable.stable_but_amplifying.tolist(),
            "stable_but_amplifying_fraction": stable.stable_but_amplifying_fraction,
        },
        "reversal": None
        if reversal is None
        else {
            "status": reversal.status,
            "positive_fraction": reversal.positive_fraction,
            "negative_fraction": reversal.negative_fraction,
            "zero_fraction": reversal.zero_fraction,
            "distance_mean": float(np.mean(reversal.distance[np.isfinite(reversal.distance)]))
            if np.isfinite(reversal.distance).any()
            else None,
        },
        "limitations": [
            "Geometry is computed on a supplied fixed Train-selected checkpoint.",
            "Latent state coordinates are not compared across independently fitted gauges.",
        ],
    }
    _write_json_atomic(attempt / "geometry.json", summary)
    return {"status": "success", "path_rows": int(len(path)), "dimension": dimension}


def _ablation(attempt: Path, stage: dict[str, Any], root: Path) -> dict[str, Any]:
    dataset = _stage_dataset(stage)
    model_root = _model_attempt(root)
    if model_root is None or not (model_root / "selected_model.npz").is_file():
        raise StageUnsupported(
            "unsupported_missing_input", "Successful model checkpoint is required"
        )
    _require_files(stage, [("dataset", dataset)])
    from core.data import load_prepared_dataset
    from experiments.ablation_falsification import (
        run_ablation_falsification,
        write_ablation_artifacts,
    )
    from model.dlcssm import DLCSSM
    from model.propagation import companion_matrix

    ds = load_prepared_dataset(dataset)
    model = DLCSSM.load(model_root / "selected_model.npz")
    filt = model.filter(ds.x, ds.u)
    params = model.params_
    lag_path = np.asarray([params.a_lags(state) for state in filt.mean])
    companions = [companion_matrix(lags) for lags in lag_path]
    # F7 is a random-invertible-gauge placebo.  Identity would only replay the
    # original parameterization and therefore cannot test gauge invariance.
    rng = np.random.default_rng(int(stage.get("seed", 0)))
    gauge = rng.normal(size=(params.state_dim, params.state_dim))
    while abs(float(np.linalg.det(gauge))) < 1e-8:
        gauge = rng.normal(size=(params.state_dim, params.state_dim))
    result = run_ablation_falsification(
        a_lags_path=lag_path,
        B=params.B,
        a0=params.A0.ravel(),
        states=filt.mean,
        companions=companions,
        params=params,
        gauge=gauge,
        horizon=min(int(stage.get("horizon", 5)), len(companions)),
        seed=int(stage.get("seed", 0)),
    )
    artifact = attempt / "artifact"
    write_ablation_artifacts(result, artifact)
    counts = Counter(row["status"] for row in result.ledger)
    if counts.get("failed_method", 0):
        return {
            "status": "failed_method",
            "reason": "Ablation/falsification cell failed",
            "status_counts": dict(counts),
        }
    if any(status.startswith("unsupported") for status in counts):
        return {
            "status": "unsupported",
            "reason": "Declared ablation/falsification cells require unavailable refits or artifacts",
            "status_counts": dict(counts),
        }
    return {"status": "success", "status_counts": dict(counts)}


def _synthetic(attempt: Path, stage: dict[str, Any], threads: int) -> dict[str, Any]:
    config_path = _resolve_path(stage.get("config", "config/synthetic_main.json"))
    _require_files(stage, [("config", config_path)])
    from experiments.synthetic_benchmark import run_benchmark, validate_config

    config = _load_mapping(config_path)
    validate_config(config)
    output = attempt / "benchmark"
    jobs = min(threads, int(stage.get("jobs", threads)))
    result = run_benchmark(
        config,
        output,
        reuse_from=stage.get("reuse_from"),
        jobs=jobs,
    )
    child_ledger = read_json(output / "ledger.json")
    counts = dict(sorted(Counter(row["status"] for row in child_ledger).items()))
    if result.get("status") == "completed":
        return {
            "status": "success",
            "status_counts": counts,
            "successful_cells": result.get("successful_cells"),
        }
    unsupported = sum(value for key, value in counts.items() if str(key).startswith("unsupported"))
    if unsupported and unsupported == result.get("attempted_cells", 0):
        return {
            "status": "unsupported_missing_input",
            "reason": "Synthetic benchmark cells are unsupported",
            "status_counts": counts,
        }
    return {
        "status": "failed_method",
        "reason": "Synthetic benchmark completed with failed or unsupported cells",
        "status_counts": counts,
        "successful_cells": result.get("successful_cells"),
    }


def _coverage(
    attempt: Path, config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_stage = {record["stage"]: record for record in records if record.get("stage") != "coverage"}
    rows = []
    for stage in STAGES:
        if stage == "coverage":
            continue
        record = by_stage.get(stage)
        declaration = config["stages"][stage]
        rows.append(
            {
                "stage": stage,
                "enabled": declaration["enabled"],
                "required": declaration["required"],
                "status": None if record is None else record["status"],
                "reason": None if record is None else record.get("reason"),
            }
        )
    counts = Counter(record["status"] for record in records if record.get("stage") != "coverage")
    payload = {
        "version": "main-coverage-v1",
        "blueprint_complete": all(row["status"] == "success" for row in rows if row["required"]),
        "required_stage_failures": [
            row["stage"] for row in rows if row["required"] and row["status"] != "success"
        ],
        "status_counts": dict(sorted(counts.items())),
        "stages": rows,
        "limitations": [
            "Coverage records execution status, not scientific validity or publication readiness by itself.",
            "Unsupported named methods remain unsupported and are not replaced.",
        ],
    }
    _write_json_atomic(attempt / "coverage.json", payload)
    return {
        "status": "success",
        "blueprint_complete": payload["blueprint_complete"],
        "status_counts": payload["status_counts"],
    }


def _stage_handler(
    stage: str,
    attempt: Path,
    stage_config: dict[str, Any],
    root: Path,
    threads: int,
    runner_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage == "synthetic":
        return _synthetic(attempt, stage_config, threads)
    if stage == "real_data":
        return _real_data(attempt, stage_config)
    if stage == "causal_support":
        return _causal_support(attempt, stage_config, threads)
    if stage == "model":
        return _model(attempt, stage_config, root, threads)
    if stage == "baselines":
        return _baseline(attempt, stage_config, root)
    if stage == "oos":
        return _oos(attempt, stage_config, root)
    if stage == "geometry":
        return _geometry(attempt, stage_config, root)
    if stage == "ablation_falsification":
        return _ablation(attempt, stage_config, root)
    if stage == "coverage":
        if runner_config is None:
            raise ValueError("coverage stage requires the complete runner configuration")
        return _coverage(attempt, runner_config, _records(root))
    raise ValueError(f"Unknown main stage: {stage}")


def _validate_resume_identity(root: Path, config: dict[str, Any]) -> None:
    stored_config = dict(read_json(root / "config.json"))
    current_config = dict(config)
    # "threads" is an execution resource, not a scientific parameter.  It is
    # intentionally allowed to change when retrying a finalized incomplete run.
    stored_config.pop("threads", None)
    current_config.pop("threads", None)
    if stored_config != current_config:
        raise ValueError("Resume configuration mismatch; start a new output directory")


def _verify_run(root: Path) -> dict[str, Any]:
    """Check that a finalized run has its completion marker and manifest."""
    manifest_path = root / "manifest.json"
    if not (root / "complete.json").is_file() or not manifest_path.is_file():
        raise ValueError("Run is not complete")
    manifest = read_json(manifest_path)
    return {
        "status": "verified",
        "run_status": manifest.get("status"),
        "artifact_files": sum(1 for path in root.rglob("*") if path.is_file()),
        "status_counts": manifest.get("status_counts", {}),
    }


def _upsert_record(root: Path, record: dict[str, Any]) -> None:
    """Atomically replace a retried cell or append a new terminal cell."""
    records = _records(root)
    replacement = next(
        (index for index, old in enumerate(records) if old.get("cell_id") == record["cell_id"]),
        None,
    )
    if replacement is None:
        records.append(record)
    else:
        records[replacement] = record
    validate_ledger(records)
    _write_json_atomic(root / "ledger.json", records)


def run_main(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    allow_incomplete: bool = False,
    publication_mode: bool | None = None,
    threads: int | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Execute or resume a main run and return its audit summary."""
    config_path = _resolve_path(config_path)
    config = load_runner_config(config_path)
    runtime_threads = config["threads"] if threads is None else _positive_int(threads, "threads")
    if publication_mode is not None:
        config["publication_mode"] = bool(publication_mode)
    if allow_incomplete:
        config["publication_mode"] = False
    environment = _set_thread_environment(runtime_threads, config["inner_threads"])
    output = _resolve_path(output)
    root_exists = output.exists()
    if root_exists and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory is not empty; use --resume or a new directory: {output}"
        )

    if resume and root_exists and (output / "run.json").is_file():
        _validate_resume_identity(output, config)
        if (output / "complete.json").exists() and not retry_failed:
            checked = _verify_run(output)
            manifest = read_json(output / "manifest.json")
            return {
                "output": str(output),
                "status": manifest["status"],
                "exit_code": 0
                if manifest["status"] == "completed" or not config["publication_mode"]
                else 1,
                "reused_cells": len(read_json(output / "ledger.json")),
                "verification": checked,
            }
        if retry_failed:
            complete = read_json(output / "complete.json")
            if complete.get("status") == "completed":
                raise ValueError("Refusing to retry a fully completed run")
            recovery = {
                "schema_version": "main-recovery-v1",
                "runtime_threads": runtime_threads,
                "reused_successful_cells": True,
            }
            _write_json_atomic(output / "recovery.json", recovery)
            (output / "complete.json").unlink()
            (output / "manifest.json").unlink()
        run = ExperimentArtifactRun(output)
        reused = 0
    else:
        if resume and root_exists and any(output.iterdir()):
            raise ValueError("--resume requires an existing main run with run.json")
        run = ExperimentArtifactRun.create(
            output,
            experiment=config["experiment"],
            config=config,
            metadata={
                "runner_schema": SCHEMA_VERSION,
                "environment_controls": environment,
                "platform": platform.platform(),
            },
        )
        reused = 0

    started = time.perf_counter()
    changed_since_resume = False
    for stage in STAGES:
        existing = _record_for(_records(output), stage)
        if (
            existing is not None
            and _successful_attempt_is_valid(output, existing)
            and not (stage == "coverage" and changed_since_resume)
        ):
            reused += 1
            continue
        stage_config = config["stages"][stage]
        attempt = _stage_attempt(output, stage)
        stage_started = time.perf_counter()
        if not stage_config["enabled"]:
            result: dict[str, Any] = {
                "status": "unsupported",
                "reason": "Stage disabled by configuration",
            }
        else:
            try:
                handler_config = dict(stage_config)
                if retry_failed and stage == "synthetic":
                    old_record = _record_for(_records(output), stage)
                    if old_record is not None:
                        old_attempt = output / str(old_record.get("attempt", ""))
                        old_benchmark = old_attempt / "benchmark"
                        if old_benchmark.is_dir():
                            handler_config["reuse_from"] = str(old_benchmark)
                result = _stage_handler(
                    stage,
                    attempt,
                    handler_config,
                    output,
                    runtime_threads,
                    config,
                )
            except StageUnsupported as exc:
                result = {"status": exc.status, "reason": exc.reason, **exc.details}
            except (ImportError, ModuleNotFoundError) as exc:
                result = {
                    "status": "unsupported_missing_dependency",
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                }
            except Exception as exc:  # retain every failure in the audit trail
                result = {
                    "status": "failed_method",
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                (attempt / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _write_json_atomic(
            attempt / "status.json",
            {
                "stage": stage,
                "status": result["status"],
                "elapsed_seconds": time.perf_counter() - stage_started,
                "result": result,
            },
        )
        record = {
            "cell_id": stage,
            "stage": stage,
            "local_cell_id": stage,
            "status": result["status"],
            "required": stage_config["required"],
            "enabled": stage_config["enabled"],
            "attempt": attempt.relative_to(output).as_posix(),
            "elapsed_seconds": time.perf_counter() - stage_started,
            "artifacts": _artifact_paths(output, attempt),
            **{key: value for key, value in result.items() if key != "status"},
        }
        _upsert_record(output, record)
        changed_since_resume = True
        stage_records = _records(output)
        _write_json_atomic(
            output / "stage_ledger.json",
            [
                {
                    "stage": name,
                    "required": config["stages"][name]["required"],
                    "enabled": config["stages"][name]["enabled"],
                    "status": next(
                        (row["status"] for row in stage_records if row["stage"] == name), None
                    ),
                    "attempt": next(
                        (row.get("attempt") for row in stage_records if row["stage"] == name), None
                    ),
                }
                for name in STAGES
            ],
        )

    records = _records(output)
    required_problems = [
        record["stage"]
        for record in records
        if record.get("required") and record.get("status") != "success"
    ]
    run_status = (
        "completed"
        if not any(record["status"] != "success" for record in records)
        else "completed_with_failures"
    )
    manifest = run.finalize(
        status=run_status,
        expected_cell_ids=STAGES,
        metadata={
            "runner_schema": SCHEMA_VERSION,
            "required_stage_failures": required_problems,
            "publication_mode": config["publication_mode"],
            "allow_incomplete": allow_incomplete,
            "elapsed_seconds": time.perf_counter() - started,
            "reused_cells": reused,
        },
    )
    checked = _verify_run(output)
    publication_failure = bool(config["publication_mode"] and required_problems)
    return {
        "output": str(output),
        "status": run_status,
        "required_stage_failures": required_problems,
        "exit_code": 1 if publication_failure else 0,
        "reused_cells": reused,
        "manifest": manifest,
        "verification": checked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse only verified successful cells"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Smoke/debug mode: retain failures but return zero unless configuration is invalid",
    )
    parser.add_argument(
        "--publication-mode",
        action="store_true",
        default=None,
        help="Override config and require every declared required stage to succeed",
    )
    parser.add_argument("--threads", type=int, help="Override configured outer parallelism")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed/unsupported cells in a finalized incomplete run; reuse verified successes",
    )
    args = parser.parse_args(argv)
    result = run_main(
        args.config,
        args.output,
        resume=args.resume,
        allow_incomplete=args.allow_incomplete,
        publication_mode=args.publication_mode,
        threads=args.threads,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
