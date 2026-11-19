"""Explicit ablation/falsification cell runner.

The runner records what can be computed from supplied fitted artifacts and
marks cells requiring unavailable refits or external methods as unsupported.
It never converts a diagnostic into a publication claim and never silently
substitutes a different estimator for an unavailable method.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model.dlcssm import DLCSSMParameters
from core.experiment_contract import ExperimentArtifactRun
from model.geometry import (
    spectral_radius,
    state_susceptibility,
)
from model.propagation import (
    frozen_pathwise_decomposition,
    pathwise_product,
)

ABLATION_NAMES = tuple(f"A{i}" for i in range(1, 13))
FALSIFICATION_NAMES = tuple(f"F{i}" for i in range(1, 9))
CELL_NAMES = ABLATION_NAMES + FALSIFICATION_NAMES


@dataclass(frozen=True)
class AblationFalsificationResult:
    report: dict[str, Any]
    ledger: list[dict[str, Any]]


def _finite(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    """
     finite.
    :param value: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :param ndim: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _positive(value: int, name: str) -> int:
    """
     positive.
    :param value: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_safe(value: Any) -> Any:
    """
     json safe.
    :param value: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer | np.floating | np.bool_):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _unsupported(cell_id: str, reason: str) -> dict[str, Any]:
    """
     unsupported.
    :param cell_id: Input argument used by this operation.
    :param reason: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return {"cell_id": cell_id, "status": "unsupported_missing_input", "reason": reason}


def _validate_lag_path(a_lags_path: np.ndarray) -> np.ndarray:
    """
     validate lag path.
    :param a_lags_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    path = _finite(a_lags_path, "a_lags_path", 4)
    if path.shape[1] < 1 or path.shape[2] != path.shape[3]:
        raise ValueError("a_lags_path must have shape (T, L, N, N)")
    return path


def _edge_centrality(a_lags_path: np.ndarray) -> dict[str, Any]:
    """
     edge centrality.
    :param a_lags_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    path = _validate_lag_path(a_lags_path)
    one_step = path[:, 0] ** 2
    indices = np.arange(one_step.shape[1])
    one_step[:, indices, indices] = 0.0
    # A[target, source] becomes source-first P[source, target].
    transmitter = one_step.sum(axis=1)
    receiver = one_step.sum(axis=2)
    return {
        "mean_transmitter": transmitter.mean(axis=0).tolist(),
        "mean_receiver": receiver.mean(axis=0).tolist(),
        "mean_net": (transmitter - receiver).mean(axis=0).tolist(),
        "definition": "one-step squared edge weight only; no companion powers or Gramian",
    }


def _spectral_only(a_lags_path: np.ndarray) -> dict[str, Any]:
    """
     spectral only.
    :param a_lags_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    from model.propagation import companion_matrix

    path = _validate_lag_path(a_lags_path)
    radii = np.asarray([spectral_radius(companion_matrix(lags)) for lags in path])
    return {"spectral_radius": radii.tolist(), "mean": float(radii.mean())}


def _frozen_and_migration(
    companions: list[np.ndarray],
    horizon: int,
) -> dict[str, Any]:
    """
     frozen and migration.
    :param companions: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if not companions:
        raise ValueError("companions must not be empty")
    horizon = _positive(horizon, "horizon")
    if horizon > len(companions):
        raise ValueError("horizon cannot exceed companion path length")
    C0 = companions[0]
    frozen = np.asarray(
        [float(np.linalg.norm(np.linalg.matrix_power(C0, h), 2)) for h in range(1, horizon + 1)]
    )
    migration = np.asarray(
        [
            float(
                np.linalg.norm(pathwise_product(companions, h) - np.linalg.matrix_power(C0, h), 2)
            )
            for h in range(1, horizon + 1)
        ]
    )
    return {
        "frozen_gain": frozen.tolist(),
        "migration_gap": migration.tolist(),
        "definition": "frozen C_0^h versus pathwise C_h...C_1",
    }


def _susceptibility(a_lags_path: np.ndarray, B: np.ndarray, horizon: int) -> dict[str, Any]:
    """
     susceptibility.
    :param a_lags_path: Input argument used by this operation.
    :param B: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    path = _validate_lag_path(a_lags_path)
    B = _finite(B, "B", 4)
    if B.shape[1:] != path.shape[1:]:
        raise ValueError("B and a_lags_path operator dimensions do not match")
    values = np.asarray(
        [state_susceptibility(lags, B, horizon)[1] for lags in path],
        dtype=float,
    )
    return {
        "aggregate_susceptibility": values.tolist(),
        "mean": float(values.mean()),
        "definition": "analytic state-to-propagation elasticity using companion derivatives",
    }


def _time_permutation(a_lags_path: np.ndarray, seed: int) -> dict[str, Any]:
    """
     time permutation.
    :param a_lags_path: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    path = _validate_lag_path(a_lags_path)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(path))
    original = np.diff(path.reshape(len(path), -1), axis=0)
    shuffled = np.diff(path[permutation].reshape(len(path), -1), axis=0)
    original_scale = float(np.linalg.norm(original))
    shuffled_scale = float(np.linalg.norm(shuffled))
    return {
        "permutation": permutation.tolist(),
        "original_increment_norm": original_scale,
        "permuted_increment_norm": shuffled_scale,
        "diagnostic_only": True,
    }


def _state_permutation(
    a0: np.ndarray,
    B: np.ndarray,
    states: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """
     state permutation.
    :param a0: Input argument used by this operation.
    :param B: Input argument used by this operation.
    :param states: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    a0 = _finite(a0, "a0", 1)
    B = _finite(B, "B")
    if B.ndim == 4:
        # Native DLC-SSM convention is (d, L, target, source); coefficient
        # paths use the same row order as DLCSSMParameters.coefficient_subspace.
        B = B.transpose(1, 2, 3, 0).reshape(-1, B.shape[0])
    elif B.ndim != 2:
        raise ValueError("B must have either (p, d) or (d, L, N, N) shape")
    states = _finite(states, "states", 2)
    if B.shape != (a0.size, states.shape[1]):
        raise ValueError("a0, B and states have incompatible dimensions")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(states))
    original = a0[None, :] + states @ B.T
    placebo = a0[None, :] + states[permutation] @ B.T
    difference = np.linalg.norm(original - placebo, axis=1)
    return {
        "permutation": permutation.tolist(),
        "mean_coefficient_path_difference": float(difference.mean()),
        "max_coefficient_path_difference": float(difference.max()),
        "diagnostic_only": True,
    }


def _gauge_placebo(
    params: DLCSSMParameters,
    states: np.ndarray,
    gauge: np.ndarray,
) -> dict[str, Any]:
    """
     gauge placebo.
    :param params: Input argument used by this operation.
    :param states: Input argument used by this operation.
    :param gauge: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    states = _finite(states, "states", 2)
    gauge = _finite(gauge, "gauge", 2)
    if states.shape[1] != params.state_dim or gauge.shape != (params.state_dim, params.state_dim):
        raise ValueError("gauge and states have incompatible dimensions")
    if abs(np.linalg.det(gauge)) < 1e-12:
        raise ValueError("gauge must be invertible")
    transformed = params.apply_gauge(gauge)
    transformed_states = states @ gauge.T
    errors = np.asarray(
        [
            np.linalg.norm(params.a_lags(state) - transformed.a_lags(state_star))
            for state, state_star in zip(states, transformed_states, strict=True)
        ]
    )
    return {
        "max_operator_error": float(errors.max()),
        "mean_operator_error": float(errors.mean()),
        "gauge_determinant": float(np.linalg.det(gauge)),
        "diagnostic_only": False,
    }


def _decomposition_check(companions: list[np.ndarray], horizon: int) -> dict[str, Any]:
    """
     decomposition check.
    :param companions: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if not companions:
        raise ValueError("companions must not be empty")
    horizon = _positive(horizon, "horizon")
    if horizon > len(companions):
        raise ValueError("horizon cannot exceed companion path length")
    residuals = []
    timewise = []
    for start in range(len(companions) - horizon + 1):
        C0 = companions[start]
        local = []
        for h in range(1, horizon + 1):
            path = companions[start : start + h]
            local.append(
                float(
                    np.linalg.norm(
                        pathwise_product(path)
                        - np.linalg.matrix_power(C0, h)
                        - frozen_pathwise_decomposition(C0, path),
                        ord="fro",
                    )
                )
            )
        timewise.append(local)
    residuals = np.max(np.asarray(timewise), axis=0).tolist()
    return {"residual_by_horizon": residuals, "max_residual": max(residuals)}


def run_ablation_falsification(
    *,
    a_lags_path: np.ndarray | None = None,
    B: np.ndarray | None = None,
    a0: np.ndarray | None = None,
    states: np.ndarray | None = None,
    companions: Iterable[np.ndarray] | None = None,
    params: DLCSSMParameters | None = None,
    gauge: np.ndarray | None = None,
    horizon: int = 5,
    seed: int = 0,
) -> AblationFalsificationResult:
    """Run available cells and explicitly ledger all unavailable cells."""
    if isinstance(seed, bool | np.bool_) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    path = None if a_lags_path is None else _validate_lag_path(a_lags_path)
    companion_list = (
        None
        if companions is None
        else [_finite(companion, "companion", 2) for companion in companions]
    )
    report: dict[str, Any] = {
        "version": "main-ablation-falsification-diagnostics-v1",
        "horizon": int(horizon),
        "seed": int(seed),
        "limitations": [
            "Cells requiring refitting or unavailable external methods are explicitly unsupported.",
            "Available cells are diagnostics on supplied fixed artifacts, not formal empirical claims.",
        ],
    }
    ledger: list[dict[str, Any]] = []

    def execute(cell_id: str, function, *args) -> None:
        """
        Execute.
        :param cell_id: Input argument used by this operation.
        :param function: Input argument used by this operation.
        :param *args: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        try:
            report[cell_id] = function(*args)
            ledger.append({"cell_id": cell_id, "status": "success"})
        except Exception as exc:
            ledger.append(
                {
                    "cell_id": cell_id,
                    "status": "failed_method",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    for cell in ABLATION_NAMES:
        if cell == "A9":
            if path is None:
                ledger.append(_unsupported(cell, "a_lags_path is required"))
            else:
                execute(cell, _edge_centrality, path)
        elif cell == "A10":
            if path is None:
                ledger.append(_unsupported(cell, "a_lags_path is required"))
            else:
                execute(cell, _spectral_only, path)
        elif cell == "A11":
            if companion_list is None:
                ledger.append(_unsupported(cell, "companion path is required"))
            else:
                execute(cell, _frozen_and_migration, companion_list, horizon)
        elif cell == "A12":
            if path is None or B is None:
                ledger.append(_unsupported(cell, "a_lags_path and B are required"))
            else:
                execute(cell, _susceptibility, path, B, horizon)
        else:
            ledger.append(_unsupported(cell, "formal refit/artifact producer is not supplied"))

    if path is None:
        ledger.append(_unsupported("F1", "a_lags_path is required"))
    else:
        execute("F1", _time_permutation, path, int(seed))
    if a0 is None or B is None or states is None:
        ledger.append(_unsupported("F2", "a0, B and states are required"))
    else:
        execute("F2", _state_permutation, a0, B, states, int(seed))
    ledger.append(_unsupported("F3", "filtered and leaked future-state paths are required"))
    ledger.append(
        _unsupported("F4", "residualized and original causal-support artifacts are required")
    )
    ledger.append(_unsupported("F5", "reversal bootstrap edge specification is required"))
    ledger.append(_unsupported("F6", "matched-spectrum null companion paths are required"))
    if params is None or states is None or gauge is None:
        ledger.append(_unsupported("F7", "params, states and an invertible gauge are required"))
    else:
        execute("F7", _gauge_placebo, params, states, gauge)
    if companion_list is None:
        ledger.append(_unsupported("F8", "companion path is required"))
    else:
        execute("F8", _decomposition_check, companion_list, horizon)
    return AblationFalsificationResult(report=_json_safe(report), ledger=ledger)


def write_ablation_artifacts(
    result: AblationFalsificationResult,
    output: str | Path,
) -> Path:
    """Persist the complete A/F ledger and diagnostics."""
    output = Path(output).resolve()
    run = ExperimentArtifactRun.create(
        output,
        experiment="main_ablation_falsification",
        config={"cells": list(CELL_NAMES), "protocol": result.report.get("version")},
    )
    (output / "report.json").write_text(
        json.dumps(_json_safe(result.report), ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    for record in result.ledger:
        run.record(record)
    status = (
        "completed"
        if all(row["status"] == "success" for row in result.ledger)
        else "completed_with_failures"
    )
    run.finalize(
        status=status, expected_cell_ids=CELL_NAMES, metadata={"producer": "ablation_falsification"}
    )
    return output
