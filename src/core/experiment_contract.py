"""Lightweight artifact helpers for the experiment matrix.

A run has three control files:

"run.json"
    Immutable identity and configuration metadata.
"manifest.json"
    Cell counts, statuses and run-level metadata.
"complete.json"
    A completion marker.

The helpers here deliberately avoid content hashing; they only record run
identity, terminal cell statuses and produce deterministic JSON.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "main-artifact-v1"
CONTROL_FILES = frozenset({"run.json", "manifest.json", "complete.json"})
TERMINAL_STATUSES = frozenset(
    {
        "success",
        "failed",
        "failed_case",
        "failed_method",
        "unsupported",
        "unsupported_missing_dependency",
        "unsupported_missing_input",
        "unsupported_missing",
        "invalid_configuration",
    }
)


def _canonical_value(value: Any) -> Any:
    """Convert common scientific Python values to JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"Unsupported value in experiment contract: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value deterministically using strict JSON."""
    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return deterministic, newline-terminated JSON for human-readable files."""
    return canonical_json_bytes(value).decode("utf-8") + "\n"


def _write_json(path: Path, value: Any) -> None:
    """
     write json.
    :param path: Input argument used by this operation.
    :param value: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    """Read strict JSON from an experiment artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_load_npz(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Load an NPZ archive without permitting Python object deserialization.

    Returning a detached dictionary also closes the underlying archive before
    the caller starts processing arrays.  Object arrays are rejected even when
    they happen to be readable, because they make the artifact format depend
    on Python pickling rather than on portable numeric/string arrays.
    """
    required = {str(key) for key in required_keys}
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"NPZ archive is missing required keys: {missing}")
        arrays = {key: archive[key].copy() for key in archive.files}
    object_keys = sorted(key for key, value in arrays.items() if value.dtype.hasobject)
    if object_keys:
        raise ValueError(f"NPZ archive contains object arrays: {object_keys}")
    return arrays


def _validate_cell_record(record: Mapping[str, Any]) -> None:
    """
     validate cell record.
    :param record: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if not isinstance(record, Mapping):
        raise TypeError("ledger records must be mappings")
    status = record.get("status")
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Unknown nonterminal or unsupported cell status: {status!r}")
    if "cell_id" not in record or not str(record["cell_id"]).strip():
        raise ValueError("ledger records require a nonempty cell_id")


def validate_ledger(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_cell_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate terminal cell records and optionally exact grid coverage."""
    normalized = [_canonical_value(record) for record in records]
    for record in normalized:
        _validate_cell_record(record)
    ids = [str(record["cell_id"]) for record in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate cell_id in experiment ledger")
    if expected_cell_ids is not None:
        expected = {str(cell_id) for cell_id in expected_cell_ids}
        if set(ids) != expected:
            missing = sorted(expected - set(ids))
            extra = sorted(set(ids) - expected)
            raise ValueError(f"Ledger coverage mismatch: missing={missing}, extra={extra}")
    return normalized


def _ledger_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """
     ledger counts.
    :param records: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


class ExperimentArtifactRun:
    """Create and finalize one experiment run directory."""

    def __init__(self, root: str | Path):
        """
          init  .
        :param root: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        experiment: str,
        config: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentArtifactRun:
        """
        Create.
        :param root: Input argument used by this operation.
        :param experiment: Input argument used by this operation.
        :param config: Input argument used by this operation.
        :param metadata: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        run_payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment": str(experiment),
            "python": sys.version,
            "platform": platform.platform(),
            "metadata": dict(metadata or {}),
        }
        _write_json(root / "config.json", config)
        _write_json(root / "run.json", run_payload)
        _write_json(root / "ledger.json", [])
        return cls(root)

    @property
    def run(self) -> dict[str, Any]:
        """
        Run.
        :return: Result produced by this operation.
        """
        return read_json(self.root / "run.json")

    def record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append one terminal cell record to "ledger.json"."""
        _validate_cell_record(record)
        path = self.root / "ledger.json"
        records = read_json(path) if path.is_file() else []
        records.append(_canonical_value(record))
        validate_ledger(records)
        _write_json(path, records)
        return records[-1]

    def finalize(
        self,
        *,
        status: str = "completed",
        expected_cell_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write manifest and completion marker after validating the ledger."""
        if status not in {"completed", "completed_with_failures", "incomplete"}:
            raise ValueError(f"Invalid run status: {status}")
        if (self.root / "complete.json").exists():
            raise FileExistsError(f"Run is already finalized: {self.root}")
        records = read_json(self.root / "ledger.json")
        validate_ledger(records, expected_cell_ids=expected_cell_ids)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "attempted_cells": len(records),
            "status_counts": _ledger_counts(records),
            "successful_cells": sum(record["status"] == "success" for record in records),
            "metadata": dict(metadata or {}),
        }
        _write_json(self.root / "manifest.json", manifest)
        complete = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
        }
        _write_json(self.root / "complete.json", complete)
        return manifest
