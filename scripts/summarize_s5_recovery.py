"""Summarize the archived S5 recovery-control JSON without fitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.synthetic_benchmark import sha256_file

REQUIRED_FIELDS = {
    "T",
    "noise_scale",
    "seed",
    "support_edges",
    "termination",
    "em_iterations",
    "principal_angles_deg",
    "projection_distance",
    "dynamic_coefficient_rmse",
    "transmitter_rank_spearman",
}
METRICS = (
    "maximum_principal_angle_deg",
    "projection_distance",
    "dynamic_coefficient_rmse",
    "transmitter_rank_spearman",
)


def _validated_rows(payload: Any) -> pd.DataFrame:
    """
    Convert validated recovery records into a tabular representation.
    :param payload: JSON-compatible recovery payload.
    :return: rows, pandas.DataFrame: One row per validated result.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError("Recovery summary must be a nonempty JSON list")
    rows = []
    for index, record in enumerate(payload):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"Recovery row {index} is missing fields: {sorted(missing)}")
        angles = np.asarray(record["principal_angles_deg"], dtype=float)
        if angles.shape != (2,) or not np.isfinite(angles).all():
            raise ValueError(f"Recovery row {index} requires two finite principal angles")
        if record["termination"] not in {"converged", "max_iter"}:
            raise ValueError(f"Unknown termination reason in recovery row {index}")
        row = dict(record)
        row["principal_angle_1_deg"] = float(angles[0])
        row["principal_angle_2_deg"] = float(angles[1])
        row["maximum_principal_angle_deg"] = float(angles.max())
        row["objective_history_length"] = int(row.pop("em_iterations"))
        row.pop("principal_angles_deg")
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["T", "noise_scale", "seed"], kind="mergesort")
    numeric = (
        "T",
        "noise_scale",
        "seed",
        "support_edges",
        "objective_history_length",
        "principal_angle_1_deg",
        "principal_angle_2_deg",
        *METRICS,
    )
    if not np.isfinite(frame[list(numeric)].to_numpy(dtype=float)).all():
        raise ValueError("Recovery summary contains nonfinite numeric values")
    if frame.duplicated(["T", "noise_scale", "seed"]).any():
        raise ValueError("Recovery summary contains duplicate condition/seed rows")
    return frame.reset_index(drop=True)


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate recovery rows by method and experimental condition.
    :param rows, pandas.DataFrame: Validated long-form result table.
    :return: summary, pandas.DataFrame: Grouped statistics for reporting.
    """
    """Aggregate each data condition with sample moments and termination counts."""
    output = []
    for (sample_size, noise), group in rows.groupby(["T", "noise_scale"], sort=True):
        row: dict[str, Any] = {
            "T": int(sample_size),
            "noise_scale": float(noise),
            "n": int(len(group)),
            "converged_count": int((group["termination"] == "converged").sum()),
            "max_iter_count": int((group["termination"] == "max_iter").sum()),
        }
        for metric in METRICS:
            values = group[metric].astype(float)
            row.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                    f"{metric}_min": float(values.min()),
                    f"{metric}_max": float(values.max()),
                }
            )
        output.append(row)
    return pd.DataFrame(output)


def summarize(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """
    Build and write the S5 recovery summary artifacts.
    :param input_path, str or pathlib.Path: Input recovery result file.
    :param output_dir, str or pathlib.Path: Destination directory.
    :return: manifest, dict: Summary metadata and output paths.
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _validated_rows(json.loads(input_path.read_text(encoding="utf-8")))
    summary = build_summary(rows)
    rows.to_csv(output_dir / "s5_recovery_rows.csv", index=False)
    summary.to_csv(output_dir / "s5_recovery_summary.csv", index=False)
    manifest = {
        "status": "completed",
        "source": str(input_path),
        "source_sha256": sha256_file(input_path),
        "row_count": int(len(rows)),
        "condition_count": int(len(summary)),
        "termination_counts": {
            key: int(value)
            for key, value in rows["termination"].value_counts().sort_index().items()
        },
        "em_iterations_semantics": "renamed to objective_history_length; includes the initial objective",
        "inference": "descriptive sample moments only; no three-seed significance tests",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """
    Parse command-line arguments and run the recovery summariser.
    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
