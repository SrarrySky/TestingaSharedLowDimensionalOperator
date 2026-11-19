"""Validate and combine fixed-fit OOS block-length sensitivity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.synthetic_benchmark import sha256_file

BLOCKS = (10, 20, 40)
REQUIRED_FILES = (
    "report.json",
    "candidates.csv",
    "metrics.csv",
    "predictions.csv",
    "comparisons.csv",
)
INTERVAL_COLUMNS = {
    "ci_low",
    "ci_high",
    "relative_ci_low",
    "relative_ci_high",
}


def _read_run(path: Path, block: int) -> dict[str, Any]:
    """
    Read one blocked OOS run and attach its block-size identity.
    :param path, pathlib.Path: Run directory or result path.
    :param block, int: Block length used by the run.
    :return: result, dict: Parsed and annotated run payload.
    """
    if not path.is_dir():
        raise NotADirectoryError(path)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Block {block} run is missing files: {missing}")
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if int(report["config"]["block_length"]) != block:
        raise ValueError(f"Expected block_length={block} in {path / 'report.json'}")
    return {
        "path": path,
        "report": report,
        "candidates": pd.read_csv(path / "candidates.csv"),
        "metrics": pd.read_csv(path / "metrics.csv"),
        "predictions": pd.read_csv(path / "predictions.csv"),
        "comparisons": pd.read_csv(path / "comparisons.csv"),
    }


def _without_block(config: dict[str, Any]) -> dict[str, Any]:
    """
    Remove block-specific settings while preserving other configuration.
    :param config, dict: Original experiment configuration.
    :return: dict: Deep-copied configuration without the block override.
    """
    result = dict(config)
    result.pop("block_length", None)
    return result


def _assert_fixed(reference: dict[str, Any], candidate: dict[str, Any], block: int) -> None:
    """
    Assert that block sensitivity compares identical non-block settings.
    :param reference, dict: Reference run metadata.
    :param candidate, dict: Candidate run metadata.
    :param block, int: Candidate block length.
    :return: None.
    """
    if _without_block(reference["report"]["config"]) != _without_block(
        candidate["report"]["config"]
    ):
        raise ValueError(f"Block {block} changes predictive configuration beyond block_length")
    for field in ("selected", "fitted", "sample_counts", "input_sha256"):
        if reference["report"].get(field) != candidate["report"].get(field):
            raise ValueError(f"Block {block} changes fixed report field {field!r}")
    for name in ("candidates", "metrics", "predictions"):
        pd.testing.assert_frame_equal(
            reference[name], candidate[name], check_dtype=False, rtol=0, atol=0
        )
    left = reference["comparisons"]
    right = candidate["comparisons"]
    fixed_columns = [column for column in left.columns if column not in INTERVAL_COLUMNS]
    if list(left.columns) != list(right.columns):
        raise ValueError(f"Block {block} comparison columns differ")
    pd.testing.assert_frame_equal(
        left[fixed_columns], right[fixed_columns], check_dtype=False, rtol=0, atol=0
    )


def combine(runs: dict[int, str | Path], output_dir: str | Path) -> dict[str, Any]:
    """
    Combine blocked OOS runs into one sensitivity artifact.
    :param runs, dict: Mapping from block length to run location.
    :param output_dir, str or pathlib.Path: Destination directory.
    :return: manifest, dict: Combined result metadata and output paths.
    """
    if set(runs) != set(BLOCKS):
        raise ValueError(f"Expected block runs {BLOCKS}, received {sorted(runs)}")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {block: _read_run(Path(runs[block]).resolve(), block) for block in BLOCKS}
    reference = loaded[20]
    for block in (10, 40):
        _assert_fixed(reference, loaded[block], block)

    combined = reference["comparisons"][
        ["reference", "candidate", "mean_loss_improvement", "relative_mse_improvement"]
    ].copy()
    for block in BLOCKS:
        comparison = loaded[block]["comparisons"]
        keys = comparison[["reference", "candidate"]]
        pd.testing.assert_frame_equal(combined[["reference", "candidate"]], keys, check_dtype=False)
        for column in sorted(INTERVAL_COLUMNS):
            combined[f"block_{block}_{column}"] = comparison[column].to_numpy()
    combined.to_csv(output_dir / "block_sensitivity.csv", index=False)

    manifest = {
        "status": "verified",
        "blocks": list(BLOCKS),
        "comparison_count": int(len(combined)),
        "fixed_predictions": True,
        "fixed_selected_alpha": True,
        "fixed_point_estimates": True,
        "runs": {
            str(block): {
                "path": str(loaded[block]["path"]),
                "report_sha256": sha256_file(loaded[block]["path"] / "report.json"),
                "comparisons_sha256": sha256_file(loaded[block]["path"] / "comparisons.csv"),
            }
            for block in BLOCKS
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """
    Parse command-line arguments and generate the block-sensitivity summary.
    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block10", type=Path, required=True)
    parser.add_argument("--block20", type=Path, required=True)
    parser.add_argument("--block40", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = combine({10: args.block10, 20: args.block20, 40: args.block40}, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
