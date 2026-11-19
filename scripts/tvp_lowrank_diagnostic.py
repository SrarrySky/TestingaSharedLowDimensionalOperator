"""Read-only low-rank diagnostic on the unrestricted TVP coefficient path.

Loads the saved TVP filtered beta path (a full-dimensional coefficient random
walk, not the DLC-SSM low-rank factor model), centers first differences, and
reports the share of increment variation explained by the leading two principal
components, the entropy effective rank, and cross-window subspace distance.

This is an estimator-conditioned, uncalibrated diagnostic: it does not fit a
matched null, refit subspaces from resampled observations, or calibrate the
coefficient-estimation error. Run from the repository root:

    python scripts/tvp_lowrank_diagnostic.py

Writes "outputs/tvp_lowrank_diagnostic.json" and never modifies source artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs/main_mid_run/stages/baselines/attempt_001/baseline_paths.npz"
MODEL = ROOT / "outputs/main_mid_run/stages/model/attempt_001/selected_model.npz"
OUT = ROOT / "outputs/tvp_lowrank_diagnostic.json"
WARMUP = 252  # observation count dropped to avoid the TVP filter initialization phase


def _subspace(
    path: np.ndarray, mask: np.ndarray | None
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    view = path if mask is None else path[:, mask]
    delta = np.diff(view, axis=0)
    delta = delta - delta.mean(axis=0)
    _, singular, right = np.linalg.svd(delta, full_matrices=False)
    weights = singular**2
    prob = weights / weights.sum()
    positive = prob > 0
    effective_rank = float(np.exp(-np.sum(prob[positive] * np.log(prob[positive]))))
    r2_two = float(prob[:2].sum())
    basis = right[:2].T
    increment_rms = float(np.sqrt(np.mean(delta * delta)))
    return r2_two, effective_rank, basis, delta, increment_rms


def main() -> None:
    with np.load(BASELINE, allow_pickle=False) as archive:
        beta = archive["tvp_beta_filtered"]
        split = archive["split"]
    with np.load(MODEL, allow_pickle=False) as archive:
        rl_support = archive["support"].astype(bool).ravel()

    n = beta.shape[2]
    # Lag-1 coefficients, intercept dropped, target x source ordering.
    path = beta[:, 1:, :].transpose(0, 2, 1).reshape(len(beta), -1)
    off_diagonal = (~np.eye(n, dtype=bool)).reshape(-1)

    train = np.flatnonzero((split == "train") & (np.arange(len(path)) >= WARMUP))
    validation = np.flatnonzero(split == "validation")

    result: dict[str, dict[str, float]] = {}
    for label, mask in (("full", None), ("rl_support", rl_support), ("off_diagonal", off_diagonal)):
        r2_train, eff_train, basis_train, delta_train, rms_train = _subspace(path[train], mask)
        r2_val, eff_val, basis_val, delta_val, rms_val = _subspace(path[validation], mask)
        singular = np.linalg.svd(basis_train.T @ basis_val, compute_uv=False)
        distance = float(np.sqrt(max(0.0, 1.0 - singular.min() ** 2)))
        residual = float(
            np.linalg.norm(delta_val - (delta_val @ basis_train) @ basis_train.T)
            / np.linalg.norm(delta_val)
        )
        result[label] = {
            "n_train": int(len(train)),
            "n_validation": int(len(validation)),
            "p": int(path.shape[1] if mask is None else int(mask.sum())),
            "r2_train": float(r2_train),
            "r2_validation": float(r2_val),
            "effective_rank_train": float(eff_train),
            "effective_rank_validation": float(eff_val),
            "cross_window_projection_distance": distance,
            "validation_relative_residual": residual,
            "increment_rms_train": rms_train,
            "increment_rms_validation": rms_val,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
