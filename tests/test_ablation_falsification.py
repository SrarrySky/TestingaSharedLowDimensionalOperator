from __future__ import annotations

import json

import numpy as np

from experiments.ablation_falsification import (
    ABLATION_NAMES,
    CELL_NAMES,
    FALSIFICATION_NAMES,
    run_ablation_falsification,
    write_ablation_artifacts,
)
from model.dlcssm import DLCSSMParameters


def _inputs():
    rng = np.random.default_rng(9)
    a_lags = rng.normal(scale=0.03, size=(12, 1, 2, 2))
    B = rng.normal(scale=0.02, size=(2, 1, 2, 2))
    a0 = a_lags[0].ravel()
    states = rng.normal(size=(12, 2))
    companions = []
    from model.propagation import companion_matrix

    for t in range(len(a_lags)):
        companions.append(companion_matrix(a_lags[t]))
    params = DLCSSMParameters(
        c=np.zeros(2),
        A0=a_lags[0],
        B=B,
        Lambda=np.zeros((2, 2)),
        D=np.zeros((2, 0)),
        F=0.8 * np.eye(2),
        G=np.zeros((2, 0)),
        Q=np.eye(2),
        R=np.eye(2),
        m0=np.zeros(2),
        P0=np.eye(2),
    )
    return a_lags, B, a0, states, companions, params


def test_available_geometry_and_falsification_cells_are_computed():
    a_lags, B, a0, states, companions, params = _inputs()
    result = run_ablation_falsification(
        a_lags_path=a_lags,
        B=B,
        a0=a0,
        states=states,
        companions=companions,
        params=params,
        gauge=np.array([[1.2, 0.1], [-0.2, 0.9]]),
        horizon=4,
        seed=3,
    )
    statuses = {row["cell_id"]: row["status"] for row in result.ledger}
    for name in ("A9", "A10", "A11", "A12", "F1", "F2", "F7", "F8"):
        assert statuses[name] == "success"
    assert result.report["F7"]["max_operator_error"] < 1e-10
    assert result.report["F8"]["max_residual"] < 1e-10
    assert len(result.report["A10"]["spectral_radius"]) == 12
    assert set(statuses) == set(CELL_NAMES)
    assert set(ABLATION_NAMES).issubset(statuses)
    assert set(FALSIFICATION_NAMES).issubset(statuses)


def test_missing_refit_inputs_are_explicitly_unsupported(tmp_path):
    result = run_ablation_falsification(seed=1)
    assert all(row["status"] == "unsupported_missing_input" for row in result.ledger)
    output = write_ablation_artifacts(result, tmp_path / "run")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_failures"
    assert manifest["status_counts"]["unsupported_missing_input"] == len(CELL_NAMES)
