"""Configuration loading and reproducibility controls for experiment runs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy and return a dedicated NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def require_equal_lag(causal_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> None:
    """Prevent an accidental mismatch between the screened and modeled lag order."""
    if int(causal_cfg["lag"]) != int(model_cfg["lag"]):
        raise ValueError(
            f"Causal lag={causal_cfg['lag']} differs from model lag={model_cfg['lag']}. "
            "Screening and estimation must use a declared compatible lag space."
        )
