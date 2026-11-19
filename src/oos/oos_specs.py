"""Declarative specifications for targets, splits, and OOS model variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oos.oos_features import OOSFeatureMatrix


@dataclass(frozen=True)
class OOSFeatureSpec:
    """Predeclared OOS predictor specification."""

    name: str
    feature_names: tuple[str, ...]


CONTROL_FEATURES = (
    "control__market_return",
    "control__realized_volatility",
    "control__turnover_proxy",
    "control__cross_sectional_dispersion",
)


def legacy_geometry_features(
    horizon: int,
) -> tuple[str, ...]:
    """Core DLC-SSM geometry for one horizon."""
    H = int(horizon)
    return (
        "geom__dynamic_nonnormality",
        f"geom__observable_peak_gain_h{H}",
        f"geom__total_path_energy_h{H}",
        f"geom__max_transmitter_h{H}",
        f"geom__transmitter_hhi_h{H}",
    )


def build_legacy_oos_specs(
    horizon: int,
) -> dict[str, OOSFeatureSpec]:
    """Default nested specifications for falsification tests."""
    H = int(horizon)
    geometry = legacy_geometry_features(H)

    return {
        # Controls only.
        "controls": OOSFeatureSpec(
            name="controls",
            feature_names=CONTROL_FEATURES,
        ),
        # Single-mechanism additions.
        "controls_plus_dynamic_nn": OOSFeatureSpec(
            name="controls_plus_dynamic_nn",
            feature_names=(
                *CONTROL_FEATURES,
                "geom__dynamic_nonnormality",
            ),
        ),
        "controls_plus_gain": OOSFeatureSpec(
            name="controls_plus_gain",
            feature_names=(
                *CONTROL_FEATURES,
                f"geom__observable_peak_gain_h{H}",
            ),
        ),
        "controls_plus_path_energy": OOSFeatureSpec(
            name="controls_plus_path_energy",
            feature_names=(
                *CONTROL_FEATURES,
                f"geom__total_path_energy_h{H}",
            ),
        ),
        "controls_plus_transmitter": OOSFeatureSpec(
            name="controls_plus_transmitter",
            feature_names=(
                *CONTROL_FEATURES,
                f"geom__max_transmitter_h{H}",
                f"geom__transmitter_hhi_h{H}",
            ),
        ),
        # Full point-geometry specification for this horizon.
        "controls_plus_geometry": OOSFeatureSpec(
            name="controls_plus_geometry",
            feature_names=(
                *CONTROL_FEATURES,
                *geometry,
            ),
        ),
    }


def select_feature_spec(
    bundle: OOSFeatureMatrix,
    spec: OOSFeatureSpec,
) -> np.ndarray:
    """Extract columns for a named OOS specification."""
    if len(set(bundle.feature_names)) != len(bundle.feature_names):
        raise ValueError("Duplicate feature names in OOS bundle")
    if len(set(spec.feature_names)) != len(spec.feature_names):
        raise ValueError("Duplicate feature names in OOS specification")
    name_to_index = {name: idx for idx, name in enumerate(bundle.feature_names)}

    missing = [name for name in spec.feature_names if name not in name_to_index]

    if missing:
        raise KeyError("Missing required OOS features: " + ", ".join(missing))

    indices = [name_to_index[name] for name in spec.feature_names]

    return bundle.X[
        :,
        indices,
    ]


SPEC_VERSION = "main-m0-m5-pathwise-v2"
PRIMARY_CONTROLS = (
    "control__market_return",
    "control__realized_volatility",
    "control__average_correlation",
    "control__turnover_proxy",
)


def _positive_integer(value: int, name: str) -> int:
    """
     positive integer.
    :param value: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def core_geometry_features(horizon: int) -> tuple[str, ...]:
    """Paper section 30 M3: full companion rho, nu and frozen gain."""
    H = _positive_integer(horizon, "horizon")
    return (
        "geom__spectral_radius",
        "geom__companion_nonnormality",
        f"geom__frozen_gain_h{H}",
    )


def required_baseline_features(
    horizon: int = 5, n_regimes: int = 3, include_bk: bool = False
) -> tuple[str, ...]:
    """
    Required baseline features.
    :param horizon: Input argument used by this operation.
    :param n_regimes: Input argument used by this operation.
    :param include_bk: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    H = _positive_integer(horizon, "horizon")
    K = _positive_integer(n_regimes, "n_regimes")
    if K < 2:
        raise ValueError("n_regimes must be >= 2")
    return (
        *(f"baseline__hmm_probability_{k}" for k in range(K - 1)),
        f"baseline__dy_connectedness_h{H}",
        f"baseline__tvp_risk_h{H}",
        *((f"baseline__bk_connectedness_h{H}",) if include_bk else ()),
    )


def build_default_oos_specs(
    horizon: int = 5, *, n_regimes: int = 3, include_bk: bool = False
) -> dict[str, OOSFeatureSpec]:
    """Predeclared M0--M5; missing baselines must never be omitted.

    K-1 filtered regime probabilities avoid a redundant simplex column with
    an intercept. BK is a separately predeclared robustness specification.
    Legacy exploratory models remain in build_legacy_oos_specs().
    """
    H = _positive_integer(horizon, "horizon")
    baseline = required_baseline_features(H, n_regimes, include_bk)
    blocks = (
        PRIMARY_CONTROLS,
        baseline[: n_regimes - 1],
        baseline[n_regimes - 1 :],
        core_geometry_features(H),
        (f"geom__state_migration_gap_h{H}",),
        (f"geom__state_susceptibility_h{H}",),
    )
    specs = {}
    names: tuple[str, ...] = ()
    for i, block in enumerate(blocks):
        names += block
        specs[f"M{i}"] = OOSFeatureSpec(f"M{i}", names)
    return specs
