"""OOS feature specification. Run with "python -m oos.oos_final"."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.data import PreparedDataset, load_prepared_dataset
from core.evaluation import fit_oos_feature_transform, future_downside_target
from model.dlcssm import DLCSSM
from model.geometry import companion_basis
from model.propagation import selection_matrix
from oos.oos_features import (
    OOSFeatureMatrix,
    append_oos_features,
    build_point_oos_feature_matrix,
    save_oos_feature_matrix,
)
from oos.oos_specs import (
    SPEC_VERSION,
    _positive_integer,
    build_default_oos_specs,
    required_baseline_features,
    select_feature_spec,
)
from oos.pathwise_oos import predictive_pathwise_summary


@dataclass(frozen=True)
class FinalOOSConfig:
    horizon: int = 5
    correlation_window: int = 20
    n_regimes: int = 3
    discount: float = 0.95
    susceptibility_epsilon: float = 1e-10
    future_u: str = "persistence"
    include_bk: bool = False
    predictive_draws: int = 256
    predictive_seed: int = 0
    predictive_credible: float = 0.95
    path_gain_threshold: float = 1.0
    migration_gap_threshold: float = 1.0

    def __post_init__(self):
        """
          post init  .
        :return: Result produced by this operation.
        """
        for name in ("horizon", "correlation_window", "n_regimes"):
            _positive_integer(getattr(self, name), name)
        if self.correlation_window < 2 or self.n_regimes < 2:
            raise ValueError("correlation_window and n_regimes must be >= 2")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if not np.isfinite(self.susceptibility_epsilon) or self.susceptibility_epsilon <= 0:
            raise ValueError("susceptibility_epsilon must be positive and finite")
        if self.future_u != "persistence":
            raise ValueError("This specification fixes future u to persistence")
        _positive_integer(self.predictive_draws, "predictive_draws")
        if (
            isinstance(self.predictive_seed, bool | np.bool_)
            or not isinstance(self.predictive_seed, int | np.integer)
            or self.predictive_seed < 0
        ):
            raise ValueError("predictive_seed must be a nonnegative integer")
        if not np.isfinite(self.predictive_credible) or not 0 < self.predictive_credible < 1:
            raise ValueError("predictive_credible must be in (0, 1)")
        for name in ("path_gain_threshold", "migration_gap_threshold"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.include_bk, bool):
            raise ValueError("include_bk must be boolean")


def _dates(values, name):
    """
     dates.
    :param values: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(values).astype(str)))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if dates.hasnans:
        raise ValueError(f"Missing {name}")
    return dates


def validate_dataset(ds: PreparedDataset):
    """
    Validate dataset.
    :param ds: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    dates = _dates(ds.dates, "dataset dates")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Dataset dates must be strictly increasing and unique")
    if ds.x.ndim != 2 or ds.x.shape[1] < 2 or ds.u.ndim != 2:
        raise ValueError("Expected at least two sectors and a 2D covariate matrix")
    if len(ds.x) != len(dates) or len(ds.u) != len(dates) or len(ds.split) != len(dates):
        raise ValueError("Dataset row alignment mismatch")
    if not np.all(np.isfinite(ds.x)) or not np.all(np.isfinite(ds.u)):
        raise ValueError("Dataset x/u must be finite")
    if len(ds.covariate_names) != ds.u.shape[1] or len(set(ds.covariate_names)) != ds.u.shape[1]:
        raise ValueError("Covariate names must be unique and match u")
    order = {"train": 0, "validation": 1, "test": 2}
    if set(ds.split) != set(order):
        raise ValueError("Require train, validation and test splits")
    codes = np.array([order[s] for s in ds.split])
    if np.any(np.diff(codes) < 0):
        raise ValueError("Splits must be chronological train/validation/test blocks")
    return dates


def load_baselines(path: str | Path, ds: PreparedDataset, config: FinalOOSConfig):
    """Read date-keyed baseline values plus conservative availability metadata.

    information_date/parameter_fit_end describe the latest date used by ANY
    baseline in that row. These declarations cannot prove producer causality.
    """
    dates = validate_dataset(ds)
    frame = pd.read_csv(path)
    names = required_baseline_features(config.horizon, config.n_regimes, config.include_bk)
    required = ("date", "information_date", "parameter_fit_end", *names)
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError("Missing baseline CSV columns: " + ", ".join(missing))
    baseline_dates = _dates(frame.date, "baseline dates")
    if baseline_dates.has_duplicates:
        raise ValueError("Duplicate baseline dates")
    frame.index = baseline_dates
    if not dates.isin(frame.index).all():
        raise ValueError("Baseline CSV must cover every dataset date (warm-up values may be NaN)")
    frame = frame.loc[dates]
    available = _dates(frame.information_date, "information_date")
    fit_end = _dates(frame.parameter_fit_end, "parameter_fit_end")
    train_end = dates[np.asarray(ds.split) == "train"][-1]
    if np.any(available > dates):
        raise ValueError("Baseline information_date is after predictor date")
    # Train rows can be retrospectively computed using a Train-fitted model.
    # Validation/Test parameters must be frozen at the Train boundary.
    if np.any(fit_end > train_end):
        raise ValueError("Baseline parameter fitting extends beyond Train")
    values = frame[list(names)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite baseline values")
    probs = values[:, : config.n_regimes - 1]
    finite = np.isfinite(probs).all(axis=1)
    if (
        np.any(probs[finite] < 0)
        or np.any(probs[finite] > 1)
        or np.any(probs[finite].sum(axis=1) > 1 + 1e-8)
    ):
        raise ValueError("Invalid HMM probability simplex")
    return {name: values[:, j] for j, name in enumerate(names)}


def average_correlation(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing Pearson average over all distinct sector pairs, including t.

    Full windows only; any constant sector makes the row unavailable.
    Train-standardized returns preserve pairwise Pearson correlations.
    """
    result = np.full(len(x), np.nan)
    pairs = np.triu_indices(x.shape[1], 1)
    for t in range(window - 1, len(x)):
        block = x[t - window + 1 : t + 1]
        if np.all(block.std(axis=0) > 1e-12):
            result[t] = np.corrcoef(block, rowvar=False)[pairs].mean()
    return result


def structural_features(params, z, u, config: FinalOOSConfig):
    """Point/mean-path diagnostics and exact propagation-log-energy derivatives."""
    C = params.companion(z)
    J = selection_matrix(params.n, params.lag)
    bases = companion_basis(params.B)
    power = np.eye(len(C))
    path = np.eye(len(C))
    derivatives = np.zeros_like(bases)
    energy = np.zeros((params.n, params.n))  # [target, source]
    denergy = np.zeros((len(z), params.n, params.n))
    predicted_z = z.copy()
    frozen_gain = migration = path_gain = 0.0
    for h in range(1, config.horizon + 1):
        # d(C^h)/dz = C_r C^(h-1) + C d(C^(h-1))/dz_r.
        derivatives = bases @ power + C @ derivatives
        power = C @ power
        predicted_z = params.F @ predicted_z + params.G @ u
        path = params.companion(predicted_z) @ path
        frozen_gain = max(frozen_gain, float(np.linalg.norm(power, 2)))
        migration = max(migration, float(np.linalg.norm(path - power, 2)))
        path_gain = max(path_gain, float(np.linalg.norm(path, 2)))
        response = J @ power @ J.T
        dresponse = J @ derivatives @ J.T
        weight = config.discount ** (h - 1)
        energy += weight * response**2
        denergy += 2 * weight * response * dresponse
    elasticities = denergy / (config.susceptibility_epsilon + energy)
    offdiag = ~np.eye(params.n, dtype=bool)
    susceptibility = float(np.linalg.norm(elasticities[:, offdiag]))
    return frozen_gain, migration, susceptibility, path_gain


def build_final_oos_feature_matrix(
    ds: PreparedDataset,
    model: DLCSSM,
    *,
    baseline_csv: str | Path,
    model_fit_end: str,
    config: FinalOOSConfig | None = None,
) -> OOSFeatureMatrix:
    """
    Build final oos feature matrix.
    :param ds: Input argument used by this operation.
    :param model: Input argument used by this operation.
    :param baseline_csv: Input argument used by this operation.
    :param model_fit_end: Input argument used by this operation.
    :param config: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if config is None:
        config = FinalOOSConfig()
    dates = validate_dataset(ds)
    train_end = dates[np.asarray(ds.split) == "train"][-1]
    if _dates([model_fit_end], "model_fit_end")[0] > train_end:
        raise ValueError("DLCSSM parameter fitting extends beyond Train")
    baseline = load_baselines(baseline_csv, ds, config)
    # Fail before filtering if a required conventional control is absent.
    for name in ("market_return", "realized_volatility", "turnover_proxy"):
        if name not in ds.covariate_names:
            raise ValueError(f"Missing conventional control: {name}")
    bundle = build_point_oos_feature_matrix(
        ds,
        model,
        horizons=(config.horizon,),
        discount=config.discount,
        # Propagate the same declared state-uncertainty budget used by the
        # predictive path calculation.  A point estimate remains available in
        # the geometry features, while uncertainty features are no longer
        # silently disconnected from the formal OOS exporter.
        posterior_draws=config.predictive_draws,
        posterior_seed=config.predictive_seed,
    )
    params = model.params_
    rows = bundle.row_indices
    values = np.array(
        [
            structural_features(params, z, ds.u[t], config)
            for t, z in zip(rows, bundle.state_mean, strict=True)
        ]
    )
    predictive = [
        predictive_pathwise_summary(
            params,
            z,
            cov,
            ds.u[t],
            horizon=config.horizon,
            draws=config.predictive_draws,
            seed=config.predictive_seed,
            row_index=int(t),
            credible=config.predictive_credible,
            gain_threshold=config.path_gain_threshold,
            migration_threshold=config.migration_gap_threshold,
        )
        for t, z, cov in zip(rows, bundle.state_mean, bundle.state_cov, strict=True)
    ]
    H = config.horizon
    features = {
        f"geom__frozen_gain_h{H}": values[:, 0],
        f"geom__state_migration_gap_h{H}": np.array(
            [p["migration_gap"]["mean"] for p in predictive]
        ),
        f"diag__mean_path_migration_gap_h{H}": values[:, 1],
        f"geom__state_susceptibility_h{H}": values[:, 2],
        f"geom__predicted_path_gain_h{H}": np.array([p["path_gain"]["mean"] for p in predictive]),
        f"diag__mean_path_gain_h{H}": values[:, 3],
        **baseline,
    }
    for metric in ("path_gain", "migration_gap"):
        for statistic in ("mean", "std", "lower", "upper", "exceedance_probability"):
            features[f"predictive__{metric}_{statistic}_h{H}"] = np.array(
                [p[metric][statistic] for p in predictive]
            )
    if "control__average_correlation" not in bundle.feature_names:
        features["control__average_correlation"] = average_correlation(
            ds.x, config.correlation_window
        )
    else:
        raise ValueError(
            "Average correlation is computed by this frozen specification; remove duplicate covariate"
        )
    bundle = append_oos_features(bundle, features)
    specs = build_default_oos_specs(H, n_regimes=config.n_regimes, include_bk=config.include_bk)
    for spec in specs.values():
        select_feature_spec(bundle, spec)
    return bundle


def prepare_designs(
    bundle: OOSFeatureMatrix, ds: PreparedDataset, config: FinalOOSConfig | None = None
):
    """Common M0--M5 sample, Train-only transform, future-label boundary purge.

    The target is the PRIMARY downside-semivariance target only. No regression
    tuning or empirical performance claim is made by this feature exporter.
    """
    if config is None:
        config = FinalOOSConfig()
    specs = build_default_oos_specs(
        config.horizon, n_regimes=config.n_regimes, include_bk=config.include_bk
    )
    names = specs["M5"].feature_names
    X = select_feature_spec(bundle, specs["M5"])
    j = list(ds.covariate_names).index("market_return")
    raw_return = ds.u[:, j] * ds.u_scale[j] + ds.u_mean[j]
    target = future_downside_target(raw_return, config.horizon)[bundle.row_indices]
    end = bundle.row_indices + config.horizon
    same_split = end < len(ds.split)
    same_split[same_split] &= ds.split[end[same_split]] == bundle.split[same_split]
    common = np.isfinite(X).all(axis=1) & np.isfinite(target) & same_split
    masks = {s: common & (bundle.split == s) for s in ("train", "validation", "test")}
    if any(mask.sum() < 2 for mask in masks.values()):
        raise ValueError("Each split requires at least two common, finite, horizon-purged rows")
    transform = fit_oos_feature_transform(X[masks["train"]], names, verbose=False)
    scaled = np.full((len(X), len(transform.kept_names)), np.nan)
    scaled[common] = transform.transform(X[common])
    designs = {}
    effective = {}
    for name, spec in specs.items():
        effective[name] = tuple(n for n in spec.feature_names if n in transform.kept_names)
        indices = [transform.kept_names.index(n) for n in effective[name]]
        designs[name] = scaled[:, indices]
    return designs, effective, target, masks, transform


def main():
    """
    Main.
    :return: Result produced by this operation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fit-end", required=True, help="Declared last parameter-fit date")
    parser.add_argument(
        "--baselines", required=True, help="Date-keyed CSV described in docs/METHODS.md"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/checkpoints/oos_final_pathwise_v2")
    args = parser.parse_args()
    config = FinalOOSConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    ds = load_prepared_dataset(args.dataset)
    bundle = build_final_oos_feature_matrix(
        ds,
        DLCSSM.load(args.model),
        baseline_csv=args.baselines,
        model_fit_end=args.model_fit_end,
        config=config,
    )
    designs, effective, target, masks, transform = prepare_designs(bundle, ds, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_oos_feature_matrix(bundle, output / "features.npz")
    np.savez_compressed(
        output / "designs.npz",
        **designs,
        y=target,
        dates=bundle.dates,
        row_indices=bundle.row_indices,
        **{f"mask_{s}": m for s, m in masks.items()},
        transform_mean=transform.mean,
        transform_scale=transform.scale,
        transform_keep_mask=transform.keep_mask,
    )
    specs = build_default_oos_specs(
        config.horizon, n_regimes=config.n_regimes, include_bk=config.include_bk
    )
    metadata = {
        "spec_version": SPEC_VERSION,
        "config": asdict(config),
        "model_fit_end_declaration": args.model_fit_end,
        "provenance_limit": "Date declarations require an independent producer audit; they do not prove causality.",
        "target": "mean(min(raw_market_return[t+1:t+H+1], 0)^2)",
        "state": "forward filtered; full chronological pass; parameters frozen after Train",
        "uncertainty": "filtered-state and future Q uncertainty; fitted parameters held fixed",
        "pathwise": {
            "version": "posterior-predictive-v1",
            "future_covariates": "standardized u_t persisted at all future steps",
            "frozen_comparator": "C(z_t_draw)^h paired with the same initial path draw",
            "M4_predictor": "posterior mean of max_h ||Phi_h - C(z_t_draw)^h||_2",
            "interval": "equal-tailed conditional posterior predictive interval",
            "exceedance": "strictly greater than predeclared config thresholds; no OOS fitting",
            "rng": "SeedSequence([predictive_seed, original_dataset_row_index])",
            "mean_path_diagnostics": "diag__mean_path_*; excluded from M0--M5",
        },
        "declared_specs": {name: spec.feature_names for name, spec in specs.items()},
        "effective_specs": effective,
        "dropped_train_constant_features": transform.dropped_names,
        "transform_original_names": transform.original_names,
        "transform_kept_names": transform.kept_names,
        "sample_counts": {s: int(m.sum()) for s, m in masks.items()},
    }
    (output / "specification.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {SPEC_VERSION}: {output}")


if __name__ == "__main__":
    main()
