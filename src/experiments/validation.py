"""Train-only fitting and chronological Validation selection with explicit information policy."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from causal.causal_support import SupportResult
from core.config import load_yaml
from core.data import PreparedDataset, load_prepared_dataset
from model.dlcssm import DLCSSM
from model.forecasting import one_step_forecasts
from model.geometry import spectral_radius
from core.parallel import parallel_map, validate_jobs


def candidate_configs(config: dict) -> list[dict]:
    """
    Candidate configs.
    :param config: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    grid = config.get("hyperparameter_grid", {})
    fields = ("state_dimension", "lag", "group_penalty", "state_l1_penalty")
    unknown = set(grid) - set(fields)
    if unknown:
        raise ValueError(f"Unknown grid fields: {sorted(unknown)}")
    defaults = (
        config["state_dimension"],
        config["lag"],
        config["penalty"]["group"],
        config["penalty"]["state_l1"],
    )
    values = [grid.get(key, [default]) for key, default in zip(fields, defaults, strict=True)]
    if any(not isinstance(v, list) or not v for v in values):
        raise ValueError("Every grid field must be a nonempty list")
    candidates = []
    for d, lag, group, state_l1 in itertools.product(*values):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (d, lag)):
            raise ValueError("state_dimension and lag must be positive integers")
        if any(not np.isfinite(v) or v < 0 for v in (group, state_l1)):
            raise ValueError("Penalties must be finite and nonnegative")
        cfg = copy.deepcopy(config)
        cfg.pop("hyperparameter_grid", None)
        cfg.update(state_dimension=d, lag=lag)
        cfg["penalty"].update(group=float(group), state_l1=float(state_l1))
        if cfg not in candidates:
            candidates.append(cfg)
    return candidates


def make_model(cfg: dict) -> DLCSSM:
    """
    Make model.
    :param cfg: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    fit, penalty = cfg["fit"], cfg["penalty"]
    return DLCSSM(
        lag=cfg["lag"],
        state_dim=cfg["state_dimension"],
        tier_b_multiplier=penalty.get("tier_b_multiplier", 2.0),
        lambda_group=penalty["group"],
        lambda_B=penalty["state_l1"],
        lambda_ridge=penalty["ridge"],
        max_iter=fit["max_em_iterations"],
        min_iter=fit["min_em_iterations"],
        tol=fit["tolerance"],
        prox_iterations=fit["prox_iterations"],
        jitter=fit["covariance_jitter"],
        diagonal_R=fit["diagonal_observation_covariance"],
        canonicalize=fit["canonicalize_after_fit"],
        seed=cfg["seed"],
    )


def _validation_candidate(task: tuple) -> dict:
    """Fit and score one Validation candidate in an isolated worker."""
    (
        index,
        cfg,
        support,
        support_method,
        x,
        u,
        train_count,
        quantiles,
        policy,
    ) = task
    trial = {
        "trial": index,
        "lag": cfg["lag"],
        "state_dimension": cfg["state_dimension"],
        "group_penalty": cfg["penalty"]["group"],
        "state_l1_penalty": cfg["penalty"]["state_l1"],
        "support_method": support_method,
        "tier_b_multiplier": cfg["penalty"].get("tier_b_multiplier", 2.0),
        "validation_score": None,
        "max_validation_spectral_radius": None,
    }
    try:
        model = make_model(cfg)
        fit = model.fit(x[:train_count].copy(), u[:train_count].copy(), copy.deepcopy(support))
        score, rho = score_validation(
            model,
            x,
            u,
            train_count,
            quantiles,
            covariate_policy=policy,
        )
        trial.update(
            validation_score=score,
            max_validation_spectral_radius=rho,
            termination_reason=fit.termination_reason,
            status="accepted"
            if rho <= cfg["stability"]["validation_reject_if_spectral_radius_above"]
            else "rejected_unstable",
        )
        return {"trial": trial, "config": cfg, "model": model, "error": None}
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        trial.update(status="failed_numerically", error=str(exc))
        return {"trial": trial, "config": cfg, "model": None, "error": str(exc)}


def validation_window(ds: PreparedDataset) -> tuple[np.ndarray, np.ndarray, int]:
    """Discard Test values before any numerical checks, fit or inference."""
    labels = np.asarray(ds.split)
    train = np.flatnonzero(labels == "train")
    val = np.flatnonzero(labels == "validation")
    if not len(train) or not len(val):
        raise ValueError("Nonempty Train and Validation splits are required")
    if not np.array_equal(train, np.arange(len(train))) or not np.array_equal(
        val, np.arange(len(train), len(train) + len(val))
    ):
        raise ValueError("Expected contiguous Train then Validation, before Test")
    end = int(val[-1]) + 1
    if np.any(labels[end:] != "test"):
        raise ValueError("Only Test rows may follow Validation")
    dates = np.asarray(ds.dates[:end], dtype="datetime64[D]")
    if np.isnat(dates).any() or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("Train/Validation dates must be unique and increasing")
    x, u = np.asarray(ds.x[:end]), np.asarray(ds.u[:end])
    if x.ndim != 2 or u.ndim != 2 or len(x) != end or len(u) != end:
        raise ValueError("Invalid Train/Validation array shapes")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Train/Validation values must be finite")
    return x, u, len(train)


def score_validation(
    model: DLCSSM,
    x: np.ndarray,
    u: np.ndarray,
    train_count: int,
    quantiles: list[float],
    *,
    covariate_policy: str = "conditional",
) -> tuple[float, float]:
    """Score every Validation date, carrying the last Train posterior forward.

    Legacy direct calls retain conditional scoring; the publication config
    explicitly selects persistence plug-in forecasts available at t-1.
    """
    p = model.params_
    if p is None:
        raise RuntimeError("Model is not fitted")
    if not p.lag < train_count < len(x):
        raise ValueError("Need Train lag context and nonempty Validation")
    forecast = one_step_forecasts(model, x, u, covariate_policy=covariate_policy)
    filtered = forecast.filtered.mean[train_count - p.lag :]
    states = np.vstack(
        (filtered, forecast.state_mean[train_count:], np.quantile(filtered, quantiles, axis=0))
    )
    radii = np.asarray([spectral_radius(p.companion(z)) for z in states])
    score = -float(forecast.nll[train_count:].mean())
    if not np.isfinite(score) or not np.isfinite(radii).all():
        raise FloatingPointError("Non-finite validation score or stability diagnostic")
    return score, float(radii.max())


def tune_validation(
    ds: PreparedDataset,
    config: dict,
    support_for_lag: Callable[[int, np.ndarray], tuple[np.ndarray | SupportResult, str]],
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
    *,
    jobs: int = 1,
    backend: str = "loky",
) -> dict:
    """Select a Train-fitted checkpoint; never refit on Validation or Test.

    support_for_lag receives ONLY Train data. Its output is cached per lag.
    Existing result files are never reused or silently overwritten.
    """
    validate_jobs(jobs)
    candidates = candidate_configs(config)
    policy = config.get("selection", {}).get("covariate_policy", "conditional")
    if policy not in ("conditional", "persistence"):
        raise ValueError("selection.covariate_policy must be conditional or persistence")
    x, u, train_count = validation_window(ds)
    if train_count <= max(c["lag"] for c in candidates) + 1:
        raise ValueError("Not enough Train observations for the requested lag grid")
    stability = config["stability"]
    threshold = float(stability["validation_reject_if_spectral_radius_above"])
    quantiles = stability["diagnostic_state_quantiles"]
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Stability threshold must be finite and positive")
    if not quantiles or any(not np.isfinite(q) or not 0 <= q <= 1 for q in quantiles):
        raise ValueError("Diagnostic quantiles must lie in [0, 1]")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ("validation_trials.json", "selected_model.yaml", "selected_model.npz")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Use a fresh output directory; tuning outputs already exist")
    report = {
        "selection_metric": (
            "mean_conditional_validation_log_predictive_density"
            if policy == "conditional"
            else "mean_persistence_plugin_validation_log_predictive_density"
        ),
        "covariate_policy": policy,
        "information_set": "through t-1 plus u_t" if policy == "conditional" else "through t-1",
        "covariate_uncertainty_integrated": False,
        "direction": "maximize",
        "fit_split": "train",
        "selection_split": "validation",
        "test_used": False,
        "refit_on_validation": False,
        "train_count": train_count,
        "validation_count": len(x) - train_count,
        "train_dates": [str(ds.dates[0]), str(ds.dates[train_count - 1])],
        "validation_dates": [str(ds.dates[train_count]), str(ds.dates[len(x) - 1])],
        "stability_threshold": threshold,
        "diagnostic_state_quantiles": quantiles,
        "tie_break": "first candidate in declared grid order",
        "config": copy.deepcopy(config),
        "trials": [],
        "selected_trial": None,
    }
    supports = {}
    tasks = []
    for i, cfg in enumerate(candidates):
        lag = cfg["lag"]
        if lag not in supports:
            mask, method = support_for_lag(lag, x[:train_count].copy())
            shape = (
                mask.support.shape if isinstance(mask, SupportResult) else np.asarray(mask).shape
            )
            if shape != (lag, x.shape[1], x.shape[1]):
                raise ValueError(f"Support shape mismatch for lag={lag}")
            supports[lag] = (mask, method)
        mask, method = supports[lag]
        tasks.append(
            (
                i,
                cfg,
                mask,
                method,
                x,
                u,
                train_count,
                quantiles,
                policy,
            )
        )

    trial_results = parallel_map(
        _validation_candidate,
        tasks,
        jobs=jobs,
        backend=backend,
        inner_max_num_threads=1,
    )
    best_model, best_cfg, best_score = None, None, -np.inf
    for result in trial_results:
        trial = result["trial"]
        report["trials"].append(trial)
        if trial["status"] == "accepted" and trial["validation_score"] > best_score:
            best_model = result["model"]
            best_cfg = result["config"]
            best_score = trial["validation_score"]
            report["selected_trial"] = trial["trial"]
        (output / names[0]).write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        if progress:
            progress(
                f"Trial {trial['trial'] + 1}/{len(candidates)}: "
                f"{trial['status']}, score={trial['validation_score']}"
            )
    if best_model is None:
        raise RuntimeError(
            "No stable finite candidate; see validation_trials.json. No model selected."
        )
    best_model.save(
        output / names[2],
        metadata={
            "fit_split": "train",
            "selection_split": "validation",
            "test_used": False,
            "selected_trial": report["selected_trial"],
            "validation_score": best_score,
            "selected_config": best_cfg,
            "covariate_policy": policy,
            "support_method": supports[best_cfg["lag"]][1],
        },
    )
    (output / names[1]).write_text(yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")
    return report


def run_tuning(
    data: Path, model_config: Path, causal_config: Path, output_dir: Path, jobs: int | None = None
) -> dict:
    """
    Run tuning.
    :param data: Input argument used by this operation.
    :param model_config: Input argument used by this operation.
    :param causal_config: Input argument used by this operation.
    :param output_dir: Input argument used by this operation.
    :param jobs: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    from causal.causal_support import discover_support, save_support

    ds = load_prepared_dataset(data)
    causal_cfg = load_yaml(causal_config)
    if jobs == 0:
        raise ValueError("jobs cannot be zero")

    def support_for_lag(lag: int, x_train: np.ndarray) -> tuple[SupportResult, str]:
        """
        Support for lag.
        :param lag: Input argument used by this operation.
        :param x_train: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        cfg = copy.deepcopy(causal_cfg)
        cfg["lag"] = lag
        if jobs is not None:
            cfg.setdefault("parallel", {})["n_jobs"] = jobs
        print(f"Discovering Train-only support for lag={lag}", flush=True)
        result = discover_support(x_train, cfg)
        save_support(result, output_dir / f"causal_support_lag{lag}.npz", ds.sector_names)
        return result, result.method

    return tune_validation(
        ds,
        load_yaml(model_config),
        support_for_lag,
        output_dir,
        progress=lambda message: print(message, flush=True),
        jobs=jobs or 1,
    )


def main() -> None:
    """
    Main.
    :return: Result produced by this operation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/dataset.npz"))
    parser.add_argument("--model-config", type=Path, default=Path("config/dlcssm.yaml"))
    parser.add_argument("--causal-config", type=Path, default=Path("config/causal.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation_tuning"))
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    run_tuning(args.data, args.model_config, args.causal_config, args.output_dir, args.jobs)


if __name__ == "__main__":
    main()
