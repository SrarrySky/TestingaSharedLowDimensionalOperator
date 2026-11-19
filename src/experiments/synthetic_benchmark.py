"""Auditable method x DGP x seed runner; not a claim of complete coverage.

Run python -m experiments.synthetic_benchmark --help. Failures remain in the ledger.
Only observed data enter fitting. Truth is used solely by recovery metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr

from baselines.connectedness import impulse_responses
from baselines.hmm_baseline import GaussianHMMBaseline, RegimeVARBaseline
from baselines.hmm_comparison import predictive_regime_probabilities
from baselines.network_baseline import FixedNetworkSSM
from baselines.var_baselines import (
    TVPVAR,
    SmoothVAR,
    StaticVAR,
    covariance,
    gaussian_nll,
    lag_matrices,
    positive_int,
)
from causal.causal_support import discover_support, save_support
from core.data import PreparedDataset
from core.parallel import validate_jobs
from experiments.baseline_experiment import BaselineConfig
from experiments.synthetic import generate_scenario, synthetic_metrics
from experiments.validation import candidate_configs, tune_validation
from model.dlcssm import DLCSSM
from model.forecasting import one_step_forecasts

METHODS = {
    "static": "Static VAR",
    "sparse": "Sparse VAR",
    "tvp": "TVP-VAR",
    "smooth": "Smooth TV-VAR (one-sided)",
    "network": "Fixed network SSM",
    "hmm": "HMM + soft regime VAR",
    "dlc": "DLC-SSM",
}
STRESSES = (
    "latent_confounder",
    "student_t",
    "heteroskedastic",
    "strong_autocorrelation",
    "missing_async",
    "structural_break",
    "support_misspecification",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an experiment input or artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """Hash every Python source file using portable project-relative keys."""
    root = Path(root).resolve()
    source = root / "src"
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(source.rglob("*.py"))
    }


def artifact_sha256(root: str | Path) -> dict[str, str]:
    """Hash finalized run artifacts, excluding the self-referential manifest."""
    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def write_json(path, data):
    """
    Write json.
    :param path: Input argument used by this operation.
    :param data: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def validate_config(config):
    """
    Validate config.
    :param config: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    required = {
        "purpose",
        "scenarios",
        "seeds",
        "T",
        "n",
        "lag",
        "train_count",
        "validation_count",
        "methods",
        "baselines",
        "dlc",
        "support",
    }
    if set(config) != required:
        raise ValueError(f"Expected configuration fields {sorted(required)}")
    if config["purpose"] not in ("engineering_smoke", "benchmark"):
        raise ValueError("purpose must be engineering_smoke or benchmark")
    for key in ("T", "n", "lag", "train_count", "validation_count"):
        positive_int(config[key], key)
    for key, allowed in (("methods", METHODS), ("scenarios", [f"s{i}" for i in range(9)])):
        values = config[key]
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
            or not set(values) <= set(allowed)
        ):
            raise ValueError(f"Invalid {key}")
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    c = BaselineConfig(**config["baselines"])
    candidates = candidate_configs(config["dlc"])
    warmup = max(
        max(c.lags), c.smooth_lag + max(c.smooth_bandwidths), max(v["lag"] + 1 for v in candidates)
    )
    if (
        config["n"] < 2
        or config["train_count"] <= warmup + 2
        or config["validation_count"] < 2
        or config["T"] - config["train_count"] - config["validation_count"] < 10
    ):
        raise ValueError("Insufficient dimensions or chronological split sizes")
    if config["dlc"].get("selection", {}).get("covariate_policy") != "persistence":
        raise ValueError("Synthetic comparisons require explicit persistence DLC selection")
    if config["support"].get("method") not in ("linear_lasso", "lpcmci"):
        raise ValueError("Unsupported screening method; no silent fallback")


def scenario_variants(scenario):
    """
    Scenario variants.
    :param scenario: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if scenario == "s6":
        return ("low", "high")
    return STRESSES if scenario == "s7" else ("base",)


def generate_case(config, scenario, variant, seed):
    """
    Generate case.
    :param config: Input argument used by this operation.
    :param scenario: Input argument used by this operation.
    :param variant: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    kwargs = dict(T=config["T"], n=config["n"], lag=config["lag"], seed=seed)
    if scenario == "s7":
        kwargs["stress"] = variant
    truth = generate_scenario(scenario, **kwargs)
    if scenario == "s6" and variant == "high":
        operators = np.repeat(truth.metadata["high_lags"][None], config["T"], axis=0)
        truth = replace(
            truth,
            x=truth.metadata["high_x"].copy(),
            operators=operators,
            a_path=operators[:, truth.support],
        )
    return truth


def prepare_case(truth, config):
    """No oracle covariates, filling missing values, or Test-fitted scaling."""
    x, u = truth.x, truth.u
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Missing/asynchronous observations are unsupported; no imputation applied")
    nt, nv = config["train_count"], config["validation_count"]
    xm, xs = x[:nt].mean(0), x[:nt].std(0)
    um, us = u[:nt].mean(0), u[:nt].std(0)
    xs, us = np.where(xs > 1e-12, xs, 1.0), np.where(us > 1e-12, us, 1.0)
    return PreparedDataset(
        x=(x - xm) / xs,
        u=(u - um) / us,
        dates=pd.date_range("2000-01-01", periods=len(x)).strftime("%Y-%m-%d").to_numpy(dtype=str),
        sector_codes=np.array([f"x{i}" for i in range(x.shape[1])]),
        sector_names=np.array([f"x{i}" for i in range(x.shape[1])]),
        covariate_names=np.array([f"u{i}" for i in range(u.shape[1])], dtype=str),
        split=np.array(["train"] * nt + ["validation"] * nv + ["test"] * (len(x) - nt - nv)),
        x_mean=xm,
        x_scale=xs,
        u_mean=um,
        u_scale=us,
    )


@dataclass
class Prediction:
    mean: np.ndarray
    nll: np.ndarray
    # Prior operators at target t, never the posterior incorporating x_t.
    operators: np.ndarray
    support: np.ndarray


@dataclass
class RegimeFit:
    hmm: GaussianHMMBaseline
    var: RegimeVARBaseline
    sigmas: np.ndarray


def predict(model, x, u, sigma=None):
    """
    Predict.
    :param model: Input argument used by this operation.
    :param x: Input argument used by this operation.
    :param u: Input argument used by this operation.
    :param sigma: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if isinstance(model, StaticVAR):
        mean = model.predict(x)
        return Prediction(
            mean,
            gaussian_nll(x, mean, model.sigma_),
            np.repeat(model.a_[None], len(x), axis=0),
            abs(model.a_) > 1e-12,
        )
    if isinstance(model, TVPVAR):
        path = model.filter(x)
        a = np.full((len(x), model.lag, x.shape[1], x.shape[1]), np.nan)
        a[model.lag] = 0
        for t in range(model.lag + 1, len(x)):
            a[t] = lag_matrices(path.beta_filtered[t - 1], model.lag)
        return Prediction(
            path.prediction,
            gaussian_nll(x, path.prediction, model.sigma_, path.variance_multiplier),
            a,
            np.ones(a.shape[1:], dtype=bool),
        )
    if isinstance(model, SmoothVAR):
        path = model.forecast_path(x)
        return Prediction(
            path.prediction,
            gaussian_nll(x, path.prediction, sigma),
            path.operators,
            np.ones(path.operators.shape[1:], dtype=bool),
        )
    if isinstance(model, FixedNetworkSSM):
        path = model.filter(x)
        a = np.full_like(path.operators, np.nan)
        a[model.lag :] = np.einsum("tk,klij->tlij", path.filtered.pred_mean, model.basis_)
        return Prediction(path.prediction, path.nll, a, np.any(abs(model.basis_) > 1e-12, axis=0))
    if isinstance(model, RegimeFit):
        probabilities = predictive_regime_probabilities(
            model.hmm, model.hmm.filtered_probabilities(x)
        )
        mean = model.var.predict(x, probabilities)
        log_components = []
        for k, covariance_k in enumerate(model.sigmas):
            weights = np.zeros_like(probabilities)
            weights[:, k] = 1.0
            component = model.var.predict(x, weights)
            log_components.append(-gaussian_nll(x, component, covariance_k))
        with np.errstate(divide="ignore"):
            nll = -logsumexp(np.log(probabilities) + np.array(log_components).T, axis=1)
        a = np.einsum("tk,klij->tlij", probabilities, model.var.coef_)
        return Prediction(mean, nll, a, np.any(abs(model.var.coef_) > 1e-12, axis=0))
    if isinstance(model, DLCSSM):
        path = one_step_forecasts(model, x, u)
        p = model.params_
        a = p.A0[None] + np.einsum("td,dlij->tlij", path.state_mean, p.B)
        support = (abs(p.A0) > 1e-12) | np.any(abs(p.B) > 1e-12, axis=0)
        return Prediction(path.mean, path.nll, a, support)
    raise TypeError(f"Unknown model class: {type(model).__name__}")


def fit_baseline(method, ds, config, output):
    """Reuse estimator APIs; isolate failed candidates and methods, without retries on Test."""
    c = BaselineConfig(**config["baselines"])
    nt, nv = config["train_count"], config["validation_count"]
    train, end = ds.x[:nt], nt + nv
    trials, best, best_loss = [], None, np.inf
    hmm = None
    if method == "hmm":
        hmm = GaussianHMMBaseline(
            c.n_regimes, config["run_seed"], c.hmm_n_iter, n_init=c.hmm_n_init
        ).fit(train)
        weights = hmm.filtered_probabilities(train)
        write_json(output / "hmm_diagnostics.json", hmm.diagnostics_)
    grid = c.smooth_bandwidths if method == "smooth" else c.lags
    for lag_or_window in grid:
        penalties = (
            c.lasso_alphas
            if method == "sparse"
            else c.process_variances
            if method in ("tvp", "network")
            else (c.smooth_ridge,)
            if method == "smooth"
            else c.ridges
        )
        for penalty in penalties:
            specification = dict(
                lag=c.smooth_lag if method == "smooth" else lag_or_window, penalty=penalty
            )
            trial = dict(specification=specification, status="failed", validation_mse=None)
            try:
                lag = specification["lag"]
                sigma = None
                if method in ("static", "sparse"):
                    model = StaticVAR(lag, penalty, method == "sparse").fit(train)
                elif method in ("tvp", "network"):
                    cls = TVPVAR if method == "tvp" else FixedNetworkSSM
                    model = cls(lag, penalty, c.tvp_p0, c.tvp_covariance_ridge).fit(train)
                elif method == "smooth":
                    specification["bandwidth"] = lag_or_window
                    model = SmoothVAR(lag, lag_or_window, penalty)
                    start = lag + lag_or_window
                    sigma = covariance((train - model.predict(train))[start:])
                else:
                    var = RegimeVARBaseline(lag, penalty).fit(train, weights)
                    sigmas = []
                    for k in range(c.n_regimes):
                        one = np.zeros_like(weights)
                        one[:, k] = 1
                        error = (train - var.predict(train, one))[lag:]
                        w = weights[lag:, k]
                        sigmas.append(
                            covariance(error * np.sqrt(len(w) * w / max(w.sum(), 1e-12))[:, None])
                        )
                    model = RegimeFit(hmm, var, np.asarray(sigmas))
                pred = predict(model, ds.x[:end], ds.u[:end], sigma)
                loss = float(np.mean((ds.x[nt:end] - pred.mean[nt:end]) ** 2))
                if not np.isfinite(loss) or not np.isfinite(pred.nll[nt:end]).all():
                    raise FloatingPointError("Nonfinite Validation prediction or density")
                trial.update(status="accepted", validation_mse=loss)
                if loss < best_loss:
                    best, best_loss = (model, sigma, specification.copy()), loss
            except (
                Exception
            ) as exc:  # Experiment ledger preserves failures, including library errors.
                trial.update(error_type=type(exc).__name__, error=str(exc))
            trials.append(trial)
            write_json(output / "trials.json", trials)
    if best is None:
        raise RuntimeError("All candidates failed; see trials.json")
    write_json(output / "selected.json", dict(specification=best[2], validation_mse=best_loss))
    return best[0], best[1]


def fit_dlc(ds, config, output, misspecification=False, *, jobs=1):
    """
    Fit dlc.
    :param ds: Input argument used by this operation.
    :param config: Input argument used by this operation.
    :param output: Input argument used by this operation.
    :param misspecification: Input argument used by this operation.
    :param jobs: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    cfg = copy.deepcopy(config["dlc"])
    cfg["seed"] = config["run_seed"]

    def support_for_lag(lag, train):
        """
        Support for lag.
        :param lag: Input argument used by this operation.
        :param train: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        support_cfg = copy.deepcopy(config["support"])
        support_cfg.update(lag=lag, seed=config["run_seed"])
        support_cfg.setdefault("parallel", {})["n_jobs"] = jobs
        result = discover_support(train, support_cfg)
        if misspecification:
            # Deliberately delete one third of DISCOVERED edges, never inspect truth.
            mask, tier = result.support.copy(), result.tier.copy()
            indices = np.argwhere(mask)[::3]
            if len(indices):
                mask[tuple(indices.T)] = False
                tier[tuple(indices.T)] = ""
            result = replace(
                result,
                support=mask,
                tier=tier,
                metadata={
                    **(result.metadata or {}),
                    "deletion_rule": "every_third_discovered_edge",
                },
            )
        save_support(result, output / f"support_lag{lag}.npz", ds.sector_names)
        return result, result.method

    tune_validation(ds, cfg, support_for_lag, output, jobs=jobs)
    return DLCSSM.load(output / "selected_model.npz"), None


def recovery_metrics(truth, estimate, support, horizon):
    """Evaluate full operator space, zero-padding omitted lags, in DGP units.

    Compare target-date prior operators with truth at that date. Kernel/gain
    use frozen operators, h=1..H, not a claim of future-path recovery.
    Support is nonzero truth anywhere in the evaluated window, not its permissive mask.
    """
    if not np.isfinite(estimate).all() or len(truth) != len(estimate):
        raise ValueError("Recovery needs finite date-aligned operator paths")
    lag = max(truth.shape[1], estimate.shape[1])
    true = np.pad(truth, ((0, 0), (0, lag - truth.shape[1]), (0, 0), (0, 0)))
    est = np.pad(estimate, ((0, 0), (0, lag - estimate.shape[1]), (0, 0), (0, 0)))
    support = np.pad(support, ((0, lag - support.shape[0]), (0, 0), (0, 0)))
    metrics = synthetic_metrics(np.any(abs(true) > 1e-12, axis=0), support, true, est)
    true_kernel = np.array([impulse_responses(a, horizon + 1)[1:] for a in true])
    est_kernel = np.array([impulse_responses(a, horizon + 1)[1:] for a in est])
    metrics["frozen_kernel_rmse"] = float(np.sqrt(np.mean((true_kernel - est_kernel) ** 2)))
    true_gain = np.linalg.norm(true_kernel, ord=2, axis=(-2, -1)).max(axis=1)
    est_gain = np.linalg.norm(est_kernel, ord=2, axis=(-2, -1)).max(axis=1)
    metrics["observable_frozen_peak_gain_rmse"] = float(
        np.sqrt(np.mean((true_gain - est_gain) ** 2))
    )
    # Off-diagonal outgoing shock energy, aggregated over held-out dates/horizons.
    energy_true, energy_est = (
        np.sum(kernel**2, axis=(0, 1)) for kernel in (true_kernel, est_kernel)
    )
    np.fill_diagonal(energy_true, 0.0)
    np.fill_diagonal(energy_est, 0.0)
    scores_true, scores_est = energy_true.sum(0), energy_est.sum(0)
    metrics["transmitter_rank_spearman"] = (
        float(spearmanr(scores_true, scores_est).statistic)
        if np.ptp(scores_true) > 1e-12 and np.ptp(scores_est) > 1e-12
        else None
    )
    return metrics


def save_truth(output, truth):
    """
    Save truth.
    :param output: Input argument used by this operation.
    :param truth: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    arrays = {
        name: getattr(truth, name)
        for name in (
            "x",
            "u",
            "z",
            "a_path",
            "support",
            "operators",
            "factors",
            "regimes",
            "observed_mask",
        )
        if getattr(truth, name) is not None
    }
    metadata = {}
    for name, value in truth.metadata.items():
        if isinstance(value, np.ndarray):
            arrays[f"metadata__{name}"] = value
            metadata[name] = {"npz_key": f"metadata__{name}", "shape": list(value.shape)}
        else:
            metadata[name] = value
    if truth.params is not None:
        arrays.update(
            {f"params__{k}": v for k, v in vars(truth.params).items() if isinstance(v, np.ndarray)}
        )
    np.savez_compressed(output / "truth.npz", **arrays)
    write_json(output / "truth.json", dict(scenario=truth.scenario, metadata=metadata))


def save_model(output, model, sigma):
    """Save numeric fitted parameters without executable pickle objects."""
    arrays = {}
    if isinstance(model, RegimeFit):
        arrays.update(
            hmm_startprob=model.hmm.fit_.startprob,
            hmm_transmat=model.hmm.fit_.transmat,
            hmm_means=model.hmm.fit_.means,
            hmm_covars=model.hmm.fit_.covars,
            regime_coef=model.var.coef_,
            regime_intercept=model.var.intercept_,
            regime_sigmas=model.sigmas,
        )
    elif isinstance(model, DLCSSM):
        return  # tune_validation already saved the full selected checkpoint.
    else:
        arrays.update({k: v for k, v in vars(model).items() if isinstance(v, np.ndarray)})
        if sigma is not None:
            arrays["predictive_sigma"] = sigma
    if arrays:
        np.savez_compressed(output / "fitted_parameters.npz", **arrays)


def evaluate_method(ds, truth, config, method, output, variant, *, jobs=1):
    """
    Evaluate method.
    :param ds: Input argument used by this operation.
    :param truth: Input argument used by this operation.
    :param config: Input argument used by this operation.
    :param method: Input argument used by this operation.
    :param output: Input argument used by this operation.
    :param variant: Input argument used by this operation.
    :param jobs: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    start = time.perf_counter()
    if method == "dlc":
        model, sigma = fit_dlc(
            ds,
            config,
            output,
            variant == "support_misspecification",
            jobs=jobs,
        )
    else:
        model, sigma = fit_baseline(method, ds, config, output)
    fit_seconds = time.perf_counter() - start
    save_model(output, model, sigma)
    pred = predict(model, ds.x, ds.u, sigma)
    nt = config["train_count"]
    if (
        not np.isfinite(pred.mean[nt:]).all()
        or not np.isfinite(pred.nll[nt:]).all()
        or not np.isfinite(pred.operators[nt:]).all()
    ):
        raise FloatingPointError("Nonfinite held-out path; refusing silent date dropping")
    raw_operators = (
        pred.operators * ds.x_scale[None, None, :, None] / ds.x_scale[None, None, None, :]
    )
    raw_mean = pred.mean * ds.x_scale + ds.x_mean
    # Density transforms with the Train scaling Jacobian, not a fresh Test scale.
    raw_nll = pred.nll + np.log(ds.x_scale).sum()
    np.savez_compressed(
        output / "predictions.npz",
        mean_raw=raw_mean,
        nll_raw=raw_nll,
        operators_raw=raw_operators,
        support=pred.support,
        dates=ds.dates.astype(str),
        split=ds.split,
    )
    rows = []
    for part in ("validation", "test"):
        mask = ds.split == part
        row = dict(
            split=part,
            n=int(mask.sum()),
            mse_standardized=float(np.mean((ds.x[mask] - pred.mean[mask]) ** 2)),
            rmse_raw=float(np.sqrt(np.mean((truth.x[mask] - raw_mean[mask]) ** 2))),
            mean_nll_raw=float(raw_nll[mask].mean()),
            fit_seconds=fit_seconds,
        )
        row.update(
            recovery_metrics(
                truth.operators[mask],
                raw_operators[mask],
                pred.support,
                config["baselines"].get("horizon", 5),
            )
        )
        if any(not np.isfinite(v) for v in row.values() if isinstance(v, float | np.floating)):
            raise FloatingPointError("Nonfinite recovery or forecast metric")
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    return rows


_SEED_SUMMARY_METRICS = (
    "mse_standardized",
    "rmse_raw",
    "mean_nll_raw",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "orientation_precision",
    "orientation_recall",
    "dynamic_coefficient_rmse",
    "frozen_kernel_rmse",
    "observable_frozen_peak_gain_rmse",
    "transmitter_rank_spearman",
)


def build_seed_summary(config, records, scores):
    """Aggregate held-out Test metrics across seeds for every declared cell.

    Only the Test split enters the summary; Validation is a selection split and
    must not appear in a final mean. "n_attempted" is the declared-cell
    denominator (including failures); "n_success" counts cells that produced a
    Test row; "n_finite" counts finite seed values for a metric. Statistics are
    seed-level sample moments and a t interval over seeds, not a bootstrap
    interval over a single fitted path.
    """
    from scipy.stats import t as t_dist

    scores_df = pd.DataFrame(scores)
    if scores_df.empty:
        scores_df = pd.DataFrame(columns=["scenario", "variant", "method", "split"])
    test = scores_df[scores_df["split"] == "test"]

    attempted: dict[tuple[str, str, str], int] = {}
    succeeded: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = (record["scenario"], record["variant"], record["method"])
        attempted[key] = attempted.get(key, 0) + 1
        if record["status"] == "success":
            succeeded[key] = succeeded.get(key, 0) + 1

    rows: list[dict[str, object]] = []
    for scenario in config["scenarios"]:
        for variant in scenario_variants(scenario):
            for method in config["methods"]:
                key = (scenario, variant, method)
                cell = test[
                    (test["scenario"] == scenario)
                    & (test["variant"] == variant)
                    & (test["method"] == method)
                ]
                for metric in _SEED_SUMMARY_METRICS:
                    if metric in cell.columns:
                        values = pd.to_numeric(cell[metric], errors="coerce").to_numpy(dtype=float)
                        values = values[~np.isnan(values)]
                    else:
                        values = np.array([], dtype=float)
                    n_finite = int(values.size)
                    if n_finite == 0:
                        rows.append(
                            dict(
                                scenario=scenario,
                                variant=variant,
                                method=method,
                                metric=metric,
                                n_attempted=attempted.get(key, 0),
                                n_success=succeeded.get(key, 0),
                                n_finite=0,
                                mean=None,
                                std=None,
                                ci_low=None,
                                ci_high=None,
                            )
                        )
                        continue
                    mean = float(values.mean())
                    std = float(values.std(ddof=1)) if n_finite > 1 else 0.0
                    ci_low = ci_high = None
                    if n_finite > 1 and std > 0:
                        half = float(t_dist.ppf(0.975, n_finite - 1) * std / np.sqrt(n_finite))
                        ci_low, ci_high = mean - half, mean + half
                    rows.append(
                        dict(
                            scenario=scenario,
                            variant=variant,
                            method=method,
                            metric=metric,
                            n_attempted=attempted.get(key, 0),
                            n_success=succeeded.get(key, 0),
                            n_finite=n_finite,
                            mean=mean,
                            std=std,
                            ci_low=ci_low,
                            ci_high=ci_high,
                        )
                    )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["scenario", "variant", "method", "metric"], kind="mergesort"
        ).reset_index(drop=True)
    return summary


def summarize_seeds(config, records, scores, output):
    """Write the across-seed Test summary to "output/seed_summary.csv"."""
    build_seed_summary(config, records, scores).to_csv(output / "seed_summary.csv", index=False)


def _reuse_compatible(source_config, config):
    """Reject reuse when any result-affecting field differs from the source run.

    Cell-grid fields ("scenarios"/"methods"/"seeds") and pure scheduling
    or provenance fields are allowed to differ because reuse is per-cell and the
    DGP/estimator RNG are fixed by "run_seed", not by the configuration's seed
    placeholders.
    """

    def normalized(cfg):
        """
        Normalized.
        :param cfg: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        c = copy.deepcopy(cfg)
        for field in ("scenarios", "methods", "seeds", "purpose"):
            c.pop(field, None)
        c.pop("run_seed", None)
        for section, ignored in (("dlc", ("seed",)), ("support", ("seed", "parallel"))):
            if isinstance(c.get(section), dict):
                for key in ignored:
                    c[section].pop(key, None)
        return c

    src = normalized(source_config)
    new = normalized(config)
    if src == new:
        return
    differing = sorted(set(src) ^ set(new))
    if not differing:
        differing = sorted(k for k in src if k in new and src[k] != new[k])
    raise ValueError(
        "Reuse source config differs from the target on result-affecting field(s): "
        f"{differing}. Run a fresh benchmark without reuse instead of mixing protocols."
    )


def run_benchmark(config, output, progress=print, reuse_from=None, *, jobs=1):
    """
    Run benchmark.
    :param config: Input argument used by this operation.
    :param output: Input argument used by this operation.
    :param progress: Input argument used by this operation.
    :param reuse_from: Input argument used by this operation.
    :param jobs: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    validate_config(config)
    jobs = validate_jobs(jobs)
    output = Path(output).resolve()
    # Never overwrite even a failed/interrupted run: it is evidence too.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    versions = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "hmmlearn", "tigramite"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    manifest = dict(
        schema_version=2,
        purpose=config["purpose"],
        status="running",
        blueprint_complete=False,
        parallel_jobs=jobs,
        config_sha256=sha256_file(output / "config.json"),
        source_sha256=source_sha256(),
        versions=versions,
        environment={
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        resolved_baseline_config=asdict(BaselineConfig(**config["baselines"])),
        expanded_scenarios={s: list(scenario_variants(s)) for s in config["scenarios"]},
        method_names={m: METHODS[m] for m in config["methods"]},
        protocol={
            "fit": "Train only; baseline Validation MSE and DLC Validation persistence NLL selection",
            "forecast": "target t uses observations through t-1; no smoother; DLC u persistence",
            "hmm": "two-stage HMM on observed Train returns, not joint MS-VAR",
            "network": "Train correlation adjacency and identity; no oracle S4 basis",
            "support": "Train discovery only; S7 misspecification deletes every third discovered DLC edge",
            "recovery": "prior operator at t vs DGP truth at t in raw units; frozen h=1..H kernels",
            "density": "plug-in density; raw units include scaling Jacobian; no future-u integration",
            "missing": "unsupported; failed ledger entries, no imputation",
            "seeds": "DGP and estimator seeds fixed before fitting; no failed-seed replacement",
            "summary": "seed_summary.csv aggregates Test metrics across seeds; Validation never enters the mean",
        },
        limitations=[
            "Missing Regime-PCMCI and PCMCI+/DYNOTEARS support comparisons.",
            "LPCMCI screening is available when configured; linear_lasso is not LPCMCI.",
            "No complete TV-GIRF inference or oracle-known S4 basis competitor.",
            "No latent/operator subspace, reversal coverage or parameter-refit bootstrap metrics here.",
            "No peak native-memory measurement or effective degrees of freedom estimate.",
            "S6 low/high simulations have matched spectra, not common random innovations.",
            "S7 misspecification directly modifies only DLC support; others are unchanged controls.",
            "Across-seed intervals in seed_summary.csv are descriptive t intervals over seeds, not bootstrap CIs.",
        ],
    )
    write_json(output / "manifest.json", manifest)
    records, scores = [], []
    reuse_from = None if reuse_from is None else Path(reuse_from).resolve()
    previous_records = {}
    if reuse_from is not None:
        previous_ledger = reuse_from / "ledger.json"
        if not previous_ledger.is_file():
            raise FileNotFoundError(f"Synthetic reuse ledger is missing: {previous_ledger}")
        source_config_path = reuse_from / "config.json"
        if not source_config_path.is_file():
            raise FileNotFoundError(f"Synthetic reuse config is missing: {source_config_path}")
        _reuse_compatible(json.loads(source_config_path.read_text(encoding="utf-8")), config)
        previous_records = {
            (row["scenario"], row["variant"], int(row["seed"]), row["method"]): row
            for row in json.loads(previous_ledger.read_text(encoding="utf-8"))
        }
        attempt_keys = {
            (s, v, int(seed), m)
            for s in config["scenarios"]
            for v in scenario_variants(s)
            for seed in config["seeds"]
            for m in config["methods"]
        }
        reusable_keys = {
            key
            for key, row in previous_records.items()
            if str(row.get("status", "")).startswith(("success", "unsupported"))
        }
        reuse_keys = attempt_keys & reusable_keys
        previous_metrics = reuse_from / "metrics.csv"
        if previous_metrics.is_file():
            previous_scores = pd.read_csv(previous_metrics)
            keep = previous_scores.apply(
                lambda r: (
                    (
                        r["scenario"],
                        r["variant"],
                        int(r["seed"]),
                        r["method"],
                    )
                    in reuse_keys
                ),
                axis=1,
            )
            scores.extend(previous_scores[keep].to_dict("records"))
    for scenario in config["scenarios"]:
        for variant in scenario_variants(scenario):
            for seed in config["seeds"]:
                case_dir = output / f"{scenario}_{variant}" / f"seed_{seed}"
                case_dir.mkdir(parents=True)
                case_error, ds, truth = None, None, None
                try:
                    truth = generate_case(config, scenario, variant, seed)
                    save_truth(case_dir, truth)
                    ds = prepare_case(truth, config)
                    np.savez_compressed(case_dir / "prepared.npz", **asdict(ds))
                except Exception as exc:
                    case_error = dict(error_type=type(exc).__name__, error=str(exc))
                    (case_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                for method in config["methods"]:
                    method_dir = case_dir / method
                    key = (scenario, variant, int(seed), method)
                    previous = previous_records.get(key)
                    if previous is not None and str(previous.get("status", "")).startswith(
                        ("success", "unsupported")
                    ):
                        source_dir = reuse_from / str(previous["relative_path"])
                        if not source_dir.is_dir():
                            raise FileNotFoundError(
                                f"Synthetic cached cell is missing: {source_dir}"
                            )
                        shutil.copytree(source_dir, method_dir)
                        record = dict(previous, reused=True)
                        records.append(record)
                        write_json(method_dir / "record.json", record)
                        write_json(output / "ledger.json", records)
                        progress(f"reused {scenario}/{variant} seed={seed} method={method}")
                        continue
                    method_dir.mkdir()
                    record = dict(
                        scenario=scenario,
                        variant=variant,
                        seed=seed,
                        method=method,
                        model=METHODS[method],
                        status="running",
                        relative_path=method_dir.relative_to(output).as_posix(),
                    )
                    records.append(record)
                    write_json(output / "ledger.json", records)
                    progress(f"{scenario}/{variant} seed={seed} method={method}")
                    start = time.perf_counter()
                    if case_error:
                        record.update(
                            status="unsupported_missing"
                            if truth is not None and not np.isfinite(truth.x).all()
                            else "failed_case",
                            **case_error,
                        )
                    else:
                        try:
                            rows = evaluate_method(
                                ds,
                                truth,
                                {**config, "run_seed": seed},
                                method,
                                method_dir,
                                variant,
                                jobs=jobs,
                            )
                            scores.extend({**record, **row, "status": "success"} for row in rows)
                            record["status"] = "success"
                        except Exception as exc:
                            record.update(
                                status="failed_method",
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                            (method_dir / "failure.txt").write_text(
                                traceback.format_exc(), encoding="utf-8"
                            )
                    record["elapsed_seconds"] = time.perf_counter() - start
                    write_json(method_dir / "record.json", record)
                    write_json(output / "ledger.json", records)
                    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    summarize_seeds(config, records, scores, output)
    manifest.update(
        status="completed"
        if all(r["status"] == "success" for r in records)
        else "completed_with_failures",
        attempted_cells=len(records),
        successful_cells=sum(r["status"] == "success" for r in records),
        artifact_sha256=artifact_sha256(output),
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    """
    Main.
    :return: Result produced by this operation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing runs are never overwritten",
    )
    parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="Optional prior benchmark directory; matching cells are copied only when the "
        "source configuration is result-compatible with --config",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel DLC validation/support workers; numerical inner threads stay at one",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_benchmark(config, args.output, reuse_from=args.reuse_from, jobs=args.jobs)
    print(f"{result['status']}: {result['successful_cells']}/{result['attempted_cells']} cells")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
