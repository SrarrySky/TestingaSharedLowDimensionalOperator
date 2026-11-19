"""Validation-selected VAR/network/HMM comparisons and optional selected DLC.

Run with python -m experiments.baseline_experiment. See docs/METHODS.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from baselines.connectedness import companion_radius, dy_connectedness, innovation_risk
from core.data import load_prepared_dataset
from core.evaluation import moving_block_bootstrap_ci
from model.forecasting import one_step_forecasts
from baselines.hmm_baseline import GaussianHMMBaseline, RegimeVARBaseline
from baselines.hmm_comparison import predictive_regime_probabilities
from baselines.network_baseline import FixedNetworkSSM
from oos.oos_final import FinalOOSConfig, load_baselines, validate_dataset
from baselines.var_baselines import (
    TVPVAR,
    SmoothVAR,
    StaticVAR,
    covariance,
    gaussian_nll,
    lag_matrices,
    positive_int,
)


@dataclass(frozen=True)
class BaselineConfig:
    lags: tuple = (1, 3)
    ridges: tuple = (0.01, 1.0, 10.0)
    lasso_alphas: tuple = (0.001, 0.01)
    process_variances: tuple = (1e-6, 1e-5, 1e-4)
    tvp_p0: float = 1.0
    tvp_covariance_ridge: float = 1.0
    smooth_lag: int = 1
    smooth_bandwidths: tuple = (126, 252)
    smooth_ridge: float = 1.0
    horizon: int = 5
    n_regimes: int = 3
    hmm_n_init: int = 5
    hmm_n_iter: int = 200
    seed: int = 42
    bootstrap_draws: int = 1000
    block_length: int = 20

    def __post_init__(self):
        """
          post init  .
        :return: Result produced by this operation.
        """
        for name in ("lags", "smooth_bandwidths"):
            if not getattr(self, name):
                raise ValueError(f"Empty {name}")
            for value in getattr(self, name):
                positive_int(value, name)
        for name in (
            "smooth_lag",
            "horizon",
            "n_regimes",
            "hmm_n_init",
            "hmm_n_iter",
            "bootstrap_draws",
            "block_length",
        ):
            positive_int(getattr(self, name), name)
        if self.n_regimes < 2 or min(self.smooth_bandwidths) < 3:
            raise ValueError("Need K>=2 and bandwidth>=3")
        for name in ("ridges", "lasso_alphas", "process_variances"):
            values = getattr(self, name)
            if not values or not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError(f"Invalid {name}")
        values = [self.tvp_p0, self.tvp_covariance_ridge, self.smooth_ridge]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Invalid positive scalar configuration")


def dlc_predictions(model, x, u):
    """One-step distribution with future covariates fixed to last close."""
    forecast = one_step_forecasts(model, x, u)
    return forecast.mean, forecast.nll


def run_experiment(ds, config=None, *, dlc_model=None, progress=None):
    """
    Run experiment.
    :param ds: Input argument used by this operation.
    :param config: Input argument used by this operation.
    :param dlc_model: Input argument used by this operation.
    :param progress: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if config is None:
        config = BaselineConfig()
    dates = validate_dataset(ds)
    emit = progress or (lambda message: None)
    c = config
    split = np.asarray(ds.split)
    nt, nv = int((split == "train").sum()), int((split == "validation").sum())
    end = nt + nv
    warmup = max(max(c.lags), c.smooth_lag + max(c.smooth_bandwidths))
    if dlc_model is not None:
        warmup = max(warmup, dlc_model.params_.lag + 1)
    if nt <= warmup + 2 or nv < 2 or (split == "test").sum() < c.block_length:
        raise ValueError("Insufficient chronological split sizes for the declared grid/windows")
    raw = ds.x * ds.x_scale + ds.x_mean
    # Match data.py's sample-standard-deviation convention (ddof=1), so
    # existing Train-fitted checkpoints remain on exactly the same scale.
    xm, xs = raw[:nt].mean(0), raw[:nt].std(0, ddof=1)
    if not np.isfinite(raw).all() or np.any(xs <= 1e-12):
        raise ValueError("Nonfinite raw returns or constant Train sector")
    x = (raw - xm) / xs
    raw_u = ds.u * ds.u_scale + ds.u_mean
    um, us = raw_u[:nt].mean(0), raw_u[:nt].std(0, ddof=1)
    keep = us > 1e-12
    if not keep.any() or not np.isfinite(raw_u).all():
        raise ValueError("HMM needs finite nonconstant Train covariates")
    us = np.where(keep, us, 1.0)
    u = (raw_u - um) / us
    if dlc_model is not None and (not np.allclose(x, ds.x) or not np.allclose(u, ds.u)):
        raise ValueError("DLC checkpoint requires the dataset's original Train scaling to match")
    candidates, selected, predictions, likelihoods = [], {}, {}, {}

    def consider(name, specification, pred, fitted):
        """
        Consider.
        :param name: Input argument used by this operation.
        :param specification: Input argument used by this operation.
        :param pred: Input argument used by this operation.
        :param fitted: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        error = x[nt:end] - pred[nt:end]
        if not np.isfinite(error).all():
            raise ValueError(f"Nonfinite Validation predictions: {name} {specification}")
        score = float(np.mean(error**2))
        row = dict(model=name, **specification, validation_mse=score)
        candidates.append(row)
        if name not in selected or score < selected[name][0]:
            selected[name] = (score, row, fitted)

    for sparse, name, grid in (
        (False, "Static VAR", c.ridges),
        (True, "Sparse VAR", c.lasso_alphas),
    ):
        emit(f"Selecting {name} on Validation")
        for lag in c.lags:
            for penalty in grid:
                model = StaticVAR(lag, penalty, sparse).fit(x[:nt])
                consider(name, dict(lag=lag, penalty=penalty), model.predict(x[:end]), model)
        model = selected[name][2]
        predictions[name] = model.predict(x)
        likelihoods[name] = gaussian_nll(x, predictions[name], model.sigma_)

    emit("Selecting unrestricted coefficient-state TVP on Validation")
    for lag in c.lags:
        for q in c.process_variances:
            model = TVPVAR(lag, q, c.tvp_p0, c.tvp_covariance_ridge).fit(x[:nt])
            consider(
                "TVP-VAR",
                dict(lag=lag, process_variance=q),
                model.filter(x[:end]).prediction,
                model,
            )
    tvp = selected["TVP-VAR"][2]
    tvp_path = tvp.filter(x)
    predictions["TVP-VAR"] = tvp_path.prediction
    likelihoods["TVP-VAR"] = gaussian_nll(
        x, tvp_path.prediction, tvp.sigma_, tvp_path.variance_multiplier
    )

    emit("Selecting fixed Train-correlation network operator state-space on Validation")
    for lag in c.lags:
        for q in c.process_variances:
            model = FixedNetworkSSM(lag, q, c.tvp_p0, c.tvp_covariance_ridge).fit(x[:nt])
            consider(
                "Fixed network SSM",
                dict(lag=lag, process_variance=q, basis_source=model.basis_source_),
                model.filter(x[:end]).prediction,
                model,
            )
    network = selected["Fixed network SSM"][2]
    network_path = network.filter(x)
    predictions["Fixed network SSM"] = network_path.prediction
    likelihoods["Fixed network SSM"] = network_path.nll

    emit("Selecting one-sided local-linear smooth VAR on Validation")
    for bandwidth in c.smooth_bandwidths:
        model = SmoothVAR(c.smooth_lag, bandwidth, c.smooth_ridge)
        consider(
            "Smooth TV-VAR (one-sided)",
            dict(lag=c.smooth_lag, bandwidth=bandwidth, ridge=c.smooth_ridge),
            model.predict(x[:end], start=nt),
            model,
        )
    smooth = selected["Smooth TV-VAR (one-sided)"][2]
    predictions["Smooth TV-VAR (one-sided)"] = smooth.predict(x)
    # A Train-only fixed innovation-covariance plug-in; not local SV likelihood.
    smooth_sigma = covariance((x - predictions["Smooth TV-VAR (one-sided)"])[warmup:nt])
    likelihoods["Smooth TV-VAR (one-sided)"] = gaussian_nll(
        x, predictions["Smooth TV-VAR (one-sided)"], smooth_sigma
    )

    emit("Fitting existing HMM and selecting soft regime VAR on Validation")
    hmm = GaussianHMMBaseline(c.n_regimes, c.seed, c.hmm_n_iter, n_init=c.hmm_n_init).fit(
        u[:nt, keep]
    )
    prob_val = hmm.filtered_probabilities(u[:end, keep])
    prior_val = predictive_regime_probabilities(hmm, prob_val)
    for lag in c.lags:
        for ridge in c.ridges:
            model = RegimeVARBaseline(lag, ridge).fit(x[:nt], prob_val[:nt])
            consider(
                "HMM + soft regime VAR",
                dict(lag=lag, ridge=ridge, states=c.n_regimes),
                model.predict(x[:end], prior_val),
                model,
            )
    probabilities = hmm.filtered_probabilities(u[:, keep])
    prior = predictive_regime_probabilities(hmm, probabilities)
    regime = selected["HMM + soft regime VAR"][2]
    predictions["HMM + soft regime VAR"] = regime.predict(x, prior)
    log_components, regime_sigmas = [], []
    for k in range(c.n_regimes):
        pure = np.zeros_like(prior)
        pure[:, k] = 1.0
        pred = regime.predict(x, pure)
        residual = (x - pred)[regime.lag : nt]
        weights = probabilities[regime.lag : nt, k]
        weighted_residual = residual * np.sqrt(weights * len(weights) / weights.sum())[:, None]
        sigma = covariance(weighted_residual)
        regime_sigmas.append(sigma)
        with np.errstate(divide="ignore"):
            log_components.append(np.log(prior[:, k]) - gaussian_nll(x, pred, sigma))
    likelihoods["HMM + soft regime VAR"] = -logsumexp(log_components, axis=0)
    if dlc_model is not None:
        emit("Scoring supplied Train-fitted DLC-SSM checkpoint")
        predictions["DLC-SSM"], likelihoods["DLC-SSM"] = dlc_predictions(dlc_model, x, u)

    emit("Exporting DY total/to/from/net and TVP innovation-risk features")
    T, n = x.shape
    H = c.horizon
    fit_end = dates[nt - 1].strftime("%Y-%m-%d")
    features = pd.DataFrame(
        dict(
            date=dates.strftime("%Y-%m-%d"),
            information_date=dates.strftime("%Y-%m-%d"),
            parameter_fit_end=fit_end,
        )
    )
    for k in range(c.n_regimes - 1):
        features[f"baseline__hmm_probability_{k}"] = probabilities[:, k]
    theta, to, from_, net = (
        np.full((T, n, n), np.nan),
        np.full((T, n), np.nan),
        np.full((T, n), np.nan),
        np.full((T, n), np.nan),
    )
    total, risk, radius = (np.full(T, np.nan) for _ in range(3))
    sigma_raw = tvp.sigma_ * xs[:, None] * xs[None, :]
    for t in range(tvp.lag, T):
        a = lag_matrices(tvp_path.beta_filtered[t], tvp.lag)
        dy = dy_connectedness(a, tvp.sigma_, H)
        theta[t], to[t], from_[t], net[t], total[t] = (
            dy["gfevd"],
            dy["to"],
            dy["from_"],
            dy["net"],
            dy["total"],
        )
        risk[t] = innovation_risk(a * xs[None, :, None] / xs[None, None, :], sigma_raw, H)
        radius[t] = companion_radius(a)
    features[f"baseline__dy_connectedness_h{H}"] = total
    features[f"baseline__tvp_risk_h{H}"] = risk
    for name in ("Static VAR", "Sparse VAR"):
        model = selected[name][2]
        dy = dy_connectedness(model.a_, model.sigma_, H)
        features[f"diagnostic__{name.lower().replace(' ', '_')}_dy_h{H}"] = dy["total"]
    # DY uses the selected TVP operator path; it is a risk summary, not a sixth predictor model.
    static = selected["Static VAR"][2]
    static_dy = dy_connectedness(static.a_, static.sigma_, H)
    losses = pd.DataFrame(dict(date=features.date, split=split))
    rows = []
    common = np.arange(T) >= warmup
    for name, pred in predictions.items():
        if not np.isfinite(pred[common]).all() or not np.isfinite(likelihoods[name][common]).all():
            raise FloatingPointError(f"Nonfinite outputs for {name}; no silent row dropping")
        loss = np.mean((x - pred) ** 2, axis=1)
        losses[name] = loss
        losses[name + "__nll"] = likelihoods[name]
        for part in ("validation", "test"):
            mask = common & (split == part)
            rows.append(
                dict(
                    model=name,
                    split=part,
                    n=int(mask.sum()),
                    mse_standardized=float(loss[mask].mean()),
                    rmse_raw=float(np.sqrt(np.mean(((x - pred)[mask] * xs) ** 2))),
                    mean_log_predictive_density=float(-likelihoods[name][mask].mean()),
                )
            )
    comparisons = []
    reference = "DLC-SSM" if dlc_model is not None else "TVP-VAR"
    for name in predictions:
        if name == reference:
            continue
        diff = (losses[reference] - losses[name]).to_numpy()[common & (split == "test")]
        low, high = moving_block_bootstrap_ci(diff, c.block_length, c.bootstrap_draws, c.seed)
        comparisons.append(
            dict(
                comparison=f"{reference} minus {name}",
                mean_mse_difference=float(diff.mean()),
                ci95_low=low,
                ci95_high=high,
            )
        )
    report = dict(
        config=asdict(c),
        parameter_fit_end=fit_end,
        selection="Train nuisance/parameter fit; baseline Validation MSE selection; DLC checkpoint selected separately (see its metadata); Test scored after selection",
        selected={name: value[1] for name, value in selected.items()},
        prediction_information="Target row t uses observations through t-1; DLC future u=persistence",
        feature_information="Close-t TVP filtered states and HMM probabilities; no backward smoother",
        tvp="Full coefficient matrix random walk; fixed Train Sigma; Q=Sigma kron (q I)",
        smooth="One-sided local-linear ridge; rolling refit on past observations; frozen window/ridge",
        fixed_network="I and Train correlation adjacency per lag; random-walk state; frozen Train c/R",
        dy="Row-normalized generalized FEVD from TVP filtered coefficients; H includes Psi_0; percent",
        tvp_risk="Mean-sector raw-unit H-step innovation FEV; frozen coefficient path; return squared",
        density="Joint one-step Gaussian; HMM mixture; smooth Train plug-in covariance; DLC/TVP/network state uncertainty included",
        train_feature_caveat="Train paths use Train-fitted nuisance quantities and Validation-selected specifications; not prequential Train performance",
        hmm_diagnostics=hmm.diagnostics_,
        hmm_features=np.asarray(ds.covariate_names)[keep].tolist(),
        stability=dict(
            tvp_radius_ge_one=int(np.sum(radius >= 1)),
            tvp_max_radius=float(np.nanmax(radius)),
            static_radius=companion_radius(static.a_),
            policy="Finite-horizon diagnostics, no clipping or hidden Test-dependent rejection",
        ),
        reference_for_bootstrap=reference,
        dlc_included=dlc_model is not None,
        versions={
            name: version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "hmmlearn")
        },
        limitations=[
            "Not Primiceri TVP-SVAR or stochastic-volatility TVP; no exact named-paper replication claim.",
            "Smooth estimator is a causal forecasting adaptation; no structural TV-GIRF confidence bands.",
            "No joint MS-VAR, Regime-PCMCI, or BK frequency connectedness.",
            "This exports M0-M5 inputs; oos_experiment fits downside regressions after oos_final prepares designs.",
            "Bootstrap conditions on fitted/selected models; no re-estimation or multiple-testing correction.",
        ],
    )
    arrays = dict(
        dates=np.asarray(ds.dates).astype(str),
        split=split,
        sector_names=np.asarray(ds.sector_names).astype(str),
        sector_codes=np.asarray(ds.sector_codes).astype(str),
        x_mean=xm,
        x_scale=xs,
        u_mean=um,
        u_scale=us,
        actual_raw=raw,
        tvp_beta_filtered=tvp_path.beta_filtered,
        tvp_sigma=tvp.sigma_,
        tvp_variance_multiplier=tvp_path.variance_multiplier,
        dy_gfevd=theta,
        dy_to=to,
        dy_from=from_,
        dy_net=net,
        dy_total=total,
        tvp_radius=radius,
        tvp_risk=risk,
        static_dy_gfevd=static_dy["gfevd"],
        static_beta=static.beta_,
        static_sigma=static.sigma_,
        sparse_beta=selected["Sparse VAR"][2].beta_,
        sparse_sigma=selected["Sparse VAR"][2].sigma_,
        hmm_probabilities=probabilities,
        hmm_startprob=hmm.fit_.startprob,
        hmm_transmat=hmm.fit_.transmat,
        hmm_means=hmm.fit_.means,
        hmm_covars=hmm.fit_.covars,
        regime_coef=regime.coef_,
        regime_intercept=regime.intercept_,
        regime_sigma=np.asarray(regime_sigmas),
        smooth_sigma=smooth_sigma,
    )
    arrays.update(
        network_basis=network.basis_,
        network_intercept=network.intercept_,
        network_sigma=network.sigma_,
        network_operators=network_path.operators,
        network_states=network_path.filtered.mean,
        network_state_cov=network_path.filtered.cov,
    )
    for name, pred in predictions.items():
        arrays[name.replace(" ", "_") + "_prediction_raw"] = pred * xs + xm
    return dict(
        features=features,
        summary=pd.DataFrame(rows),
        candidates=pd.DataFrame(candidates),
        comparisons=pd.DataFrame(comparisons),
        losses=losses,
        report=report,
        arrays=arrays,
    )


def main():
    """
    Main.
    :return: Result produced by this operation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/dataset.npz")
    parser.add_argument("--config", default="config/baselines.json")
    parser.add_argument("--output", default="outputs/baseline_comparison")
    parser.add_argument(
        "--dlc-model", help="Optional Train-only checkpoint on the same prepared dataset"
    )
    parser.add_argument(
        "--model-fit-end", help="Required last fit date when --dlc-model is supplied"
    )
    args = parser.parse_args()
    c = BaselineConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    ds = load_prepared_dataset(args.dataset)
    model = None
    if bool(args.dlc_model) != bool(args.model_fit_end):
        parser.error("--dlc-model and --model-fit-end must be supplied together")
    if args.dlc_model:
        from model.dlcssm import DLCSSM

        dates = validate_dataset(ds)
        declared = pd.Timestamp(args.model_fit_end)
        if pd.isna(declared) or declared > dates[np.asarray(ds.split) == "train"][-1]:
            parser.error("DLC fit date must be on or before Train end")
        model = DLCSSM.load(args.dlc_model)
    result = run_experiment(ds, c, dlc_model=model, progress=lambda m: print(m, flush=True))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result["features"].to_csv(out / "baselines.csv", index=False)
    # Validate against the UNMODIFIED existing consumer before delivering results.
    load_baselines(
        out / "baselines.csv", ds, FinalOOSConfig(horizon=c.horizon, n_regimes=c.n_regimes)
    )
    for name in ("summary", "candidates", "comparisons", "losses"):
        result[name].to_csv(out / f"{name}.csv", index=False)
    result["report"]["dlc_fit_end_declaration"] = args.model_fit_end
    np.savez_compressed(out / "baseline_paths.npz", **result["arrays"])
    (out / "report.json").write_text(
        json.dumps(result["report"], indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(result["summary"].to_string(index=False))
    print(f"Saved baselines and comparison to {out}")


if __name__ == "__main__":
    main()
