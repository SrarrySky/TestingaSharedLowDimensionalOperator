"""Train-only HMM + soft regime VAR / static VAR / DLC-SSM comparison.

Run: python -m baselines.hmm_comparison --help
This two-stage baseline is NOT a jointly estimated Markov-switching VAR.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.hmm_baseline import GaussianHMMBaseline, RegimeVARBaseline
from core.config import load_yaml
from core.data import load_prepared_dataset
from core.evaluation import moving_block_bootstrap_ci
from model.dlcssm import DLCSSM
from model.forecasting import one_step_forecasts
from oos.oos_final import validate_dataset


def predictive_regime_probabilities(hmm, filtered):
    """Row t conditions on observations through t-1 only."""
    out = np.empty_like(filtered)
    out[0] = hmm.fit_.startprob
    out[1:] = filtered[:-1] @ hmm.fit_.transmat
    return out


def dlc_next_day_predictions(model, x, u):
    """Origin t-1 filtering; forecast u_t as u_{t-1} for BOTH equations."""
    return one_step_forecasts(model, x, u).mean


def run_comparison(
    ds,
    *,
    states=(2, 3, 4),
    lags=(1, 3),
    ridges=(0.01, 1.0),
    seed=42,
    n_init=5,
    n_iter=200,
    dlc_options=None,
    bootstrap_draws=1000,
    block_length=20,
):
    """
    Run comparison.
    :param ds: Input argument used by this operation.
    :param states: Input argument used by this operation.
    :param lags: Input argument used by this operation.
    :param ridges: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :param n_init: Input argument used by this operation.
    :param n_iter: Input argument used by this operation.
    :param dlc_options: Input argument used by this operation.
    :param bootstrap_draws: Input argument used by this operation.
    :param block_length: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    dates = validate_dataset(ds)
    if (
        not states
        or not lags
        or not ridges
        or min(states) < 2
        or min(lags) < 1
        or not np.isfinite(ridges).all()
        or min(ridges) <= 0
    ):
        raise ValueError("Nonempty positive grids required; HMM K >= 2")
    options = dict(dlc_options or {})
    options.setdefault("seed", seed)
    options.setdefault("lag", 3)
    warmup = max(max(lags), options["lag"] + 1)
    train = np.asarray(ds.split) == "train"
    validation = np.asarray(ds.split) == "validation"
    test = np.asarray(ds.split) == "test"
    ntrain = int(train.sum())
    if ntrain <= warmup + max(states) or validation.sum() < 2 or test.sum() < 10:
        raise ValueError("Insufficient chronological train/validation/test rows")
    if bootstrap_draws < 1 or not 1 <= block_length <= test.sum():
        raise ValueError("Invalid bootstrap draws or block length")
    # Guard against a differently standardized input: recover raw values and
    # independently fit scales using Train only, identically for all models.
    raw_x = ds.x * ds.x_scale + ds.x_mean
    raw_u = ds.u * ds.u_scale + ds.u_mean
    xm, xs = raw_x[train].mean(0), raw_x[train].std(0)
    um, us = raw_u[train].mean(0), raw_u[train].std(0)
    xs = np.where(xs > 1e-12, xs, 1.0)
    keep = us > 1e-12
    if not keep.any():
        raise ValueError("No nonconstant Train covariates available for HMM")
    us = np.where(us > 1e-12, us, 1.0)
    x, u = (raw_x - xm) / xs, (raw_u - um) / us
    v = u[:, keep]
    # Only Train+Validation is touched while choosing specifications.
    end = ntrain + int(validation.sum())
    candidates, diagnostics = [], {}
    best_hmm, best_static = None, None

    def val_loss(pred):
        """
        Val loss.
        :param pred: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        value = float(np.mean((x[:end][validation[:end]] - pred[validation[:end]]) ** 2))
        if not np.isfinite(value):
            raise ValueError("Nonfinite validation loss")
        return value

    for lag in lags:
        for ridge in ridges:
            model = RegimeVARBaseline(lag, ridge).fit(x[train], np.ones((ntrain, 1)))
            loss = val_loss(model.predict(x[:end], np.ones((end, 1))))
            record = dict(model="Static VAR", lag=lag, ridge=ridge, validation_mse=loss)
            candidates.append(record)
            if best_static is None or loss < best_static[0]:
                best_static = (loss, model, record)
    for k in states:
        hmm = GaussianHMMBaseline(k, seed, n_iter, n_init=n_init).fit(v[train])
        diagnostics[str(k)] = hmm.diagnostics_
        filtered = hmm.filtered_probabilities(v[:end])
        predictive = predictive_regime_probabilities(hmm, filtered)
        for lag in lags:
            for ridge in ridges:
                # Soft Train responsibilities may use v_t to estimate state-t
                # operators; forecasting must instead use state-t PRIOR weights.
                model = RegimeVARBaseline(lag, ridge).fit(x[train], filtered[:ntrain])
                loss = val_loss(model.predict(x[:end], predictive))
                record = dict(
                    model="HMM + soft regime VAR",
                    states=k,
                    lag=lag,
                    ridge=ridge,
                    validation_mse=loss,
                )
                candidates.append(record)
                if best_hmm is None or loss < best_hmm[0]:
                    best_hmm = (loss, hmm, model, record)
    # DLC hyperparameters are fixed by the supplied config, never Test-selected.
    dlc = DLCSSM(**options)
    dlc.fit(x[train], u[train])
    hmm, regime = best_hmm[1:3]
    probabilities = hmm.filtered_probabilities(v)
    predictions = {
        "Static VAR": best_static[1].predict(x, np.ones((len(x), 1))),
        "HMM + soft regime VAR": regime.predict(
            x, predictive_regime_probabilities(hmm, probabilities)
        ),
        "DLC-SSM": dlc_next_day_predictions(dlc, x, u),
    }
    rows, loss_columns = [], {}
    common = np.arange(len(x)) >= warmup
    for name, pred in predictions.items():
        if not np.isfinite(pred[common]).all():
            raise RuntimeError(f"Nonfinite predictions for {name}; refusing partial comparison")
        loss = np.mean((x - pred) ** 2, axis=1)
        loss_columns[name] = loss
        for split in ("validation", "test"):
            mask = common & (np.asarray(ds.split) == split)
            rows.append(
                dict(
                    model=name,
                    split=split,
                    n=int(mask.sum()),
                    mse_standardized=float(loss[mask].mean()),
                    rmse_raw=float(np.sqrt(np.mean(((x[mask] - pred[mask]) * xs) ** 2))),
                )
            )
    comparisons = []
    for other in ("Static VAR", "HMM + soft regime VAR"):
        diff = (loss_columns["DLC-SSM"] - loss_columns[other])[test & common]
        low, high = moving_block_bootstrap_ci(diff, block_length, bootstrap_draws, seed)
        comparisons.append(
            dict(
                comparison=f"DLC-SSM minus {other}",
                mean_mse_difference=float(diff.mean()),
                ci95_low=low,
                ci95_high=high,
                block_length=block_length,
                bootstrap_draws=bootstrap_draws,
            )
        )
    features = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "information_date": dates.strftime("%Y-%m-%d"),
            "parameter_fit_end": dates[ntrain - 1].strftime("%Y-%m-%d"),
        }
    )
    for k in range(hmm.n_states - 1):
        features[f"baseline__hmm_probability_{k}"] = probabilities[:, k]
    losses = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "split": ds.split, **loss_columns})
    report = dict(
        protocol="Train fit; Validation MSE selection; frozen Test evaluation",
        target="next-day sector returns; information through prior close; future u=persistence",
        primary_metric="mean daily mean-sector squared error in Train-standardized units",
        parameter_fit_end=str(dates[ntrain - 1].date()),
        selected_hmm=best_hmm[3],
        selected_static=best_static[2],
        dlc_options=options,
        hmm_features=np.asarray(ds.covariate_names)[keep].tolist(),
        versions={name: version(name) for name in ("numpy", "scipy", "hmmlearn", "scikit-learn")},
        seed=seed,
        n_init=n_init,
        n_iter=n_iter,
        hmm_restarts=diagnostics,
        dlc_termination=dlc.fit_result_.termination_reason,
        limitations=[
            "Two-stage soft regime VAR, not joint MS-VAR or Regime-PCMCI.",
            "Bootstrap intervals condition on fitted models; no parameter re-estimation.",
            "Does not test within-regime operator variation against posterior/bootstrap noise.",
            "HMM feature CSV alone does not provide DY/TVP/BK required by oos_final.",
        ],
    )
    arrays = dict(
        probabilities=probabilities,
        x_mean=xm,
        x_scale=xs,
        u_mean=um,
        u_scale=us,
        hmm_feature_mask=keep,
        startprob=hmm.fit_.startprob,
        transmat=hmm.fit_.transmat,
        means=hmm.fit_.means,
        covars=hmm.fit_.covars,
        regime_coef=regime.coef_,
        regime_intercept=regime.intercept_,
        static_coef=best_static[1].coef_,
        static_intercept=best_static[1].intercept_,
        dates=np.asarray(ds.dates).astype(str),
        actual_raw=raw_x,
    )
    for name, pred in predictions.items():
        arrays[name.replace(" ", "_") + "_prediction_raw"] = pred * xs + xm
    return dict(
        summary=pd.DataFrame(rows),
        candidates=pd.DataFrame(candidates),
        comparisons=pd.DataFrame(comparisons),
        features=features,
        losses=losses,
        report=report,
        arrays=arrays,
        dlc=dlc,
    )


def main():
    """
    Main.
    :return: Result produced by this operation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/dataset.npz")
    parser.add_argument("--dlc-config", default="config/dlcssm.yaml")
    parser.add_argument("--output", default="outputs/hmm_comparison")
    parser.add_argument("--states", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--ridges", nargs="+", type=float, default=[0.01, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=20)
    args = parser.parse_args()
    cfg = load_yaml(args.dlc_config)
    fit, penalty = cfg["fit"], cfg["penalty"]
    options = dict(
        lag=cfg["lag"],
        state_dim=cfg["state_dimension"],
        seed=cfg["seed"],
        max_iter=fit["max_em_iterations"],
        min_iter=fit["min_em_iterations"],
        tol=fit["tolerance"],
        prox_iterations=fit["prox_iterations"],
        jitter=fit["covariance_jitter"],
        diagonal_R=fit["diagonal_observation_covariance"],
        canonicalize=fit["canonicalize_after_fit"],
        lambda_group=penalty["group"],
        lambda_B=penalty["state_l1"],
        lambda_ridge=penalty["ridge"],
    )
    print(
        "Fitting baselines and DLC-SSM on Train; DLC-SSM uses the supplied training budget.",
        flush=True,
    )
    result = run_comparison(
        load_prepared_dataset(args.dataset),
        states=args.states,
        lags=args.lags,
        ridges=args.ridges,
        seed=args.seed,
        n_init=args.n_init,
        n_iter=args.n_iter,
        dlc_options=options,
        bootstrap_draws=args.bootstrap_draws,
        block_length=args.block_length,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("summary", "candidates", "comparisons", "features", "losses"):
        result[name].to_csv(out / f"{name}.csv", index=False)
    (out / "report.json").write_text(
        json.dumps(result["report"], indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    np.savez_compressed(out / "hmm_predictions.npz", **result["arrays"])
    result["dlc"].save(out / "dlcssm_train.npz", metadata=result["report"])
    print(result["summary"].to_string(index=False))
    print(f"Saved comparison to {out}")


if __name__ == "__main__":
    main()
