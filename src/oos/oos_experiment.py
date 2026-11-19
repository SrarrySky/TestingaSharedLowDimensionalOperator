"""Primary downside experiment: python -m oos.oos_experiment --help.

Consumes the existing strict M0--M5 exporter without changing model code.
All forecast coefficients are fitted on Train; Validation chooses ridge alpha.
Test labels are used only for scoring. See docs/METHODS.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from oos.oos_specs import (
    SPEC_VERSION,
    _positive_integer,
    build_default_oos_specs,
)


@dataclass(frozen=True)
class PredictiveConfig:
    alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)
    block_length: int = 20
    bootstrap_draws: int = 2000
    seed: int = 42
    confidence_level: float = 0.95
    hac_lags: int | None = None
    nonnegative_prediction: bool = True

    def __post_init__(self):
        """
          post init  .
        :return: Result produced by this operation.
        """
        if (
            not self.alphas
            or any(isinstance(a, bool) for a in self.alphas)
            or not np.isfinite(self.alphas).all()
            or min(self.alphas) <= 0
            or len(set(self.alphas)) != len(self.alphas)
        ):
            raise ValueError("alphas must contain unique positive finite numbers")
        for name in ("block_length", "bootstrap_draws"):
            _positive_integer(getattr(self, name), name)
        for name in ("seed", "hac_lags"):
            value = getattr(self, name)
            if value is None and name == "hac_lags":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not np.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if not isinstance(self.nonnegative_prediction, bool):
            raise ValueError("nonnegative_prediction must be boolean")


def _ratio(numerator, denominator):
    """
     ratio.
    :param numerator: Input argument used by this operation.
    :param denominator: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    return None if denominator <= 0 else float(1 - numerator / denominator)


def _dm(difference, lags):
    """Two-sided asymptotic DM, Bartlett HAC; no absolute variance floor.

    Positive difference means reference loss exceeds candidate loss. This is
    a descriptive comparison of selected forecasts, not a nested-model proof.
    """
    d = np.asarray(difference, dtype=float)
    centered = d - d.mean()
    n = len(d)
    variance = float(centered @ centered / n)
    for k in range(1, lags + 1):
        variance += 2 * (1 - k / (lags + 1)) * float(centered[k:] @ centered[:-k] / n)
    if variance <= 0:
        return None, None, "zero_long_run_variance"
    statistic = float(d.mean() / np.sqrt(variance / n))
    return statistic, float(2 * stats.norm.sf(abs(statistic))), "ok"


def _holm(pvalues):
    """Holm correction over the entire predeclared comparison family."""
    result = [None] * len(pvalues)
    order = sorted((i for i, p in enumerate(pvalues) if p is not None), key=lambda i: pvalues[i])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        result[i] = previous
    return result


def evaluate_designs(designs, y, masks, rows, dates, effective, *, horizon, config=None):
    """Fit/select/score the frozen primary experiment on an aligned export."""
    if config is None:
        config = PredictiveConfig()
    _positive_integer(horizon, "horizon")
    names = tuple(f"M{i}" for i in range(6))
    if set(designs) != set(names) or set(effective) != set(names):
        raise ValueError("Exactly M0--M5 are required")
    y, rows, dates = np.asarray(y, float), np.asarray(rows), np.asarray(dates)
    if y.ndim != 1 or rows.shape != y.shape or dates.shape != y.shape:
        raise ValueError("Misaligned target, row indices or dates")
    if rows.dtype.kind not in "iu" or np.any(np.diff(rows) <= 0):
        raise ValueError("Row indices must be strictly increasing integers")
    dates_index = pd.DatetimeIndex(pd.to_datetime(dates))
    if dates_index.hasnans or dates_index.has_duplicates or not dates_index.is_monotonic_increasing:
        raise ValueError("Dates must be strictly increasing and unique")
    if set(masks) != {"train", "validation", "test"}:
        raise ValueError("Require train/validation/test masks")
    for mask in masks.values():
        if np.asarray(mask).dtype != bool or np.asarray(mask).shape != y.shape:
            raise ValueError("Masks must be aligned boolean arrays")
    if np.any(sum(m.astype(int) for m in masks.values()) > 1):
        raise ValueError("Split masks overlap")
    for part in ("train", "validation", "test"):
        if masks[part].sum() < 2:
            raise ValueError(f"Insufficient {part} rows")
    for left, right in (("train", "validation"), ("validation", "test")):
        if rows[masks[left]][-1] + horizon >= rows[masks[right]][0]:
            raise ValueError("Chronology/horizon purge violated")
    common = masks["train"] | masks["validation"] | masks["test"]
    if not np.isfinite(y[common]).all() or np.any(y[common] < 0):
        raise ValueError("Primary downside targets must be finite and nonnegative")
    previous = ()
    for name in names:
        X = np.asarray(designs[name])
        features = tuple(effective[name])
        if len(set(features)) != len(features) or features[: len(previous)] != previous:
            raise ValueError("Effective feature specifications must remain nested")
        if X.shape != (len(y), len(features)) or not np.isfinite(X[common]).all():
            raise ValueError(f"Invalid {name} design")
        previous = features
    c = config
    test_rows = rows[masks["test"]]
    if np.any(np.diff(test_rows) != 1):
        raise ValueError(
            "Test must be consecutive trading rows for HAC/block bootstrap; do not compress gaps"
        )
    ntest = len(test_rows)
    lags = horizon - 1 if c.hac_lags is None else c.hac_lags
    if lags < horizon - 1 or lags >= ntest or ntest < 10:
        raise ValueError("Require H-1 <= HAC lags < Test size and at least 10 Test rows")
    if not horizon <= c.block_length < ntest:
        raise ValueError("Require H <= block_length < Test size")
    train, val, test = (masks[s] for s in ("train", "validation", "test"))
    candidates, selected, fitted, prediction = [], {}, {}, {}
    # No Test outcomes enter this fitting/selection loop.
    for name in names:
        X = designs[name]
        best = None
        for alpha in sorted(c.alphas):
            if X.shape[1]:
                model = Ridge(alpha=alpha, solver="svd").fit(X[train], y[train])
                coef, intercept = model.coef_, float(model.intercept_)
            else:
                coef, intercept = np.empty(0), float(y[train].mean())
            pred = X[val] @ coef + intercept
            if c.nonnegative_prediction:
                pred = np.maximum(pred, 0)
            mse = float(np.mean((y[val] - pred) ** 2))
            if not np.isfinite(mse):
                raise FloatingPointError(f"Nonfinite Validation loss in {name}")
            candidates.append(dict(model=name, alpha=float(alpha), validation_mse=mse))
            if best is None or mse < best[0]:
                best = mse, float(alpha), coef.copy(), intercept
        mse, alpha, coef, intercept = best
        selected[name] = dict(
            alpha=alpha, validation_mse=mse, n_features=len(coef), intercept_only=not len(coef)
        )
        fitted[name] = dict(coef=coef.tolist(), intercept=intercept, features=list(effective[name]))
        pred = designs[name][common] @ coef + intercept
        if c.nonnegative_prediction:
            pred = np.maximum(pred, 0)
        if not np.isfinite(pred).all():
            raise FloatingPointError(f"Nonfinite predictions in {name}")
        prediction[name] = np.full(len(y), np.nan)
        prediction[name][common] = pred
    mean = float(y[train].mean())
    records, metrics = [], []
    losses = {}
    for name in names:
        losses[name] = (y - prediction[name]) ** 2
        for part in ("validation", "test"):
            mask = masks[part]
            mse = float(losses[name][mask].mean())
            benchmark = float(np.mean((y[mask] - mean) ** 2))
            metrics.append(
                dict(
                    model=name,
                    split=part,
                    n=int(mask.sum()),
                    mse=mse,
                    rmse=float(np.sqrt(mse)),
                    mae=float(np.mean(np.abs(y[mask] - prediction[name][mask]))),
                    r2_oos_train_mean=_ratio(mse, benchmark),
                    r2_oos_m0=_ratio(mse, float(losses["M0"][mask].mean())),
                )
            )
            for i in np.flatnonzero(mask):
                records.append(
                    dict(
                        model=name,
                        split=part,
                        date=str(dates[i]),
                        row_index=int(rows[i]),
                        target_end_row=int(rows[i] + horizon),
                        actual=float(y[i]),
                        prediction=float(prediction[name][i]),
                        squared_loss=float(losses[name][i]),
                        train_mean_prediction=mean,
                    )
                )
    # Identical draws across all comparisons: paired moving blocks, not iid rows.
    rng = np.random.default_rng(c.seed)
    blocks = int(np.ceil(ntest / c.block_length))
    boot_indices = (
        rng.integers(0, ntest - c.block_length + 1, size=(c.bootstrap_draws, blocks))[:, :, None]
        + np.arange(c.block_length)
    ).reshape(c.bootstrap_draws, -1)[:, :ntest]
    pairs = [(f"M{i - 1}", f"M{i}") for i in range(1, 6)]
    pairs += [("M2", "M4"), ("M2", "M5"), ("M0", "M5")]
    comparisons = []
    q = (1 - c.confidence_level) / 2
    for reference, candidate in pairs:
        a, b = losses[reference][test], losses[candidate][test]
        d = a - b
        sampled_a, sampled_b = a[boot_indices].mean(1), b[boot_indices].mean(1)
        lower, upper = np.quantile(sampled_a - sampled_b, [q, 1 - q])
        relative = None
        if np.all(sampled_a > 0):
            relative = np.quantile(1 - sampled_b / sampled_a, [q, 1 - q]).tolist()
        statistic, pvalue, status = _dm(d, lags)
        comparisons.append(
            dict(
                reference=reference,
                candidate=candidate,
                mean_loss_improvement=float(d.mean()),
                ci_low=float(lower),
                ci_high=float(upper),
                relative_mse_improvement=_ratio(float(b.mean()), float(a.mean())),
                relative_ci_low=None if relative is None else relative[0],
                relative_ci_high=None if relative is None else relative[1],
                dm_stat=statistic,
                dm_p_two_sided=pvalue,
                dm_status=status,
                identical_predictions=bool(
                    np.array_equal(prediction[reference][test], prediction[candidate][test])
                ),
            )
        )
    for row, adjusted in zip(
        comparisons, _holm([r["dm_p_two_sided"] for r in comparisons]), strict=True
    ):
        row["dm_p_holm"] = adjusted
    return dict(
        metrics=pd.DataFrame(metrics),
        predictions=pd.DataFrame(records),
        candidates=pd.DataFrame(candidates),
        comparisons=pd.DataFrame(comparisons),
        fitted=fitted,
        selected=selected,
        hac_lags=lags,
        sample_counts={s: int(m.sum()) for s, m in masks.items()},
    )


def _validate_export_metadata(metadata):
    """
     validate export metadata.
    :param metadata: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if metadata["spec_version"] != SPEC_VERSION:
        raise ValueError("Only the current strict pathwise feature export is supported")
    feature_config = metadata["config"]
    expected_specs = {
        name: list(spec.feature_names)
        for name, spec in build_default_oos_specs(
            feature_config["horizon"],
            n_regimes=feature_config["n_regimes"],
            include_bk=feature_config["include_bk"],
        ).items()
    }
    if metadata.get("declared_specs") != expected_specs:
        raise ValueError("Feature export label is compatible but its M0--M5 contract differs")


def score_export(directory, output, config=None):
    """
    Score export.
    :param directory: Input argument used by this operation.
    :param output: Input argument used by this operation.
    :param config: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if config is None:
        config = PredictiveConfig()
    directory, output = Path(directory), Path(output)
    metadata = json.loads((directory / "specification.json").read_text(encoding="utf-8"))
    _validate_export_metadata(metadata)
    with np.load(directory / "designs.npz", allow_pickle=False) as z:
        result = evaluate_designs(
            {f"M{i}": z[f"M{i}"] for i in range(6)},
            z["y"],
            {s: z[f"mask_{s}"] for s in ("train", "validation", "test")},
            z["row_indices"],
            z["dates"],
            metadata["effective_specs"],
            horizon=metadata["config"]["horizon"],
            config=config,
        )
    output.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "predictions", "candidates", "comparisons"):
        result[name].to_csv(output / f"{name}.csv", index=False)
    report = dict(
        version="primary-downside-predictive-v1",
        config=asdict(config),
        target=metadata["target"],
        horizon=metadata["config"]["horizon"],
        selected=result["selected"],
        fitted=result["fitted"],
        sample_counts=result["sample_counts"],
        hac_lags=result["hac_lags"],
        feature_specification=metadata,
        protocol="Train-only coefficients and transforms; Validation MSE alpha selection; no refit; frozen Test forecasts",
        tie_break="smallest alpha; per-model selection",
        positive_improvement="reference squared loss minus candidate squared loss",
        benchmark="constant mean of purged common Train targets; R2=1-SSE_model/SSE_benchmark",
        bootstrap="paired overlapping moving blocks on consecutive Test rows; percentile CI; fixed fitted forecasts",
        limitations=[
            "Primary future downside semivariance / squared-loss experiment only. ES, drawdown, quantile regression and tail classification are separate experiments.",
            "DM is asymptotic and descriptive for nested, Validation-selected ridge forecasts; not a calibrated nested-model significance proof.",
            "Holm adjusts the eight declared DM comparisons. Bootstrap intervals are pointwise, not simultaneous.",
            "Bootstrap conditions on fitted parameters and selected specifications, without retraining.",
            "Baseline/model provenance is declared, not independently established. Validation-selected upstream features make Validation scores selection scores, not unbiased OOS evidence.",
            "A completed run does not establish incremental value; constant/dropped geometry can produce identical nested forecasts.",
        ],
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    """
    Main.
    :return: Result produced by this operation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-dir", help="Score an existing oos_final export; skips feature generation"
    )
    parser.add_argument("--dataset", default="data/processed/dataset.npz")
    parser.add_argument("--model")
    parser.add_argument("--model-fit-end")
    parser.add_argument(
        "--baselines",
        help="Existing date-keyed baselines.csv; otherwise generate using baseline_experiment",
    )
    parser.add_argument("--baseline-config", default="config/baselines.json")
    parser.add_argument("--feature-config", default="config/oos_final.json")
    parser.add_argument("--config", default="config/oos_predictive.json")
    parser.add_argument("--output", default="outputs/oos_downside")
    args = parser.parse_args()
    config = PredictiveConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error(
            "Output directory is not empty; use a new run directory to preserve prior results"
        )
    if args.design_dir:
        if args.model or args.model_fit_end or args.baselines:
            parser.error("--design-dir cannot be combined with model/baseline inputs")
        directory = Path(args.design_dir)
    else:
        if not args.model or not args.model_fit_end:
            parser.error("--model and --model-fit-end are required for feature generation")
        baselines = args.baselines
        if baselines is None:
            fc = json.loads(Path(args.feature_config).read_text(encoding="utf-8"))
            bc = json.loads(Path(args.baseline_config).read_text(encoding="utf-8"))
            if (
                fc.get("include_bk", False)
                or fc["horizon"] != bc["horizon"]
                or fc["n_regimes"] != bc["n_regimes"]
            ):
                parser.error("Automatic baselines require matching H/K and include_bk=false")
            print("Generating Train-fitted / Validation-selected baselines...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.baseline_experiment",
                    "--dataset",
                    args.dataset,
                    "--config",
                    args.baseline_config,
                    "--output",
                    str(out / "baselines"),
                ],
                check=True,
            )
            baselines = str(out / "baselines" / "baselines.csv")
        directory = out / "features"
        print("Building strictly filtered pathwise features and purged designs...", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "oos.oos_final",
                "--dataset",
                args.dataset,
                "--model",
                args.model,
                "--model-fit-end",
                args.model_fit_end,
                "--baselines",
                baselines,
                "--config",
                args.feature_config,
                "--output",
                str(directory),
            ],
            check=True,
        )
    print("Fitting Train, selecting on Validation, scoring Test...", flush=True)
    result = score_export(directory, out, config)
    print(result["metrics"].to_string(index=False))
    print(f"Saved primary downside experiment: {out}")


if __name__ == "__main__":
    main()
