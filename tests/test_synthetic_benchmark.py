import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

import experiments.synthetic_benchmark as sb
from baselines.var_baselines import SmoothVAR


def config():
    path = Path(__file__).resolve().parents[1] / "config/synthetic_smoke.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.update(scenarios=["s0"], seeds=[7], T=110, n=3, train_count=65, validation_count=20)
    cfg["baselines"].update(
        lags=[2],
        ridges=[0.1],
        lasso_alphas=[0.03],
        process_variances=[0.001],
        smooth_bandwidths=[15],
        hmm_n_init=1,
        hmm_n_iter=15,
    )
    cfg["dlc"]["hyperparameter_grid"] = {"state_dimension": [1]}
    cfg["dlc"]["fit"].update(max_em_iterations=2, prox_iterations=5)
    cfg["support"]["bootstrap_count"] = 2
    return cfg


def test_case_expansion_and_s6_high_truth():
    cfg = config()
    assert len(sb.scenario_variants("s7")) == 7
    assert sb.scenario_variants("s6") == ("low", "high")
    low = sb.generate_case(cfg, "s6", "low", 7)
    high = sb.generate_case(cfg, "s6", "high", 7)
    assert_allclose(high.x, low.metadata["high_x"])
    assert_allclose(high.operators[0], low.metadata["high_lags"])
    assert not np.allclose(low.operators, high.operators)
    # S6 has no covariates; it must not receive fabricated ones for the HMM.
    assert sb.prepare_case(high, cfg).u.shape == (cfg["T"], 0)


def test_train_scaling_not_changed_by_future():
    cfg = config()
    truth = sb.generate_case(cfg, "s0", "base", 7)
    ds = sb.prepare_case(truth, cfg)
    x, u = truth.x.copy(), truth.u.copy()
    x[65:] += 100
    u[65:] -= 100
    changed = sb.prepare_case(replace(truth, x=x, u=u), cfg)
    for field in ("x_mean", "x_scale", "u_mean", "u_scale"):
        assert_allclose(getattr(ds, field), getattr(changed, field))
    assert_allclose(ds.x[:65], changed.x[:65])


@pytest.mark.parametrize("method", list(sb.METHODS))
def test_method_adapters_use_prior_operators_and_ignore_future(tmp_path, method):
    cfg = {**config(), "run_seed": 7}
    truth = sb.generate_case(cfg, "s0", "base", 7)
    ds = sb.prepare_case(truth, cfg)
    model, sigma = (
        sb.fit_dlc(ds, cfg, tmp_path)
        if method == "dlc"
        else sb.fit_baseline(method, ds, cfg, tmp_path)
    )
    original = sb.predict(model, ds.x, ds.u, sigma)
    x, u = ds.x.copy(), ds.u.copy()
    x[90:] += 3
    u[90:] -= 3
    changed = sb.predict(model, x, u, sigma)
    assert_allclose(original.mean[:91], changed.mean[:91])
    assert_allclose(original.operators[:91], changed.operators[:91])
    assert_allclose(original.nll[:90], changed.nll[:90])
    prefix = sb.predict(model, ds.x[:90], ds.u[:90], sigma)
    assert_allclose(prefix.mean, original.mean[:90])
    assert_allclose(prefix.operators, original.operators[:90])
    if method == "static":
        raw_cov = model.sigma_ * ds.x_scale[:, None] * ds.x_scale[None, :]
        raw_mean = original.mean[90] * ds.x_scale + ds.x_mean
        assert original.nll[90] + np.log(ds.x_scale).sum() == pytest.approx(
            -multivariate_normal.logpdf(truth.x[90], raw_mean, raw_cov)
        )


@pytest.mark.parametrize("method", ["static", "hmm", "dlc"])
def test_selection_does_not_use_test_data(tmp_path, method):
    cfg = {**config(), "run_seed": 7}
    truth = sb.generate_case(cfg, "s0", "base", 7)
    ds = sb.prepare_case(truth, cfg)
    x, u = ds.x.copy(), ds.u.copy()
    x[85:] += 100
    u[85:] -= 100
    for name, data in (("original", ds), ("changed", replace(ds, x=x, u=u))):
        out = tmp_path / name
        out.mkdir()
        if method == "dlc":
            sb.fit_dlc(data, cfg, out)
        else:
            sb.fit_baseline(method, data, cfg, out)
    filename = "validation_trials.json" if method == "dlc" else "selected.json"
    a = json.loads((tmp_path / "original" / filename).read_text())
    b = json.loads((tmp_path / "changed" / filename).read_text())
    assert a == b


def test_recovery_zero_padding_and_permissive_support():
    true = np.repeat(np.array([[[0.3, 0.0], [0.2, 0.1]]])[None], 20, axis=0)
    estimate = np.pad(true, ((0, 0), (0, 1), (0, 0), (0, 0)))
    support = np.any(abs(estimate) > 1e-12, axis=0)
    metrics = sb.recovery_metrics(true, estimate, support, 4)
    assert metrics["edge_f1"] == pytest.approx(1)
    assert metrics["dynamic_coefficient_rmse"] == 0
    assert metrics["frozen_kernel_rmse"] == 0
    assert metrics["observable_frozen_peak_gain_rmse"] == 0
    assert metrics["transmitter_rank_spearman"] == pytest.approx(1)


def test_smooth_export_matches_prediction_equation():
    x = np.random.default_rng(2).normal(size=(80, 3))
    model = SmoothVAR(lag=2, bandwidth=15)
    path = model.forecast_path(x)
    assert_allclose(model.predict(x), path.prediction)
    # Local-linear fit is independently tested against an explicit weighted solve.
    from baselines.var_baselines import design, lag_matrices

    X, Y = design(x, 2)
    t, k = 40, X.shape[1]
    d = np.arange(-15, 0) / 16
    w = 0.75 * (1 - d * d)
    xx, yy = X[t - 2 - 15 : t - 2], Y[t - 2 - 15 : t - 2]
    z = np.column_stack([xx, xx * d[:, None]])
    penalty = np.eye(2 * k)
    penalty[0, 0] = 0
    b = np.linalg.solve(z.T @ (w[:, None] * z) + penalty, z.T @ (w[:, None] * yy))
    assert_allclose(path.operators[t], lag_matrices(b[:k], 2))
    assert_allclose(path.prediction[t], X[t - 2] @ b[:k])


def test_runner_preserves_failures_and_checksums(tmp_path, monkeypatch):
    cfg = config()
    cfg["methods"] = ["static", "network"]
    original = sb.fit_baseline

    def fail_network(method, *args):
        if method == "network":
            raise RuntimeError("deliberate failure")
        return original(method, *args)

    monkeypatch.setattr(sb, "fit_baseline", fail_network)
    output = tmp_path / "run"
    result = sb.run_benchmark(cfg, output, progress=lambda _: None)
    assert result["attempted_cells"] == 2
    assert result["successful_cells"] == 1
    assert result["status"] == "completed_with_failures"
    assert not result["blueprint_complete"]
    records = json.loads((output / "ledger.json").read_text())
    assert records[1]["error"] == "deliberate failure"
    assert records[1]["status"] == "failed_method"
    assert len(pd.read_csv(output / "metrics.csv")) == 2
    with pytest.raises(FileExistsError):
        sb.run_benchmark(cfg, output)


def test_missing_observations_are_not_imputed(tmp_path):
    cfg = config()
    cfg.update(scenarios=["s7"], methods=["static"])
    result = sb.run_benchmark(cfg, tmp_path / "stress", progress=lambda _: None)
    assert result["attempted_cells"] == 7
    records = json.loads((tmp_path / "stress/ledger.json").read_text())
    missing = next(r for r in records if r["variant"] == "missing_async")
    assert missing["status"] == "unsupported_missing"
    assert result["successful_cells"] == 6
    manifest = json.loads((tmp_path / "stress/manifest.json").read_text())
    assert manifest["attempted_cells"] == 7


def test_safe_npz_roundtrip(tmp_path):
    cfg = config()
    cfg["methods"] = ["static"]
    output = tmp_path / "run"
    sb.run_benchmark(cfg, output, progress=lambda _: None)
    for path in output.rglob("*.npz"):
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                assert not z[key].dtype.hasobject, (path, key)


@pytest.mark.parametrize(
    "key,value",
    [("methods", ["Regime-PCMCI"]), ("seeds", [True]), ("scenarios", ["s9"]), ("train_count", 2)],
)
def test_invalid_config_does_not_create_output(tmp_path, key, value):
    cfg = copy.deepcopy(config())
    cfg[key] = value
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        sb.run_benchmark(cfg, out)
    assert not out.exists()


def test_build_seed_summary_uses_test_split_and_counts_attempted():
    cfg = {"scenarios": ["s5"], "methods": ["static", "sparse"]}
    records = [
        {"scenario": "s5", "variant": "base", "method": "static", "status": "success"},
        {"scenario": "s5", "variant": "base", "method": "static", "status": "success"},
        {"scenario": "s5", "variant": "base", "method": "static", "status": "failed_method"},
        {"scenario": "s5", "variant": "base", "method": "sparse", "status": "success"},
    ]
    scores = [
        {
            "scenario": "s5",
            "variant": "base",
            "method": "static",
            "split": "test",
            "mse_standardized": 1.0,
        },
        {
            "scenario": "s5",
            "variant": "base",
            "method": "static",
            "split": "validation",
            "mse_standardized": 99.0,
        },
        {
            "scenario": "s5",
            "variant": "base",
            "method": "static",
            "split": "test",
            "mse_standardized": 3.0,
        },
        {
            "scenario": "s5",
            "variant": "base",
            "method": "static",
            "split": "validation",
            "mse_standardized": 98.0,
        },
        {
            "scenario": "s5",
            "variant": "base",
            "method": "sparse",
            "split": "test",
            "mse_standardized": 5.0,
        },
        {
            "scenario": "s5",
            "variant": "base",
            "method": "sparse",
            "split": "validation",
            "mse_standardized": 97.0,
        },
    ]
    summary = sb.build_seed_summary(cfg, records, scores)
    static_mse = summary[
        (summary["method"] == "static") & (summary["metric"] == "mse_standardized")
    ]
    assert len(static_mse) == 1
    row = static_mse.iloc[0]
    assert row["n_attempted"] == 3
    assert row["n_success"] == 2
    assert row["n_finite"] == 2
    assert row["mean"] == pytest.approx(2.0)
    assert row["std"] == pytest.approx(float(np.std([1.0, 3.0], ddof=1)))
    # Validation rows must not leak into the Test-only mean.
    sparse_mse = summary[
        (summary["method"] == "sparse") & (summary["metric"] == "mse_standardized")
    ]
    assert sparse_mse.iloc[0]["mean"] == pytest.approx(5.0)
    assert sparse_mse.iloc[0]["std"] == pytest.approx(0.0)


def test_reuse_rejects_result_affecting_config_differences():
    base = {
        "purpose": "benchmark",
        "scenarios": ["s5"],
        "seeds": [7, 19],
        "T": 1200,
        "n": 8,
        "lag": 2,
        "train_count": 720,
        "validation_count": 240,
        "methods": ["static"],
        "baselines": {"lags": [1]},
        "dlc": {"seed": 42},
        "support": {"seed": 42, "parallel": {"n_jobs": 2}},
    }
    # Cell-grid and scheduling-only differences must be tolerated.
    compatible = copy.deepcopy(base)
    compatible["seeds"] = [7, 19, 31]
    compatible["scenarios"] = ["s0", "s5"]
    compatible["methods"] = ["static", "sparse"]
    compatible["support"]["parallel"] = {"n_jobs": 8}
    sb._reuse_compatible(base, compatible)

    incompatible = copy.deepcopy(base)
    incompatible["T"] = 999
    with pytest.raises(ValueError, match="differs"):
        sb._reuse_compatible(base, incompatible)


def test_run_benchmark_reuse_from_scopes_metrics_and_matches_records(tmp_path):
    cfg = config()
    cfg["methods"] = ["static", "sparse"]
    first = tmp_path / "first"
    sb.run_benchmark(cfg, first, progress=lambda _: None)

    second_cfg = copy.deepcopy(cfg)
    second_cfg["seeds"] = [7, 19]
    second = tmp_path / "second"
    sb.run_benchmark(second_cfg, second, progress=lambda _: None, reuse_from=first)

    ledger = json.loads((second / "ledger.json").read_text(encoding="utf-8"))
    assert len(ledger) == 4  # 2 seeds x 2 methods
    assert sum(1 for r in ledger if r.get("reused")) == 2  # seed 7 cells only
    # Every cell's record.json must match its ledger entry, including the reused flag.
    for record in ledger:
        record_path = second / record["relative_path"] / "record.json"
        assert json.loads(record_path.read_text(encoding="utf-8")) == record
    # metrics.csv must stay scoped to the declared grid, not inherit other cells.
    metrics = pd.read_csv(second / "metrics.csv")
    assert len(metrics) == 8  # 2 seeds x 2 methods x 2 splits
    assert set(metrics["seed"]) == {7, 19}
    assert set(metrics["method"]) == {"static", "sparse"}
    summary = pd.read_csv(second / "seed_summary.csv")
    static_mse = summary[
        (summary["method"] == "static") & (summary["metric"] == "mse_standardized")
    ]
    assert static_mse.iloc[0]["n_attempted"] == 2
    assert static_mse.iloc[0]["n_success"] == 2
    assert static_mse.iloc[0]["n_finite"] == 2
