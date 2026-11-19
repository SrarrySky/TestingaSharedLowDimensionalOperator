"""Numerical identities, target-date leakage guards, and existing OOS integration."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from baselines.connectedness import (
    companion_radius,
    dy_connectedness,
    generalized_fevd,
    impulse_responses,
    innovation_risk,
)
from baselines.var_baselines import TVPVAR, SmoothVAR, StaticVAR, design, gaussian_nll
from core.data import PreparedDataset
from experiments.baseline_experiment import BaselineConfig, run_experiment
from oos.oos_final import FinalOOSConfig, load_baselines


def sample(seed=17, rows=220, n=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, n))
    for t in range(1, rows):
        x[t] += 0.35 * x[t - 1]
    return x


def test_static_matches_independent_ols_and_predicts_lags_only():
    x = sample()
    model = StaticVAR(2, 0.0).fit(x[:120])
    X, Y = design(x[:120], 2)
    expected = np.linalg.lstsq(X, Y, rcond=None)[0]
    np.testing.assert_allclose(model.beta_, expected)
    target = 150
    manual = model.beta_[0].copy()
    for ell in range(2):
        manual += model.a_[ell] @ x[target - ell - 1]
    np.testing.assert_allclose(model.predict(x)[target], manual)
    changed = x.copy()
    changed[target:] += 100
    np.testing.assert_allclose(model.predict(x)[: target + 1], model.predict(changed)[: target + 1])


def test_lasso_is_actually_sparse_and_intercept_unpenalized():
    x = sample() + 4
    model = StaticVAR(1, 10.0, sparse=True).fit(x)
    assert np.count_nonzero(model.beta_[1:]) == 0
    np.testing.assert_allclose(model.beta_[0], x[1:].mean(0))


def test_separable_tvp_matches_full_vector_kalman():
    x = sample(rows=50, n=2)
    model = TVPVAR(2, 1e-3, 0.7).fit(x[:30])
    path = model.filter(x)
    X, Y = design(x, 2)
    k, n = X.shape[1], Y.shape[1]
    m = np.zeros(k * n)
    P = np.kron(model.sigma_, 0.7 * np.eye(k))
    Q = np.kron(model.sigma_, 1e-3 * np.eye(k))
    for t, (h, y) in enumerate(zip(X, Y, strict=True), 2):
        P += Q
        H = np.kron(np.eye(n), h[None, :])
        mu = H @ m
        S = H @ P @ H.T + model.sigma_
        K = np.linalg.solve(S, H @ P).T
        m += K @ (y - mu)
        P -= K @ H @ P
        np.testing.assert_allclose(mu, path.prediction[t], atol=2e-10)
        np.testing.assert_allclose(m.reshape(n, k).T, path.beta_filtered[t], atol=2e-10)
        np.testing.assert_allclose(S, model.sigma_ * path.variance_multiplier[t], atol=2e-10)


@pytest.mark.parametrize("model", [TVPVAR(2), SmoothVAR(2, 25)])
def test_dynamic_forecast_prefix_invariance(model):
    x = sample()
    target = 155
    changed = x.copy()
    changed[target:] *= -20
    if isinstance(model, TVPVAR):
        model.fit(x[:100])
        original, other = model.filter(x), model.filter(changed)
        np.testing.assert_allclose(
            original.prediction[: target + 1], other.prediction[: target + 1]
        )
        np.testing.assert_allclose(original.beta_filtered[:target], other.beta_filtered[:target])
    else:
        np.testing.assert_allclose(
            model.predict(x)[: target + 1], model.predict(changed)[: target + 1]
        )


def test_var2_impulses_match_companion_powers():
    rng = np.random.default_rng(1)
    a = rng.normal(scale=0.1, size=(2, 3, 3))
    c = np.block([[a[0], a[1]], [np.eye(3), np.zeros((3, 3))]])
    psi = impulse_responses(a, 7)
    for h in range(7):
        np.testing.assert_allclose(psi[h], np.linalg.matrix_power(c, h)[:3, :3], atol=1e-12)
    np.testing.assert_allclose(companion_radius(a), max(abs(np.linalg.eigvals(c))))


def test_dy_closed_form_direction_and_conservation():
    # Only source 0 -> target 1, at horizon 2.
    a = np.array([[[0.0, 0.0], [0.5, 0.0]]])
    dy = dy_connectedness(a, np.eye(2), 2)
    np.testing.assert_allclose(dy["gfevd"], [[1.0, 0.0], [0.2, 0.8]])
    assert dy["total"] == pytest.approx(10.0)
    np.testing.assert_allclose(dy["to"], [20.0, 0.0])
    np.testing.assert_allclose(dy["from_"], [0.0, 20.0])
    np.testing.assert_allclose(dy["net"], [20.0, -20.0])
    assert dy["net"].sum() == pytest.approx(0.0)
    assert dy_connectedness(a, np.eye(2), 1)["total"] == pytest.approx(0.0)
    assert innovation_risk(a, np.eye(2), 2) == pytest.approx(1.125)


def test_gfevd_correlated_shocks_scaling_and_permutation():
    a = np.array([[[0.3, 0.1], [0.4, 0.2]]])
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    raw, theta = generalized_fevd(a, sigma, 4)
    assert not np.allclose(raw.sum(1), 1.0)
    np.testing.assert_allclose(theta.sum(1), 1.0)
    scaled_a = a * np.array([3.0, 0.2])[None, :, None] / np.array([3.0, 0.2])[None, None, :]
    scaled_s = sigma * np.array([3.0, 0.2])[:, None] * np.array([3.0, 0.2])[None, :]
    np.testing.assert_allclose(generalized_fevd(scaled_a, scaled_s, 4)[1], theta)
    np.testing.assert_allclose(
        generalized_fevd(a[:, ::-1, ::-1], sigma[::-1, ::-1], 4)[1], theta[::-1, ::-1]
    )


def test_density_matches_scipy():
    from scipy.stats import multivariate_normal

    x = sample(rows=5, n=2)
    S = np.array([[1.0, 0.3], [0.3, 2.0]])
    np.testing.assert_allclose(
        gaussian_nll(x, np.zeros_like(x), S, np.full(5, 1.7)),
        -multivariate_normal.logpdf(x, cov=1.7 * S),
    )


def small_dataset():
    raw = sample()
    u = sample(seed=8, n=3)
    nt = 120
    return PreparedDataset(
        x=(raw - raw[:nt].mean(0)) / raw[:nt].std(0),
        u=(u - u[:nt].mean(0)) / u[:nt].std(0),
        dates=pd.date_range("2020-01-01", periods=len(raw)).strftime("%Y-%m-%d").to_numpy(),
        sector_codes=np.array(["a", "b", "c"]),
        sector_names=np.array(["A", "B", "C"]),
        covariate_names=np.array(["market_return", "realized_volatility", "turnover_proxy"]),
        split=np.array(["train"] * 120 + ["validation"] * 50 + ["test"] * 50),
        x_mean=raw[:nt].mean(0),
        x_scale=raw[:nt].std(0),
        u_mean=u[:nt].mean(0),
        u_scale=u[:nt].std(0),
    )


def small_config():
    return BaselineConfig(
        lags=(1, 2),
        ridges=(0.1,),
        lasso_alphas=(0.03,),
        process_variances=(1e-4, 1e-3),
        smooth_bandwidths=(20, 30),
        n_regimes=2,
        hmm_n_init=1,
        hmm_n_iter=20,
        bootstrap_draws=20,
        block_length=5,
    )


def test_experiment_consumer_and_test_perturbation(tmp_path):
    pytest.importorskip("hmmlearn")
    ds, cfg = small_dataset(), small_config()
    result = run_experiment(ds, cfg)
    path = tmp_path / "baselines.csv"
    result["features"].to_csv(path, index=False)
    values = load_baselines(path, ds, FinalOOSConfig(n_regimes=2))
    assert np.isfinite(values["baseline__dy_connectedness_h5"][120:]).all()
    assert len(result["summary"]) == 12
    assert result["summary"].n.eq(50).all()
    changed = ds.x.copy()
    changed[185:] += 30
    changed_u = ds.u.copy()
    changed_u[185:] -= 10
    other = run_experiment(replace(ds, x=changed, u=changed_u), cfg)
    assert result["report"]["selected"] == other["report"]["selected"]
    np.testing.assert_allclose(
        result["features"].iloc[:185, 3:].to_numpy(dtype=float),
        other["features"].iloc[:185, 3:].to_numpy(dtype=float),
    )
    for key in result["arrays"]:
        if key.endswith("_prediction_raw"):
            np.testing.assert_allclose(result["arrays"][key][:186], other["arrays"][key][:186])


@pytest.mark.parametrize(
    "call",
    [
        lambda: BaselineConfig(lags=()),
        lambda: BaselineConfig(process_variances=(0.0,)),
        lambda: TVPVAR(p0=0),
        lambda: impulse_responses(np.zeros((1, 2, 2)), 0),
        lambda: generalized_fevd(np.zeros((1, 2, 2)), np.zeros((2, 2))),
    ],
)
def test_invalid_inputs_fail(call):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        call()
