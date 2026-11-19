import numpy as np
import pytest

from baselines.hmm_baseline import GaussianHMMBaseline, HMMFit, RegimeVARBaseline
from baselines.hmm_comparison import (
    dlc_next_day_predictions,
    predictive_regime_probabilities,
    run_comparison,
)
from core.data import PreparedDataset
from model.dlcssm import DLCSSM


def fixed_hmm():
    model = GaussianHMMBaseline(2)
    model.fit_ = HMMFit(
        np.array([0.6, 0.4]),
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([[-1.0], [1.0]]),
        np.ones((2, 1, 1)),
    )
    return model


def test_filter_chunk_and_future_invariance():
    hmm = fixed_hmm()
    v = np.random.default_rng(2).normal(size=(25, 1))
    full = hmm.filtered_probabilities(v)
    np.testing.assert_allclose(full.sum(1), 1)
    first = hmm.filtered_probabilities(v[:10])
    np.testing.assert_allclose(full[:10], first)
    np.testing.assert_allclose(full[10:], hmm.filtered_probabilities(v[10:], previous=first[-1]))
    assert hmm.filtered_probabilities(np.empty((0, 1))).shape == (0, 2)
    prior = predictive_regime_probabilities(hmm, full)
    np.testing.assert_allclose(prior[1:], full[:-1] @ hmm.fit_.transmat)


def test_regime_var_predicts_analytic_lag():
    model = RegimeVARBaseline(lag=1)
    model.coef_ = np.array([[[[2.0]]], [[[-1.0]]]])
    model.intercept_ = np.array([[1.0], [3.0]])
    x = np.arange(5.0)[:, None]
    probabilities = np.tile([0.25, 0.75], (5, 1))
    pred = model.predict(x, probabilities)
    np.testing.assert_allclose(pred[1:, 0], 2.5 - 0.25 * x[:-1, 0])
    changed = x.copy()
    changed[-1] = 1000
    np.testing.assert_allclose(model.predict(changed, probabilities), pred)
    with pytest.raises(ValueError):
        model.predict(x, probabilities * 2)


def dataset():
    rng = np.random.default_rng(13)
    u = rng.normal(size=(110, 2))
    x = rng.normal(scale=0.3, size=(110, 2))
    for t in range(1, len(x)):
        x[t] += 0.4 * x[t - 1] + 0.1 * u[t - 1]
    return PreparedDataset(
        x=x,
        u=u,
        dates=np.arange("2020-01-01", "2020-04-20", dtype="datetime64[D]").astype(str),
        sector_codes=np.array(["a", "b"]),
        sector_names=np.array(["a", "b"]),
        covariate_names=np.array(["return", "vol"]),
        split=np.array(["train"] * 65 + ["validation"] * 20 + ["test"] * 25),
        x_mean=np.zeros(2),
        x_scale=np.ones(2),
        u_mean=np.zeros(2),
        u_scale=np.ones(2),
    )


def test_dlc_forecast_does_not_use_target_day_information():
    ds = dataset()
    model = DLCSSM(lag=1, state_dim=1, max_iter=2, min_iter=1, prox_iterations=3)
    model.fit(ds.x[:65], ds.u[:65])
    pred = dlc_next_day_predictions(model, ds.x, ds.u)
    x, u = ds.x.copy(), ds.u.copy()
    x[90:] += 20
    u[90:] -= 20
    changed = dlc_next_day_predictions(model, x, u)
    np.testing.assert_allclose(pred[:91], changed[:91])


def test_actual_hmm_filter_matches_prefix_likelihood():
    from hmmlearn.hmm import GaussianHMM

    v = np.random.default_rng(3).normal(size=(80, 1))
    hmm = GaussianHMMBaseline(2, n_iter=30, n_init=2).fit(v)
    f = hmm.fit_
    ref = GaussianHMM(2, covariance_type="full")
    ref.startprob_, ref.transmat_, ref.means_, ref.covars_ = (
        f.startprob,
        f.transmat,
        f.means,
        f.covars,
    )
    np.testing.assert_allclose(hmm.train_loglik_, ref.score(v))
    for stop in (1, 15, 80):
        np.testing.assert_allclose(
            hmm.filtered_probabilities(v)[stop - 1], ref.predict_proba(v[:stop])[-1], atol=1e-10
        )


def test_comparison_and_test_data_cannot_select_parameters():
    from dataclasses import replace

    ds = dataset()
    opts = dict(
        states=(2,),
        lags=(1,),
        ridges=(0.1, 1.0),
        n_init=2,
        n_iter=25,
        dlc_options=dict(lag=1, state_dim=1, max_iter=2, min_iter=1, prox_iterations=3),
        bootstrap_draws=30,
        block_length=5,
    )
    result = run_comparison(ds, **opts)
    changed_x, changed_u = ds.x.copy(), ds.u.copy()
    changed_x[85:] *= 2
    changed_u[85:] += 1
    changed = run_comparison(replace(ds, x=changed_x, u=changed_u), **opts)
    assert len(result["summary"]) == 6
    assert set(result["summary"].query("split == 'test'").n) == {25}
    assert result["report"]["selected_hmm"] == changed["report"]["selected_hmm"]
    assert result["report"]["selected_static"] == changed["report"]["selected_static"]
    np.testing.assert_allclose(result["arrays"]["transmat"], changed["arrays"]["transmat"])
    np.testing.assert_allclose(result["dlc"].params_.A0, changed["dlc"].params_.A0)
    assert result["features"].filter(like="baseline__").shape == (110, 1)
