"""Independent one-step density and information-set regression checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from baselines.hmm_comparison import dlc_next_day_predictions
from experiments.baseline_experiment import dlc_predictions
from experiments.synthetic import generate_shared_low_rank
from experiments.validation import score_validation
from model.dlcssm import DLCSSM
from model.forecasting import one_step_forecasts


def fixture():
    ds = generate_shared_low_rank(T=70, n=3, lag=2, d=2, seed=473)
    model = DLCSSM(lag=2, state_dim=2)
    model.params_ = ds.params
    return model, ds.x, ds.u


def test_forecast_matches_independent_prefix_distribution_and_density():
    model, x, u = fixture()
    p = model.params_
    forecast = one_step_forecasts(model, x, u)
    assert np.isnan(forecast.mean[: p.lag + 1]).all()
    for t in (30, 41, 69):
        prefix = model.filter(x[:t], u[:t])
        state = p.F @ prefix.mean[-1] + p.G @ u[t - 1]
        state_cov = p.F @ prefix.cov[-1] @ p.F.T + p.Q
        offset = p.c + p.D @ u[t - 1]
        loading = p.Lambda.copy()
        for ell in range(p.lag):
            offset += p.A0[ell] @ x[t - ell - 1]
            for d in range(p.state_dim):
                loading[:, d] += p.B[d, ell] @ x[t - ell - 1]
        mean = offset + loading @ state
        cov = p.R + loading @ state_cov @ loading.T
        assert_allclose(forecast.mean[t], mean)
        assert_allclose(forecast.covariance[t], cov)
        assert forecast.nll[t] == pytest.approx(-multivariate_normal.logpdf(x[t], mean, cov))
    score, _ = score_validation(model, x, u, 40, [0.05, 0.5, 0.95], covariate_policy="persistence")
    assert score == pytest.approx(-forecast.nll[40:].mean())


def test_forecasts_do_not_consume_target_day_x_or_u():
    model, x, u = fixture()
    baseline = one_step_forecasts(model, x, u)
    x[41:] += 100
    u[41:] -= 100
    changed = one_step_forecasts(model, x, u)
    for name in ("mean", "covariance", "state_mean"):
        assert_allclose(getattr(baseline, name)[:42], getattr(changed, name)[:42])
    # Scores, unlike forecasts, must of course depend on the realized target.
    assert_allclose(baseline.nll[:41], changed.nll[:41])
    assert baseline.nll[41] != changed.nll[41]


def test_public_forecast_consumers_share_exactly_the_same_policy():
    model, x, u = fixture()
    forecast = one_step_forecasts(model, x, u)
    mean, nll = dlc_predictions(model, x, u)
    assert_allclose(mean, forecast.mean)
    assert_allclose(nll, forecast.nll)
    assert_allclose(dlc_next_day_predictions(model, x, u), mean)
    conditional = one_step_forecasts(model, x, u, covariate_policy="conditional")
    assert not np.allclose(conditional.mean[3:], mean[3:])
    with pytest.raises(ValueError, match="covariate_policy"):
        one_step_forecasts(model, x, u, covariate_policy="future")
