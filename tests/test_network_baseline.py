import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from baselines.network_baseline import FixedNetworkSSM


def sample():
    return np.random.default_rng(129).normal(size=(90, 3))


def test_fixed_network_operator_identity_and_density():
    x = sample()
    model = FixedNetworkSSM(lag=2).fit(x[:50])
    path = model.filter(x)
    assert path.operators.shape == (90, 2, 3, 3)
    for t in (2, 45, 89):
        operator = sum(z * w for z, w in zip(path.filtered.mean[t - 2], model.basis_, strict=True))
        assert_allclose(path.operators[t], operator)
        assert path.nll[t] == pytest.approx(
            -multivariate_normal.logpdf(x[t], path.prediction[t], path.predictive_covariance[t])
        )
    assert -np.sum(path.nll[2:]) == pytest.approx(path.filtered.loglik)


def test_fixed_network_no_future_data_or_target_day_prediction_leakage():
    x = sample()
    model = FixedNetworkSSM(lag=2).fit(x[:50])
    basis = model.basis_.copy()
    path = model.filter(x)
    for end in (51, 65, 90):
        prefix = model.filter(x[:end])
        assert_allclose(prefix.prediction, path.prediction[:end])
        assert_allclose(prefix.operators, path.operators[:end])
    x[65:] += 100
    other = model.filter(x)
    assert_allclose(path.prediction[:66], other.prediction[:66])
    assert_allclose(path.predictive_covariance[:66], other.predictive_covariance[:66])
    assert_allclose(path.operators[:65], other.operators[:65])
    assert_allclose(basis, model.basis_)


def test_network_filter_agrees_with_scalar_recursive_bayes():
    x = np.random.default_rng(12).normal(size=(60, 1))
    model = FixedNetworkSSM(process_variance=0, p0=0.7).fit(x[:30], np.ones((1, 1, 1, 1)))
    path = model.filter(x)
    precision, numerator = 1 / 0.7, 0.0
    variance = model.sigma_.item()
    for t in range(1, len(x)):
        h, y = x[t - 1, 0], x[t, 0] - model.intercept_[0]
        assert path.prediction[t, 0] == pytest.approx(
            model.intercept_[0] + h * numerator / precision
        )
        precision += h * h / variance
        numerator += h * y / variance
        assert path.filtered.mean[t - 1, 0] == pytest.approx(numerator / precision)


def test_invalid_network_basis_rejected():
    with pytest.raises(ValueError, match="basis"):
        FixedNetworkSSM().fit(sample(), np.ones((2, 2)))
