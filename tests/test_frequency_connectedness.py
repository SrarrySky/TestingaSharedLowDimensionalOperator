import numpy as np
import pytest

from baselines.connectedness import frequency_connectedness


def test_frequency_bands_recombine_to_full_generalized_decomposition():
    a = np.array([[[0.35, 0.08], [0.02, 0.25]]])
    sigma = np.array([[1.0, 0.2], [0.2, 0.8]])
    result = frequency_connectedness(
        a,
        sigma,
        bands=((0.0, np.pi / 3), (np.pi / 3, np.pi)),
        horizon=64,
    )
    shares = result["frequency_gfevd"]
    assert shares.shape == (2, 2, 2)
    assert np.isfinite(shares).all()
    combined = shares.sum(axis=0)
    assert np.allclose(combined.sum(axis=1), 1.0, atol=2e-3)
    assert not np.allclose(combined, np.ones((2, 2)), atol=2e-3)


def test_frequency_grid_and_input_validation():
    a = np.array([[[0.2]]])
    sigma = np.array([[1.0]])
    result = frequency_connectedness(
        a,
        sigma,
        bands=((0.0, np.pi),),
        frequency_grid=np.linspace(0.0, np.pi, 9),
    )
    assert np.allclose(result["frequency_gfevd"], 1.0)
    with pytest.raises(ValueError, match="frequency_grid"):
        frequency_connectedness(a, sigma, frequency_grid=[0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="horizon"):
        frequency_connectedness(a, sigma, horizon=4)
