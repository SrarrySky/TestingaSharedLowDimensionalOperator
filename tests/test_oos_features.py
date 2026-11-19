from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.data import PreparedDataset
from model.propagation import companion_matrix
from oos.oos_features import build_point_oos_feature_matrix


class _FakeParams:
    lag = 2
    n = 2

    def companion(
        self,
        z: np.ndarray,
    ) -> np.ndarray:
        A = np.zeros(
            (
                self.lag,
                self.n,
                self.n,
            )
        )
        return companion_matrix(A)


class _FakeModel:
    def __init__(
        self,
    ) -> None:
        self.params_ = _FakeParams()

    def filter(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ):
        M = len(x) - self.params_.lag
        return SimpleNamespace(
            mean=np.zeros(
                (
                    M,
                    1,
                )
            ),
            cov=np.repeat(
                np.eye(1)[
                    None,
                    :,
                    :,
                ],
                M,
                axis=0,
            ),
        )


def test_oos_feature_alignment() -> None:
    T = 8
    ds = PreparedDataset(
        x=np.zeros(
            (
                T,
                2,
            )
        ),
        u=np.arange(
            T,
            dtype=float,
        )[:, None],
        dates=np.asarray([f"d{i}" for i in range(T)]),
        sector_codes=np.asarray(
            [
                "a",
                "b",
            ]
        ),
        sector_names=np.asarray(
            [
                "A",
                "B",
            ]
        ),
        covariate_names=np.asarray(["market_return"]),
        split=np.asarray(
            [
                "train",
                "train",
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ]
        ),
        x_mean=np.zeros(2),
        x_scale=np.ones(2),
        u_mean=np.zeros(1),
        u_scale=np.ones(1),
    )

    model = _FakeModel()
    extra = np.arange(
        T,
        dtype=float,
    )
    bundle = build_point_oos_feature_matrix(
        ds,
        model,
        horizons=(2,),
        extra_features={
            "extra__example": extra,
        },
    )
    # lag=2 means the first feature date corresponds to dataset row 2.
    assert np.array_equal(
        bundle.row_indices,
        np.arange(
            2,
            T,
        ),
    )
    assert np.array_equal(
        bundle.dates,
        ds.dates[2:],
    )
    assert np.array_equal(
        bundle.split,
        ds.split[2:],
    )
    assert bundle.state_mean.shape[0] == (T - 2)
    j = bundle.feature_names.index("extra__example")
    assert np.array_equal(
        bundle.X[:, j],
        extra[2:],
    )
    assert np.all(bundle.valid_rows)
