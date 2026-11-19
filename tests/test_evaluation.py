from __future__ import annotations

import numpy as np

from core.evaluation import (
    fit_oos_feature_transform,
    is_degenerate_feature,
)


def test_degenerate_feature_detection() -> None:
    constant = np.zeros(100)
    varying = np.linspace(
        0.0,
        1.0,
        100,
    )
    assert is_degenerate_feature(constant)
    assert not is_degenerate_feature(varying)


def test_oos_feature_transform_drops_constant_rho() -> None:
    # spectral_radius is constant.
    # dynamic_nonnormality and observable_gain vary.
    X_train = np.array(
        [
            [0.0, 0.60, 0.05],
            [0.0, 0.70, 0.07],
            [0.0, 0.80, 0.09],
            [0.0, 0.90, 0.11],
        ]
    )

    X_test = np.array(
        [
            [0.0, 0.75, 0.08],
            [0.0, 0.85, 0.10],
        ]
    )

    names = [
        "spectral_radius",
        "dynamic_nonnormality",
        "observable_gain",
    ]
    prep = fit_oos_feature_transform(
        X_train,
        names,
        standardize=True,
        verbose=False,
    )
    assert prep.dropped_names == ("spectral_radius",)
    assert prep.kept_names == (
        "dynamic_nonnormality",
        "observable_gain",
    )
    X_train_t = prep.transform(X_train)
    X_test_t = prep.transform(X_test)
    assert X_train_t.shape == (
        4,
        2,
    )
    assert X_test_t.shape == (
        2,
        2,
    )
    # Train means should be zero after Train-only standardization.
    assert np.allclose(
        X_train_t.mean(axis=0),
        0.0,
        atol=1e-12,
    )
    # Train population std should be one.
    assert np.allclose(
        X_train_t.std(
            axis=0,
            ddof=0,
        ),
        1.0,
        atol=1e-12,
    )


def test_oos_feature_transform_roundtrip(
    tmp_path,
) -> None:
    from core.evaluation import (
        fit_oos_feature_transform,
        load_oos_feature_transform,
        save_oos_feature_transform,
    )

    X = np.array(
        [
            [0.0, 1.0, 10.0],
            [0.0, 2.0, 20.0],
            [0.0, 3.0, 30.0],
            [0.0, 4.0, 40.0],
        ],
        dtype=float,
    )
    names = [
        "constant",
        "x1",
        "x2",
    ]
    original = fit_oos_feature_transform(
        X,
        names,
        verbose=False,
    )
    path = tmp_path / "transform.npz"
    save_oos_feature_transform(
        original,
        path,
    )
    restored = load_oos_feature_transform(path)
    assert np.array_equal(
        original.keep_mask,
        restored.keep_mask,
    )
    assert original.original_names == restored.original_names
    assert original.kept_names == restored.kept_names
    assert original.dropped_names == restored.dropped_names
    assert np.allclose(
        original.mean,
        restored.mean,
    )
    assert np.allclose(
        original.scale,
        restored.scale,
    )
    assert np.allclose(
        original.transform(X),
        restored.transform(X),
    )
