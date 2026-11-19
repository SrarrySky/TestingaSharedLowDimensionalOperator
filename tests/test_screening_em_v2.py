"""Regression checks for Version2 screening, weighted GEM and public entrypoints."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize

from causal.causal_support import SupportResult, discover_support, load_support, save_support
from core.config import load_yaml
from experiments.validation import tune_validation
from model.dlcssm import (
    DLCSSM,
    DLCSSMParameters,
    _expected_objective,
    _observation_m_step,
    _pack_theta,
    _penalty,
    _unpack_theta,
)
from model.kalman_em import SmoothResult

ROOT = Path(__file__).resolve().parents[1]


def support_fixture(lag=1):
    mask = np.ones((lag, 2, 2), bool)
    tier = np.full(mask.shape, "A", dtype="U16")
    tier[:, 0, 1] = "B"
    mask[:, 1, 0] = False
    tier[:, 1, 0] = "none"
    return SupportResult(mask, mask * 0.8, tier, "lpcmci_union")


@pytest.mark.parametrize("method", ["lpcmci", "linear_lasso"])
@pytest.mark.parametrize("mode", ["stable", "union"])
def test_global_context_uses_mode_threshold(method, mode):
    config_name = (
        "config/exploration/causal_explore.yaml" if method == "lpcmci" else "config/causal.yaml"
    )
    cfg = load_yaml(ROOT / config_name)
    cfg.update(
        method=method,
        support_mode=mode,
        lag=1,
        bootstrap_count=2,
        block_length=5,
        rolling_contexts=2,
    )
    if mode == "union":
        cfg["union_screen_threshold"] = 0.5
    # This test exercises the public per-context entrypoint.  The publication
    # default uses the separate shared worker-pool scheduler, which deliberately
    # runs below lpcmci_stability() and therefore should not be mocked here.
    cfg["parallel"]["context_scope"] = "per_context"
    seen = []

    def fake(x, lag, B, block, alpha, threshold, seed, **kwargs):
        seen.append(threshold)
        # A global-only edge at exactly union threshold must be retained.
        freq = np.zeros((1, 2, 2))
        if len(seen) == 1:
            freq[0, 0, 1] = 0.5
        mask = freq >= threshold
        tier = np.where(mask, "A" if method == "lpcmci" else "predictive", "none")
        return SupportResult(mask, freq, tier, method)

    name = "lpcmci_stability" if method == "lpcmci" else "linear_lasso_stability"
    with patch("causal.causal_support." + name, side_effect=fake):
        result = discover_support(np.zeros((160, 2)), cfg)
    expected = 0.5 if mode == "union" else 0.7
    assert seen == [expected] * len(seen)
    assert bool(result.support[0, 0, 1]) == (mode == "union")
    assert result.metadata["context_thresholds"] == seen
    assert_allclose(result.global_frequency[0, 0, 1], 0.5)


def test_weighted_prox_matches_independent_convex_minimizer():
    rng = np.random.default_rng(97)
    x = rng.normal(size=(51, 1))
    u = np.empty((51, 0))
    p = DLCSSMParameters(
        np.zeros(1),
        np.ones((1, 1, 1)) * 0.1,
        np.ones((1, 1, 1, 1)) * 0.15,
        np.ones((1, 1)) * 0.2,
        np.empty((1, 0)),
        np.eye(1) * 0.5,
        np.empty((1, 0)),
        np.eye(1),
        np.eye(1),
        np.zeros(1),
        np.eye(1),
    )
    smooth = SmoothResult(
        rng.normal(size=(50, 1)),
        np.ones((50, 1, 1)) * 0.2,
        np.zeros((50, 1, 1)),
        np.zeros((49, 1, 1)),
    )
    weights = np.full((1, 1, 1), 3.0)
    shape = _pack_theta(p).shape

    def objective(flat):
        trial = deepcopy(p)
        _unpack_theta(flat.reshape(shape), trial)
        # Independent formula, not the production penalty helper.
        pen = 0.012 * 3 * np.sqrt(trial.A0.item() ** 2 + trial.B.item() ** 2)
        pen += 0.006 * 3 * abs(trial.B.item()) + 0.01 / 2 * np.sum(flat**2)
        return _expected_objective(x, u, smooth, trial, pen)

    opt = minimize(
        objective,
        _pack_theta(p).ravel(),
        method="Powell",
        options={"xtol": 1e-11, "ftol": 1e-11, "maxiter": 3000},
    )
    candidate = deepcopy(p)
    _observation_m_step(
        x,
        u,
        smooth,
        candidate,
        np.ones_like(weights, bool),
        0.012,
        0.006,
        0.01,
        10000,
        1e-6,
        True,
        weights,
    )
    assert objective(_pack_theta(candidate).ravel()) <= opt.fun + 1e-7


def test_weighted_gem_gauge_checkpoint_and_legacy(tmp_path):
    rng = np.random.default_rng(419)
    x = rng.normal(size=(65, 2))
    u = rng.normal(size=(65, 1))
    support = support_fixture()
    save_support(support, tmp_path / "support.npz", ["a", "b"])
    model = DLCSSM(
        lag=1,
        state_dim=1,
        max_iter=5,
        prox_iterations=100,
        lambda_group=0.04,
        lambda_B=0.02,
        tier_b_multiplier=3.0,
    )
    fit = model.fit(x, u, load_support(tmp_path / "support.npz"))
    assert len(fit.penalized_objective_history) > 1
    assert np.all(np.diff(fit.penalized_objective_history) >= 0)
    assert_allclose(model.edge_weights_[0], [[1, 3], [1, 1]])
    assert_allclose(model.penalty_value(), fit.penalty_history[-1], atol=1e-10)
    p = fit.params.apply_gauge(np.linalg.inv(model.penalty_gauge_))
    assert_allclose(p.Q, np.eye(1), atol=1e-10)
    assert_allclose(p.A0[~support.support], 0)
    assert_allclose(p.B[:, ~support.support], 0)
    expected = 0.04 * np.sum(model.edge_weights_ * np.sqrt(p.A0**2 + (p.B**2).sum(0)))
    expected += 0.02 * np.sum(model.edge_weights_[None] * abs(p.B))
    expected += model.lambda_ridge / 2 * np.sum(_pack_theta(p) ** 2)
    assert_allclose(expected, model.penalty_value())
    model.save(tmp_path / "model.npz")
    loaded = DLCSSM.load(tmp_path / "model.npz")
    assert_allclose(loaded.penalty_value(), expected)
    assert_array_equal(loaded.support_tier_, support.tier)
    assert_allclose(loaded.filter(x, u).mean, fit.filtered.mean, atol=1e-9)
    # A pre-patch checkpoint without tier fields remains a uniform-weight model.
    with np.load(tmp_path / "model.npz") as z:
        old = {
            k: z[k]
            for k in z.files
            if k not in ("edge_penalty_weights", "support_tier", "tier_b_multiplier")
        }
    np.savez(tmp_path / "old.npz", **old)
    legacy = DLCSSM.load(tmp_path / "old.npz")
    assert_allclose(legacy.edge_weights_, 1)
    assert_allclose(legacy.penalty_value(), _penalty(p, 0.04, 0.02, model.lambda_ridge))


def test_tuning_preserves_tiers_and_test_isolation(tmp_path):
    rng = np.random.default_rng(72)
    ds = SimpleNamespace(
        x=rng.normal(size=(60, 2)),
        u=rng.normal(size=(60, 1)),
        dates=np.arange("2020-01-01", "2020-03-01", dtype="datetime64[D]"),
        split=np.array(["train"] * 40 + ["validation"] * 12 + ["test"] * 8),
    )
    cfg = load_yaml(ROOT / "config/dlcssm.yaml")
    cfg.update(lag=1, state_dimension=1, hyperparameter_grid={})
    cfg["fit"].update(max_em_iterations=2, min_em_iterations=1, prox_iterations=20)
    cfg["stability"]["validation_reject_if_spectral_radius_above"] = 100

    def get_support(lag, train):
        assert_array_equal(train, ds.x[:40])
        return support_fixture(lag), "lpcmci_union"

    a = tune_validation(ds, cfg, get_support, tmp_path / "a")
    model = DLCSSM.load(tmp_path / "a/selected_model.npz")
    assert model.edge_weights_[0, 0, 1] == 2
    ds.x[52:] = np.nan
    ds.u[52:] = np.inf
    b = tune_validation(ds, cfg, get_support, tmp_path / "b")
    assert a == b


@pytest.mark.parametrize("value", [1.0, 0.5, float("nan"), float("inf")])
def test_invalid_tier_multiplier(value):
    with pytest.raises(ValueError, match="tier_b_multiplier"):
        DLCSSM(tier_b_multiplier=value)


def test_cli_fit_retains_tiers(tmp_path):
    import yaml

    from cli import fit as cli_fit

    rng = np.random.default_rng(73)
    ds = SimpleNamespace(
        x=rng.normal(size=(45, 2)),
        u=rng.normal(size=(45, 1)),
        split=np.array(["train"] * 35 + ["test"] * 10),
    )
    cfg = load_yaml(ROOT / "config/dlcssm.yaml")
    cfg.update(lag=1, state_dimension=1)
    cfg["fit"].update(max_em_iterations=2, min_em_iterations=1, prox_iterations=20)
    cfg["penalty"]["tier_b_multiplier"] = 4.0
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    save_support(support_fixture(), tmp_path / "support.npz", ["a", "b"])
    with patch("cli.load_prepared_dataset", return_value=ds):
        cli_fit(
            data=tmp_path / "unused.npz",
            model_config=tmp_path / "config.yaml",
            support=tmp_path / "support.npz",
            output=tmp_path / "model.npz",
        )
    assert DLCSSM.load(tmp_path / "model.npz").edge_weights_[0, 0, 1] == 4.0


