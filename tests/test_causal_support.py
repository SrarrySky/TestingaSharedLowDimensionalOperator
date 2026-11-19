from __future__ import annotations

import numpy as np
import pandas as pd

from causal.causal_support import (
    SupportResult,
    _bh_adjust_lagged_pvalues,
    _classify_lpcmci_mark,
    _merge_union_results,
    _shared_lpcmci_stability,
    discover_support,
    load_support,
    lpcmci_stability,
    save_support,
)


def test_shared_lpcmci_pool_keeps_context_results_separate() -> None:
    contexts = [np.zeros((40, 2)), np.ones((40, 2))]
    seeds = [[11, 12, 13], [21, 22, 23]]

    def fake_worker(
        context_index,
        bootstrap_index,
        bootstrap_seed,
        *,
        x,
        **kwargs,
    ):
        assert x.shape == (40, 2)
        shape = (1, 2, 2)
        a = np.zeros(shape, dtype=bool)
        b = np.zeros(shape, dtype=bool)
        c = np.zeros(shape, dtype=bool)
        d = np.zeros(shape, dtype=bool)
        a[0, 0, 1] = context_index == 0
        b[0, 1, 0] = context_index == 1
        return context_index, bootstrap_index, 0.0, (a, b, c, d)

    import causal.causal_support as cs

    original = cs._lpcmci_context_bootstrap_worker
    cs._lpcmci_context_bootstrap_worker = fake_worker
    try:
        results = _shared_lpcmci_stability(
            contexts,
            ["global", "rolling_1"],
            bootstrap_count=3,
            lag=1,
            block_length=5,
            pc_alpha=0.05,
            threshold=0.5,
            seeds=seeds,
            tau_min=1,
            ci_test="parcorr",
            significance="analytic",
            fdr_method="fdr_bh",
            fdr_alpha=0.05,
            tier_a_marks=("-->",),
            tier_b_marks=("o->",),
            tier_c_marks=("<->",),
            tier_d_marks=("o-o",),
            include_tier_b=True,
            n_jobs=2,
            backend="threading",
            inner_max_num_threads=1,
            parallel_verbose=0,
        )
    finally:
        cs._lpcmci_context_bootstrap_worker = original

    assert len(results) == 2
    assert results[0].support[0, 0, 1]
    assert not results[0].support[0, 1, 0]
    assert results[1].support[0, 1, 0]
    assert not results[1].support[0, 0, 1]
    assert np.isclose(results[0].tier_a_frequency[0, 0, 1], 1.0)
    assert np.isclose(results[1].tier_b_frequency[0, 1, 0], 1.0)


def test_union_lpcmci_tier_a_upgrades_tier_b() -> None:
    shape = (1, 2, 2)

    support_1 = np.zeros(shape, dtype=bool)
    freq_1 = np.zeros(shape, dtype=float)
    tier_1 = np.full(shape, "none", dtype="U16")
    a_freq_1 = np.zeros(shape, dtype=float)
    b_freq_1 = np.zeros(shape, dtype=float)
    c_freq_1 = np.zeros(shape, dtype=float)
    d_freq_1 = np.zeros(shape, dtype=float)

    support_1[0, 1, 0] = True
    freq_1[0, 1, 0] = 0.80
    tier_1[0, 1, 0] = "B"
    b_freq_1[0, 1, 0] = 0.80

    global_result = SupportResult(
        support=support_1,
        frequency=freq_1,
        tier=tier_1,
        method="lpcmci",
        tier_a_frequency=a_freq_1,
        tier_b_frequency=b_freq_1,
        tier_c_frequency=c_freq_1,
        tier_d_frequency=d_freq_1,
    )

    support_2 = np.zeros(shape, dtype=bool)
    freq_2 = np.zeros(shape, dtype=float)
    tier_2 = np.full(shape, "none", dtype="U16")
    a_freq_2 = np.zeros(shape, dtype=float)
    b_freq_2 = np.zeros(shape, dtype=float)
    c_freq_2 = np.zeros(shape, dtype=float)
    d_freq_2 = np.zeros(shape, dtype=float)

    support_2[0, 1, 0] = True
    freq_2[0, 1, 0] = 0.91
    tier_2[0, 1, 0] = "A"
    a_freq_2[0, 1, 0] = 0.91

    rolling_result = SupportResult(
        support=support_2,
        frequency=freq_2,
        tier=tier_2,
        method="lpcmci",
        tier_a_frequency=a_freq_2,
        tier_b_frequency=b_freq_2,
        tier_c_frequency=c_freq_2,
        tier_d_frequency=d_freq_2,
    )

    merged = _merge_union_results(
        [global_result, rolling_result],
        method="lpcmci",
    )

    assert merged.support[0, 1, 0]
    assert np.isclose(merged.frequency[0, 1, 0], 0.91)
    assert merged.tier[0, 1, 0] == "A"
    assert merged.tier_a_frequency is not None
    assert merged.tier_b_frequency is not None
    assert np.isclose(merged.tier_a_frequency[0, 1, 0], 0.91)
    assert np.isclose(merged.tier_b_frequency[0, 1, 0], 0.80)


def test_lpcmci_mark_policy_is_exact() -> None:
    kwargs = dict(
        tier_a_marks=("-->",),
        tier_b_marks=("o->",),
        tier_c_marks=("<->",),
        tier_d_marks=("o-o",),
    )

    assert _classify_lpcmci_mark("-->", **kwargs) == "A"
    assert _classify_lpcmci_mark("o->", **kwargs) == "B"
    assert _classify_lpcmci_mark("<->", **kwargs) == "C"
    assert _classify_lpcmci_mark("o-o", **kwargs) == "D"
    assert _classify_lpcmci_mark("-?>", **kwargs) == "D"
    assert _classify_lpcmci_mark("", **kwargs) == "none"


def test_bh_adjust_lagged_pvalues() -> None:
    p = np.ones((1, 1, 5), dtype=float)
    p[0, 0, 1:] = np.asarray([0.001, 0.010, 0.040, 0.200])

    q = _bh_adjust_lagged_pvalues(
        p,
        tau_min=1,
        tau_max=4,
    )

    assert np.isclose(q[0, 0, 0], 1.0)
    np.testing.assert_allclose(
        q[0, 0, 1:],
        np.asarray([0.004, 0.020, 0.05333333333333334, 0.200]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_lpcmci_rejects_zero_or_nonfinite_stability_threshold() -> None:
    x = np.zeros((100, 2), dtype=float)
    for threshold in (0.0, np.nan, np.inf):
        with np.testing.assert_raises(ValueError):
            lpcmci_stability(
                x,
                lag=1,
                bootstrap_count=1,
                block_length=20,
                pc_alpha=0.05,
                threshold=threshold,
                seed=1,
            )


def test_tier_a_and_b_are_thresholded_separately(monkeypatch) -> None:
    import causal.causal_support as cs

    call_index = {"value": 0}

    def fake_worker(**kwargs):
        idx = call_index["value"]
        call_index["value"] += 1

        shape = (1, 2, 2)
        a = np.zeros(shape, dtype=bool)
        b = np.zeros(shape, dtype=bool)
        c = np.zeros(shape, dtype=bool)
        d = np.zeros(shape, dtype=bool)

        if idx < 7:
            a[0, 1, 0] = True
        elif idx < 14:
            b[0, 1, 0] = True
        else:
            d[0, 1, 0] = True

        return a, b, c, d

    monkeypatch.setattr(cs, "_lpcmci_bootstrap_worker", fake_worker)

    x = np.zeros((100, 2), dtype=float)
    result = lpcmci_stability(
        x,
        lag=1,
        bootstrap_count=20,
        block_length=20,
        pc_alpha=0.05,
        threshold=0.70,
        seed=42,
        n_jobs=1,
    )

    assert result.tier_a_frequency is not None
    assert result.tier_b_frequency is not None
    assert np.isclose(result.tier_a_frequency[0, 1, 0], 0.35)
    assert np.isclose(result.tier_b_frequency[0, 1, 0], 0.35)
    assert not result.support[0, 1, 0]
    assert result.tier[0, 1, 0] == "none"


def test_support_publication_fields_roundtrip(tmp_path) -> None:
    shape = (2, 2, 2)

    support = np.zeros(shape, dtype=bool)
    support[0, 1, 0] = True

    frequency = np.zeros(shape, dtype=float)
    frequency[0, 1, 0] = 0.90

    tier = np.full(shape, "none", dtype="U16")
    tier[0, 1, 0] = "A"

    a = np.zeros(shape, dtype=float)
    b = np.zeros(shape, dtype=float)
    c = np.zeros(shape, dtype=float)
    d = np.zeros(shape, dtype=float)
    a[0, 1, 0] = 0.90

    global_frequency = np.zeros(shape, dtype=float)
    global_frequency[0, 1, 0] = 0.75

    context_frequency = np.stack(
        [global_frequency, frequency],
        axis=0,
    )

    result = SupportResult(
        support=support,
        frequency=frequency,
        tier=tier,
        method="lpcmci_union",
        tier_a_frequency=a,
        tier_b_frequency=b,
        tier_c_frequency=c,
        tier_d_frequency=d,
        global_frequency=global_frequency,
        context_frequency=context_frequency,
        context_names=("global", "rolling_1"),
        metadata={
            "fit_split": "train_only",
            "fdr_method": "fdr_bh",
        },
    )

    path = tmp_path / "support.npz"
    save_support(
        result,
        path,
        names=np.asarray(["A", "B"]),
    )

    loaded = load_support(path)

    np.testing.assert_array_equal(loaded.support, result.support)
    np.testing.assert_allclose(loaded.frequency, result.frequency)
    np.testing.assert_allclose(loaded.tier_a_frequency, result.tier_a_frequency)
    np.testing.assert_allclose(loaded.tier_b_frequency, result.tier_b_frequency)
    np.testing.assert_allclose(loaded.tier_c_frequency, result.tier_c_frequency)
    np.testing.assert_allclose(loaded.tier_d_frequency, result.tier_d_frequency)
    np.testing.assert_allclose(loaded.global_frequency, result.global_frequency)
    np.testing.assert_allclose(loaded.context_frequency, result.context_frequency)
    assert loaded.context_names == result.context_names
    assert loaded.metadata == result.metadata

    edge_table = pd.read_csv(path.with_suffix(".edges.csv"))
    expected_columns = {
        "source",
        "target",
        "lag",
        "frequency",
        "global_frequency",
        "best_context",
        "best_context_frequency",
        "tier",
        "tier_a_frequency",
        "tier_b_frequency",
        "tier_c_warning_frequency",
        "tier_d_warning_frequency",
    }
    assert expected_columns <= set(edge_table.columns)


def test_lpcmci_publication_config_is_forwarded(monkeypatch) -> None:
    import causal.causal_support as cs

    captured = {}

    def fake_lpcmci_stability(
        x,
        lag,
        bootstrap_count,
        block_length,
        pc_alpha,
        threshold,
        seed,
        **kwargs,
    ):
        captured.update(
            {
                "lag": lag,
                "bootstrap_count": bootstrap_count,
                "block_length": block_length,
                "pc_alpha": pc_alpha,
                "threshold": threshold,
                "seed": seed,
                **kwargs,
            }
        )

        shape = (lag, x.shape[1], x.shape[1])
        support = np.zeros(shape, dtype=bool)
        frequency = np.zeros(shape, dtype=float)
        tier = np.full(shape, "none", dtype="U16")

        return SupportResult(
            support=support,
            frequency=frequency,
            tier=tier,
            method="lpcmci",
            tier_a_frequency=frequency.copy(),
            tier_b_frequency=frequency.copy(),
            tier_c_frequency=frequency.copy(),
            tier_d_frequency=frequency.copy(),
        )

    monkeypatch.setattr(cs, "lpcmci_stability", fake_lpcmci_stability)

    config = {
        "method": "lpcmci",
        "lag": 3,
        "seed": 123,
        "bootstrap_count": 4,
        "block_length": 20,
        "bootstrap_mode": "moving_block",
        "stability_threshold": 0.70,
        "union_screen_threshold": 0.50,
        "support_mode": "stable",
        "parallel": {
            "n_jobs": 1,
            "backend": "loky",
            "inner_max_num_threads": 1,
            "verbose": 0,
        },
        "lpcmci": {
            "pc_alpha": 0.04,
            "tau_min": 1,
            "ci_test": "parcorr",
            "significance": "analytic",
            "fdr_method": "fdr_bh",
            "fdr_alpha": 0.03,
            "tier_a_marks": ["-->"],
            "tier_b_marks": ["o->"],
            "tier_c_marks": ["<->"],
            "tier_d_marks": ["o-o", "x-x", "x?x"],
            "include_tier_b": True,
        },
        "linear_lasso": {
            "alpha": 0.0025,
            "max_iter": 10000,
            "coef_zero_tolerance": 1e-12,
        },
    }

    x = np.zeros((200, 3), dtype=float)
    discover_support(x, config)

    assert captured["block_length"] == 20
    assert captured["tau_min"] == 1
    assert captured["ci_test"] == "parcorr"
    assert captured["significance"] == "analytic"
    assert captured["fdr_method"] == "fdr_bh"
    assert np.isclose(captured["fdr_alpha"], 0.03)
    assert captured["tier_a_marks"] == ("-->",)
    assert captured["tier_b_marks"] == ("o->",)
    assert captured["tier_c_marks"] == ("<->",)
    assert captured["tier_d_marks"] == ("o-o", "x-x", "x?x")
    assert captured["include_tier_b"] is True
