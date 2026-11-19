from __future__ import annotations

import numpy as np

from experiments.synthetic import (
    generate_scenario,
    generate_shared_low_rank,
    operator_subspace_metrics,
    theory_checks,
)


def test_synthetic_shared_low_rank_shapes() -> None:
    ds = generate_shared_low_rank(T=250, n=5, lag=2, d=2, seed=9)
    assert ds.x.shape == (250, 5)
    assert ds.u.shape[0] == 250
    assert ds.z.shape == (250, 2)
    assert ds.a_path.shape[0] == 250
    assert ds.support.shape == (2, 5, 5)
    assert ds.operators is not None


def test_competing_synthetic_dispatch_and_truth() -> None:
    for scenario in ("s0", "s1", "s2", "s3", "s4", "s5", "s6"):
        ds = generate_scenario(scenario, T=64, n=4, lag=2, seed=3)
        assert ds.operators is not None
        assert ds.operators.shape == (64, 2, 4, 4)

    pair = generate_scenario("s6", T=64, seed=3)
    assert np.allclose(
        np.linalg.eigvals(pair.metadata["low_operator"]),
        np.linalg.eigvals(pair.metadata["high_operator"]),
    )
    assert pair.metadata["high_gain"].max() > pair.metadata["low_gain"].max()


def test_financial_stress_is_one_factor_at_a_time() -> None:
    ds = generate_scenario("stress_missing_async", T=64, n=4, lag=2, seed=3)
    assert ds.observed_mask is not None
    assert np.isnan(ds.x).any()
    assert ds.metadata["stress"] == "missing_async"


def test_theory_checks() -> None:
    checks = theory_checks(seed=3, horizon=5)
    assert checks["gauge_invariance_error"] < 1e-10
    assert checks["sensitivity_finite_difference_error"] < 1e-7
    assert checks["frozen_pathwise_residual"] < 1e-10


def test_operator_subspace_rejects_dimension_and_rank_mismatch() -> None:
    with np.testing.assert_raises(ValueError):
        operator_subspace_metrics(np.ones((4, 2)), np.ones((5, 2)))
    with np.testing.assert_raises(ValueError):
        operator_subspace_metrics(np.eye(4)[:, :2], np.eye(4)[:, :1])


def test_all_s0_s7_scenarios_have_auditable_truth_metadata() -> None:
    for scenario in ("s0", "s1", "s2", "s3", "s4", "s5", "s6"):
        dataset = generate_scenario(scenario, T=48, n=4, lag=2, seed=11)
        assert dataset.operators is not None
        assert dataset.operators.shape == (48, 2, 4, 4)
        assert dataset.a_path.shape[0] == 48
        assert dataset.metadata
    stress = generate_scenario("s7", stress="student_t", T=48, n=4, lag=2, seed=11)
    assert stress.metadata["stress"] == "student_t"
    assert stress.observed_mask is not None
