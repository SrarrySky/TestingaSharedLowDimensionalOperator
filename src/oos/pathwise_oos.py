"""Filtered posterior predictive companion geometry (paper section 14.7)."""

from __future__ import annotations

import numpy as np

from model.uncertainty import predictive_state_paths


def predictive_pathwise_summary(
    params,
    mean,
    cov,
    u,
    *,
    horizon=5,
    draws=256,
    seed=0,
    row_index=0,
    credible=0.95,
    gain_threshold=1.0,
    migration_threshold=1.0,
):
    """Simulate conditional on information at t, with future u fixed at u_t.

    Each path uses its own sampled z_t for the frozen comparator. Compute
    max/norm BEFORE averaging; a product of mean operators is not this mean.
    Intervals describe latent/process uncertainty conditional on fitted parameters
    and the persistence scenario, not parameter-estimation uncertainty.
    """
    for name, value in (("horizon", horizon), ("draws", draws)):
        if (
            isinstance(value, bool | np.bool_)
            or not isinstance(value, int | np.integer)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("seed", seed), ("row_index", row_index)):
        if (
            isinstance(value, bool | np.bool_)
            or not isinstance(value, int | np.integer)
            or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer")
    if not np.isfinite(credible) or not 0 < credible < 1:
        raise ValueError("credible must be in (0, 1)")
    if any(not np.isfinite(v) or v < 0 for v in (gain_threshold, migration_threshold)):
        raise ValueError("exceedance thresholds must be finite and nonnegative")
    mean, cov, u = (np.asarray(v, dtype=float) for v in (mean, cov, u))
    d = params.state_dim
    if mean.shape != (d,) or cov.shape != (d, d) or u.shape != (params.G.shape[1],):
        raise ValueError("Filtered mean/covariance or current u shape mismatch")
    for name, matrix in (("filtered covariance", cov), ("Q", params.Q)):
        if (
            matrix.shape != (d, d)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10)
            or np.linalg.eigvalsh(matrix).min() < -1e-10
        ):
            raise ValueError(f"{name} must be finite, symmetric positive semidefinite")
    if not all(np.isfinite(v).all() for v in (mean, u, params.F, params.G, params.A0, params.B)):
        raise ValueError("State and transition/operator inputs must be finite")
    # Independent per-origin stream: extending the dataset cannot change earlier draws.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(row_index)]))
    initial, paths = predictive_state_paths(
        mean,
        cov,
        params.F,
        params.G,
        params.Q,
        np.repeat(u[None, :], horizon, axis=0),
        draws,
        rng,
        return_initial=True,
    )
    C0 = np.stack([params.companion(z) for z in initial])
    frozen = np.broadcast_to(np.eye(C0.shape[-1]), C0.shape).copy()
    product = frozen.copy()
    gains = np.zeros(draws)
    gaps = np.zeros(draws)
    for h in range(horizon):
        future_C = np.stack([params.companion(z) for z in paths[:, h]])
        product = future_C @ product
        frozen = C0 @ frozen
        gains = np.maximum(gains, np.linalg.norm(product, ord=2, axis=(-2, -1)))
        gaps = np.maximum(gaps, np.linalg.norm(product - frozen, ord=2, axis=(-2, -1)))
    if not np.isfinite(gains).all() or not np.isfinite(gaps).all():
        raise ValueError("Non-finite predictive geometry; inspect fitted dynamics/horizon")
    alpha = (1 - credible) / 2
    result = {}
    for name, values, threshold in (
        ("path_gain", gains, gain_threshold),
        ("migration_gap", gaps, migration_threshold),
    ):
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if draws > 1 else 0.0,
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
            "exceedance_probability": float(np.mean(values > threshold)),
        }
    return result
