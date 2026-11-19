"""S5 recovery control: does subspace recovery improve with signal strength?

Generates shared-low-rank data with adjustable observation noise and sample size,
fits DLC-SSM at the true configuration (L=2, d=2) with the oracle support, and
reports the full principal-angle vector, projector distance, coefficient-path RMSE,
and transmitter-order recovery. The purpose is to check whether the estimator can
recover the target operator subspace when information is sufficient, separating an
estimation-flow failure from the representation hypothesis.

Run from the repository root:

    python scripts/s5_recovery_control.py

Writes "outputs/s5_recovery_control/summary.json".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

from experiments.synthetic import (
    _dataset,
    _simulate,
    _stability_lag_factors,
    _stable_random_var,
)
from experiments.synthetic_benchmark import recovery_metrics
from model.dlcssm import DLCSSM, DLCSSMParameters
from model.forecasting import one_step_forecasts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/s5_recovery_control"
HORIZON = 10


def generate_s5(T: int, n: int, lag: int, d: int, seed: int, noise_scale: float):
    """
    Generate one controlled S5 dynamic latent process.
    :param T, int: Number of time points.
    :param n, int: Number of observed series.
    :param lag, int: Autoregressive lag order.
    :param d, int: Latent coefficient-state dimension.
    :param seed, int: Random seed for reproducible simulation.
    :param noise_scale, float: Standard deviation of state and observation noise.
    :return: synthetic payload: Arrays and ground-truth parameters.
    """
    """Mirror the audited shared-low-rank generator with adjustable noise."""
    rng = np.random.default_rng(seed)
    a0, support = _stable_random_var(n, lag, rng)
    basis = rng.normal(0, 0.012, size=(d, lag, n, n)) * support[None, ...]
    transition = np.diag(np.linspace(0.92, 0.75, d))
    g = rng.normal(0, 0.05, size=(d, 2))
    q = 0.03 * np.eye(d)
    lam = rng.normal(0, 0.08, size=(n, d))
    dm = rng.normal(0, 0.03, size=(n, 2))
    u = rng.normal(size=(T, 2))
    z = np.zeros((T, d))
    for t in range(1, T):
        z[t] = transition @ z[t - 1] + g @ u[t] + rng.multivariate_normal(np.zeros(d), q)
    operators = np.asarray([a0 + np.einsum("r,rlij->lij", z[t], basis) for t in range(T)])
    lag_factors = _stability_lag_factors(operators)
    a0 *= lag_factors[:, None, None]
    basis *= lag_factors[None, :, None, None]
    operators *= lag_factors[None, :, None, None]
    x, mask, _ = _simulate(
        operators,
        support,
        rng,
        u=u,
        factors=z.copy(),
        Lambda=lam,
        D=dm,
        noise_scale=noise_scale,
    )
    params = DLCSSMParameters(
        c=np.zeros(n),
        A0=a0,
        B=basis,
        Lambda=lam,
        D=dm,
        F=transition,
        G=g[:, :2],
        Q=q,
        R=noise_scale**2 * np.eye(n),
        m0=z[lag].copy(),
        P0=np.eye(d),
    )
    return _dataset(
        operators,
        support,
        x,
        u,
        z,
        "shared_low_rank",
        factors=z.copy(),
        observed_mask=mask,
        params=params,
    )


def main() -> None:
    """
    Execute the S5 recovery-control grid and persist the summary.
    :return: None.
    """
    configs = [(1200, 0.25), (1200, 0.10), (1200, 0.05), (2400, 0.10)]
    records: list[dict[str, object]] = []
    for T, noise in configs:
        nt = int(round(T * 0.6))
        nv = int(round(T * 0.2))
        for seed in (7, 19, 31):
            ds = generate_s5(T, 8, 2, 2, seed, noise)
            x_raw, u_raw = ds.x, ds.u
            operators, support, b_true = ds.operators, ds.support, ds.params.B
            xm, xs = x_raw[:nt].mean(0), x_raw[:nt].std(0)
            um, us = u_raw[:nt].mean(0), u_raw[:nt].std(0)
            xs = np.where(xs > 1e-12, xs, 1.0)
            us = np.where(us > 1e-12, us, 1.0)
            x = (x_raw - xm) / xs
            u = (u_raw - um) / us
            model = DLCSSM(
                lag=2,
                state_dim=2,
                tier_b_multiplier=2.0,
                lambda_group=1e-4,
                lambda_B=0.0,
                lambda_ridge=1e-6,
                max_iter=400,
                min_iter=30,
                tol=1e-8,
                prox_iterations=300,
                jitter=1e-6,
                diagonal_R=True,
                canonicalize=True,
                seed=seed,
            )
            try:
                fit = model.fit(x[:nt], u[:nt], support.astype(bool))
                params = model.params_
                b_raw = params.B * xs[None, None, :, None] / xs[None, None, None, :]
                a0_raw = params.A0 * xs[None, :, None] / xs[None, None, :]
                bt = b_true.reshape(b_true.shape[0], -1).T
                be = b_raw.reshape(b_raw.shape[0], -1).T
                angles = np.degrees(subspace_angles(bt, be))
                ut = np.linalg.svd(bt, full_matrices=False)[0]
                ue = np.linalg.svd(be, full_matrices=False)[0]
                proj = float(np.linalg.norm(ut @ ut.T - ue @ ue.T, 2))
                forecasts = one_step_forecasts(model, x, u)
                prior = a0_raw[None] + np.einsum("td,dlij->tlij", forecasts.state_mean, b_raw)
                test_mask = np.arange(len(x)) >= nt + nv
                rec = recovery_metrics(
                    operators[test_mask], prior[test_mask], support.astype(bool), HORIZON
                )
                row = dict(
                    T=T,
                    noise_scale=noise,
                    seed=seed,
                    support_edges=int(support.sum()),
                    termination=str(fit.termination_reason),
                    em_iterations=int(len(fit.loglik_history)),
                    principal_angles_deg=[float(v) for v in angles],
                    projection_distance=proj,
                    dynamic_coefficient_rmse=float(rec["dynamic_coefficient_rmse"]),
                    transmitter_rank_spearman=rec["transmitter_rank_spearman"],
                )
            except Exception as exc:  # noqa: BLE001
                row = dict(T=T, noise_scale=noise, seed=seed, error=f"{type(exc).__name__}: {exc}")
            records.append(row)
            print(json.dumps(row))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
