"""S5 positive control: structural recovery with known (oracle) support at L=2, d=2.

Loads the audited S5 truth and standardized prepared data, fits DLC-SSM at the
generating configuration (lag=2, state_dim=2) using (i) the oracle true support
and (ii) the linear-Lasso screened support, and reports operator-subspace recovery
(principal angles, projector distance) and path recovery (coefficient/kernel/gain
RMSE, transmitter-rank correlation).

This is read-only with respect to existing artifacts and writes to
"outputs/s5_positive_control/". Run from the repository root:

    python scripts/s5_positive_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from causal.causal_support import discover_support
from experiments.synthetic import operator_subspace_metrics
from experiments.synthetic_benchmark import recovery_metrics
from model.dlcssm import DLCSSM
from model.forecasting import one_step_forecasts

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs/main_mid_run/stages/synthetic/attempt_001/benchmark"
OUT = ROOT / "outputs/s5_positive_control"
SEEDS = (7, 19, 31)
HORIZON = 10
TRAIN_COUNT, VALIDATION_COUNT = 720, 240

SUPPORT_CFG = dict(
    method="linear_lasso",
    bootstrap_count=100,
    block_length=20,
    bootstrap_mode="moving_block",
    stability_threshold=0.7,
    union_screen_threshold=0.5,
    support_mode="union",
    rolling_contexts=3,
    linear_lasso=dict(alpha=0.0025, max_iter=10000, coef_zero_tolerance=1e-12),
)


def _raw_units(params, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover raw latent coordinates and coefficient vectors from fitted parameters.
    :param params: Fitted dynamic latent state-space parameter object.
    :param xs, numpy.ndarray: Latent state samples to transform.
    :return: tuple[numpy.ndarray, numpy.ndarray]: Raw coordinates and flattened coefficients.
    """
    """Map standardized-coordinate coefficients back to DGP raw units."""
    b_raw = params.B * xs[None, None, :, None] / xs[None, None, None, :]
    a0_raw = params.A0 * xs[None, :, None] / xs[None, None, :]
    return a0_raw, b_raw


def _fit(x: np.ndarray, u: np.ndarray, support: np.ndarray, seed: int):
    """
    Fit the positive-control model under a prescribed causal support.
    :param x, numpy.ndarray: Observed response series.
    :param u, numpy.ndarray: Exogenous covariate series.
    :param support, numpy.ndarray: Boolean mask of admissible lagged coefficients.
    :param seed, int: Random seed used by the optimizer.
    :return: fitted model: Fitted dynamic latent state-space model.
    """
    model = DLCSSM(
        lag=2,
        state_dim=2,
        tier_b_multiplier=2.0,
        lambda_group=1e-4,
        lambda_B=0.0,
        lambda_ridge=1e-6,
        max_iter=300,
        min_iter=20,
        tol=1e-7,
        prox_iterations=300,
        jitter=1e-6,
        diagonal_R=True,
        canonicalize=True,
        seed=seed,
    )
    fit = model.fit(x, u, support)
    return model, fit


def main() -> None:
    """
    Run the S5 positive-control experiment and write its diagnostic artifact.
    :return: None.
    """
    records = []
    for seed in SEEDS:
        case = BENCH / f"s5_base/seed_{seed}"
        with np.load(case / "truth.npz", allow_pickle=False) as archive:
            x_raw = archive["x"]
            u_raw = archive["u"]
            operators = archive["operators"]
            true_support = archive["support"].astype(bool)
            b_true = archive["params__B"]

        nt = TRAIN_COUNT
        x_mean, x_scale = x_raw[:nt].mean(0), x_raw[:nt].std(0)
        u_mean, u_scale = u_raw[:nt].mean(0), u_raw[:nt].std(0)
        x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        u_scale = np.where(u_scale > 1e-12, u_scale, 1.0)
        x = (x_raw - x_mean) / x_scale
        u = (u_raw - u_mean) / u_scale

        # Oracle support: the true generating support at lag 2.
        screened_result = discover_support(x[:nt], dict(SUPPORT_CFG, lag=2, seed=seed))

        for support_label, support in (
            ("oracle", true_support),
            ("screened", screened_result.support.astype(bool)),
        ):
            row: dict[str, object] = {
                "seed": seed,
                "support": support_label,
                "support_edges": int(support.sum()),
            }
            try:
                model, fit = _fit(x[:nt], u[:nt], support, seed)
                params = model.params_
                a0_raw, b_raw = _raw_units(params, x_scale)
                row["termination_reason"] = str(fit.loglik_history and fit.termination_reason)
                row["em_iterations"] = int(len(fit.loglik_history))
                try:
                    subspace = operator_subspace_metrics(b_true, b_raw)
                    row["principal_angle_max_deg"] = float(
                        np.degrees(subspace["principal_angle_max"])
                    )
                    row["projection_distance"] = float(subspace["projection_distance"])
                except ValueError as exc:
                    row["subspace_error"] = str(exc)

                forecasts = one_step_forecasts(model, x, u)
                prior_operators = a0_raw[None] + np.einsum(
                    "td,dlij->tlij", forecasts.state_mean, b_raw
                )
                test_mask = np.arange(len(x)) >= nt + VALIDATION_COUNT
                recovery = recovery_metrics(
                    operators[test_mask], prior_operators[test_mask], support, HORIZON
                )
                row["dynamic_coefficient_rmse"] = float(recovery["dynamic_coefficient_rmse"])
                row["frozen_kernel_rmse"] = float(recovery["frozen_kernel_rmse"])
                row["observable_frozen_peak_gain_rmse"] = float(
                    recovery["observable_frozen_peak_gain_rmse"]
                )
                row["transmitter_rank_spearman"] = recovery["transmitter_rank_spearman"]
                row["edge_precision"] = recovery["edge_precision"]
                row["edge_recall"] = recovery["edge_recall"]
            except Exception as exc:  # noqa: BLE001 - record every failure honestly.
                row["error"] = f"{type(exc).__name__}: {exc}"
            records.append(row)
            print(json.dumps(row))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
