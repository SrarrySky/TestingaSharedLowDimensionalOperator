"""Command-line entry point for data preparation and paper experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from causal.causal_support import discover_from_config, load_support, save_support
from core.config import load_yaml
from core.data import download_from_tushare, load_prepared_dataset, prepare_dataset
from experiments.synthetic import generate_scenario
from model.dlcssm import DLCSSM, DLCSSMParameters
from model.geometry import low_rank_diagnostics, matrix_power_derivative
from model.propagation import (
    frozen_pathwise_residual,
)

app = typer.Typer(help="DLC-SSM risk-geometry research CLI.", no_args_is_help=True)


@app.command()
def download(config: Path = typer.Option(Path("config/data.yaml"), exists=True)) -> None:
    """
    Download.
    :param config: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    paths = download_from_tushare(config)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def prepare(config: Path = typer.Option(Path("config/data.yaml"), exists=True)) -> None:
    """
    Prepare.
    :param config: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    ds = prepare_dataset(config)
    typer.echo(
        f"Prepared T={len(ds.x)}, N={ds.x.shape[1]}, q={ds.u.shape[1]} "
        f"with splits train={np.sum(ds.split=='train')}, "
        f"validation={np.sum(ds.split=='validation')}, test={np.sum(ds.split=='test')}"
    )


@app.command("causal")
def causal_command(
    data: Path = typer.Option(
        Path("data/processed/dataset.npz"),
        exists=True,
    ),
    config: Path = typer.Option(
        Path("config/causal.yaml"),
        exists=True,
    ),
    output: Path = typer.Option(Path("outputs/checkpoints/causal_support.npz")),
    jobs: int | None = typer.Option(
        None,
        "--jobs",
        "-j",
        help=(
            "Parallel causal-screening workers. " "1=serial, -1=all CPUs. " "Overrides config/causal.yaml."
        ),
    ),
) -> None:
    """
    Causal command.
    :param data: Input argument used by this operation.
    :param config: Input argument used by this operation.
    :param output: Input argument used by this operation.
    :param jobs: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    ds = load_prepared_dataset(data)
    train = ds.split == "train"
    result = discover_from_config(
        ds.x[train],
        config,
        n_jobs=jobs,
    )
    save_support(result, output, ds.sector_names)
    typer.echo(
        f"Saved {result.method} support: {result.support.sum()} directed lag coefficients -> {output}"
    )


@app.command()
def fit(
    data: Path = typer.Option(Path("data/processed/dataset.npz"), exists=True),
    model_config: Path = typer.Option(Path("config/dlcssm.yaml"), exists=True),
    support: Path | None = typer.Option(None),
    output: Path = typer.Option(Path("outputs/checkpoints/dlcssm_train.npz")),
) -> None:
    """
    Fit.
    :param data: Input argument used by this operation.
    :param model_config: Input argument used by this operation.
    :param support: Input argument used by this operation.
    :param output: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    ds = load_prepared_dataset(data)
    cfg = load_yaml(model_config)
    train = ds.split == "train"
    x, u = ds.x[train], ds.u[train]
    lag = int(cfg["lag"])

    if support is None:
        support_array = np.ones((lag, x.shape[1], x.shape[1]), dtype=bool)
        support_method = "unrestricted"
    else:
        s = load_support(support)
        support_array = s
        support_method = s.method

    model = DLCSSM(
        lag=lag,
        state_dim=int(cfg["state_dimension"]),
        tier_b_multiplier=float(cfg["penalty"].get("tier_b_multiplier", 2.0)),
        lambda_group=float(cfg["penalty"]["group"]),
        lambda_B=float(cfg["penalty"]["state_l1"]),
        lambda_ridge=float(cfg["penalty"]["ridge"]),
        max_iter=int(cfg["fit"]["max_em_iterations"]),
        min_iter=int(cfg["fit"]["min_em_iterations"]),
        tol=float(cfg["fit"]["tolerance"]),
        prox_iterations=int(cfg["fit"]["prox_iterations"]),
        jitter=float(cfg["fit"]["covariance_jitter"]),
        diagonal_R=bool(cfg["fit"]["diagonal_observation_covariance"]),
        canonicalize=bool(cfg["fit"]["canonicalize_after_fit"]),
        seed=int(cfg["seed"]),
    )
    result = model.fit(x, u, support_array)
    model.save(
        output,
        metadata={
            "fit_split": "train",
            "support_method": support_method,
            "max_spectral_radius": result.max_spectral_radius,
            "loglik_history": result.loglik_history,
        },
    )
    typer.echo(
        f"Fit complete: iterations={len(result.loglik_history) - 1}, "
        f"termination={result.termination_reason}, "
        f"penalized objective={result.penalized_objective_history[-1]:.6f}, "
        f"max sampled spectral radius={result.max_spectral_radius:.4f}. Saved -> {output}"
    )


@app.command("tune")
def tune(
    data: Path = typer.Option(Path("data/processed/dataset.npz"), exists=True),
    model_config: Path = typer.Option(Path("config/dlcssm.yaml"), exists=True),
    causal_config: Path = typer.Option(Path("config/causal.yaml"), exists=True),
    output_dir: Path = typer.Option(Path("outputs/validation_tuning")),
    jobs: int | None = typer.Option(None, "--jobs", "-j"),
) -> None:
    """Select hyperparameters on Validation, fitting candidates on Train only."""
    from experiments.validation import run_tuning

    run_tuning(data, model_config, causal_config, output_dir, jobs)
    typer.echo(f"Validation selection complete -> {output_dir}")


@app.command()
def synthetic(
    scenario: str = typer.Option("shared_low_rank"),
    output: Path = typer.Option(Path("outputs/checkpoints/synthetic_demo.npz")),
    seed: int = typer.Option(7),
) -> None:
    """
    Synthetic.
    :param scenario: Input argument used by this operation.
    :param output: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    ds = generate_scenario(scenario, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": ds.x,
        "u": ds.u,
        "z": ds.z,
        "a_path": ds.a_path,
        "support": ds.support.astype(np.uint8),
        "scenario": np.asarray(ds.scenario),
    }
    # Persist the complete ground truth needed for structural and geometry
    # evaluation; older consumers can continue reading the original keys.
    if ds.operators is not None:
        payload["operators"] = ds.operators
    if ds.factors is not None:
        payload["factors"] = ds.factors
    if ds.regimes is not None:
        payload["regimes"] = ds.regimes
    if ds.observed_mask is not None:
        payload["observed_mask"] = ds.observed_mask.astype(np.uint8)
    for name, value in ds.metadata.items():
        if isinstance(value, np.ndarray):
            payload[f"truth_{name}"] = value
    np.savez_compressed(output, **payload)
    typer.echo(f"Synthetic {scenario}: x={ds.x.shape}, a_path={ds.a_path.shape} -> {output}")


@app.command("theory-checks")
def theory_checks(seed: int = typer.Option(11)) -> None:
    """
    Theory checks.
    :param seed: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    rng = np.random.default_rng(seed)

    # 1) Gauge invariance.
    n, L, d, q = 3, 2, 2, 1
    p = DLCSSMParameters(
        c=rng.normal(size=n),
        A0=rng.normal(scale=0.05, size=(L, n, n)),
        B=rng.normal(scale=0.03, size=(d, L, n, n)),
        Lambda=rng.normal(size=(n, d)),
        D=rng.normal(size=(n, q)),
        F=0.7 * np.eye(d),
        G=rng.normal(size=(d, q)),
        Q=np.eye(d),
        R=np.eye(n),
        m0=np.zeros(d),
        P0=np.eye(d),
    )
    Rg = np.array([[1.2, 0.3], [-0.2, 0.9]])
    z = rng.normal(size=d)
    z_star = Rg @ z
    err_gauge = np.linalg.norm(p.a_lags(z) - p.apply_gauge(Rg).a_lags(z_star))

    # 2) Low rank.
    Bmat = rng.normal(size=(20, d))
    states = rng.normal(size=(500, d))
    apath = states @ Bmat.T
    diag = low_rank_diagnostics(apath, d)

    # 3) Matrix-power derivative.
    C = rng.normal(scale=0.1, size=(5, 5))
    Cr = rng.normal(scale=0.05, size=(5, 5))
    h = 4
    eps = 1e-6
    analytic = matrix_power_derivative(C, Cr, h)
    numeric = (
        np.linalg.matrix_power(C + eps * Cr, h) - np.linalg.matrix_power(C - eps * Cr, h)
    ) / (2 * eps)
    err_deriv = np.linalg.norm(analytic - numeric)

    # 4) Exact telescoping.
    C0 = rng.normal(scale=0.1, size=(4, 4))
    path = [C0 + rng.normal(scale=0.01, size=(4, 4)) for _ in range(5)]
    err_tel = frozen_pathwise_residual(C0, path)

    result = {
        "gauge_invariance_error": err_gauge,
        "low_rank_explained_ratio_d": diag.explained_ratio_d,
        "sensitivity_finite_difference_error": err_deriv,
        "frozen_pathwise_residual": err_tel,
    }
    typer.echo(json.dumps(result, indent=2))
    if err_gauge > 1e-10 or err_deriv > 1e-7 or err_tel > 1e-10:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
