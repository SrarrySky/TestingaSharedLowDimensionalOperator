"""Causal-support construction and diagnostics for lagged dynamic models."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import Lasso

from core.config import load_yaml


@dataclass(frozen=True)
class SupportResult:
    support: np.ndarray
    frequency: np.ndarray
    tier: np.ndarray
    method: str

    # Orientation-specific stability frequencies.
    tier_a_frequency: np.ndarray | None = None
    tier_b_frequency: np.ndarray | None = None
    tier_c_frequency: np.ndarray | None = None
    tier_d_frequency: np.ndarray | None = None

    # Publication/audit provenance.
    global_frequency: np.ndarray | None = None
    context_frequency: np.ndarray | None = None
    context_names: tuple[str, ...] = ()

    metadata: dict[str, Any] | None = None


BootstrapResult = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _moving_block_sample(
    x: np.ndarray, block_length: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
     moving block sample.
    :param x: Input argument used by this operation.
    :param block_length: Input argument used by this operation.
    :param rng: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    T = len(x)
    blocks = int(np.ceil(T / block_length))
    pieces = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for _ in range(blocks):
        start = int(rng.integers(0, max(1, T - block_length + 1)))
        stop = min(start + block_length, T)
        piece = x[start:stop]
        pieces.append(piece)
        ranges.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    out = np.concatenate(pieces, axis=0)[:T]
    trimmed: list[tuple[int, int]] = []
    for a, b in ranges:
        if a >= T:
            break
        trimmed.append((a, min(b, T)))
    return out, trimmed


def _lag_design_with_segments(
    x: np.ndarray, lag: int, segments: list[tuple[int, int]] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
     lag design with segments.
    :param x: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param segments: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if segments is None:
        segments = [(0, len(x))]
    Xrows, Yrows = [], []
    for start, stop in segments:
        for t in range(start + lag, stop):
            Xrows.append(np.concatenate([x[t - ell] for ell in range(1, lag + 1)]))
            Yrows.append(x[t])
    if not Xrows:
        raise ValueError("No valid lagged rows; increase sample size or reduce lag/block length")
    return np.asarray(Xrows), np.asarray(Yrows)


def linear_lasso_stability(
    x: np.ndarray,
    lag: int,
    bootstrap_count: int,
    block_length: int,
    alpha: float,
    threshold: float,
    seed: int,
    max_iter: int = 10000,
    coef_zero_tolerance: float = 1e-12,
) -> SupportResult:
    """Conditional lagged predictive-direction screen via multivariate Lasso stability."""
    T, n = x.shape
    if block_length <= lag:
        raise ValueError("block_length must be larger than lag")
    rng = np.random.default_rng(seed)
    counts = np.zeros((lag, n, n), dtype=float)

    for _ in range(bootstrap_count):
        xb, segments = _moving_block_sample(x, block_length, rng)
        X, Y = _lag_design_with_segments(xb, lag, segments)
        for target in range(n):
            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter, random_state=seed)
            model.fit(X, Y[:, target])
            coef = model.coef_.reshape(lag, n)
            counts[:, target, :] += np.abs(coef) > coef_zero_tolerance

    freq = counts / bootstrap_count
    support = freq >= threshold
    tier = np.where(support, "predictive", "none").astype("U16")
    return SupportResult(support, freq, tier, "linear_lasso")


def _bh_adjust_lagged_pvalues(
    p_matrix: np.ndarray,
    *,
    tau_min: int,
    tau_max: int,
) -> np.ndarray:
    """Benjamini-Hochberg-adjust lagged LPCMCI p-values.

    This local helper intentionally avoids "LPCMCI.get_corrected_pvalues".
    Some Tigramite releases expose that inherited method even though LPCMCI
    lacks an internal helper that it calls, which raises "AttributeError".

    The correction family contains all source-target hypotheses at lags
    "tau_min" through "tau_max". In the publication path "tau_min >= 1",
    so contemporaneous hypotheses are excluded by construction.
    """
    from scipy.stats import false_discovery_control

    p = np.asarray(p_matrix, dtype=float)

    if p.ndim != 3:
        raise ValueError("p_matrix must have shape (source, target, lag)")

    n_source, n_target, n_lags = p.shape
    if n_source != n_target:
        raise ValueError("p_matrix must be square in its first two dimensions")

    if not (0 <= tau_min <= tau_max < n_lags):
        raise ValueError("tau_min/tau_max are inconsistent with p_matrix.shape")

    selected = p[:, :, tau_min : tau_max + 1]
    flat = selected.reshape(-1)

    if not np.all(np.isfinite(flat)):
        raise ValueError("LPCMCI p-values used for BH correction must be finite")

    if np.any((flat < 0.0) | (flat > 1.0)):
        raise ValueError("LPCMCI p-values must lie in [0, 1]")

    q = np.array(p, dtype=float, copy=True)

    if flat.size:
        adjusted = false_discovery_control(
            flat,
            method="bh",
        ).reshape(selected.shape)
        q[:, :, tau_min : tau_max + 1] = adjusted

    return q


def _classify_lpcmci_mark(
    mark: str,
    tier_a_marks: tuple[str, ...],
    tier_b_marks: tuple[str, ...],
    tier_c_marks: tuple[str, ...],
    tier_d_marks: tuple[str, ...],
) -> str:
    """Map LPCMCI endpoint marks to predeclared publication tiers."""
    if mark in tier_a_marks:
        return "A"
    if mark in tier_b_marks:
        return "B"
    if mark in tier_c_marks:
        return "C"
    if mark in tier_d_marks:
        return "D"
    if mark:
        # Any unknown/conflicting nonempty mark is diagnostic only.
        return "D"

    return "none"


def _lpcmci_once(
    x: np.ndarray,
    lag: int,
    pc_alpha: float,
    *,
    tau_min: int,
    ci_test: str,
    significance: str,
    fdr_method: str,
    fdr_alpha: float,
    block_length: int,
    bootstrap_seed: int,
    tier_a_marks: tuple[str, ...],
    tier_b_marks: tuple[str, ...],
    tier_c_marks: tuple[str, ...],
    tier_d_marks: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run LPCMCI on one genuine moving-block bootstrap draw."""
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.lpcmci import LPCMCI

    except ImportError as exc:
        raise RuntimeError(
            "LPCMCI requires the optional dependency: " "`pip install -e '.[causal]'`."
        ) from exc

    if ci_test.lower() != "parcorr":
        raise ValueError(
            f"Unsupported lpcmci.ci_test={ci_test!r}. "
            "The publication path currently implements only 'parcorr'."
        )

    n = x.shape[1]

    dataframe = pp.DataFrame(
        x,
        var_names=[f"x{i}" for i in range(n)],
    )

    # ------------------------------------------------------------
    # TRUE MOVING-BLOCK BOOTSTRAP
    # This is the same DataFrame bootstrap mechanism used internally by Tigramite's run_bootstrap_of().
    # ------------------------------------------------------------
    dataframe.bootstrap = {
        "boot_blocklength": int(block_length),
        "random_state": np.random.default_rng(int(bootstrap_seed)),
    }
    ci = ParCorr(
        significance=significance,
    )
    method = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=ci,
        verbosity=0,
    )
    result = method.run_lpcmci(
        tau_min=tau_min,
        tau_max=lag,
        pc_alpha=pc_alpha,
    )
    graph = result["graph"]

    # ------------------------------------------------------------
    # BH/FDR gate.
    #
    # Do not call LPCMCI.get_corrected_pvalues() here. Some
    # Tigramite releases expose that inherited method while LPCMCI
    # lacks an internal helper it calls, causing AttributeError.
    # Apply standard BH adjustment directly to this replicate's
    # lagged p-value family instead.
    # ------------------------------------------------------------
    fdr_name = str(fdr_method).strip().lower()

    if fdr_name in {"", "none"}:
        q_matrix = None
    elif fdr_name == "fdr_bh":
        q_matrix = _bh_adjust_lagged_pvalues(
            result["p_matrix"],
            tau_min=tau_min,
            tau_max=lag,
        )
    else:
        raise ValueError(
            f"Unsupported LPCMCI FDR method: {fdr_method!r}. "
            "Supported values are 'none' and 'fdr_bh'."
        )

    masks = {
        tier: np.zeros(
            (lag, n, n),
            dtype=bool,
        )
        for tier in (
            "A",
            "B",
            "C",
            "D",
        )
    }

    for ell in range(
        tau_min,
        lag + 1,
    ):
        for source in range(n):
            for target in range(n):
                if (
                    q_matrix is not None
                    and q_matrix[
                        source,
                        target,
                        ell,
                    ]
                    > fdr_alpha
                ):
                    continue
                mark = str(
                    graph[
                        source,
                        target,
                        ell,
                    ]
                )
                tier = _classify_lpcmci_mark(
                    mark,
                    tier_a_marks,
                    tier_b_marks,
                    tier_c_marks,
                    tier_d_marks,
                )
                if tier in masks:
                    # Internal convention: [lag, target, source]
                    masks[tier][
                        ell - 1,
                        target,
                        source,
                    ] = True

    return (
        masks["A"],
        masks["B"],
        masks["C"],
        masks["D"],
    )


def _lpcmci_bootstrap_worker(
    x: np.ndarray,
    bootstrap_seed: int,
    lag: int,
    block_length: int,
    pc_alpha: float,
    tau_min: int,
    ci_test: str,
    significance: str,
    fdr_method: str,
    fdr_alpha: float,
    tier_a_marks: tuple[str, ...],
    tier_b_marks: tuple[str, ...],
    tier_c_marks: tuple[str, ...],
    tier_d_marks: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """One moving-block-bootstrap LPCMCI replicate."""
    return _lpcmci_once(
        x,
        lag,
        pc_alpha,
        tau_min=tau_min,
        ci_test=ci_test,
        significance=significance,
        fdr_method=fdr_method,
        fdr_alpha=fdr_alpha,
        block_length=block_length,
        bootstrap_seed=bootstrap_seed,
        tier_a_marks=tier_a_marks,
        tier_b_marks=tier_b_marks,
        tier_c_marks=tier_c_marks,
        tier_d_marks=tier_d_marks,
    )


def _lpcmci_context_bootstrap_worker(
    context_index: int,
    bootstrap_index: int,
    bootstrap_seed: int,
    *,
    x: np.ndarray,
    **worker_kwargs: Any,
) -> tuple[int, int, float, BootstrapResult]:
    """Run one bootstrap task and return its context/replicate coordinates."""
    started = time.perf_counter()
    result = _lpcmci_bootstrap_worker(
        x=x,
        bootstrap_seed=bootstrap_seed,
        **worker_kwargs,
    )
    return context_index, bootstrap_index, time.perf_counter() - started, result


def _bootstrap_seeds(seed: int, bootstrap_count: int) -> list[int]:
    """Create deterministic per-bootstrap seeds for one screening context."""
    children = np.random.SeedSequence(seed).spawn(bootstrap_count)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def _aggregate_lpcmci_results(
    results: list[BootstrapResult],
    lag: int,
    n: int,
    threshold: float,
    include_tier_b: bool,
) -> SupportResult:
    """Aggregate ordered LPCMCI bootstrap results into a support result."""
    if not results:
        raise ValueError("results must contain at least one LPCMCI bootstrap result")

    tier_a_count = np.zeros((lag, n, n), dtype=float)
    tier_b_count = np.zeros_like(tier_a_count)
    tier_c_count = np.zeros_like(tier_a_count)
    tier_d_count = np.zeros_like(tier_a_count)

    for a, b, c, d in results:
        tier_a_count += a
        tier_b_count += b
        tier_c_count += c
        tier_d_count += d

    bootstrap_count = len(results)
    a_freq = tier_a_count / bootstrap_count
    b_freq = tier_b_count / bootstrap_count
    c_freq = tier_c_count / bootstrap_count
    d_freq = tier_d_count / bootstrap_count

    # Exact publication stability semantics:
    # Tier A means "-->" itself is stable.
    # Tier B means "o->" itself is stable.
    # Do not pool A and B before applying the stability threshold.
    a_stable = a_freq >= threshold
    b_stable = b_freq >= threshold

    if include_tier_b:
        support = a_stable | b_stable
        frequency = np.maximum(a_freq, b_freq)
    else:
        support = a_stable.copy()
        frequency = a_freq.copy()

    tier = np.full((lag, n, n), "none", dtype="U16")

    if include_tier_b:
        tier[b_stable] = "B"

    # Tier A has precedence if both marks cross the stability threshold.
    tier[a_stable] = "A"

    return SupportResult(
        support=support,
        frequency=frequency,
        tier=tier,
        method="lpcmci",
        tier_a_frequency=a_freq,
        tier_b_frequency=b_freq,
        tier_c_frequency=c_freq,
        tier_d_frequency=d_freq,
    )


def _shared_lpcmci_stability(
    contexts: list[np.ndarray],
    context_names: list[str],
    bootstrap_count: int,
    lag: int,
    block_length: int,
    pc_alpha: float,
    threshold: float,
    seeds: list[list[int]],
    *,
    tau_min: int,
    ci_test: str,
    significance: str,
    fdr_method: str,
    fdr_alpha: float,
    tier_a_marks: tuple[str, ...],
    tier_b_marks: tuple[str, ...],
    tier_c_marks: tuple[str, ...],
    tier_d_marks: tuple[str, ...],
    include_tier_b: bool,
    n_jobs: int,
    backend: str,
    inner_max_num_threads: int,
    parallel_verbose: int,
) -> list[SupportResult]:
    """Run all context/bootstrap tasks through one shared process pool.

    Tasks are submitted round-robin by bootstrap index.  This is important for
    uneven LPCMCI runtimes: a slow replicate in one context cannot prevent the
    other contexts from using the available workers.
    """
    if not contexts:
        raise ValueError("contexts must contain at least one array")
    if len(seeds) != len(contexts) or any(
        len(context_seeds) != bootstrap_count for context_seeds in seeds
    ):
        raise ValueError("seeds must contain one bootstrap seed list per context")
    if bootstrap_count < 1:
        raise ValueError("bootstrap_count must be >= 1")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if len(context_names) != len(contexts):
        raise ValueError("context_names must contain one name per context")

    n = contexts[0].shape[1]
    for context in contexts:
        if context.ndim != 2 or context.shape[1] != n:
            raise ValueError("all contexts must be two-dimensional with equal width")
        if block_length <= lag:
            raise ValueError("block_length must be larger than lag")
        if 2 * block_length > len(context):
            raise ValueError(
                "LPCMCI moving-block bootstrap needs at least two blocks " "in every context"
            )

    worker_kwargs = dict(
        lag=lag,
        block_length=block_length,
        pc_alpha=pc_alpha,
        tau_min=tau_min,
        ci_test=ci_test,
        significance=significance,
        fdr_method=fdr_method,
        fdr_alpha=fdr_alpha,
        tier_a_marks=tier_a_marks,
        tier_b_marks=tier_b_marks,
        tier_c_marks=tier_c_marks,
        tier_d_marks=tier_d_marks,
    )

    # Round-robin ordering ensures that the initial dispatch contains every
    # context instead of filling the dispatch window with global-context work.
    tasks = [
        (
            context_index,
            bootstrap_index,
            seeds[context_index][bootstrap_index],
        )
        for bootstrap_index in range(bootstrap_count)
        for context_index in range(len(contexts))
    ]
    total_tasks = len(tasks)

    config_kwargs: dict[str, Any] = {"backend": backend, "n_jobs": n_jobs}
    if backend == "loky":
        config_kwargs["inner_max_num_threads"] = inner_max_num_threads

    completed = 0
    ordered: list[list[BootstrapResult | None]] = [[None] * bootstrap_count for _ in contexts]

    if n_jobs == 1:
        task_results = (
            _lpcmci_context_bootstrap_worker(
                *task,
                x=contexts[task[0]],
                **worker_kwargs,
            )
            for task in tasks
        )
        for context_index, bootstrap_index, elapsed, result in task_results:
            ordered[context_index][bootstrap_index] = result
            completed += 1
            print(
                f"[causal] shared {completed}/{total_tasks} | "
                f"{context_names[context_index]} bootstrap "
                f"{bootstrap_index + 1}/{bootstrap_count} | "
                f"{elapsed:.1f}s",
                flush=True,
            )
    else:
        parallel_kwargs: dict[str, Any] = {
            "batch_size": 1,
            "verbose": parallel_verbose,
        }
        # "return_as" was added after joblib 1.3.  The list fallback keeps
        # the declared dependency range working, while recent joblib versions
        # can report tasks as soon as they complete.
        if "return_as" in signature(Parallel).parameters:
            parallel_kwargs["return_as"] = "generator_unordered"

        with parallel_config(**config_kwargs):
            task_results = Parallel(**parallel_kwargs)(
                delayed(_lpcmci_context_bootstrap_worker)(
                    task[0],
                    task[1],
                    task[2],
                    x=contexts[task[0]],
                    **worker_kwargs,
                )
                for task in tasks
            )
            for context_index, bootstrap_index, elapsed, result in task_results:
                ordered[context_index][bootstrap_index] = result
                completed += 1
                print(
                    f"[causal] shared {completed}/{total_tasks} | "
                    f"{context_names[context_index]} bootstrap "
                    f"{bootstrap_index + 1}/{bootstrap_count} | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

    results: list[SupportResult] = []
    for context_results in ordered:
        if any(result is None for result in context_results):
            raise RuntimeError("shared LPCMCI scheduling returned incomplete results")
        results.append(
            _aggregate_lpcmci_results(
                [result for result in context_results if result is not None],
                lag,
                n,
                threshold,
                include_tier_b,
            )
        )
    return results


def lpcmci_stability(
    x: np.ndarray,
    lag: int,
    bootstrap_count: int,
    block_length: int,
    pc_alpha: float,
    threshold: float,
    seed: int,
    *,
    tau_min: int = 1,
    ci_test: str = "parcorr",
    significance: str = "analytic",
    fdr_method: str = "fdr_bh",
    fdr_alpha: float = 0.05,
    tier_a_marks: tuple[str, ...] = ("-->",),
    tier_b_marks: tuple[str, ...] = ("o->",),
    tier_c_marks: tuple[str, ...] = ("<->",),
    tier_d_marks: tuple[str, ...] = ("o-o",),
    include_tier_b: bool = True,
    n_jobs: int = 1,
    backend: str = "loky",
    inner_max_num_threads: int = 1,
    parallel_verbose: int = 0,
) -> SupportResult:
    """
    Lpcmci stability.
    :param x: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param bootstrap_count: Input argument used by this operation.
    :param block_length: Input argument used by this operation.
    :param pc_alpha: Input argument used by this operation.
    :param threshold: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :param tau_min: Input argument used by this operation.
    :param ci_test: Input argument used by this operation.
    :param significance: Input argument used by this operation.
    :param fdr_method: Input argument used by this operation.
    :param fdr_alpha: Input argument used by this operation.
    :param tier_a_marks: Input argument used by this operation.
    :param tier_b_marks: Input argument used by this operation.
    :param tier_c_marks: Input argument used by this operation.
    :param tier_d_marks: Input argument used by this operation.
    :param include_tier_b: Input argument used by this operation.
    :param n_jobs: Input argument used by this operation.
    :param backend: Input argument used by this operation.
    :param inner_max_num_threads: Input argument used by this operation.
    :param parallel_verbose: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    T, n = x.shape

    if bootstrap_count < 1:
        raise ValueError("bootstrap_count must be >= 1")
    if block_length <= lag:
        raise ValueError("block_length must be larger than lag")
    if 2 * block_length > T:
        raise ValueError("LPCMCI moving-block bootstrap " "needs at least two blocks")
    if not (1 <= tau_min <= lag):
        raise ValueError("tau_min must satisfy " "1 <= tau_min <= lag")
    if not (0 <= pc_alpha <= 1 and 0 <= fdr_alpha <= 1):
        raise ValueError("pc_alpha and fdr_alpha " "must lie in [0, 1]")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be finite and lie in (0, 1]")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    # ------------------------------------------------------------
    # Deterministic child seeds.
    # Parallel scheduling cannot change the bootstrap samples.
    # ------------------------------------------------------------
    bootstrap_seeds = _bootstrap_seeds(seed, bootstrap_count)

    worker_kwargs = dict(
        x=x,
        lag=lag,
        block_length=block_length,
        pc_alpha=pc_alpha,
        tau_min=tau_min,
        ci_test=ci_test,
        significance=significance,
        fdr_method=fdr_method,
        fdr_alpha=fdr_alpha,
        tier_a_marks=tier_a_marks,
        tier_b_marks=tier_b_marks,
        tier_c_marks=tier_c_marks,
        tier_d_marks=tier_d_marks,
    )

    if n_jobs == 1:
        results = [
            _lpcmci_bootstrap_worker(
                bootstrap_seed=bs,
                **worker_kwargs,
            )
            for bs in bootstrap_seeds
        ]
    else:
        config_kwargs: dict[
            str,
            Any,
        ] = {
            "backend": backend,
            "n_jobs": n_jobs,
        }

        if backend == "loky":
            config_kwargs["inner_max_num_threads"] = inner_max_num_threads

        with parallel_config(**config_kwargs):
            results = Parallel(
                batch_size=1,
                verbose=parallel_verbose,
            )(
                delayed(_lpcmci_bootstrap_worker)(
                    bootstrap_seed=bs,
                    **worker_kwargs,
                )
                for bs in bootstrap_seeds
            )

    return _aggregate_lpcmci_results(
        results,
        lag,
        n,
        threshold,
        include_tier_b,
    )


def _pcmciplus_once(
    x: np.ndarray,
    lag: int,
    pc_alpha: float,
    *,
    tau_min: int,
    ci_test: str,
    significance: str,
    fdr_method: str,
    fdr_alpha: float,
    block_length: int,
    bootstrap_seed: int,
    tier_a_marks: tuple[str, ...],
    tier_b_marks: tuple[str, ...],
    tier_c_marks: tuple[str, ...],
    tier_d_marks: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run PCMCI+ on one genuine moving-block bootstrap draw.

    PCMCI+ ("PCMCI.run_pcmciplus") is the no-latent-confounder sibling of
    LPCMCI: it shares ParCorr, the same "graph"/"p_matrix" layout, and the
    same endpoint-mark vocabulary, but skips the latent-confounder search.  The
    downstream FDR gate and A/B/C/D tier classification are therefore identical
    to the LPCMCI branch.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

    except ImportError as exc:
        raise RuntimeError(
            "PCMCI+ requires the optional dependency: " "`pip install -e '.[causal]'`."
        ) from exc

    if ci_test.lower() != "parcorr":
        raise ValueError(
            f"Unsupported pcmciplus.ci_test={ci_test!r}. "
            "The publication path currently implements only 'parcorr'."
        )

    n = x.shape[1]

    dataframe = pp.DataFrame(
        x,
        var_names=[f"x{i}" for i in range(n)],
    )

    # TRUE MOVING-BLOCK BOOTSTRAP (same DataFrame bootstrap mechanism as LPCMCI).
    dataframe.bootstrap = {
        "boot_blocklength": int(block_length),
        "random_state": np.random.default_rng(int(bootstrap_seed)),
    }
    ci = ParCorr(
        significance=significance,
    )
    method = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ci,
        verbosity=0,
    )
    result = method.run_pcmciplus(
        tau_min=tau_min,
        tau_max=lag,
        pc_alpha=pc_alpha,
    )
    graph = result["graph"]

    # BH/FDR gate.  Same local helper as LPCMCI: apply BH directly to this
    # replicate's lagged p-value family instead of Tigramite's FDR wrapper.
    fdr_name = str(fdr_method).strip().lower()

    if fdr_name in {"", "none"}:
        q_matrix = None
    elif fdr_name == "fdr_bh":
        q_matrix = _bh_adjust_lagged_pvalues(
            result["p_matrix"],
            tau_min=tau_min,
            tau_max=lag,
        )
    else:
        raise ValueError(
            f"Unsupported PCMCI+ FDR method: {fdr_method!r}. "
            "Supported values are 'none' and 'fdr_bh'."
        )

    masks = {
        tier: np.zeros(
            (lag, n, n),
            dtype=bool,
        )
        for tier in (
            "A",
            "B",
            "C",
            "D",
        )
    }

    for ell in range(
        tau_min,
        lag + 1,
    ):
        for source in range(n):
            for target in range(n):
                if (
                    q_matrix is not None
                    and q_matrix[
                        source,
                        target,
                        ell,
                    ]
                    > fdr_alpha
                ):
                    continue
                mark = str(
                    graph[
                        source,
                        target,
                        ell,
                    ]
                )
                tier = _classify_lpcmci_mark(
                    mark,
                    tier_a_marks,
                    tier_b_marks,
                    tier_c_marks,
                    tier_d_marks,
                )
                if tier in masks:
                    # Internal convention: [lag, target, source]
                    masks[tier][
                        ell - 1,
                        target,
                        source,
                    ] = True

    return (
        masks["A"],
        masks["B"],
        masks["C"],
        masks["D"],
    )


def _pcmciplus_bootstrap_worker(
    x: np.ndarray,
    bootstrap_seed: int,
    lag: int,
    block_length: int,
    pc_alpha: float,
    tau_min: int,
    ci_test: str,
    significance: str,
    fdr_method: str,
    fdr_alpha: float,
    tier_a_marks: tuple[str, ...],
    tier_b_marks: tuple[str, ...],
    tier_c_marks: tuple[str, ...],
    tier_d_marks: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """One moving-block-bootstrap PCMCI+ replicate."""
    return _pcmciplus_once(
        x,
        lag,
        pc_alpha,
        tau_min=tau_min,
        ci_test=ci_test,
        significance=significance,
        fdr_method=fdr_method,
        fdr_alpha=fdr_alpha,
        block_length=block_length,
        bootstrap_seed=bootstrap_seed,
        tier_a_marks=tier_a_marks,
        tier_b_marks=tier_b_marks,
        tier_c_marks=tier_c_marks,
        tier_d_marks=tier_d_marks,
    )


def _aggregate_pcmciplus_results(
    results: list[BootstrapResult],
    lag: int,
    n: int,
    threshold: float,
    include_tier_b: bool,
) -> SupportResult:
    """Aggregate PCMCI+ bootstrap results (same A/B stability semantics as LPCMCI)."""
    return replace(
        _aggregate_lpcmci_results(
            results,
            lag,
            n,
            threshold,
            include_tier_b,
        ),
        method="pcmciplus",
    )


def pcmciplus_stability(
    x: np.ndarray,
    lag: int,
    bootstrap_count: int,
    block_length: int,
    pc_alpha: float,
    threshold: float,
    seed: int,
    *,
    tau_min: int = 1,
    ci_test: str = "parcorr",
    significance: str = "analytic",
    fdr_method: str = "fdr_bh",
    fdr_alpha: float = 0.05,
    tier_a_marks: tuple[str, ...] = ("-->",),
    tier_b_marks: tuple[str, ...] = ("o->",),
    tier_c_marks: tuple[str, ...] = ("<->",),
    tier_d_marks: tuple[str, ...] = ("o-o",),
    include_tier_b: bool = True,
    n_jobs: int = 1,
    backend: str = "loky",
    inner_max_num_threads: int = 1,
    parallel_verbose: int = 0,
) -> SupportResult:
    """
    Pcmciplus stability.
    :param x: Input argument used by this operation.
    :param lag: Input argument used by this operation.
    :param bootstrap_count: Input argument used by this operation.
    :param block_length: Input argument used by this operation.
    :param pc_alpha: Input argument used by this operation.
    :param threshold: Input argument used by this operation.
    :param seed: Input argument used by this operation.
    :param tau_min: Input argument used by this operation.
    :param ci_test: Input argument used by this operation.
    :param significance: Input argument used by this operation.
    :param fdr_method: Input argument used by this operation.
    :param fdr_alpha: Input argument used by this operation.
    :param tier_a_marks: Input argument used by this operation.
    :param tier_b_marks: Input argument used by this operation.
    :param tier_c_marks: Input argument used by this operation.
    :param tier_d_marks: Input argument used by this operation.
    :param include_tier_b: Input argument used by this operation.
    :param n_jobs: Input argument used by this operation.
    :param backend: Input argument used by this operation.
    :param inner_max_num_threads: Input argument used by this operation.
    :param parallel_verbose: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    T, n = x.shape

    if bootstrap_count < 1:
        raise ValueError("bootstrap_count must be >= 1")
    if block_length <= lag:
        raise ValueError("block_length must be larger than lag")
    if 2 * block_length > T:
        raise ValueError("PCMCI+ moving-block bootstrap " "needs at least two blocks")
    if not (1 <= tau_min <= lag):
        raise ValueError("tau_min must satisfy " "1 <= tau_min <= lag")
    if not (0 <= pc_alpha <= 1 and 0 <= fdr_alpha <= 1):
        raise ValueError("pc_alpha and fdr_alpha " "must lie in [0, 1]")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be finite and lie in (0, 1]")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    bootstrap_seeds = _bootstrap_seeds(seed, bootstrap_count)

    worker_kwargs = dict(
        x=x,
        lag=lag,
        block_length=block_length,
        pc_alpha=pc_alpha,
        tau_min=tau_min,
        ci_test=ci_test,
        significance=significance,
        fdr_method=fdr_method,
        fdr_alpha=fdr_alpha,
        tier_a_marks=tier_a_marks,
        tier_b_marks=tier_b_marks,
        tier_c_marks=tier_c_marks,
        tier_d_marks=tier_d_marks,
    )

    if n_jobs == 1:
        results = [
            _pcmciplus_bootstrap_worker(
                bootstrap_seed=bs,
                **worker_kwargs,
            )
            for bs in bootstrap_seeds
        ]
    else:
        config_kwargs: dict[
            str,
            Any,
        ] = {
            "backend": backend,
            "n_jobs": n_jobs,
        }

        if backend == "loky":
            config_kwargs["inner_max_num_threads"] = inner_max_num_threads

        with parallel_config(**config_kwargs):
            results = Parallel(
                batch_size=1,
                verbose=parallel_verbose,
            )(
                delayed(_pcmciplus_bootstrap_worker)(
                    bootstrap_seed=bs,
                    **worker_kwargs,
                )
                for bs in bootstrap_seeds
            )

    return _aggregate_pcmciplus_results(
        results,
        lag,
        n,
        threshold,
        include_tier_b,
    )


def _rolling_contexts(x: np.ndarray, count: int, min_size: int) -> list[np.ndarray]:
    """
     rolling contexts.
    :param x: Input argument used by this operation.
    :param count: Input argument used by this operation.
    :param min_size: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if count <= 1:
        return [x]
    T = len(x)
    width = max(min_size, int(np.ceil(T / count * 1.5)))
    starts = np.linspace(0, max(0, T - width), count, dtype=int)
    return [x[s : min(T, s + width)] for s in starts]


def _merge_union_results(
    results: list[SupportResult],
    method: str,
    *,
    threshold: float | None = None,
) -> SupportResult:
    """
     merge union results.
    :param results: Input argument used by this operation.
    :param method: Input argument used by this operation.
    :param threshold: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if not results:
        raise ValueError("results must contain at least one SupportResult")

    frequency = np.maximum.reduce([r.frequency for r in results])
    # A union is defined on the observed bootstrap frequencies, not on a
    # pre-thresholded boolean mask.  Re-applying the declared union threshold
    # here prevents a context-local implementation detail from changing the
    # final support semantics and makes this helper safe for independently
    # constructed SupportResult objects.
    support = np.logical_or.reduce([r.support for r in results])
    if threshold is not None:
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be finite and in (0, 1]")
        support = frequency >= threshold

    # LPCMCI and PCMCI+ share explicit A/B orientation tiers.
    if method in ("lpcmci", "pcmciplus"):
        tier = np.full(support.shape, "none", dtype="U16")

        any_b = np.logical_or.reduce([r.support & (r.tier == "B") for r in results])

        any_a = np.logical_or.reduce([r.support & (r.tier == "A") for r in results])

        # B first, then A has precedence and upgrades B -> A.
        tier[support & any_b] = "B"
        tier[support & any_a] = "A"

    else:
        # Preserve fallback semantics such as "predictive" for linear_lasso.
        tier = np.full(support.shape, "none", dtype="U16")

        for result in results:
            fill = (tier == "none") & result.support & (result.tier != "none")
            tier[fill] = result.tier[fill]

    def merge_optional_max(attribute: str) -> np.ndarray | None:
        """
        Merge optional max.
        :param attribute: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        values = [getattr(result, attribute) for result in results]

        if all(value is None for value in values):
            return None

        if any(value is None for value in values):
            raise ValueError(
                "Inconsistent union results: " f"{attribute} is missing from some contexts"
            )

        return np.maximum.reduce(values)

    return SupportResult(
        support=support,
        frequency=frequency,
        tier=tier,
        method=method + "_union",
        tier_a_frequency=merge_optional_max("tier_a_frequency"),
        tier_b_frequency=merge_optional_max("tier_b_frequency"),
        tier_c_frequency=merge_optional_max("tier_c_frequency"),
        tier_d_frequency=merge_optional_max("tier_d_frequency"),
    )


def discover_support(
    x_train: np.ndarray,
    config: dict[str, Any],
) -> SupportResult:
    """Train-only support discovery with optional union-context screening."""
    method = str(config["method"]).lower()
    lag = int(config["lag"])
    seed = int(config["seed"])
    B = int(config["bootstrap_count"])
    block = int(config["block_length"])
    stable_threshold = float(config["stability_threshold"])
    union_threshold = float(config["union_screen_threshold"])
    mode = str(config.get("support_mode", "stable")).strip().lower()
    bootstrap_mode = str(config.get("bootstrap_mode", "moving_block")).strip().lower()

    if mode not in {"stable", "union"}:
        raise ValueError("support_mode must be either 'stable' or 'union'")

    if bootstrap_mode != "moving_block":
        raise ValueError("Only bootstrap_mode='moving_block' is implemented")
    parallel_cfg = config.get("parallel", {})
    n_jobs = int(parallel_cfg.get("n_jobs", 1))
    backend = str(parallel_cfg.get("backend", "loky"))
    inner_max_num_threads = int(
        parallel_cfg.get(
            "inner_max_num_threads",
            1,
        )
    )
    parallel_verbose = int(
        parallel_cfg.get(
            "verbose",
            0,
        )
    )
    context_scope = str(parallel_cfg.get("context_scope", "per_context")).strip().lower()
    if context_scope not in {"per_context", "shared"}:
        raise ValueError("parallel.context_scope must be either 'per_context' or 'shared'")
    contexts = [x_train]
    context_names = ["global"]
    if mode == "union":
        rolling = _rolling_contexts(
            x_train,
            int(
                config.get(
                    "rolling_contexts",
                    3,
                )
            ),
            min_size=max(
                block * 5,
                lag + 50,
            ),
        )
        contexts.extend(rolling)
        context_names.extend([f"rolling_{i + 1}" for i in range(len(rolling))])

    # Union applies the SAME threshold to every Train context, including global.
    threshold = union_threshold if mode == "union" else stable_threshold
    for value in (stable_threshold, union_threshold):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("screening thresholds must be finite and in (0, 1]")
    results: list[SupportResult]
    if context_scope == "shared" and method == "lpcmci" and len(contexts) > 1:
        lc = config["lpcmci"]
        print(
            f"[causal] shared context pool | contexts={len(contexts)} | "
            f"bootstrap={B} each | n_jobs={n_jobs}",
            flush=True,
        )
        results = _shared_lpcmci_stability(
            contexts,
            context_names,
            B,
            lag,
            block,
            float(lc["pc_alpha"]),
            threshold,
            [_bootstrap_seeds(seed + idx, B) for idx in range(len(contexts))],
            tau_min=int(
                lc.get(
                    "tau_min",
                    1,
                )
            ),
            ci_test=str(
                lc.get(
                    "ci_test",
                    "parcorr",
                )
            ),
            significance=str(
                lc.get(
                    "significance",
                    "analytic",
                )
            ),
            fdr_method=str(
                lc.get(
                    "fdr_method",
                    "fdr_bh",
                )
            ),
            fdr_alpha=float(
                lc.get(
                    "fdr_alpha",
                    0.05,
                )
            ),
            tier_a_marks=tuple(
                str(v)
                for v in lc.get(
                    "tier_a_marks",
                    ["-->"],
                )
            ),
            tier_b_marks=tuple(
                str(v)
                for v in lc.get(
                    "tier_b_marks",
                    ["o->"],
                )
            ),
            tier_c_marks=tuple(
                str(v)
                for v in lc.get(
                    "tier_c_marks",
                    ["<->"],
                )
            ),
            tier_d_marks=tuple(
                str(v)
                for v in lc.get(
                    "tier_d_marks",
                    ["o-o"],
                )
            ),
            include_tier_b=bool(
                lc.get(
                    "include_tier_b",
                    True,
                )
            ),
            n_jobs=n_jobs,
            backend=backend,
            inner_max_num_threads=inner_max_num_threads,
            parallel_verbose=parallel_verbose,
        )
    else:
        if context_scope == "shared" and method != "lpcmci":
            print(
                "[causal] parallel.context_scope='shared' is only implemented "
                "for method='lpcmci'; using per-context scheduling",
                flush=True,
            )
        results = []
        for idx, context in enumerate(contexts):
            print(
                f"[causal] context {idx + 1}/{len(contexts)} | "
                f"T={len(context)} | "
                f"bootstrap={B} | "
                f"n_jobs={n_jobs}",
                flush=True,
            )
            if method == "lpcmci":
                lc = config["lpcmci"]
                res = lpcmci_stability(
                    context,
                    lag,
                    B,
                    block,
                    float(lc["pc_alpha"]),
                    threshold,
                    seed + idx,
                    tau_min=int(
                        lc.get(
                            "tau_min",
                            1,
                        )
                    ),
                    ci_test=str(
                        lc.get(
                            "ci_test",
                            "parcorr",
                        )
                    ),
                    significance=str(
                        lc.get(
                            "significance",
                            "analytic",
                        )
                    ),
                    fdr_method=str(
                        lc.get(
                            "fdr_method",
                            "fdr_bh",
                        )
                    ),
                    fdr_alpha=float(
                        lc.get(
                            "fdr_alpha",
                            0.05,
                        )
                    ),
                    tier_a_marks=tuple(
                        str(v)
                        for v in lc.get(
                            "tier_a_marks",
                            ["-->"],
                        )
                    ),
                    tier_b_marks=tuple(
                        str(v)
                        for v in lc.get(
                            "tier_b_marks",
                            ["o->"],
                        )
                    ),
                    tier_c_marks=tuple(
                        str(v)
                        for v in lc.get(
                            "tier_c_marks",
                            ["<->"],
                        )
                    ),
                    tier_d_marks=tuple(
                        str(v)
                        for v in lc.get(
                            "tier_d_marks",
                            ["o-o"],
                        )
                    ),
                    include_tier_b=bool(
                        lc.get(
                            "include_tier_b",
                            True,
                        )
                    ),
                    n_jobs=n_jobs,
                    backend=backend,
                    inner_max_num_threads=inner_max_num_threads,
                    parallel_verbose=parallel_verbose,
                )
            elif method == "linear_lasso":
                lc = config["linear_lasso"]
                res = linear_lasso_stability(
                    context,
                    lag,
                    B,
                    block,
                    float(lc["alpha"]),
                    threshold,
                    seed + idx,
                    max_iter=int(lc["max_iter"]),
                    coef_zero_tolerance=float(lc["coef_zero_tolerance"]),
                )
            elif method == "pcmciplus":
                lc = config["pcmciplus"]
                res = pcmciplus_stability(
                    context,
                    lag,
                    B,
                    block,
                    float(lc["pc_alpha"]),
                    threshold,
                    seed + idx,
                    tau_min=int(lc.get("tau_min", 1)),
                    ci_test=str(lc.get("ci_test", "parcorr")),
                    significance=str(lc.get("significance", "analytic")),
                    fdr_method=str(lc.get("fdr_method", "fdr_bh")),
                    fdr_alpha=float(lc.get("fdr_alpha", 0.05)),
                    tier_a_marks=tuple(str(v) for v in lc.get("tier_a_marks", ["-->"])),
                    tier_b_marks=tuple(str(v) for v in lc.get("tier_b_marks", ["o->"])),
                    tier_c_marks=tuple(str(v) for v in lc.get("tier_c_marks", ["<->"])),
                    tier_d_marks=tuple(str(v) for v in lc.get("tier_d_marks", ["o-o"])),
                    include_tier_b=bool(lc.get("include_tier_b", True)),
                    n_jobs=n_jobs,
                    backend=backend,
                    inner_max_num_threads=inner_max_num_threads,
                    parallel_verbose=parallel_verbose,
                )
            else:
                raise ValueError(f"Unknown causal support method: {method}")
            results.append(res)

    context_frequency = np.stack(
        [r.frequency for r in results],
        axis=0,
    )

    metadata = {
        "fit_split": "train_only",
        "method": method,
        "lag": lag,
        "seed": seed,
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_count": B,
        "block_length": block,
        "stability_threshold": stable_threshold,
        "union_screen_threshold": union_threshold,
        "support_mode": mode,
        "context_names": context_names,
        "context_thresholds": [threshold] * len(contexts),
        "parallel_context_scope": context_scope,
        "screening_rule_version": 2,
    }

    try:
        tigramite_version = package_version("tigramite")
    except PackageNotFoundError:
        tigramite_version = "not-installed"

    metadata["software"] = {
        "tigramite": tigramite_version,
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
    }

    if method == "lpcmci":
        lc = config["lpcmci"]
        metadata["lpcmci"] = {
            "pc_alpha": float(lc["pc_alpha"]),
            "tau_min": int(lc.get("tau_min", 1)),
            "ci_test": str(lc.get("ci_test", "parcorr")),
            "significance": str(lc.get("significance", "analytic")),
            "fdr_method": str(lc.get("fdr_method", "fdr_bh")),
            "fdr_alpha": float(lc.get("fdr_alpha", 0.05)),
            "tier_a_marks": list(lc.get("tier_a_marks", ["-->"])),
            "tier_b_marks": list(lc.get("tier_b_marks", ["o->"])),
            "tier_c_marks": list(lc.get("tier_c_marks", ["<->"])),
            "tier_d_marks": list(lc.get("tier_d_marks", ["o-o"])),
            "include_tier_b": bool(lc.get("include_tier_b", True)),
        }

    if method == "pcmciplus":
        lc = config["pcmciplus"]
        metadata["pcmciplus"] = {
            "pc_alpha": float(lc["pc_alpha"]),
            "tau_min": int(lc.get("tau_min", 1)),
            "ci_test": str(lc.get("ci_test", "parcorr")),
            "significance": str(lc.get("significance", "analytic")),
            "fdr_method": str(lc.get("fdr_method", "fdr_bh")),
            "fdr_alpha": float(lc.get("fdr_alpha", 0.05)),
            "tier_a_marks": list(lc.get("tier_a_marks", ["-->"])),
            "tier_b_marks": list(lc.get("tier_b_marks", ["o->"])),
            "tier_c_marks": list(lc.get("tier_c_marks", ["<->"])),
            "tier_d_marks": list(lc.get("tier_d_marks", ["o-o"])),
            "include_tier_b": bool(lc.get("include_tier_b", True)),
        }

    if mode == "stable":
        return replace(
            results[0],
            global_frequency=results[0].frequency.copy(),
            context_frequency=context_frequency,
            context_names=tuple(context_names),
            metadata=metadata,
        )

    merged = _merge_union_results(results, method, threshold=union_threshold)

    return replace(
        merged,
        global_frequency=results[0].frequency.copy(),
        context_frequency=context_frequency,
        context_names=tuple(context_names),
        metadata=metadata,
    )


def save_support(
    result: SupportResult,
    path: str | Path,
    names: np.ndarray | None = None,
) -> None:
    """
    Save support.
    :param result: Input argument used by this operation.
    :param path: Input argument used by this operation.
    :param names: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "support": result.support.astype(np.uint8),
        "frequency": result.frequency,
        "tier": result.tier,
        "method": np.asarray(result.method),
        "context_names": np.asarray(result.context_names, dtype="U64"),
        "metadata_json": np.asarray(
            json.dumps(
                result.metadata or {},
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
    }

    optional_arrays = {
        "tier_a_frequency": result.tier_a_frequency,
        "tier_b_frequency": result.tier_b_frequency,
        "tier_c_frequency": result.tier_c_frequency,
        "tier_d_frequency": result.tier_d_frequency,
        "global_frequency": result.global_frequency,
        "context_frequency": result.context_frequency,
    }

    for key, value in optional_arrays.items():
        if value is not None:
            payload[key] = value

    np.savez_compressed(path, **payload)

    if result.metadata is not None:
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                result.metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if names is None:
        return

    lag_count, n, n_source = result.support.shape
    if n != n_source:
        raise ValueError("support must be square in target/source dimensions")
    if len(names) != n:
        raise ValueError(f"names has length {len(names)}, expected {n}")

    edge_columns = [
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
    ]

    rows: list[dict[str, Any]] = []

    def optional_value(
        array: np.ndarray | None,
        ell: int,
        target: int,
        source: int,
    ) -> float:
        """
        Optional value.
        :param array: Input argument used by this operation.
        :param ell: Input argument used by this operation.
        :param target: Input argument used by this operation.
        :param source: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        if array is None:
            return float("nan")
        return float(array[ell, target, source])

    for ell in range(lag_count):
        for target in range(n):
            for source in range(n):
                if not result.support[ell, target, source]:
                    continue

                global_frequency = optional_value(
                    result.global_frequency,
                    ell,
                    target,
                    source,
                )

                best_context = ""
                best_context_frequency = float("nan")

                if result.context_frequency is not None:
                    context_values = result.context_frequency[:, ell, target, source]
                    best_index = int(np.argmax(context_values))
                    best_context_frequency = float(context_values[best_index])

                    if best_index < len(result.context_names):
                        best_context = str(result.context_names[best_index])

                rows.append(
                    {
                        "source": str(names[source]),
                        "target": str(names[target]),
                        "lag": ell + 1,
                        "frequency": float(result.frequency[ell, target, source]),
                        "global_frequency": global_frequency,
                        "best_context": best_context,
                        "best_context_frequency": best_context_frequency,
                        "tier": str(result.tier[ell, target, source]),
                        "tier_a_frequency": optional_value(
                            result.tier_a_frequency,
                            ell,
                            target,
                            source,
                        ),
                        "tier_b_frequency": optional_value(
                            result.tier_b_frequency,
                            ell,
                            target,
                            source,
                        ),
                        "tier_c_warning_frequency": optional_value(
                            result.tier_c_frequency,
                            ell,
                            target,
                            source,
                        ),
                        "tier_d_warning_frequency": optional_value(
                            result.tier_d_frequency,
                            ell,
                            target,
                            source,
                        ),
                    }
                )

    edge_table = pd.DataFrame(rows, columns=edge_columns)

    if not edge_table.empty:
        edge_table = edge_table.sort_values(
            ["frequency", "lag", "source", "target"],
            ascending=[False, True, True, True],
        )

    edge_table.to_csv(
        path.with_suffix(".edges.csv"),
        index=False,
    )


def load_support(path: str | Path) -> SupportResult:
    """
    Load support.
    :param path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    with np.load(path, allow_pickle=False) as z:
        return SupportResult(
            support=z["support"].astype(bool),
            frequency=z["frequency"],
            tier=z["tier"],
            method=str(z["method"].item()),
            tier_a_frequency=z.get("tier_a_frequency", None),
            tier_b_frequency=z.get("tier_b_frequency", None),
            tier_c_frequency=z.get("tier_c_frequency", None),
            tier_d_frequency=z.get("tier_d_frequency", None),
            global_frequency=z.get("global_frequency", None),
            context_frequency=z.get("context_frequency", None),
            context_names=(
                tuple(str(v) for v in z["context_names"].tolist()) if "context_names" in z else ()
            ),
            metadata=(json.loads(z["metadata_json"].item()) if "metadata_json" in z else None),
        )


def discover_from_config(
    x_train: np.ndarray,
    config_path: str | Path,
    n_jobs: int | None = None,
) -> SupportResult:
    """
    Discover from config.
    :param x_train: Input argument used by this operation.
    :param config_path: Input argument used by this operation.
    :param n_jobs: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    cfg = load_yaml(config_path)

    if n_jobs is not None:
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")

        cfg.setdefault(
            "parallel",
            {},
        )["n_jobs"] = n_jobs

    return discover_support(
        x_train,
        cfg,
    )
