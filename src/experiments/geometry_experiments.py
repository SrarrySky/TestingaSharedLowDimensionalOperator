"""Auditable fixed-fit producers for geometry experiments.

This module contains pure, deterministic producers rather than a real-data
experiment command.  They consume already fitted coefficient/state paths and
return portable arrays and JSON-safe summaries.  In particular, they do not
smooth test states, refit parameters inside a bootstrap draw, or interpret a
latent coordinate system across independently fitted models.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from model.geometry import (
    LowRankDiagnostics,
    low_rank_diagnostics,
    nonnormality,
    reversal_distance,
    reversal_hyperplane,
    spectral_radius,
    transient_gain_profile,
)
from model.propagation import companion_matrix


def _finite_array(value: np.ndarray, name: str, ndim: int | None = None) -> np.ndarray:
    """
     finite array.
    :param value: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :param ndim: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    value = np.asarray(value, dtype=float)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return value


def _positive_integer(value: int, name: str) -> int:
    """
     positive integer.
    :param value: Input argument used by this operation.
    :param name: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def coefficient_path_from_states(
    a0: np.ndarray,
    B: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Evaluate "a_t = a0 + B @ z_t" for a complete state path."""
    a0 = _finite_array(a0, "a0", 1)
    B = _finite_array(B, "B", 2)
    states = _finite_array(states, "states", 2)
    if B.shape[0] != a0.size or B.shape[1] != states.shape[1]:
        raise ValueError("a0, B and states have incompatible dimensions")
    return a0[None, :] + states @ B.T


def _increment_subspace(path: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """
     increment subspace.
    :param path: Input argument used by this operation.
    :param dimension: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    delta = np.diff(path, axis=0)
    centered = delta - delta.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    rank = min(dimension, right_vectors.shape[0])
    return singular_values, right_vectors[:rank].T


def _principal_angle_sines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
     principal angle sines.
    :param left: Input argument used by this operation.
    :param right: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    if left.shape[1] != right.shape[1]:
        raise ValueError("Subspaces must have the same declared dimension")
    if left.shape[1] == 0:
        return np.empty(0)
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - singular**2))


def _moving_block_index_stream(n: int, block_length: int, seed: int):
    """Yield moving-block indices in the same order as "moving_block_indices"."""
    rng = np.random.default_rng(int(seed))
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=np.int64)
    while True:
        starts = rng.integers(0, n, size=blocks)
        yield np.concatenate([(start + offsets) % n for start in starts])[:n]


def _projection_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return ||UU' - VV'||_2 without constructing p-by-p projectors."""
    if left.shape != right.shape:
        raise ValueError("Subspace bases must have the same shape")
    residual = right - left @ (left.T @ right)
    return float(np.linalg.svd(residual, compute_uv=False)[0])


@dataclass(frozen=True)
class LowRankPathReport:
    diagnostics: LowRankDiagnostics
    bootstrap_singular_values: np.ndarray
    bootstrap_max_sin_theta: np.ndarray
    bootstrap_projection_distance: np.ndarray
    reference_basis: np.ndarray
    bootstrap_count: int
    block_length: int
    seed: int
    uncertainty: str = "fixed-fit moving-block bootstrap over coefficient-path rows"

    def summary(self, level: float = 0.95) -> dict[str, Any]:
        """
        Summary.
        :param level: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        alpha = (1.0 - level) / 2.0
        return {
            "diagnostics": {
                **asdict(self.diagnostics),
                "singular_values": self.diagnostics.singular_values.tolist(),
            },
            "reference_basis_shape": list(self.reference_basis.shape),
            "bootstrap_count": self.bootstrap_count,
            "block_length": self.block_length,
            "seed": self.seed,
            "uncertainty": self.uncertainty,
            "max_sin_theta": {
                "mean": float(self.bootstrap_max_sin_theta.mean()),
                "lower": float(np.quantile(self.bootstrap_max_sin_theta, alpha)),
                "upper": float(np.quantile(self.bootstrap_max_sin_theta, 1.0 - alpha)),
            },
            "projection_distance": {
                "mean": float(self.bootstrap_projection_distance.mean()),
                "lower": float(np.quantile(self.bootstrap_projection_distance, alpha)),
                "upper": float(np.quantile(self.bootstrap_projection_distance, 1.0 - alpha)),
            },
        }


def low_rank_path_report(
    coefficient_path: np.ndarray,
    dimension: int,
    *,
    block_length: int = 20,
    bootstrap_count: int = 1000,
    seed: int = 0,
    checkpoint: str | os.PathLike[str] | None = None,
    progress: bool = False,
    timeout_seconds: float | None = None,
    memory_gb: float | None = None,
) -> LowRankPathReport:
    """Measure increment low-rank structure and fixed-fit subspace stability."""
    path = _finite_array(coefficient_path, "coefficient_path", 2)
    dimension = _positive_integer(dimension, "dimension")
    if dimension > path.shape[1]:
        raise ValueError("dimension cannot exceed coefficient dimension")
    block_length = _positive_integer(block_length, "block_length")
    bootstrap_count = _positive_integer(bootstrap_count, "bootstrap_count")
    if block_length > path.shape[0]:
        raise ValueError("block_length cannot exceed coefficient-path length")
    if isinstance(seed, bool | np.bool_) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    diagnostics = low_rank_diagnostics(path, dimension)
    _, reference_basis = _increment_subspace(path, dimension)
    boot_singular = np.zeros((bootstrap_count, min(path.shape[0] - 1, path.shape[1])))
    max_angles = np.zeros(bootstrap_count)
    distances = np.zeros(bootstrap_count)
    completed = 0
    if checkpoint is not None and Path(checkpoint).is_file():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if saved["path_shape"].tolist() != list(path.shape) or int(saved["seed"]) != int(seed):
                raise ValueError("Geometry checkpoint does not match path or seed")
            completed = int(saved["completed"])
            boot_singular[:completed] = saved["singular_values"][:completed]
            max_angles[:completed] = saved["max_sin_theta"][:completed]
            distances[:completed] = saved["projection_distance"][:completed]
    stream = _moving_block_index_stream(path.shape[0], block_length, int(seed))
    for _ in range(completed):
        next(stream)
    started = time.monotonic()
    process = psutil.Process(os.getpid())
    for draw in range(completed, bootstrap_count):
        index = next(stream)
        sampled = path[index]
        singular, basis = _increment_subspace(sampled, dimension)
        boot_singular[draw, : len(singular)] = singular
        angles = _principal_angle_sines(reference_basis, basis)
        max_angles[draw] = float(angles.max(initial=0.0))
        # The paper's projection distance is the operator norm of the
        # difference between the orthogonal projectors.  This is not the
        # Frobenius distance (which would be sqrt(2 * sum(sin(theta)^2))).
        distances[draw] = _projection_distance(reference_basis, basis)
        completed = draw + 1
        if checkpoint is not None:
            target = Path(checkpoint)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    path_shape=np.asarray(path.shape),
                    seed=int(seed),
                    completed=completed,
                    singular_values=boot_singular,
                    max_sin_theta=max_angles,
                    projection_distance=distances,
                )
            os.replace(temporary, target)
        if progress:
            print(f"[geometry] bootstrap {completed}/{bootstrap_count}", flush=True)
        if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
            raise TimeoutError("Geometry bootstrap timeout exceeded")
        if memory_gb is not None and process.memory_info().rss > memory_gb * 1024**3:
            raise MemoryError("Geometry bootstrap memory limit exceeded")
    return LowRankPathReport(
        diagnostics=diagnostics,
        bootstrap_singular_values=boot_singular,
        bootstrap_max_sin_theta=max_angles,
        bootstrap_projection_distance=distances,
        reference_basis=reference_basis,
        bootstrap_count=bootstrap_count,
        block_length=block_length,
        seed=int(seed),
    )


@dataclass(frozen=True)
class ReversalMap:
    normal: np.ndarray
    offset: float
    signed_margin: np.ndarray
    distance: np.ndarray
    positive_fraction: float
    negative_fraction: float
    zero_fraction: float
    status: str


def reversal_map(a0: float, b: np.ndarray, states: np.ndarray) -> ReversalMap:
    """Map state observations to a coefficient sign-reversal hyperplane."""
    if not np.isfinite(a0):
        raise ValueError("a0 must be finite")
    b = _finite_array(b, "b", 1)
    states = _finite_array(states, "states", 2)
    if states.shape[1] != b.size:
        raise ValueError("b and states have incompatible dimensions")
    norm = float(np.linalg.norm(b))
    if norm == 0:
        return ReversalMap(
            normal=np.zeros_like(b),
            offset=float("nan"),
            signed_margin=np.full(states.shape[0], a0),
            distance=np.full(states.shape[0], np.inf),
            positive_fraction=float(a0 > 0),
            negative_fraction=float(a0 < 0),
            zero_fraction=float(a0 == 0),
            status="constant",
        )
    normal, offset = reversal_hyperplane(float(a0), b)
    signed = a0 + states @ b
    distances = np.asarray([reversal_distance(a0, b, state) for state in states])
    return ReversalMap(
        normal=normal,
        offset=offset,
        signed_margin=signed,
        distance=distances,
        positive_fraction=float(np.mean(signed > 0)),
        negative_fraction=float(np.mean(signed < 0)),
        zero_fraction=float(np.mean(signed == 0)),
        status="reversal-capable"
        if np.any(signed > 0) and np.any(signed < 0)
        else "no-observed-reversal",
    )


@dataclass(frozen=True)
class StableAmplificationReport:
    spectral_radii: np.ndarray
    nonnormalities: np.ndarray
    peak_gains: np.ndarray
    peak_horizons: np.ndarray
    stable_but_amplifying: np.ndarray
    horizon: int
    amplification_threshold: float

    @property
    def stable_but_amplifying_fraction(self) -> float:
        """
        Stable but amplifying fraction.
        :return: Result produced by this operation.
        """
        return float(np.mean(self.stable_but_amplifying))


def stable_amplification_report(
    a_lags_path: np.ndarray,
    *,
    horizon: int = 10,
    amplification_threshold: float = 1.0,
    stability_threshold: float = 1.0,
) -> StableAmplificationReport:
    """Diagnose stable individual companions with finite-horizon amplification."""
    a_lags_path = _finite_array(a_lags_path, "a_lags_path", 4)
    horizon = _positive_integer(horizon, "horizon")
    if not np.isfinite(amplification_threshold) or amplification_threshold < 0:
        raise ValueError("amplification_threshold must be finite and nonnegative")
    if not np.isfinite(stability_threshold) or stability_threshold <= 0:
        raise ValueError("stability_threshold must be finite and positive")
    _, lag, n, n2 = a_lags_path.shape
    if n != n2:
        raise ValueError("a_lags_path must contain square lag matrices")
    radii = np.empty(len(a_lags_path))
    nonnormalities = np.empty(len(a_lags_path))
    gains = np.empty(len(a_lags_path))
    peak_horizons = np.empty(len(a_lags_path), dtype=int)
    for t, a_lags in enumerate(a_lags_path):
        companion = companion_matrix(a_lags)
        profile = transient_gain_profile(companion, horizon)
        radii[t] = spectral_radius(companion)
        nonnormalities[t] = nonnormality(companion)
        gains[t] = profile.peak_gain
        peak_horizons[t] = profile.peak_horizon
    stable = radii < stability_threshold
    return StableAmplificationReport(
        spectral_radii=radii,
        nonnormalities=nonnormalities,
        peak_gains=gains,
        peak_horizons=peak_horizons,
        stable_but_amplifying=stable & (gains > amplification_threshold),
        horizon=horizon,
        amplification_threshold=float(amplification_threshold),
    )
