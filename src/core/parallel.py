"""Small, deterministic helpers for bounded experiment parallelism."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from joblib import Parallel, delayed, parallel_config

T = TypeVar("T")
R = TypeVar("R")


def validate_jobs(jobs: int) -> int:
    """Validate a job count while retaining Joblib's "-1" convention."""
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs == 0:
        raise ValueError("jobs must be a nonzero integer")
    return jobs


def parallel_map(
    function: Callable[[T], R],
    tasks: Iterable[T],
    *,
    jobs: int,
    backend: str = "loky",
    inner_max_num_threads: int = 1,
    verbose: int = 0,
    batch_size: int | str = "auto",
) -> list[R]:
    """Map independent tasks with a bounded, deterministic worker pool.

    Results are returned in input order.  The function is deliberately a
    top-level API around Joblib so callers can keep all file writes and result
    aggregation in the parent process.
    """
    validate_jobs(jobs)
    task_list = list(tasks)
    if jobs == 1 or len(task_list) <= 1:
        return [function(task) for task in task_list]

    config: dict[str, Any] = {"backend": backend, "n_jobs": jobs}
    if backend == "loky":
        if isinstance(inner_max_num_threads, bool) or inner_max_num_threads < 1:
            raise ValueError("inner_max_num_threads must be a positive integer")
        config["inner_max_num_threads"] = inner_max_num_threads

    with parallel_config(**config):
        return Parallel(batch_size=batch_size, verbose=verbose)(
            delayed(function)(task) for task in task_list
        )
