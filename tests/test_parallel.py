from __future__ import annotations

import time

import pytest

from core.parallel import parallel_map, validate_jobs


def test_validate_jobs_rejects_zero_and_boolean() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        validate_jobs(0)
    with pytest.raises(ValueError, match="nonzero"):
        validate_jobs(True)


def test_parallel_map_keeps_input_order_and_supports_serial_execution() -> None:
    def work(value: int) -> int:
        return value * value

    assert parallel_map(work, [3, 1, 2], jobs=1) == [9, 1, 4]


def test_parallel_map_keeps_input_order_with_threading_backend() -> None:
    def work(value: int) -> int:
        # Deliberately complete in reverse order. Joblib must still return the
        # declared task order because downstream selection uses that order for
        # deterministic tie-breaking.
        time.sleep((3 - value) * 0.01)
        return value

    assert parallel_map(work, [0, 1, 2, 3], jobs=2, backend="threading") == [0, 1, 2, 3]


def test_parallel_map_accepts_joblib_all_cpu_convention() -> None:
    assert parallel_map(lambda value: value + 1, [1, 2], jobs=-1) == [2, 3]
