"""Regression checks for public package release entry points."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tracked_paths(*roots: str) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", *roots],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [path for path in completed.stdout.decode().split("\0") if path]


def test_module_entrypoint_exposes_cli() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "cli", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "theory-checks" in completed.stdout


@pytest.mark.parametrize("name", ["requirements.txt"])
def test_requirements_are_utf8_and_nonempty(name: str) -> None:
    path = ROOT / name
    raw = path.read_bytes()
    content = raw.decode("utf-8")
    assert content.strip()
    assert b"\x00" not in raw


def test_release_boundary_does_not_track_local_artifacts() -> None:
    tracked = _tracked_paths("outputs", "data/raw", "data/interim", "data/processed", "Write")
    allowed = {
        "outputs/README.md",
        "outputs/checkpoints/.gitkeep",
        "outputs/figures/.gitkeep",
        "outputs/tables/.gitkeep",
        "data/raw/.gitkeep",
        "data/interim/.gitkeep",
        "data/processed/.gitkeep",
    }
    assert sorted(set(tracked) - allowed) == []


def test_release_source_has_no_machine_absolute_paths() -> None:
    tracked = _tracked_paths(
        "scripts",
        "src",
        "tests",
        "config",
        "README.md",
        "docs",
        "paper_supplement",
        "pyproject.toml",
        "requirements.txt",
    )
    pattern = re.compile(r"(?:[A-Za-z]:\\Users\\|[A-Za-z]:\\Study\\|/home/|/Users/)")
    matches = []
    for relative in tracked:
        if relative == "tests/test_release_tooling.py":
            continue
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            matches.append(relative)
    assert matches == []
