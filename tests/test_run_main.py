from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_main


def _config(*, publication: bool = False, real_data: bool = False) -> dict:
    stages = {name: {"enabled": False, "required": False} for name in run_main.STAGES}
    stages["coverage"] = {"enabled": True, "required": True}
    if real_data:
        stages["real_data"] = {
            "enabled": True,
            "required": publication,
            "dataset": "missing/dataset.npz",
        }
    return {
        "version": run_main.SCHEMA_VERSION,
        "experiment": "runner-test",
        "purpose": "publication" if publication else "debug",
        "seed": 3,
        "threads": 1,
        "inner_threads": 1,
        "publication_mode": publication,
        "stages": stages,
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "runner.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runner_config_rejects_required_disabled_and_unknown_stage():
    payload = _config()
    payload["stages"]["coverage"]["enabled"] = False
    with pytest.raises(ValueError, match="coverage"):
        run_main.validate_runner_config(payload)

    payload = _config()
    payload["stages"]["unknown"] = {}
    with pytest.raises(ValueError, match="Unknown stages"):
        run_main.validate_runner_config(payload)


def test_runner_records_missing_input_and_publication_exit(tmp_path):
    config_path = _write_config(tmp_path, _config(publication=True, real_data=True))
    result = run_main.run_main(config_path, tmp_path / "run")

    assert result["exit_code"] == 1
    assert "real_data" in result["required_stage_failures"]
    ledger = json.loads((tmp_path / "run" / "ledger.json").read_text(encoding="utf-8"))
    record = next(row for row in ledger if row["cell_id"] == "real_data")
    assert record["status"] == "unsupported_missing_input"
    assert (tmp_path / "run" / "coverage.json").exists() is False
    coverage = list((tmp_path / "run" / "stages" / "coverage").glob("**/coverage.json"))
    assert len(coverage) == 1
    assert json.loads(coverage[0].read_text(encoding="utf-8"))["blueprint_complete"] is False
    assert result["verification"]["status"] == "verified"


def test_runner_resume_reuses_only_verified_success_cells(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, _config())
    calls: list[str] = []

    def fake_handler(stage, attempt, stage_config, root, threads, runner_config=None):
        calls.append(stage)
        (attempt / "producer.txt").write_text(stage, encoding="utf-8")
        return {"status": "success"}

    # Make one enabled stage deterministic; disabled stages remain explicit
    # unsupported records and are therefore retried rather than reused.
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["stages"]["synthetic"] = {"enabled": True, "required": False}
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(run_main, "_stage_handler", fake_handler)

    first = run_main.run_main(config_path, tmp_path / "run", allow_incomplete=True)
    assert first["status"] == "completed_with_failures"
    assert calls.count("synthetic") == 1

    calls.clear()
    second = run_main.run_main(
        config_path,
        tmp_path / "run",
        resume=True,
        allow_incomplete=True,
    )
    assert second["reused_cells"] >= 1
    assert calls.count("synthetic") == 0
    assert second["verification"]["status"] == "verified"
    records = json.loads((tmp_path / "run" / "ledger.json").read_text(encoding="utf-8"))
    assert len(records) == len(run_main.STAGES)
    assert len({row["cell_id"] for row in records}) == len(run_main.STAGES)
