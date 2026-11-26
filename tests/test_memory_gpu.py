"""GPU choice and accounting must obey the authorized sharing and budget policy."""

import runpy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

stage = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_memory_gpu.py"))


def test_prefer_idle_but_allow_sufficient_shared_gpu():
    rows = [{"index": "1", "uuid": "shared", "free_mib": 30000, "utilization": 20},
            {"index": "2", "uuid": "idle", "free_mib": 22000, "utilization": 0}]
    select = stage["select_gpu"]
    assert select(rows, {"shared"}, 20000)["index"] == "2"
    assert select(rows, {"shared"}, 25000)["index"] == "1"
    assert select(rows, set(), 40000) is None


def test_budget_counts_failed_stages_and_rejects_invalid_or_duplicate_durations():
    used = stage["used_seconds"]
    assert used([{"label": "failed", "exit_code": 1, "wall_seconds": 100},
                 {"label": "retry", "exit_code": 0, "wall_seconds": 200}]) == 300
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Invalid GPU ledger"):
            used([{"label": "invalid", "wall_seconds": value}])
    with pytest.raises(ValueError, match="Duplicate stage"):
        used([{"label": "same", "wall_seconds": 1}, {"label": "same", "wall_seconds": 1}])


def test_optional_total_cap_and_failed_stage_accounting(tmp_path, monkeypatch):
    run = stage["run"]
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
    monkeypatch.setitem(run.__globals__, "available_gpu", lambda minimum: {"index": "0", "uuid": "idle"})
    monkeypatch.setattr(stage["time"], "sleep", lambda seconds: None)
    commands = []

    class FailedProcess:
        pid = 123

        def __init__(self, command, **kwargs):
            commands.append(command)

        def wait(self, timeout=None):
            return 2

        def poll(self):
            return 2

    monkeypatch.setattr(stage["subprocess"], "Popen", FailedProcess)
    ledger = tmp_path / "budget.jsonl"
    ledger.write_text(json.dumps({"label": "previous", "wall_seconds": 31 * 3600}) + "\n")
    args = SimpleNamespace(label="failed", max_hours=2.0, total_hours=None,
                           min_free_mib=20000, command=["--", "fake"], budget_dir=tmp_path)
    assert run(args) == 2
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 2 and rows[1]["exit_code"] == 2 and rows[1]["wall_seconds"] >= 0
    assert rows[1]["round_limit_seconds"] is None and rows[1]["remaining_before_seconds"] is None
    assert rows[1]["stage_limit_seconds"] == 7200
    assert commands[0][:4] == ["timeout", "--signal=TERM", "--kill-after=10s", "7180"]
    assert not (tmp_path / "active-stage.json").exists()

    args.label = "within_cap"
    args.total_hours = 32
    assert run(args) == 2
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows[-1]["round_limit_seconds"] == 32 * 3600
    assert 3599 < rows[-1]["stage_limit_seconds"] <= 3600
    assert rows[-1]["remaining_before_seconds"] == rows[-1]["stage_limit_seconds"]

    args.label = "capped"
    args.total_hours = 30
    with pytest.raises(RuntimeError, match="Insufficient remaining GPU budget"):
        run(args)
    assert len(ledger.read_text().splitlines()) == 3


@pytest.mark.parametrize("hours", [0, -1, float("inf"), float("nan"), 1e308])
def test_reject_invalid_total_cap(hours, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace())
    args = SimpleNamespace(label="stage", max_hours=1, total_hours=hours,
                           min_free_mib=20000, command=["fake"], budget_dir=tmp_path)
    with pytest.raises(ValueError, match="Total hours"):
        stage["run"](args)
