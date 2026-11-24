"""GPU choice and accounting must obey the authorized sharing and budget policy."""

import runpy
from pathlib import Path

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
