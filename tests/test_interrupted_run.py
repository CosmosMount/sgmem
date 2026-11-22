import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recover_interrupted_run as recovery


def dump(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def rows(path, items):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")


@pytest.fixture
def interrupted(tmp_path):
    run = tmp_path / "test-hybrid"
    folder = run / "hybrid"
    folder.mkdir(parents=True)
    source = {"commit": "a" * 40, "status": ""}
    dump(run / "run-config.json", {"finished": False, "sessions": [], "project_git": source,
                                   "arguments": {"methods": ["hybrid"], "partition": "test"}})
    dump(folder / "manifest.json", {"status": "running", "project_git": source})
    questions = [{"domain": "Finance", "category": "temporal", "question_id": q} for q in ("done", "cut")]
    dump(run / "split.json", {"dev": [], "test": questions})
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    completed = {**questions[0], "method": "hybrid", "partition": "test", "attempt": 1, "status": "ok",
                 "question": "Unicode\u2028separator\u2029inside a message"}
    calls = []
    for index, question in enumerate(questions):
        for stage in (("answer", "judge") if index == 0 else ("answer",)):
            calls.append({"question": {**question, "method": "hybrid", "attempt": 1}, "stage": stage,
                          "usage": usage, "finish_reason": "stop", "seconds": 0.5})
            if index == 0:
                completed.update({stage + "_usage": usage, stage + "_finish_reason": "stop"})
    rows(folder / "attempts.jsonl", [completed])
    rows(run / "generation-audit.jsonl", calls)
    ledger = tmp_path / "budget.jsonl"
    rows(ledger, [{"command": ["python", "--output", str(run)], "exit_code": 124, "wall_seconds": 1800}])
    return run, ledger


def test_recovery_preserves_prefixes_and_retains_call_without_inventing_answer(interrupted):
    run, ledger = interrupted
    original = (run / "hybrid/attempts.jsonl").read_bytes()
    call_bytes = (run / "generation-audit.jsonl").read_bytes()
    receipt = recovery.prepare(run, ledger)
    assert (run / "hybrid/attempts.jsonl").read_bytes() == original
    assert receipt["error_attempt"]["question_id"] == "cut"
    assert "answer" not in receipt["error_attempt"] and "prediction" not in receipt["error_attempt"]
    recovery.apply(run, receipt)
    assert (run / "hybrid/attempts.jsonl").read_bytes().startswith(original)
    assert (run / "generation-audit.jsonl").read_bytes() == call_bytes
    assert recovery.verify(run)["budget_stage"]["wall_seconds"] == 1800
    with pytest.raises(ValueError, match="already recorded"):
        recovery.prepare(run, ledger)


def test_interruption_before_answer_needs_no_synthetic_attempt(interrupted):
    run, ledger = interrupted
    calls = recovery.read_rows(run / "generation-audit.jsonl")
    rows(run / "generation-audit.jsonl", calls[:2])
    receipt = recovery.prepare(run, ledger)
    assert receipt["error_attempt"] is None
    recovery.apply(run, receipt)
    assert len(recovery.read_rows(run / "hybrid/attempts.jsonl")) == 1


def test_recovery_rejects_duplicate_calls(interrupted):
    run, ledger = interrupted
    calls = recovery.read_rows(run / "generation-audit.jsonl")
    rows(run / "generation-audit.jsonl", calls + calls[-1:])
    with pytest.raises(ValueError, match="duplicate generation"):
        recovery.prepare(run, ledger)


def test_recovery_rejects_wrong_budget_exit(interrupted):
    run, ledger = interrupted
    stage = recovery.read_rows(ledger)[0]
    rows(ledger, [{**stage, "exit_code": 0}])
    with pytest.raises(ValueError, match="hard budget timeout"):
        recovery.prepare(run, ledger)


def test_recovery_audit_rejects_changed_original_history(interrupted):
    run, ledger = interrupted
    recovery.apply(run, recovery.prepare(run, ledger))
    path = run / "generation-audit.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"seconds": 0.5', b'"seconds": 0.6', 1))
    with pytest.raises(ValueError, match="evidence prefix changed"):
        recovery.verify(run)
