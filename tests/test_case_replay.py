"""Archived demonstration must preserve recorded answers without executing their text."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replay_cases", ROOT / "scripts/replay_cases.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_replay_preserves_all_recorded_answers(tmp_path):
    from html import unescape
    output = tmp_path / "demo.html"
    assert replay.build(output) == 6
    html = output.read_text(encoding="utf-8")
    assert "Offline replay, not live inference" in html
    decoded = unescape(html)
    for case in json.loads(replay.SOURCE.read_text(encoding="utf-8"))["cases"]:
        assert f"{case['domain']}/{case['question_id']}" in html
        assert case["question"] in decoded and case["reference_answer"] in decoded
        for row in case["methods"].values():
            assert row["answer"] in decoded and row["judge_output"] in decoded
        for message in case["source_messages"]:
            assert message["content"] in decoded
    assert html.count("<section>") == 6
    assert "<script" not in html and '<iframe' not in html


def test_recorded_text_is_escaped_and_invalid_selection_does_not_write(tmp_path):
    case = json.loads(replay.SOURCE.read_text(encoding="utf-8"))["cases"][0]
    case["methods"]["hybrid"]["answer"] = '<script>alert("example")</script>'
    html = replay.render_case(case)
    assert "<script>" not in html and "&lt;script&gt;" in html
    output = tmp_path / "unknown.html"
    with pytest.raises(ValueError, match="Unknown saved case"):
        replay.build(output, "Finance/not-a-saved-case")
    assert not output.exists()
