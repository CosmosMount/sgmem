"""New memory systems must receive public inputs with unambiguous source identity."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_memory_inputs", ROOT / "scripts/prepare_memory_inputs.py")
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "DOMAINS", ("Finance", "Technology"))
    monkeypatch.setattr(inputs, "CATEGORIES", ("multi_hop",))
    split = {"dev": [], "test": []}
    for domain in inputs.DOMAINS:
        path = tmp_path / f"data/final/{domain}"
        path.mkdir(parents=True)
        # Identical text belongs to different people and groups, and IDs recur across domains.
        channels = {f"group-{i}": [{"msg_node": f"m{i}", "author": f"user-{i}",
                                    "content": "I agree.", "timestamp": "2026-01-01",
                                    "reply_to": None, "answer": "DO NOT EXPORT"}]
                    for i in (1, 2)}
        (path / f"synthetic_domain_channels_rolevariants_{domain}.json").write_text(json.dumps(channels))
        path = tmp_path / "questions" / domain
        path.mkdir(parents=True)
        rows = [{"id": f"q{i}", "question": f"Question {i}?", "answer": "SECRET GOLD",
                 "asking_user_id": "user-1", "supporting_facts": ["SECRET EVIDENCE"]} for i in (1, 2)]
        (path / "multi_hop.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        for name, i in (("dev", 1), ("test", 2)):
            split[name].append({"domain": domain, "category": "multi_hop", "id": f"q{i}"})
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    return tmp_path, split_path


def test_export_preserves_duplicate_sources_and_hides_labels(source, tmp_path):
    benchmark, split = source
    output = tmp_path / "export"
    manifest = inputs.prepare(output, benchmark, split, audit_per_domain=2)
    assert manifest["status"] == "complete" and manifest["question_count"] == 2
    sources = []
    for domain in inputs.DOMAINS:
        rows = [json.loads(line) for line in (output / f"messages-{domain}.jsonl").read_text().splitlines()]
        assert [row["speaker"] for row in rows] == ["user-1", "user-2"]
        assert len({row["content"] for row in rows}) == 1
        assert manifest["domains"][domain]["duplicate_body_occurrences"] == 1
        sources.extend((row["domain"], row["id"]) for row in rows)
    assert len(set(sources)) == 4
    queries = [json.loads(line) for line in (output / "queries-dev.jsonl").read_text().splitlines()]
    assert {row["id"] for row in queries} == {"q1"}
    assert not (output / "queries-test.jsonl").exists()
    for name, digest in manifest["files"].items():
        raw = (output / name).read_bytes()
        assert b"SECRET" not in raw and b"DO NOT EXPORT" not in raw and b'"answer"' not in raw
        assert hashlib.sha256(raw).hexdigest() == digest


def test_source_sample_is_repeatable_and_existing_output_is_preserved(source, tmp_path):
    benchmark, split = source
    a, b = tmp_path / "a", tmp_path / "b"
    inputs.prepare(a, benchmark, split, audit_per_domain=1)
    inputs.prepare(b, benchmark, split, audit_per_domain=1)
    assert (a / "extraction-audit.jsonl").read_bytes() == (b / "extraction-audit.jsonl").read_bytes()
    before = (a / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        inputs.prepare(a, benchmark, split)
    assert (a / "manifest.json").read_bytes() == before


def test_bad_split_is_rejected_before_writing(source, tmp_path):
    benchmark, split = source
    data = json.loads(split.read_text())
    data["test"].append(data["dev"][0])
    split.write_text(json.dumps(data))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="overlapping"):
        inputs.prepare(output, benchmark, split)
    assert not output.exists()
