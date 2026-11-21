import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_frozen_test as audit
from sgmem.qa import summarize
from sgmem.runner import CATEGORIES, DOMAINS


SOURCE = "a" * 40
RUN_COMMIT = "b" * 40
BENCHMARK = "c" * 40
SELECTED = "hybrid-meta"
CONTROL = "hybrid"


def write(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def mutate(path, update):
    value = audit.read(path)
    update(value)
    write(path, value)


def make_questions(prefix, count):
    cells = [(domain, category) for domain in DOMAINS for category in CATEGORIES]
    return [{"domain": cells[n % len(cells)][0], "category": cells[n % len(cells)][1], "id": f"{prefix}-{n:03d}"}
            for n in range(count)]


def score(method, i):
    if method == SELECTED:
        return i in {0, 1, 2}
    if method == CONTROL:
        return i in {2}
    return i in {1, 3}


def make_run(path, method, split, protocol, runtime, split_hash):
    folder = path / method
    folder.mkdir(parents=True)
    write(path / "full-split.json", split)
    write(path / "split.json", {**split, "dev": []})
    project_git, benchmark_git = {"commit": RUN_COMMIT, "status": ""}, {"commit": BENCHMARK, "status": ""}
    args = {k: protocol[k] for k in ("domains", "categories", "device")}
    args.update(methods=[method], partition="test", full=True, shards=1, shard_index=0, seed=42,
                model_revision="qa", embedding_revision="e5", model="/models/qa", embedding_model="/models/e5")
    run_meta = {**runtime, "arguments": args, "project_git": project_git, "benchmark_git": benchmark_git,
                "started_at_utc": "2026-11-21T04:00:01Z", "source_split_sha256": split_hash,
                "finished": True, "embedding_weight_sha256": "e5-sha", "script_sha256": audit.sha256(b"launcher\n").hexdigest(),
                "tracked_patch_sha256": audit.sha256(b"").hexdigest(), "gpu_name": "synthetic GPU", "wall_seconds": 3600,
                "cpu_seconds": 195, "gpu_hours": 1.0, "peak_reserved_gpu_bytes": 1024, "peak_allocated_gpu_bytes": 512}
    write(path / "run-config.json", run_meta)
    cfg = {k: v for k, v in protocol.items() if k not in {"inputs", "benchmark_commit"}}
    cfg.update(partition="test", method=method)
    write(folder / "manifest.json", {"config": cfg, "project_git": project_git, "benchmark_git": benchmark_git,
                                     "status": "complete", "started_at_utc": "2026-11-21T04:00:02Z",
                                     "expected_questions": len(split["test"]), "question_ids": split["test"],
                                     "input_sha256": {**protocol["inputs"], "split": audit.digest(path / "split.json")},
                                     "index_seconds_by_domain": dict.fromkeys(DOMAINS, 1.0)})
    predictions, attempts, calls = [], [], []
    for i, q in enumerate(split["test"]):
        mark = i < 6 if method == SELECTED else i in {2, 6}
        judgment = "Correct" if mark else "Incorrect"
        retrieved = [{"message": {"id": f"Msg_{n}", "content": "Synthetic document"}, "score": 1.0}
                     for n in range(protocol["top_k"])]
        record = {**q, "question_id": q["id"], "question": "Synthetic\u2028question?", "asking_user_id": None,
                  "reference_answer": "Synthetic reference", "partition": "test", "method": method,
                  "retrieval": retrieved, "retrieved_docs": [audit.format_message(h["message"]) for h in retrieved],
                  "answer": "Synthetic answer", "prediction": "Synthetic answer",
                  "agent_output": "Final: Synthetic answer", "judge_answer": judgment, "judge_output": f"Final: {judgment}",
                  "correct": mark, "official_correct": mark, "retrieval_seconds": 0.1, "answer_seconds": 1.0,
                  "judge_seconds": 0.4, "status": "ok", "attempt": 2 if i == 0 and method == SELECTED else 1}
        call_context = {**q, "question_id": q["id"], "method": method, "attempt": record["attempt"]}
        for stage in audit.STAGES:
            limit = protocol["generation_config"][stage + "_max_tokens"]
            truncated = method == SELECTED and i == 0 and stage == "answer"
            used = limit if truncated else 2
            usage = {"prompt_tokens": 3, "completion_tokens": used, "total_tokens": used + 3}
            reason = "length" if truncated else "stop"
            record.update({stage + "_usage": usage, stage + "_finish_reason": reason})
            call = {"question": call_context, "stage": stage, "usage": usage, "max_tokens": limit,
                    "output_tokens": used, "finish_reason": reason, "seconds": 0.3,
                    "prompt_sha256": audit.sha256(stage.encode()).hexdigest()}
            if record["attempt"] == 2 and stage == "answer":
                attempts.append({**record, "attempt": 1, "status": "error", "error": "Synthetic judge failure"})
                calls.append({**call, "question": {**call_context, "attempt": 1}})
            calls.append(call)
        attempts.append(record)
        predictions.append(record)
    write_rows(folder / "predictions.jsonl", predictions)
    write_rows(folder / "attempts.jsonl", attempts)
    write_rows(path / "generation-audit.jsonl", calls)
    metrics = {**summarize(predictions), "complete": True, "expected_questions": len(predictions)}
    write(folder / "metrics.json", metrics)
    write(path / "benchmark-summary.json", {method: metrics})


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "implementation_signature", lambda root, commit: {"implementation": "frozen"})
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: b"launcher\n")
    evidence = tmp_path / "part12"
    evidence.mkdir()
    split = {"dev": make_questions("dev", 179), "test": make_questions("test", 536), "seed": 42}
    write(evidence / "split.json", split)
    split_hash = audit.digest(evidence / "split.json")
    generation = {"seed_per_call": 42, "answer_max_tokens": 512, "judge_max_tokens": 256, "temperature": 0.2}
    protocol = {"domains": list(DOMAINS), "categories": list(CATEGORIES), "device": "cuda", "limit": None, "top_k": 10,
                "partition": "all", "generation_config": generation, "benchmark_commit": BENCHMARK,
                "model_provenance": {"qa_revision": "qa", "embedding_revision": "e5", "files": {"qa": {}, "embedding": {}}},
                "inputs": {"prompts/example.txt": "pin"}}
    runtime = {**generation, "packages": {"torch": "pinned"}, "model_files": protocol["model_provenance"]["files"]}
    write(evidence / "provenance.json", {"dev": 179, "test": 536, "benchmark_commit": BENCHMARK,
                                         "split_sha256": split_hash, "protocol": protocol, "runtime": runtime})
    write(tmp_path / "ordered-reply-dev.json", {"model": {"revision": "e5", "weight_sha256": "e5-sha"}})
    methods = ("bm25", "dense", *audit.REPLY_REPORTS, *audit.STRUCTURED_METHODS)
    metrics = {m: {"overall": {"correct": sum(score(m, i) for i in range(179)), "total": 179, "completed": 179, "errors": 0}}
               for m in methods}

    def dev_report(method):
        names = (method, "bm25", "dense")
        rows = [{**q, **{name: score(name, i) for name in names}} for i, q in enumerate(split["dev"])]
        return {"project_commit": SOURCE, "benchmark_commit": BENCHMARK, "source_split_sha256": split_hash,
                "questions": rows, "metrics": {m: metrics[m] for m in names}}

    reply_hashes = {}
    for m, filename in audit.REPLY_REPORTS.items():
        path = tmp_path / filename
        write(path, dev_report(m))
        reply_hashes[m] = audit.digest(path)
    comparison, plan, freeze = (tmp_path / name for name in ("comparison.json", "plan.md", "freeze.json"))
    write(comparison, {"complete_matrix": True, "runs": {m: dev_report(m) for m in audit.STRUCTURED_METHODS},
                       "metrics": metrics, "reply_control_sha256": reply_hashes})
    plan.write_text("Development-only synthetic plan\n", encoding="utf-8")
    write(freeze, {"selected_method": SELECTED, "control_method": CONTROL, "source_commit": SOURCE,
                   "benchmark_commit": BENCHMARK, "split_sha256": split_hash, "dev_comparison_sha256": audit.digest(comparison),
                   "selection_plan_sha256": audit.digest(plan), "frozen_at_utc": "2026-11-21T04:00:00Z",
                   "selection_rule": "Choose on dev", "dev_selection": {"selected_correct": 3, "control_correct": 1,
                                                                         "total": 179, "reason": "Synthetic delta"}})
    selected, control = tmp_path / "selected", tmp_path / "control"
    make_run(selected, SELECTED, split, protocol, runtime, split_hash)
    make_run(control, CONTROL, split, protocol, runtime, split_hash)
    return selected, control, freeze, comparison, plan, evidence


def test_frozen_test_recomputes_paired_results_and_retains_retry_cost(experiment):
    result = audit.analyze(*experiment)
    overall = result["paired"]["overall"]
    assert overall == {"questions": 536, "gained": 5, "lost": 1, "net": 4,
                       "accuracy_delta": 4 / 536, "mcnemar_exact_two_sided_p": 0.21875}
    assert set(result["paired"]["by_category"]) == set(CATEGORIES)
    assert set(result["paired"]["by_domain"]) == set(DOMAINS)
    assert [row["outcome"] for row in result["questions"]].count("gain") == 5
    run = result["runs"][SELECTED]
    assert run["metrics"]["overall"]["truncated_answers"] == 1
    assert run["cost"]["retry_calls"] == run["cost"]["failed_attempts"] == 1
    assert run["cost"]["generation"]["answer"]["calls"] == 537
    assert result["total_test_gpu_hours"] == 2.0


@pytest.mark.parametrize(("artifact", "field", "value", "message"), [
    ("freeze", "selection_plan_sha256", "changed", "comparison or selection plan hash"),
    ("freeze", "frozen_at_utc", "2026-11-21T04:00:01Z", "started before"),
    ("manifest", "started_at_utc", "2026-11-20T04:00:00Z", "original attempt started before"),
    ("runtime", "script_sha256", "changed", "launcher or tracked patch"),
    ("runtime", "temperature", 0.9, "QA protocol changed"),
    ("runtime", "embedding_weight_sha256", "changed", "hybrid E5 identity differs"),
    ("metrics", "complete", False, "Archived metrics differ"),
])
def test_frozen_test_rejects_manifest_and_protocol_changes(experiment, artifact, field, value, message):
    selected, _, freeze, *_ = experiment
    files = {"freeze": freeze, "manifest": selected / SELECTED / "manifest.json",
             "runtime": selected / "run-config.json", "metrics": selected / SELECTED / "metrics.json"}
    mutate(files[artifact], lambda document: document.update({field: value}))
    with pytest.raises(ValueError, match=message):
        audit.analyze(*experiment)


@pytest.mark.parametrize(("file", "index", "action", "message"), [
    (f"{SELECTED}/predictions.jsonl", 0, "duplicate", "duplicate or extra question"),
    (f"{SELECTED}/attempts.jsonl", 0, "drop", "Invalid attempt sequence"),
    ("generation-audit.jsonl", 0, "drop", "Incomplete generation audit"),
    ("generation-audit.jsonl", 1, "drop", "Judge before answer"),
])
def test_frozen_test_rejects_broken_append_only_evidence(experiment, file, index, action, message):
    path = experiment[0] / file
    rows = audit.read_jsonl(path)
    if action == "duplicate":
        rows[-1] = rows[index]
    else:
        rows.pop(index)
    write_rows(path, rows)
    with pytest.raises(ValueError, match=message):
        audit.analyze(*experiment)


def test_test_source_must_match_frozen_implementation(experiment, monkeypatch):
    monkeypatch.setattr(audit, "implementation_signature", lambda root, commit: {"signature": commit})
    with pytest.raises(ValueError, match="Implementation differs from the freeze"):
        audit.analyze(*experiment)


def test_frozen_dev_count_must_match_per_question_results(experiment):
    _, _, freeze, comparison, *_ = experiment
    def alter_score(report):
        report["runs"][SELECTED]["questions"][0][SELECTED] = False
    mutate(comparison, alter_score)
    mutate(freeze, lambda f: f.update(dev_comparison_sha256=audit.digest(comparison)))
    with pytest.raises(ValueError, match="Development metrics changed"):
        audit.analyze(*experiment)


def test_unchanged_context_must_have_the_same_prompt(experiment):
    path = experiment[0] / "generation-audit.jsonl"
    calls = audit.read_jsonl(path)
    calls[1]["prompt_sha256"] = "d" * 64
    write_rows(path, calls)
    with pytest.raises(ValueError, match="Paired answer prompt hashes disagree"):
        audit.analyze(*experiment)


@pytest.mark.parametrize("change", ["duplicate", "reformat"])
def test_test_context_must_preserve_ten_unique_raw_passages(experiment, change):
    path = experiment[0] / SELECTED / "predictions.jsonl"
    rows = audit.read_jsonl(path)
    if change == "duplicate":
        rows[0]["retrieval"][-1] = rows[0]["retrieval"][0]
    else:
        rows[0]["retrieved_docs"][0] = "Changed passage"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="retrieved passage"):
        audit.analyze(*experiment)
