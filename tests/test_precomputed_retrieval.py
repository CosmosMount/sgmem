import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest

from sgmem import runner
from sgmem.data import load_messages, load_questions
from sgmem.memory_retrieval import CONTEXT_BUDGET, export_rankings, public_sources
from sgmem.qa import build_answer_prompt, format_message
from test_full_run import Client, _args
from test_memory_context import FramedTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json")


@pytest.fixture
def replay_run(tmp_path):
    args = _args(tmp_path)
    args.method, args.top_k = "densex", 10
    benchmark = Path(args.benchmark)
    messages = load_messages(benchmark / SOURCE)
    sources = public_sources([{"domain": "Finance", **message} for message in messages])
    write_rows = runpy.run_path(str(ROOT / "scripts/prepare_memory_inputs.py"))["write_rows"]
    digest = write_rows(tmp_path / "input.jsonl", sources.values())
    questions = load_questions(benchmark / "questions/Finance/multi_hop.jsonl", "Finance", "multi_hop")
    questions = [{key: value for key, value in row.items() if key != "answer"} for row in questions]

    class Retriever:
        def retrieve(self, query, top_k, *, asking_user_id):
            return [{"message": row, "score": 1.0, "trace": {"native": "GENERATED_NOT_QA"}}
                    for row in reversed(list(sources.values()))][:top_k]

    ranking = tmp_path / "rankings"
    export_rankings(Retriever(), sources, questions, args.method, ranking, {"messages_sha256": digest})
    args.precomputed = [str(ranking)]
    return args, ranking, messages


def test_replay_resolves_source_ids_with_identical_answer_and_judge_inputs(replay_run, monkeypatch):
    args, ranking, messages = replay_run
    args.limit = 1  # The complete ranking can cover a scoped QA pilot.
    calls = []

    class Capture(Client):
        def complete(self, system, prompt, max_tokens):
            calls.append((system, prompt, max_tokens))
            return super().complete(system, prompt, max_tokens)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "build_retriever", lambda *args: pytest.fail("replay builds no retriever"))
        output = runner.run(args, client_factory=Capture)
    actual = calls[:]
    calls.clear()
    prediction = json.loads((output / "predictions.jsonl").read_text())
    assert [row["message"] for row in prediction["retrieval"]] == messages[::-1]
    assert prediction["retrieval"][0]["trace"] == {"native": "GENERATED_NOT_QA"}
    assert prediction["query_trace"]["base_order"] == [row["id"] for row in messages[::-1]]
    assert all("GENERATED_NOT_QA" not in prompt for _, prompt, _ in actual)
    assert "Budget & schedule" in actual[0][1]
    assert "SECRET_GOLD" not in actual[0][1] and "SECRET_GOLD" in actual[1][1]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["precomputed"]["Finance"]["ranking_sha256"] == hashlib.sha256((ranking / "densex.jsonl").read_bytes()).hexdigest()
    assert manifest["index_seconds_by_domain"] == {}

    class InlineRetriever:
        def retrieve(self, query, top_k, *, asking_user_id):
            return [{"message": original, "score": 1.0} for original in messages[::-1]]

    monkeypatch.setattr(runner, "build_retriever", lambda *args: InlineRetriever())
    del args.precomputed
    args.method, args.output = "bm25", str(Path(args.output).with_name("inline"))
    inline = runner.run(args, client_factory=Capture)
    assert calls == actual
    manifest = json.loads((inline / "manifest.json").read_text())
    assert "precomputed" not in manifest and "retrieval_timing" not in manifest


def _rewrite(path, change):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(("bad", "match"), [
    ("incomplete", "incomplete or unsupported"), ("source", "source digest"),
    ("text", "wording"), ("asker", "asker"), ("method", "method"),
    ("wrong-domain", "ranking question identity"), ("dropped", "manifest membership"),
    ("support", "fields differ"), ("identifier", "Unknown or repeated"),
    ("repeat", "Unknown or repeated"), ("generated", "raw source ID"),
    ("score", "score or trace"), ("scope", "omit 1 QA-selected"),
])
def test_preflight_rejects_invalid_artifacts_before_evaluator(replay_run, bad, match):
    args, ranking, _ = replay_run
    path = ranking / "densex.jsonl"
    manifest_path = ranking / "manifest.json"
    if bad == "source":
        source = Path(args.benchmark) / SOURCE
        source.write_text(source.read_text().replace("&amp;", "&"))
    elif bad == "incomplete":
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "incomplete"
        manifest_path.write_text(json.dumps(manifest))
    elif bad == "scope":
        manifest = json.loads(manifest_path.read_text())
        manifest["expected_queries"].pop()
        manifest["completed_queries"] -= 1
        manifest_path.write_text(json.dumps(manifest))
        _rewrite(path, lambda rows: rows.pop())
    else:
        def edit(rows):
            row = rows[0]
            hit = row["retrieval"][0]
            if bad == "dropped":
                rows.pop()
            elif bad == "repeat":
                row["retrieval"][1]["message_id"] = hit["message_id"]
            elif bad in {"identifier", "generated", "score"}:
                hit.update({"identifier": {"message_id": "missing"}, "generated": {"message": "fake"},
                            "score": {"score": float("nan")}}[bad])
            else:
                row.update({"text": {"question": "changed"}, "asker": {"asking_user_id": None},
                            "method": {"method": "densex-meta"}, "wrong-domain": {"domain": "Healthcare"},
                            "support": {"supporting_messages": ["Msg_1"]}}[bad])

        _rewrite(path, edit)
    with pytest.raises(ValueError, match=match):
        runner.run(args, client_factory=lambda *a, **k: pytest.fail("evaluator must not be initialized"))
    assert not Path(args.output).exists()


def test_resume_rejects_mutated_ranking_bytes(replay_run):
    args, ranking, _ = replay_run
    runner.run(args, client_factory=Client)
    args.resume = True
    _rewrite(ranking / "densex.jsonl", lambda rows: rows[0]["query_trace"].update(note="drift"))
    with pytest.raises(ValueError, match="Resume rejected"):
        runner.run(args, client_factory=lambda *a, **k: pytest.fail("resume must reject fingerprint drift"))


def _context_replay(replay_run, view):
    args, ranking, messages = replay_run
    sources = public_sources([{"domain": "Finance", **message} for message in messages])
    questions = load_questions(Path(args.benchmark) / "questions/Finance/multi_hop.jsonl", "Finance", "multi_hop")
    questions = [{key: value for key, value in row.items() if key != "answer"} for row in questions]

    class Retriever:
        def retrieve(self, query, top_k, *, asking_user_id):
            return [{"message": row, "score": 1.0} for row in list(sources.values())[::-1]][:top_k]

        def retrieve_native(self, query, top_k, *, asking_user_id):
            return [
                {"unit_id": "N0", "text": "word " * 1800, "score": 1.0, "source_ids": ["Msg_1"]},
                {"unit_id": "N1", "text": "word " * 1700, "score": 0.5, "source_ids": ["Msg_2", "Msg_3"]},
                {"unit_id": "N2", "text": "Small.", "score": 0.0, "source_ids": ["Msg_3"]},
            ]

    args.method = f"densex-{view}"
    context_dir = ranking.with_name(view)
    original_manifest = json.loads((ranking / "manifest.json").read_text())
    export_rankings(Retriever(), sources, questions, "densex", context_dir,
                    {"messages_sha256": original_manifest["messages_sha256"]}, context_view=view)
    args.precomputed = [str(context_dir)]
    args.limit = 1
    return args, messages, context_dir


class ContextClient(Client):
    tokenizer = FramedTokenizer()
    drift = 0

    def complete(self, system, prompt, max_tokens):
        if system == "agent":
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            self.last_usage = {"prompt_tokens": self.tokenizer(rendered, return_tensors="pt").input_ids.shape[1] + self.drift}
        return super().complete(system, prompt, max_tokens)


@pytest.mark.parametrize("view", ["native", "raw-budget"])
def test_context_replay_uses_exact_rendered_usage_and_persists_lineage(replay_run, view):
    args, messages, _ = _context_replay(replay_run, view)
    answer_inputs = []

    class Capture(ContextClient):
        def complete(self, system, prompt, max_tokens):
            if system == "agent":
                answer_inputs.append(prompt)
            return super().complete(system, prompt, max_tokens)

    output = runner.run(args, client_factory=Capture)
    row = json.loads((output / "predictions.jsonl").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    trace = row["context_trace"]
    assert manifest["config"]["top_k"] is None
    assert manifest["context_protocol"] == {"view": view, "budget_tokens": 3500, "candidate_depth": 50,
                                           "count": "rendered_prompt_delta"}
    assert trace["view"] == view and trace["evidence_tokens"] <= CONTEXT_BUDGET
    assert trace["prompt_tokens"] == row["answer_usage"]["prompt_tokens"]
    assert answer_inputs == [build_answer_prompt(row, row["retrieved_docs"])]
    assert all(set(hit) == {"unit_id", "text", "score", "source_ids"} for hit in row["retrieval"])
    if view == "native":
        assert len(row["retrieval"]) == 1 and row["retrieval"][0]["source_ids"] == ["Msg_1"]
        assert trace["dropped"][0]["source_ids"] == ["Msg_2", "Msg_3"]
        assert trace["first_rejected_prompt_tokens"] - trace["empty_prompt_tokens"] > CONTEXT_BUDGET
    else:
        assert row["retrieved_docs"] == [format_message(message) for message in reversed(messages)]
        assert not trace["dropped"]


@pytest.mark.parametrize(("view", "invalid", "match"), [
    ("native", "unknown", "source lineage"), ("native", "generated-key", "context unit fields"),
    ("native", "question", "wording"), ("native", "source", "source digest"),
    ("native", "budget", "context protocol"), ("raw-budget", "fake-text", "differs from its original source"),
])
def test_context_preflight_rejects_changed_artifacts_before_client(replay_run, view, invalid, match):
    args, _, directory = _context_replay(replay_run, view)
    path = directory / f"{args.method}.jsonl"
    if invalid == "source":
        source = Path(args.benchmark) / SOURCE
        source.write_text(source.read_text().replace("&amp;", "&"))
    elif invalid == "budget":
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["context_budget"] -= 1
        manifest_path.write_text(json.dumps(manifest))
    elif invalid == "question":
        _rewrite(path, lambda rows: rows[0].update(question="Tampered question"))
    else:
        bad = {"unknown": {"source_ids": ["elsewhere"]}, "generated-key": {"message": {"content": "fake"}},
               "fake-text": {"text": "Generated content pretending to be original"}}[invalid]
        _rewrite(path, lambda rows: rows[0]["retrieval"][-1].update(bad))
    with pytest.raises(ValueError, match=match):
        runner.run(args, client_factory=lambda *a, **k: pytest.fail("malformed contexts must fail before clients"))
    assert not Path(args.output).exists()


def test_context_replay_requires_tokenizer_and_pins_actual_usage(replay_run):
    args, _, _ = _context_replay(replay_run, "native")
    with pytest.raises(ValueError, match="answer client tokenizer"):
        runner.run(args, client_factory=Client)
    assert json.loads((Path(args.output) / "manifest.json").read_text())["status"] == "incomplete"
    args.output += "-usage"

    class Drift(ContextClient):
        drift = 1

    with pytest.raises(RuntimeError, match="Model call failed"):
        runner.run(args, client_factory=Drift)
    row = json.loads((Path(args.output) / "predictions.jsonl").read_text())
    assert row["status"] == "error" and "differs from actual answer prompt" in row["error"]
    assert row["answer_usage"]["prompt_tokens"] == row["context_trace"]["prompt_tokens"] + 1


def test_formal_launcher_preserves_clean_gate_for_precomputed(tmp_path, monkeypatch):
    model, ranking = tmp_path / "model", tmp_path / "ranking"
    model.mkdir()
    ranking.mkdir()
    script = runpy.run_path(str(ROOT / "scripts/run_local_baselines.py"))
    monkeypatch.setitem(script["main"].__globals__, "load_frozen_split", lambda *a: {
        "dev": [{"domain": "Finance", "category": "multi_hop", "id": "Q1"}], "test": [],
    })
    monkeypatch.setattr(runner, "_git_info", lambda path: {"commit": "c", "status": " M file"})
    monkeypatch.setattr(sys, "argv", ["run_local_baselines.py", "--methods", "densex", "--precomputed", str(ranking),
                                     "--model", str(model), "--embedding-model", str(model),
                                     "--full", "--output", str(tmp_path / "formal")])
    with pytest.raises(SystemExit) as result:
        script["main"]()
    assert result.value.code == 2 and not (tmp_path / "formal").exists()
