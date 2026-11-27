import json
from collections import Counter
import re
from types import SimpleNamespace

import numpy as np
import pytest

from sgmem.lightmem_memory import (
    DIMENSIONS, E5_MAX_TOKENS, PATCH, SLEEP_RESPONSE_POLICY, E5Encoder, LightMemIndex, _build_index,
    _check_updates, _complete_response, _endpoint, _normalize, _turn,
)


def message(identity, group="room-a", domain="Finance", timestamp="2026-01-02 12:00:00"):
    return {
        "domain": domain, "group": group, "id": identity, "content": f"Text &amp; details: {identity}",
        "speaker": "Morgan", "timestamp": timestamp, "role": "analyst",
        "reply_to": "a" if identity == "b" else None, "phase_name": "planning", "topic": "notes",
    }


def vector(index=0):
    return np.eye(1, DIMENSIONS, index, dtype=np.float32)[0]


def test_truncated_generation_retries_once_without_changing_prompt_and_counts_usage():
    calls = []

    def create(**params):
        calls.append(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length" if len(calls) == 1 else "stop",
                                     message=SimpleNamespace(content='{"action":"ignore"}'))],
            usage=SimpleNamespace(prompt_tokens=4456, completion_tokens=20, total_tokens=4476),
        )

    params = {"messages": [{"role": "user", "content": "Keep source facts"}],
              "max_tokens": 4096, "seed": 42}
    result = _complete_response(create, params)
    assert [call["max_tokens"] for call in calls] == [4096, 8192]
    assert all(call["messages"] == params["messages"] and call["seed"] == 42 for call in calls)
    assert params["max_tokens"] == 4096
    assert result.usage.prompt_tokens == 8912 and result.usage.total_tokens == 8952


@pytest.mark.parametrize("reason,expected_calls", [("length", 2), ("content_filter", 1)])
def test_incomplete_generation_stops_with_diagnostic(reason, expected_calls):
    calls = []

    def create(**params):
        calls.append(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content="partial"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    messages = [{"role": "system", "content": "Return a JSON action"},
                {"role": "user", "content": "Target\n候选 \"one\""}]
    with pytest.raises(ValueError, match=f"finish_reason={reason}") as failure:
        _complete_response(create, {"max_tokens": 4096, "messages": messages})
    assert len(calls) == expected_calls
    diagnostic, request = str(failure.value).split("request_messages=", 1)
    assert f"attempts={expected_calls}" in diagnostic
    assert f"'total_tokens': {12 * expected_calls}" in diagnostic
    assert "response_tail='partial'" in diagnostic
    assert json.loads(request) == messages


def test_missing_usage_is_not_retried_or_treated_as_success():
    def create(**params):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")], usage=None)

    with pytest.raises(ValueError, match="token usage"):
        _complete_response(create, {"max_tokens": 4096})


@pytest.fixture
def patched_sleep_update():
    # Exercise the complete replacement method without importing heavyweight dependencies.
    patch = PATCH.read_text(encoding="utf-8")
    marker = "     def _call_update_llm(self, system_prompt, target_entry, candidate_sources):"
    _, found, rest = patch.partition(marker)
    assert found
    body = (found + rest).split("\ndiff --git ", 1)[0]
    source = "\n".join(line[1:] for line in body.splitlines() if line.startswith((" ", "+")))
    scope = {"json": json, "SLEEP_REWRITE_PROMPT": "Rewrite only the selected evidence"}
    import textwrap
    exec(compile(textwrap.dedent(source), str(PATCH), "exec"), scope)
    return scope["_call_update_llm"]


def sleep_case(method, outputs):
    requests = []
    responses = iter(outputs)

    def generate_response(**request):
        requests.append(request)
        value = next(responses)
        return (value if isinstance(value, str) else json.dumps(value)), {
            "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
        }

    result = method(SimpleNamespace(generate_response=generate_response), "Classify the relationship",
                    {"payload": {"memory": "original target"}},
                    [{"payload": {"memory": text}} for text in ("irrelevant source", "selected evidence")])
    return result, requests


@pytest.mark.parametrize("relation,index,action", [
    ("unrelated", -1, "ignore"), ("equivalent", 1, "ignore"), ("contradiction", 1, "delete"),
])
def test_sleep_classifies_without_rewriting(patched_sleep_update, relation, index, action):
    result, requests = sleep_case(patched_sleep_update, [{"relation": relation, "candidate_index": index}])
    assert result["action"] == action and result["relation"] == relation
    assert len(requests) == 1 and result["usage"]["total_tokens"] == 12
    schema = requests[0]["response_format"]["json_schema"]
    assert schema["strict"] and schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["properties"]["candidate_index"]["enum"] == list(range(-1, 10))


def test_sleep_rewrites_only_selected_evidence_and_counts_both_calls(patched_sleep_update):
    result, requests = sleep_case(patched_sleep_update, [
        {"relation": "refinement", "candidate_index": 1}, {"new_memory": "Target with its supported detail."},
    ])
    assert result["action"] == "update" and result["selected_candidate_index"] == 1
    assert result["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert len(requests) == 2
    assert json.loads(requests[1]["messages"][1]["content"]) == {
        "target": "original target", "later_evidence": "selected evidence",
    }
    assert requests[1]["response_format"]["json_schema"]["schema"]["properties"]["new_memory"]["maxLength"] == 1024


@pytest.mark.parametrize("selection", [
    '{"relation":', {"relation": "unknown", "candidate_index": 1},
    {"relation": "refinement", "candidate_index": 2}, {"relation": "refinement", "candidate_index": True},
    {"relation": "unrelated", "candidate_index": 0}, {"relation": "equivalent", "candidate_index": -1},
    {"relation": "refinement", "candidate_index": 1, "extra": 0},
])
def test_sleep_rejects_invalid_selection(patched_sleep_update, selection):
    with pytest.raises(ValueError):
        sleep_case(patched_sleep_update, [selection])


@pytest.mark.parametrize("revision", [
    '{"new_memory":', {"new_memory": {}}, {"new_memory": " "},
    {"new_memory": "x" * 1025}, {"new_memory": "updated", "extra": 0},
])
def test_sleep_rejects_invalid_revision(patched_sleep_update, revision):
    with pytest.raises(ValueError):
        sleep_case(patched_sleep_update, [{"relation": "refinement", "candidate_index": 1}, revision])


def test_sleep_records_an_exact_unchanged_revision_as_noop(patched_sleep_update):
    result, requests = sleep_case(patched_sleep_update, [
        {"relation": "refinement", "candidate_index": 1}, {"new_memory": "original target"},
    ])
    assert result["action"] == "ignore" and result["unchanged_revision"] is True
    assert result["relation"] == "refinement" and result["usage"]["total_tokens"] == 24
    assert len(requests) == 2


def test_sleep_rejects_a_schema_boundary_revision_without_losing_the_target(patched_sleep_update):
    result, requests = sleep_case(patched_sleep_update, [
        {"relation": "refinement", "candidate_index": 1}, {"new_memory": "x" * 1024},
    ])
    assert result == {"action": "ignore", "relation": "refinement", "selected_candidate_index": 1,
                      "revision_rejected": "length_limit", "usage": {
                          "prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24,
                      }}
    assert len(requests) == 2
    before = [{"id": key, "source_ids": [key], "time_stamp": f"2026-01-02 12:00:0{i}"}
              for i, key in enumerate(("target", "selected"))]
    event = {**result, "target": "target", "source_ids": ["target"],
             "selected_candidate": "selected", "candidates": ["selected"], "candidate_source_ids": [["selected"]]}
    _check_updates(before, before, [event])
    with pytest.raises(RuntimeError, match="Unexpected change"):
        _check_updates(before, before[1:], [event])


def test_sleep_lineage_includes_only_the_selected_candidate():
    before = [{"id": key, "source_ids": [key], "time_stamp": f"2026-01-02 12:00:0{i}"}
              for i, key in enumerate(("target", "unused", "selected"))]
    event = {"target": "target", "source_ids": ["target"], "action": "update",
             "candidates": ["unused", "selected"], "candidate_source_ids": [["unused"], ["selected"]],
             "selected_candidate": "selected"}
    updated = {**before[0], "source_ids": ["target", "selected"]}
    _check_updates(before, [updated, *before[1:]], [event])
    with pytest.raises(RuntimeError, match="provenance"):
        _check_updates(before, [{**updated, "source_ids": ["target", "unused", "selected"]}, *before[1:]], [event])
    with pytest.raises(RuntimeError, match="outside the candidates"):
        _check_updates(before, [updated, *before[1:]], [{**event, "selected_candidate": "absent"}])


class FakeNative:
    """Implements the patched boundary with deterministic CPU data and no model imports."""

    def __init__(self, name, *, fail=False, corrupt=False, omit_turn=False, fail_sleep=False):
        self.name = name
        self.points = {}
        self.calls = []
        self.fail = fail
        self.fail_sleep = fail_sleep
        self.corrupt = corrupt
        self.omit_turn = omit_turn
        self.buffer = []
        self.compressor = SimpleNamespace(compress=self.compress)
        self.segmenter = SimpleNamespace(propose_cut=lambda _: [1])
        self.text_embedder = SimpleNamespace(calls=Counter(), length_stats=Counter())
        self.senmem_buffer_manager = SimpleNamespace(
            buffer=self.buffer, big_buffer=[], cut_with_segmenter=self.cut,
        )
        self.shortmem_buffer_manager = SimpleNamespace(buffer=[])
        self.embedding_retriever = SimpleNamespace(
            get_all=lambda: [self.points[key] for key in sorted(self.points)], insert=self.insert,
        )
        self.manager = SimpleNamespace(
            meta_text_extract=self.extract,
            _call_update_llm=lambda *_: {
                "action": "update", "new_memory": "Consolidated fact", "usage": {"total_tokens": 2},
            },
        )
        self.tokens = Counter()

    def insert(self, *, vectors, payloads, ids):
        self.points.update({identity: {"id": identity, "vector": vector, "payload": payload}
                            for identity, vector, payload in zip(ids, vectors, payloads, strict=True)})

    @staticmethod
    def compress(messages, tokenizer):
        for row in messages:
            if row["content"]:
                row["content"] = row["content"].split()[-1]
        return messages

    def get_token_statistics(self):
        return {"llm": {phase: {"total_tokens": self.tokens[phase]}
                        for phase in ("add_memory", "update")}}

    def cut(self):
        self.segmenter.propose_cut([row["content"] for row in self.buffer])
        self.text_embedder.calls["turn"] += len(self.buffer)
        return [self.buffer.copy()]

    def add_memory(self, messages, *, force_segment=False, force_extract=False):
        if self.fail:
            raise RuntimeError("extraction problem")
        if messages:
            user, empty = messages
            assert user["role"] == "user" and user["speaker_id"] == "Morgan"
            assert empty == {"role": "assistant", "content": "", "time_stamp": user["time_stamp"],
                             "source_time_stamp": user["time_stamp"]}
            assert " & " in user["content"]
            self.calls.append(user["dia_id"])
            self.compressor.compress(messages, SimpleNamespace(encode=str.split))
            self.buffer.append(user)
            return {"api_call_nums": 0}
        assert force_segment and force_extract
        self.compressor.compress([], SimpleNamespace(encode=str.split))
        segments = self.senmem_buffer_manager.cut_with_segmenter()
        if self.omit_turn:
            segments[0].pop()
        self.manager.meta_text_extract(extract_list=[segments])
        self.buffer.clear()
        self.tokens["add_memory"] += 8
        self.calls.append("flush")
        return {"api_call_nums": 1}

    def extract(self, *, extract_list):
        for user in extract_list[0][0]:
            identity = f"{self.name}:{user['dia_id']}"
            self.embedding_retriever.insert(vectors=[vector()], ids=[identity], payloads=[{
                "memory": f"fact {user['dia_id']}", "anchor_dia_id": user["dia_id"],
                "source_ids": [user["dia_id"]], "entry_type": "factual", "topic_id": 0,
            }])

    def construct_update_queue_all_entries(self, *, top_k, keep_top_n):
        assert (top_k, keep_top_n) == (20, 10)
        assert not self.calls or self.calls[-1] == "flush"
        self.calls.append("queue")

    def offline_update_all_entries(self, *, score_threshold, max_candidate_sources):
        assert score_threshold == 0.9
        assert max_candidate_sources == 10
        assert self.calls[-1] == "queue"
        self.calls.append("sleep")
        if self.fail_sleep:
            # Simulate a partially applied pass. Resume must restore the pre-sleep store.
            self.points.clear()
            raise RuntimeError("sleep generation problem")
        if len(self.points) < 2:
            return
        target, candidate, *_ = self.points.values()
        result = self.manager._call_update_llm("update", target, [candidate])
        self.tokens["update"] += result["usage"]["total_tokens"]
        payload = {**target["payload"], "memory": result["new_memory"]}
        if not self.corrupt:
            payload["source_ids"] = [*payload["source_ids"], *candidate["payload"]["source_ids"]]
        self.points[target["id"]] = {**target, "payload": payload, "vector": vector(1)}


def test_full_boundary_group_isolation_resume_and_backtracking(tmp_path):
    rows = [message("z", "room-b"), message("b", timestamp="2026-01-02T12:00:01Z"), message("a")]
    calls = []

    def new_group(name):
        instance = FakeNative(name)
        calls.append(instance)
        return instance

    path = tmp_path / "memory"
    manifest = _build_index(rows, path, {"source": "fixture"}, new_group)
    assert [stage.calls for stage in calls] == [
        ["a", "b", "flush", "queue", "sleep"], ["z", "flush", "queue", "sleep"],
    ]
    metric = manifest["metrics"][0]
    assert metric["input_turns"] == metric["changed_turns"] == 2
    assert metric["compressed_tokens"] < metric["input_tokens"]
    assert metric["segmentation_calls"] == metric["released_segments"] == 1
    assert metric["llm"]["update"]["total_tokens"] == 2
    assert metric["embedding_calls"]["turn"] == 2
    assert metric["update_actions"] == {"update": 1}
    assert metric["extraction_source_turns"] == 2
    assert metric["source_coverage"] == {"pre": {"anchors": 2, "lineage": 2}, "post": {"anchors": 2, "lineage": 2}}
    assert metric["embedding_length"] == {"long_passages": 0, "long_passage_chunks": 0}
    assert sorted(json.loads(line)["id"] for line in (path / "sources.jsonl").read_text().splitlines()) == ["a", "b", "z"]

    encoder = SimpleNamespace(query=lambda text: vector(1) if text == "Morgan question" else vector())
    post = LightMemIndex.load(path, "Finance", encoder=encoder)
    hits = post.retrieve("question", 3, asking_user_id="Morgan")
    assert [hit["message"]["id"] for hit in hits] == ["a", "b", "z"]
    assert hits[0]["score"] == hits[1]["score"] == 1.0
    assert hits[1]["message"]["reply_to"] == "a"
    assert hits[1]["message"]["content"] == "Text &amp; details: b"
    assert hits[0]["unit_id"] == hits[1]["unit_id"]
    unit = post.retrieve_native("question", 1, asking_user_id="Morgan")[0]
    assert unit["text"] == "Consolidated fact"
    assert unit["source_ids"] == ["a", "b"]
    pre = LightMemIndex.load(path, "Finance", stage="pre", encoder=encoder)
    assert all(entry["text"] != "Consolidated fact" for entry in pre.retrieve_native("question", 3))
    assert all(len(entry["source_ids"]) == 1 for entry in pre.entries)
    with pytest.raises(ValueError, match="fewer than top_k=4"):
        post.retrieve("question", 4)

    assert _build_index(rows, path, {"source": "fixture"}, new_group, resume=True) == manifest
    assert len(calls) == 2
    with pytest.raises(ValueError, match="configuration changed"):
        _build_index(rows, path, {"source": "other"}, new_group, resume=True)
    with pytest.raises(ValueError, match="configuration changed"):
        _build_index(rows, path, {"source": "fixture", "sleep_response_policy": SLEEP_RESPONSE_POLICY},
                     new_group, resume=True)
    old_manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    del old_manifest["protocol"]["e5_length_policy"]
    (path / "manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Rebuild this LightMem index"):
        LightMemIndex.load(path, "Finance", encoder=encoder)
    with pytest.raises(ValueError, match="configuration changed"):
        _build_index(rows, path, {"source": "fixture"}, new_group, resume=True)


def test_checkpoint_keeps_only_completed_channels(tmp_path):
    rows = [message("a"), message("b", "room-b")]
    path = tmp_path / "memory"
    calls = []

    def failing(name):
        calls.append(name)
        return FakeNative(name, fail=len(calls) == 2)

    with pytest.raises(RuntimeError, match="extraction problem"):
        _build_index(rows, path, {"source": "fixture"}, failing)
    with pytest.raises(ValueError, match="incomplete"):
        LightMemIndex.load(path, "Finance")
    assert len(list(path.glob("*.npz"))) == 1

    def resume(name):
        calls.append(name)
        return FakeNative(name)

    _build_index(rows, path, {"source": "fixture"}, resume, resume=True)
    assert len(calls) == 3 and calls[1] == calls[2] != calls[0]
    statuses = [json.loads(line)["status"] for line in (path / "attempts.jsonl").read_text().splitlines()]
    assert statuses == ["complete", "failed", "complete"]


def test_missing_update_provenance_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="record update provenance"):
        _build_index([message("a"), message("b", timestamp="2026-01-02 12:00:01")], tmp_path / "bad", {},
                     lambda name: FakeNative(name, corrupt=True))


def test_sleep_failure_resumes_exact_pre_store_without_reextracting(tmp_path):
    rows = [message("a"), message("b", timestamp="2026-01-02 12:00:01")]
    path = tmp_path / "memory"
    with pytest.raises(RuntimeError, match="sleep generation problem"):
        _build_index(rows, path, {}, lambda name: FakeNative(name, fail_sleep=True))
    assert len(list(path.glob("*.pre.npz"))) == 1
    assert len(list(path.glob("*.npz"))) == 1
    with pytest.raises(ValueError, match="incomplete"):
        LightMemIndex.load(path, "Finance")
    restored = []

    def resume(name):
        worker = FakeNative(name, fail=True)  # Any accidental new extraction must fail.
        restored.append(worker)
        return worker

    manifest = _build_index(rows, path, {}, resume, resume=True)
    assert restored[0].calls == ["queue", "sleep"]
    metric, = manifest["metrics"]
    assert metric["reused_ingestion"] is True
    assert metric["extraction_source_turns"] == 2
    assert metric["llm"] == {"add_memory": {"total_tokens": 8}, "update": {"total_tokens": 2}}
    assert metric["embedding_calls"]["turn"] == 2
    assert not list(path.glob("*.pre.npz"))
    encoder = SimpleNamespace(query=lambda _: vector())
    before = LightMemIndex.load(path, "Finance", stage="pre", encoder=encoder)
    after = LightMemIndex.load(path, "Finance", stage="post", encoder=encoder)
    assert len(before.entries) == len(after.entries) == 2
    assert {entry["text"] for entry in before.entries} == {"fact a", "fact b"}
    assert "Consolidated fact" in {entry["text"] for entry in after.entries}


def test_changed_pre_sleep_checkpoint_identity_fails_before_restore(tmp_path):
    rows = [message("a")]
    path = tmp_path / "memory"
    with pytest.raises(RuntimeError, match="sleep generation problem"):
        _build_index(rows, path, {}, lambda name: FakeNative(name, fail_sleep=True))
    checkpoint, = path.glob("*.pre.npz")
    with np.load(checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        vectors = saved["vectors"]
        stored = saved["stored_vectors"]
    metadata["sources_sha256"] = "wrong"
    np.savez(checkpoint, vectors=vectors, stored_vectors=stored, metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match="checkpoint identity"):
        _build_index(rows, path, {}, FakeNative, resume=True)


def test_empty_buffers_do_not_hide_a_turn_missing_from_extraction(tmp_path):
    with pytest.raises(RuntimeError, match="every source turn in channel order"):
        _build_index([message("a")], tmp_path / "missing", {}, lambda name: FakeNative(name, omit_turn=True))


def test_source_instants_sort_chronologically_with_stable_id_ties():
    rows = [
        message("z", timestamp="2026-01-01T10:00:00-02:00"),
        message("a", timestamp="2026-01-01T12:00:00Z"),
        message("next", timestamp="2026-01-01T11:00:01-01:00"),
        message("b", timestamp="2026-01-01 12:00:00"),
    ]
    normalized = _normalize(rows)
    assert [row["id"] for row in normalized] == ["a", "b", "z", "next"]
    assert [row["timestamp"] for row in normalized] == [rows[i]["timestamp"] for i in (1, 3, 0, 2)]
    timing = [user["source_time_stamp"] for user, _ in map(_turn, normalized)]
    assert timing == ["2026-01-01T12:00:00+00:00"] * 3 + ["2026-01-01T12:00:01+00:00"]


def test_sleep_delete_and_mismatched_candidate_ledger():
    first = {"id": "one", "source_ids": ["a"], "time_stamp": "2026-01-02 12:00:00"}
    second = {"id": "two", "source_ids": ["b"], "time_stamp": "2026-01-02 12:00:01"}
    deletion = {"target": "one", "source_ids": ["a"], "action": "delete",
                "candidates": ["two"], "candidate_source_ids": [["b"]]}
    _check_updates([first, second], [second], [deletion])
    with pytest.raises(RuntimeError, match="update inputs"):
        _check_updates([first, second], [second], [dict(deletion, candidate_source_ids=[["a"]])])
    tied = dict(second, time_stamp="2026-01-02T04:00:00-08:00")
    with pytest.raises(RuntimeError, match="not strictly newer"):
        _check_updates([first, tied], [tied], [deletion])


def test_rejects_sidecar_change_and_label_fields(tmp_path):
    path = tmp_path / "index"
    with pytest.raises(ValueError, match="label-free"):
        _build_index([dict(message("a"), answer="hidden")], path, {}, FakeNative)
    assert not path.exists()
    _build_index([message("a")], path, {}, FakeNative)
    with (path / "sources.jsonl").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="sidecar changed"):
        LightMemIndex.load(path, "Finance")


class FakeTokenizer:
    is_fast = True

    def __call__(self, text, *, add_special_tokens, truncation, return_offsets_mapping):
        assert not truncation and return_offsets_mapping
        offsets = [item.span() for item in re.finditer(r"\w+|[^\s\w]", text)]
        if add_special_tokens:
            offsets = [(0, 0), *offsets, (0, 0)]
        return {"offset_mapping": offsets}

    def encode(self, text, *, add_special_tokens, truncation):
        return list(range(len(self(text, add_special_tokens=add_special_tokens, truncation=truncation,
                                   return_offsets_mapping=True)["offset_mapping"])))


def fake_e5(encode):
    return E5Encoder(SimpleNamespace(get_sentence_embedding_dimension=lambda: DIMENSIONS, encode=encode,
                                     tokenizer=FakeTokenizer(), max_seq_length=E5_MAX_TOKENS))


def test_e5_prefixes():
    calls = []

    def fake_encode(text, **kwargs):
        calls.append((text, kwargs))
        return vector()

    encoder = fake_e5(fake_encode)
    encoder.embed("fact")
    encoder.embed_turn("turn")
    encoder.query("question")
    assert [text for text, _ in calls] == ["passage: fact", "query: turn", "query: question"]
    assert all(options == {"normalize_embeddings": True, "convert_to_numpy": True} for _, options in calls)


def test_oversized_e5_passage_keeps_exact_text_and_weights_including_tail():
    calls = []

    def fake_encode(texts, **kwargs):
        calls.append(texts)
        return np.stack([vector(1) if "β終点" in text else vector() for text in texts])

    encoder = fake_e5(fake_encode)
    text = "  αβ \t" * 551 + "\nβ終点  "
    original = np.asarray(encoder.embed(text))
    parts = calls[0]
    assert len(parts) == 2
    assert all(part.startswith("passage: ") and encoder._token_count(part) <= E5_MAX_TOKENS for part in parts)
    assert "".join(part[len("passage: "):] for part in parts) == text
    weights = [encoder._token_count(part[len("passage: "):], special=False) for part in parts]
    expected = weights[0] * vector() + weights[1] * vector(1)
    np.testing.assert_allclose(original, expected / np.linalg.norm(expected), rtol=1e-6)
    assert not np.allclose(original, encoder.embed(text.replace("β終点", "without-tail")))
    assert encoder.length_stats == {"long_passages": 2, "long_passage_chunks": 4}
    for use, encode in (("query", encoder.query), ("turn", encoder.embed_turn)):
        with pytest.raises(ValueError, match=f"{use} has .* exceeds {E5_MAX_TOKENS}"):
            encode(text)
    assert len(calls) == 2


def test_local_endpoint(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _endpoint("http://127.0.0.1:8765/v1/") == "http://127.0.0.1:8765/v1"
    for invalid in ("https://server/v1", "http://127.0.0.1/v1", "http://127.0.0.1:8765/v1?redirect=1"):
        with pytest.raises(ValueError, match="loopback"):
            _endpoint(invalid)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    with pytest.raises(ValueError, match="Unset OPENROUTER"):
        _endpoint("http://127.0.0.1:8765/v1")
