import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sgmem import hipporag_memory as adapter


def message(message_id, content, domain="Finance"):
    return {"id": message_id, "domain": domain, "content": content, "speaker": f"speaker-{message_id}",
            "group": "G1", "timestamp": "2026-01-01", "reply_to": None}


@pytest.fixture
def upstream(monkeypatch):
    """Exercise the real adapter against the official API shape without model calls."""
    fact = ("x", "connects", "y")
    body = '[[ ## fact_after_filter ## ]]\n{"fact": [["x", "connects", "y"]]}\n[[ ## completed ## ]]'
    state = SimpleNamespace(output=body, finish="stop", embeddings=[], indexes={}, builds=0, engine_calls=0,
                            calls=[], failures=False, have_facts=True, ppr_runs=0, dense_runs=0,
                            openie_calls=[], openie_rendered=[], openie_requests=[], extracted_triples=[],
                            ner=('{"named_entities": ["x", "y"]}', "stop"),
                            triple=('{"triples": [["x", "connects", "y"]]}', "stop"))

    class Encoder:
        def __init__(self, path, *, device, local_files_only):
            assert local_files_only and device == "cpu"

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] and kwargs["convert_to_numpy"]
            state.embeddings.append(texts)
            vectors = np.zeros((len(texts), 384), np.float32)
            vectors[:, 0] = 1
            return vectors

    class BaseEmbedding:
        def __init__(self, config):
            self.global_config = config

    class Filter:
        def __init__(self, engine):
            self.llm_infer_fn = engine.extraction_llm.infer

        def llm_call(self, query, fact_input):
            assert json.loads(fact_input) == {"fact": [list(fact)]}
            response, _, _ = self.llm_infer_fn(messages=[{"role": "user", "content": query}])
            return response

    def text_processing(values):
        return [" ".join("".join(char if char.isalnum() or char.isspace() else " "
                                 for char in value.casefold()).split()) for value in values]

    def infer(*, messages, response_format=None, max_new_tokens=None, frequency_penalty=None):
        if response_format is not None:
            assert frequency_penalty == 0.15
            state.openie_requests.append((messages, response_format, max_new_tokens))
            ner = response_format["json_schema"]["name"] == "hipporag_ner"
            raw, finish = state.ner if ner else state.triple
            return raw, {"finish_reason": finish}, True
        state.calls.append(messages)
        assert frequency_penalty is None
        if state.failures:
            raise ConnectionError("server disconnected")
        return state.output, {"finish_reason": state.finish, "completion_tokens": 9}, False

    def render(*, name, passage, named_entity_json=None):
        messages = [
            {"role": "system", "content": f"Official {name} prompt"},
            {"role": "user", "content": "First example source"},
            {"role": "assistant", "content": "First example output"},
            {"role": "user", "content": "Second example source"},
            {"role": "assistant", "content": "Second example output"},
            {"role": "user", "content": f"Passage: {passage}\nNamed entities: {named_entity_json}"},
        ]
        state.openie_rendered.append(messages)
        return messages

    class OpenIE:
        def __init__(self, llm_model, max_workers=8, ner_max_tokens=2048, triple_max_tokens=4096):
            self.llm_model = llm_model
            self.max_workers = max_workers
            self.ner_max_tokens = ner_max_tokens
            self.triple_max_tokens = triple_max_tokens
            self.prompt_template_manager = SimpleNamespace(render=render)

    class Engine:
        def __init__(self, *, global_config, embedding_model, index_identity):
            state.engine_calls += 1
            assert global_config.openie_mode == "online" and global_config.linking_top_k == 5
            assert global_config.vector_store_type == "parquet"
            assert global_config.llm_base_url == "http://127.0.0.1:18951/v1"
            self.config = global_config
            self.embedding = embedding_model
            assert len(index_identity) == 64
            self.entity_embedding_store = SimpleNamespace()
            self.extraction_llm = SimpleNamespace(infer=infer)
            self.openie = OpenIE(self.extraction_llm, max_workers=global_config.openie_max_workers)

        def index(self, chunks):
            state.builds += 1
            self.chunk_metadata = {f"chunk-{i}": chunk.metadata for i, chunk in enumerate(chunks)}
            for i, chunk in enumerate(chunks):
                chunk_key = f"chunk-{i}"
                state.openie_calls.append(("ner", chunk_key))
                ner = self.openie.ner(chunk_key, chunk.content)
                state.openie_calls.append(("triple_extraction", chunk_key))
                triple = self.openie.triple_extraction(chunk_key, chunk.content, ner.unique_entities)
                state.extracted_triples.append(triple.triples)
            self.embedding.batch_encode([chunk.content for chunk in chunks])
            self.entity_embedding_store.embedding_model.batch_encode(["entity"])
            self.embedding.batch_encode([str(fact)])
            state.indexes[self.config.save_dir] = chunks

        def prepare_retrieval_objects(self):
            self.chunks = state.indexes[self.config.save_dir]
            self.passage_node_keys = list(range(len(self.chunks)))
            self.chunk_metadata = {key: self.chunks[key].metadata for key in self.passage_node_keys}
            self.chunk_embedding_store = SimpleNamespace(
                get_row=lambda key: {"content": self.chunks[key].content},
            )

        def rerank_facts(self, query, scores):
            candidates = [fact] if state.have_facts else []
            try:
                indices, selected, _ = self.rerank_filter.rerank(query, candidates, [0], 5) if candidates else ([], [], {})
                trace = {"facts_before_rerank": candidates, "facts_after_rerank": selected}
            except Exception as exc:
                return [], [], {"error": str(exc)}
            return indices, selected, trace

        def retrieve(self, queries, num_to_retrieve):
            query, = queries
            self.embedding.batch_encode(query, instruction="query_to_passage", norm=True)
            _, selected, _ = self.rerank_facts(query, [1])
            state.ppr_runs += bool(selected)
            state.dense_runs += not selected
            return self.rank_passages(num_to_retrieve)

        def retrieve_dpr(self, queries, num_to_retrieve):
            query, = queries
            self.embedding.batch_encode(query, instruction="query_to_passage", norm=True)
            state.dense_runs += 1
            return self.rank_passages(num_to_retrieve)

        def rank_passages(self, num_to_retrieve):
            rows = list(reversed(self.chunks))[:num_to_retrieve]
            scores = [0.9 if row.content != "Low" else 0.3 for row in rows]
            pairs = sorted(zip(rows, scores), key=lambda pair: -pair[1])
            return [SimpleNamespace(docs=[row.content for row, _ in pairs], doc_scores=[score for _, score in pairs],
                                    doc_metadata=[row.metadata for row, _ in pairs])]

        def close(self):
            pass

        def get_graph_info(self):
            return {"num_passage_nodes": len(self.chunks)}

    monkeypatch.setattr(adapter, "_verify_upstream", lambda module: None)
    modules = {"hipporag": SimpleNamespace(HippoRAG=Engine, Chunk=SimpleNamespace),
               "hipporag.embedding_model": SimpleNamespace(BaseEmbeddingModel=BaseEmbedding),
               "hipporag.information_extraction.openie_openai": SimpleNamespace(OpenIE=OpenIE),
               "hipporag.llm": SimpleNamespace(find_gateway=lambda name: None),
               "hipporag.rerank": SimpleNamespace(DSPyFilter=Filter),
               "hipporag.utils.config_utils": SimpleNamespace(BaseConfig=SimpleNamespace),
               "hipporag.utils.misc_utils": SimpleNamespace(text_processing=text_processing,
                                                            NerRawOutput=SimpleNamespace,
                                                            TripleRawOutput=SimpleNamespace),
               "sentence_transformers": SimpleNamespace(SentenceTransformer=Encoder)}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


@pytest.fixture
def pack(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"placeholder")
    rows = [message("M1", "Shared &amp; safe"), message("M2", "Equal"),
            message("M3", "Shared & safe"), message("M4", "Low")]
    kwargs = {"embedding_path": model, "embedding_revision": "e5-test", "llm_revision": "qwen-test"}
    return rows, tmp_path / "cache", kwargs


def test_mapping_threshold_ties_and_e5_namespace(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    hits, trace = memory.retrieve_with_trace("question", 2, asking_user_id="User_2")
    assert upstream.embeddings[:4] == [
        ["passage: Shared & safe", "passage: Equal", "passage: Low"], ["query: entity"],
        ["passage: ('x', 'connects', 'y')"], ["query: User_2 question"],
    ]
    assert [hit["message"]["id"] for hit in hits] == ["M1", "M2"]
    assert {tuple(row["message_ids"]) for row in trace["passages"]} == {("M1", "M3"), ("M2",)}
    assert trace["expanded_candidates"] == 3 and trace["passages_scored"] == 3
    recognition = trace["recognition"]
    assert recognition["mode"] == "ppr" and recognition["cache_hit"] is False
    assert recognition["parsed_triples"] == [["x", "connects", "y"]]
    assert recognition["selected_triples"] == recognition["candidate_triples"] == [("x", "connects", "y")]
    assert [hit["message"]["id"] for hit in memory.retrieve("question", 9)] == ["M1", "M2", "M3", "M4"]
    assert upstream.ppr_runs == 2 and not upstream.dense_runs
    memory.close()


def test_scores_expand_all_sources_before_the_cap(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory([rows[0], rows[2], rows[1], rows[3]], path, **kwargs).build()
    hits, _ = memory.retrieve_with_trace("question", 2)
    assert [hit["message"]["id"] for hit in hits] == ["M1", "M3"]
    assert hits[1]["message"]["speaker"] == "speaker-M3"


def test_valid_empty_and_no_facts_are_explicit(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    upstream.output = "[[ ## fact_after_filter ## ]]\n{'fact': []}\n[[ ## completed ## ]]"
    _, trace = memory.retrieve_with_trace("question", 2)
    assert trace["recognition"]["mode"] == "dense_valid_empty"
    assert trace["recognition"]["parsed_triples"] == []
    upstream.have_facts = False
    _, trace = memory.retrieve_with_trace("question", 2)
    assert trace["recognition"] == {"mode": "dense_no_facts", "candidate_triples": [], "selected_triples": []}
    assert upstream.dense_runs == 2 and len(upstream.calls) == 1


def test_dpr_control_reuses_same_index_and_source_expansion_without_recognition(upstream, pack):
    rows, path, kwargs = pack
    adapter.HippoRAGMemory(rows, path, **kwargs).build().close()
    control = adapter.HippoRAGMemory(rows, path, **kwargs, dense_only=True).build(require_existing=True)
    upstream.failures = True
    hits, trace = control.retrieve_with_trace("question", 3, asking_user_id="User_2")
    assert [hit["message"]["id"] for hit in hits] == ["M1", "M2", "M3"]
    assert trace["recognition"] == {"mode": "dense_control", "candidate_triples": [], "selected_triples": []}
    assert upstream.dense_runs == 1 and upstream.ppr_runs == 0 and not upstream.calls
    assert upstream.builds == 1 and trace["query"] == "User_2 question"
    control.close()


@pytest.mark.parametrize("failure", ["network", "invalid", "missing_field", "length"])
def test_bad_filter_never_becomes_dense_fallback(upstream, pack, failure):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    if failure == "network":
        upstream.failures = True
    elif failure == "length":
        upstream.finish = "length"
    elif failure == "missing_field":
        upstream.output = '{"fact": []}'
    else:
        upstream.output = '[[ ## fact_after_filter ## ]]\n{"fact": [["only one part"]]}'
    with pytest.raises(adapter.RecognitionError, match="recognition failed") as error:
        memory.retrieve("question", 2)
    assert error.value.trace["mode"] == "error" and error.value.trace["candidate_triples"]
    assert "raw_output" in error.value.trace or failure == "network"
    assert upstream.dense_runs == upstream.ppr_runs == 0


def test_recognition_uses_official_normalization_and_deduplicates(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    upstream.output = '[[ ## fact_after_filter ## ]]\n{"fact": [[" X!", "CONNECTS", "Y."], ["x", "connects", "y"]]}'
    _, trace = memory.retrieve_with_trace("question", 2)
    assert trace["recognition"]["mode"] == "ppr"
    assert trace["recognition"]["selected_triples"] == [("x", "connects", "y")]


def test_recognition_rejects_unknown_triple_even_with_valid_selection(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    triples = [["x", "connects", "y"], ["invented", "uses", "unknown"]]
    upstream.output = '[[ ## fact_after_filter ## ]]\n' + json.dumps({"fact": triples})
    with pytest.raises(adapter.RecognitionError, match="outside its candidates") as error:
        memory.retrieve("question", 2)
    assert error.value.trace["parsed_triples"] == triples
    assert error.value.trace["candidate_triples"] == [("x", "connects", "y")]
    assert upstream.dense_runs == upstream.ppr_runs == 0


def test_reopen_reuses_index_and_rejects_drift(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    memory.close()
    adapter.HippoRAGMemory(rows, path, **kwargs).build(require_existing=True).close()
    assert upstream.builds == 1
    assert json.loads((path / "sgmem-hipporag.json").read_text())["graph"] == {"num_passage_nodes": 3}
    with pytest.raises(ValueError, match="identity changed"):
        adapter.HippoRAGMemory([*rows[:-1], message("M4", "New body")], path, **kwargs).build()
    (kwargs["embedding_path"] / "model.safetensors").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="identity changed"):
        adapter.HippoRAGMemory(rows, path, **kwargs).build()
    with pytest.raises(ValueError, match="one domain"):
        adapter.HippoRAGMemory([rows[0], message("M1", "Another", "Healthcare")], path, **kwargs)
    with pytest.raises(ValueError, match="loopback"):
        adapter.HippoRAGMemory(rows, path, llm_base_url="https://api.openai.com/v1", **kwargs)


@pytest.mark.parametrize(("stage", "response", "finish", "failure"), [
    ("ner", '{"named_entities":["Alice","report",', "length", "finish reason"),
    ("triple_extraction", '{"triples":[["Alice","owns","report"],', "length", "finish reason"),
    ("triple_extraction", '{"triples":["abc",["X","Y"],["Z","owns",42]]}', "stop", "three-string arrays"),
    ("triple_extraction", '{"triples":[["@Ops","and","@Data","can confirm current mappings"]]}',
     "stop", "three-string arrays"),
    ("ner", '{"named_entities":["Alice",42]}', "stop", "nonempty strings"),
    ("ner", '["Alice"]', "stop", "object with only named_entities"),
    ("ner", json.dumps({"named_entities": ["Alice"] * 65}), "stop", "64-item limit"),
    ("triple_extraction", json.dumps({"triples": [["Alice", "owns", "report"]] * 33}),
     "stop", "32-item limit"),
])
def test_openie_guards_reject_original_outputs_before_graph_completion(
        upstream, pack, stage, response, finish, failure):
    rows, path, kwargs = pack
    setattr(upstream, "ner" if stage == "ner" else "triple", (response, finish))
    memory = adapter.HippoRAGMemory(rows, path, **kwargs)
    with pytest.raises(adapter.OpenIEError, match=failure) as error:
        memory.build()
    trace = error.value.trace
    assert trace["stage"] == stage and trace["chunk_id"] == "chunk-0"
    assert trace["message_ids"] == ["M1", "M3"] and trace["raw_output"] == response
    assert trace["metadata"]["finish_reason"] == finish and trace["metadata"]["cache_hit"] is True
    assert upstream.openie_calls == ([('ner', 'chunk-0')] if stage == "ner" else
                                    [('ner', 'chunk-0'), ('triple_extraction', 'chunk-0')])
    assert json.loads((path / "openie-errors.jsonl").read_text()) == trace
    assert json.loads((path / "sgmem-hipporag.json").read_text())["status"] == "building"
    assert memory.engine is None and not upstream.indexes


def test_openie_valid_empty_responses_complete_index(upstream, pack):
    rows, path, kwargs = pack
    upstream.ner = '{"named_entities":[]}', "stop"
    upstream.triple = '{"triples":[]}', "stop"
    adapter.HippoRAGMemory(rows, path, **kwargs).build().close()
    assert json.loads((path / "sgmem-hipporag.json").read_text())["status"] == "complete"
    assert not (path / "openie-errors.jsonl").exists()


def test_openie_drops_only_blank_triples_and_records_raw_output(upstream, pack):
    rows, path, kwargs = pack
    valid = ["Alice", "owns", "report"]
    raw = json.dumps({"triples": [valid, ["Alice", "owns", "   "]]})
    upstream.triple = raw, "stop"
    adapter.HippoRAGMemory(rows, path, **kwargs).build().close()
    assert upstream.extracted_triples == [[valid]] * 3
    records = [json.loads(line) for line in (path / "openie-blank-triples.jsonl").read_text().splitlines()]
    assert records[0] == {"chunk_id": "chunk-0", "raw_output": raw, "dropped_blank_triples": 1}
    assert len(records) == 3


def test_openie_requests_keep_examples_and_bind_schemas_to_index(upstream, pack, monkeypatch):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    memory.close()
    assert len(upstream.openie_requests) == len(upstream.openie_rendered) == 6
    for i, ((messages, response_format, max_new_tokens), official) in enumerate(
        zip(upstream.openie_requests, upstream.openie_rendered, strict=True)
    ):
        stage = "triple_extraction" if i % 2 else "ner"
        field = "triples" if i % 2 else "named_entities"
        assert messages[1:] == official[1:]
        assert official[0]["content"] == f"Official {stage} prompt"
        assert messages[0]["content"].startswith(official[0]["content"])
        assert adapter.OPENIE_SCOPE in messages[0]["content"]
        assert max_new_tokens == (4096 if i % 2 else 2048)
        assert response_format["type"] == "json_schema"
        spec = response_format["json_schema"]
        assert spec["name"] == f"hipporag_{stage}" and spec["strict"] is True
        schema = spec["schema"]
        assert schema["required"] == [field] and schema["additionalProperties"] is False
        assert schema["properties"][field]["maxItems"] == (32 if stage == "triple_extraction" else 64)
        items = schema["properties"][field]["items"]
        if stage == "triple_extraction":
            assert items["minItems"] == items["maxItems"] == 3
            assert '{"named_entities": []}' in messages[-1]["content"]
            items = items["items"]
        assert items == {"type": "string", "minLength": 1}
        assert response_format == memory.identity["openie_profile"][stage]["response_format"]
    monkeypatch.setattr(adapter, "OPENIE_SCOPE", "A changed instruction")
    changed = adapter.HippoRAGMemory(rows, path, **kwargs)
    assert changed.identity["openie_profile_sha256"] != memory.identity["openie_profile_sha256"]
    with pytest.raises(ValueError, match="identity changed"):
        changed.build()


def test_ner_excludes_unmentioned_example_entities_and_records_them(upstream, pack):
    rows, path, kwargs = pack
    upstream.ner = json.dumps({"named_entities": ["Shared", "Radio City", "Shared"]}), "stop"
    adapter.HippoRAGMemory(rows, path, **kwargs).build().close()
    records = [json.loads(line) for line in (path / "ner-grounding.jsonl").read_text().splitlines()]
    assert records[0] == {"chunk_id": "chunk-0", "accepted": ["Shared"], "unmatched": ["Radio City"]}
    assert '{"named_entities": ["Shared"]}' in upstream.openie_requests[1][0][-1]["content"]
    assert adapter._entity_mentioned("Ops", "Ask @Ops by Friday")
    assert not adapter._entity_mentioned("Ops", "workshops tomorrow")


def test_require_existing_rejects_missing_or_building_before_engine(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs)
    with pytest.raises(ValueError, match="completed HippoRAG index"):
        memory.build(require_existing=True)
    assert not path.exists()
    path.mkdir()
    manifest = path / "sgmem-hipporag.json"
    manifest.write_text(json.dumps({"status": "building", "identity": memory.identity}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed HippoRAG index"):
        memory.build(require_existing=True)
    assert upstream.engine_calls == 0
    assert json.loads(manifest.read_text())["status"] == "building"


def test_provenance_mismatch_fails_instead_of_selecting_an_unrelated_source(upstream, pack):
    rows, path, kwargs = pack
    memory = adapter.HippoRAGMemory(rows, path, **kwargs).build()
    memory.engine.chunks[0].metadata["message_ids"] = ["M2"]
    with pytest.raises(ValueError, match="passage provenance"):
        memory.retrieve("question", 2)


def test_cli_writes_only_selected_domain_and_keeps_failures_explicit(upstream, pack, monkeypatch):
    rows, path, kwargs = pack
    messages, queries, output = (path.parent / name for name in ("messages.jsonl", "queries.jsonl", "out.jsonl"))
    messages.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    questions = [{"domain": domain, "category": "multi_hop", "id": "1", "question": domain,
                  "asking_user_id": "User_1", "answer": "gold"} for domain in ("Finance", "Healthcare")]
    queries.write_text("\n".join(json.dumps(row) for row in questions), encoding="utf-8")
    argv = ["hipporag_memory", "--messages", str(messages), "--index", str(path),
            "--embedding-path", str(kwargs["embedding_path"]), "--embedding-revision", "e5-test",
            "--llm-revision", "qwen-test", "--queries", str(queries), "--top-k", "3", "--output"]
    monkeypatch.setattr(sys, "argv", [*argv, str(output)])
    adapter.main()
    row, = [json.loads(line) for line in output.read_text().splitlines()]
    assert (row["domain"], row["question_id"], row["status"]) == ("Finance", "1", "ok")
    assert [hit["message"]["id"] for hit in row["hits"]] == ["M1", "M2", "M3"]
    assert upstream.calls == [[{"role": "user", "content": "User_1 Finance"}]]

    bad_output = output.with_name("bad.jsonl")
    monkeypatch.setattr(sys, "argv", [*argv, str(bad_output)])
    upstream.output = "malformed"
    with pytest.raises(adapter.RecognitionError):
        adapter.main()
    rejected = json.loads(bad_output.read_text())
    assert rejected["status"] == "error" and rejected["trace"]["recognition"]["raw_output"] == "malformed"
