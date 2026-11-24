import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sgmem.memory_retrieval import (ABLATIONS, CONTEXT_BASES, CONTEXT_METHODS, PairedMemoryRetriever,
                                    export_rankings, public_questions, public_sources, sentence_records)
from sgmem.precomputed_retrieval import PRECOMPUTED_METHODS, RANKING_FIELDS
from sgmem.propositions import PropositionRetriever


def corpus():
    return [{"domain": "Finance", "id": str(index), "content": f"Original message {index}",
             "speaker": "User_2" if index in {21, 30, 53} else "User_1",
             "group": "Project Blue" if index in {22, 30, 53} else "Project Green",
             "role": "Supervisor" if index == 30 else "Analyst", "timestamp": "2026-01-01"}
            for index in range(55)]


def question(text="What happened?", *, domain="Finance", id="A", **fields):
    return {"domain": domain, "category": "multi_hop", "id": id, "question": text,
            "asking_user_id": "User_1", **fields}


class Retriever:
    def __init__(self, messages):
        self.hits = [{"message": row, "score": -float(index), "unit_id": f"unit-{index}"}
                     for index, row in enumerate(messages[:50])]
        self.calls = []

    def retrieve_with_trace(self, query, limit, *, asking_user_id):
        self.calls.append((query, limit, asking_user_id))
        return self.hits[:limit], {"backend": "fixture"}


def test_metadata_intersection_uses_same_fifty_and_fixed_rank_fusion():
    rows = corpus()
    retriever = Retriever(rows)
    paired = PairedMemoryRetriever(retriever, public_sources(rows))
    query = "What did User_2 report for Project Blue?"
    (base, base_trace), (meta, meta_trace) = paired.rank(query, "User_1")
    assert retriever.calls == [(query, 50, "User_1")]
    assert [item["message_id"] for item in base] == list(map(str, range(10)))
    assert meta[0] == {"message_id": "30", "score": pytest.approx(1 / 91 + 1 / 61),
                       "trace": {"unit_id": "unit-30"}}
    assert meta_trace["metadata"]["matched_ids"] == ["30"]  # And never the rank-54 source 53.
    assert meta_trace["base_order"] == base_trace["base_order"] == list(map(str, range(50)))
    assert base_trace["base_scores"]["30"] == -30
    assert meta_trace["retriever"] == {"backend": "fixture"}


@pytest.mark.parametrize("method", ABLATIONS)
def test_raw_ablation_exports_only_its_method_and_keeps_50_pool_10_hits(tmp_path, monkeypatch, method):
    rows = corpus()
    retriever = Retriever(rows)
    query = "What did User_2 report for Project Blue?"
    output = tmp_path / "out"
    monkeypatch.setattr(PairedMemoryRetriever, "query_plan", lambda *_a: pytest.fail("raw ablations do not rerank"))
    manifest = export_rankings(retriever, public_sources(rows), [question(query)], method, output, {})
    assert method in PRECOMPUTED_METHODS and f"{method}-meta" not in PRECOMPUTED_METHODS
    assert manifest["status"] == "complete" and manifest["methods"] == [method]
    assert set(path.name for path in output.iterdir()) == {"manifest.json", f"{method}.jsonl"}
    row = json.loads((output / f"{method}.jsonl").read_text())
    assert row["method"] == method and "metadata" not in row["query_trace"]
    assert [hit["message_id"] for hit in row["retrieval"]] == list(map(str, range(10)))
    assert len(row["query_trace"]["base_order"]) == 50 and retriever.calls == [(query, 50, "User_1")]


def test_sentence_units_use_boundaries_and_same_parent_max_path():
    first = {**corpus()[0], "content": 'Dr. Tan measured 3.5%. "Ready?" Keep A. Smith informed &amp; stay ready.'}
    second = {**corpus()[1], "content": "No terminal punctuation"}
    rows = [first, second]
    records = sentence_records(rows)
    assert [record["units"] for record in records] == [
        ["Dr. Tan measured 3.5%.", '"Ready?"', "Keep A. Smith informed & stay ready."],
        ["No terminal punctuation"],
    ]
    assert records[0]["input_tokens"] == records[0]["output_tokens"] == 0

    def encode(texts, **_kwargs):
        return np.asarray([[1, 0] if text in {'passage: "Ready?"', "query: User_1 What happened?"} else [0, 1]
                           for text in texts], dtype=np.float32)

    retriever = PropositionRetriever(rows, records, SimpleNamespace(encode=encode))
    winner, = retriever.retrieve("What happened?", 1, asking_user_id="User_1")
    assert winner["score"] == 1 and winner["message"] == first
    assert winner["winning_unit"]["text"] == '"Ready?"'


def test_densex_native_ranks_all_units_before_source_backtracking():
    messages = [{**corpus()[0], "content": "First. Second."}, {**corpus()[1], "content": "Third."}]
    records = sentence_records(messages)

    def encode(texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return np.asarray([[0.6, 0.8] if text == "passage: Third." else [1, 0] for text in texts])

    retriever = PropositionRetriever(messages, records, SimpleNamespace(encode=encode))
    raw = retriever.retrieve("What happened?", 2)
    native = retriever.retrieve_native("What happened?", 2)
    assert [hit["message"]["id"] for hit in raw] == ["0", "1"]
    assert native == [
        {"unit_id": "0:0", "text": "First.", "score": 1.0, "source_ids": ["0"]},
        {"unit_id": "0:1", "text": "Second.", "score": 1.0, "source_ids": ["0"]},
    ]


@pytest.mark.parametrize("method", CONTEXT_BASES)
def test_native_export_uses_one_canonical_pool_without_early_budgeting(tmp_path, method):
    rows = corpus()
    calls = []
    id_field = {"densex": "unit_id", "lightmem": "id", "raptor": "node_id"}[method]
    identifier = (lambda i: i) if method == "raptor" else (lambda i: f"u{i}")

    def native(query, top_k, **kwargs):
        calls.append((query, top_k, kwargs))
        return [{id_field: identifier(i), "text": "long unit " * 3600 if not i else f"unit {i}",
                 "score": 1 - i / 100, "source_ids": ["53", "0"],
                 **({"layer": 2, "children": [3, 4]} if method == "raptor" else {})} for i in range(top_k)]

    backend = SimpleNamespace(retrieve_nodes=native, retrieve_native=native)
    out = tmp_path / "out"
    manifest = export_rankings(backend, public_sources(rows), [question(), question(domain="Healthcare")],
                               method, out, {"messages_sha256": "a" * 64}, context_view="native")
    name = f"{method}-native"
    assert name in CONTEXT_METHODS and manifest["methods"] == [name] and manifest["top_k"] is None
    assert manifest["schema_version"] == 1 and manifest["context_view"] == "native"
    assert manifest["messages_sha256"] == "a" * 64 and manifest["context_budget"] == 3500
    assert manifest["status"] == "complete" and manifest["candidate_depth"] == 50
    assert set(p.name for p in out.iterdir()) == {"manifest.json", f"{name}.jsonl"}
    row = json.loads((out / f"{name}.jsonl").read_text(encoding="utf-8"))
    assert set(row) == RANKING_FIELDS and row["question_id"] == "A" and len(row["retrieval"]) == 50
    assert row["retrieval"][0] == {"unit_id": identifier(0), "text": "long unit " * 3600,
                                   "score": 1.0, "source_ids": ["53", "0"]}
    assert all(unit["source_ids"] == ["53", "0"] for unit in row["retrieval"])
    expected = {"asking_user_id": "User_1", **({"max_context_tokens": None} if method == "raptor" else {})}
    assert calls == [("What happened?", 50, expected)]
    if method == "raptor":
        assert row["query_trace"]["unit_metadata"][0] == {"unit_id": 0, "layer": 2, "children": [3, 4]}


def test_raw_budget_uses_fifty_validated_raw_candidates(tmp_path):
    rows = corpus()
    rows[0]["content"] = "A &amp; B"
    retriever = Retriever(rows)
    out = tmp_path / "out"
    manifest = export_rankings(retriever, public_sources(rows), [question()], "densex", out,
                               {"messages_sha256": "b" * 64}, context_view="raw-budget")
    assert manifest["context_view"] == "raw-budget" and manifest["methods"] == ["densex-raw-budget"]
    row = json.loads((out / "densex-raw-budget.jsonl").read_text(encoding="utf-8"))
    assert len(row["retrieval"]) == 50
    assert row["retrieval"][0] == {"unit_id": "0", "text": "A & B", "score": 0.0, "source_ids": ["0"]}
    assert [unit["unit_id"] for unit in row["retrieval"]] == list(map(str, range(50)))
    assert retriever.calls == [("What happened?", 50, "User_1")]


def test_invalid_native_lineage_leaves_export_incomplete(tmp_path):
    sources = public_sources(corpus())
    backend = SimpleNamespace(retrieve_native=lambda query, *_, **__: [
        {"id": "native", "text": "evidence", "score": 0.2, "source_ids": ["0" if query != "bad" else "foreign"]},
    ])
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="wrong-domain source lineage"):
        export_rankings(backend, sources, [question(), question("bad", id="B")], "lightmem", out,
                        {"messages_sha256": "a" * 64}, context_view="native")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete" and manifest["completed_queries"] == 1
    assert len((out / "lightmem-native.jsonl").read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize("query", ["What happened?", "What did User_2 say in Project Blue? after close?"])
def test_no_match_preserves_raw_order_and_scores(query):
    rows = corpus()
    rows[30] = {**rows[30], "group": "Project Green"}
    (base, _), (meta, trace) = PairedMemoryRetriever(Retriever(rows), public_sources(rows)).rank(query, "User_1")
    assert meta == base
    assert trace["metadata"]["triggered"] is False
    assert trace["metadata"]["matched_ids"] == []


@pytest.mark.parametrize("issue", ["foreign", "future", "synthetic", "duplicate", "unknown", "nan", "short"])
def test_rejects_invalid_raw_candidates(issue):
    rows = corpus()
    backend = Retriever(rows)
    hit = backend.hits[3]
    if issue == "duplicate":
        backend.hits[3] = backend.hits[0]
    elif issue == "short":
        backend.hits.pop()
    elif issue == "nan":
        hit["score"] = float("nan")
    else:
        changed = {"foreign": {"domain": "Healthcare"}, "future": {"timestamp": "2030-01-01"},
                   "synthetic": {"content": "Invented summary"}, "unknown": {"id": "missing"}}[issue]
        hit["message"] = {**hit["message"], **changed}
    with pytest.raises(ValueError, match="original source|raw candidates|retrieval score"):
        PairedMemoryRetriever(backend, public_sources(rows)).rank("What happened?")


def test_public_pack_checks_label_and_identity_for_other_domains():
    first = question()
    assert public_questions([first, question(domain="Healthcare")], "Finance") == [first]
    for label in ("answer", "reference_answer", "supporting_messages"):
        with pytest.raises(ValueError, match="without answers or support"):
            public_questions([first, question(domain="Healthcare", **{label: "secret"})], "Finance")
    with pytest.raises(ValueError, match="Duplicate"):
        public_questions([first, *[question(domain="Healthcare")] * 2], "Finance")
    with pytest.raises(ValueError, match="one domain"):
        public_sources([*corpus(), {**corpus()[0], "domain": "Healthcare"}])
    with pytest.raises(ValueError, match="label fields"):
        public_sources([{**corpus()[0], "answer": "secret"}])


def test_paired_retriever_checks_immutable_source_even_if_backend_mutates_its_input():
    rows = corpus()
    backend = Retriever(rows)
    paired = PairedMemoryRetriever(backend, public_sources(rows))
    backend.hits[0]["message"]["content"] = "mutated original"
    with pytest.raises(ValueError, match="modified original source"):
        paired.rank("What happened?")


def test_durable_pair_leaves_an_explicit_incomplete_manifest(tmp_path):
    rows = corpus()
    backend = Retriever(rows)
    original = backend.retrieve_with_trace

    def interrupted(query, *args, **kwargs):
        if query == "stop":
            raise RuntimeError("simulated query failure")
        return original(query, *args, **kwargs)

    backend.retrieve_with_trace = interrupted
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="simulated query failure"):
        export_rankings(backend, public_sources(rows), [question(), question("stop", id="B")], "lightmem", output, {})
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "incomplete" and manifest["completed_queries"] == 1
    assert [key["question_id"] for key in manifest["expected_queries"]] == ["A", "B"]
    for name in ("lightmem", "lightmem-meta"):
        row = json.loads((output / f"{name}.jsonl").read_text())
        assert row["method"] == name and row["question_id"] == "A"
        assert all(set(hit) == {"message_id", "score", "trace"} for hit in row["retrieval"])
    with pytest.raises(FileExistsError):
        export_rankings(backend, public_sources(rows), [question()], "lightmem", output, {})


def test_cli_keeps_full_questions_hash_and_only_retrieves_selected_domain(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/retrieve_memories.py"
    run = runpy.run_path(str(script))
    messages, questions, output = (tmp_path / name for name in ("messages.jsonl", "questions.jsonl", "out"))
    messages.write_text("\n".join(map(json.dumps, corpus())), encoding="utf-8")
    questions.write_text("\n".join(map(json.dumps, [question(), question(domain="Healthcare")])), encoding="utf-8")
    backend = Retriever(corpus())
    monkeypatch.setitem(run["main"].__globals__, "load_retriever", lambda *_: (backend, {"pin": "fixture"}, {}))
    monkeypatch.setattr(sys, "argv", [str(script), "--method", "raptor", "--messages", str(messages),
                        "--questions", str(questions), "--index", str(tmp_path), "--output", str(output)])
    run["main"]()
    data = json.loads((output / "manifest.json").read_text())
    assert data["status"] == "complete" and data["completed_queries"] == 1
    assert data["questions_sha256"] == hashlib.sha256(questions.read_bytes()).hexdigest()
    assert data["index"] == {"pin": "fixture"}
    assert backend.calls == [("What happened?", 50, "User_1")]

    context_out = tmp_path / "context-out"
    backend.calls.clear()
    monkeypatch.setattr(sys, "argv", [str(script), "--method", "raptor", "--context-view", "raw-budget",
                        "--messages", str(messages), "--questions", str(questions), "--index", str(tmp_path),
                        "--output", str(context_out)])
    run["main"]()
    data = json.loads((context_out / "manifest.json").read_text(encoding="utf-8"))
    assert data["methods"] == ["raptor-raw-budget"] and data["completed_queries"] == 1
    assert data["messages_sha256"] == hashlib.sha256(messages.read_bytes()).hexdigest()
    assert backend.calls == [("What happened?", 50, "User_1")]


@pytest.mark.parametrize("method", ["densex", "lightmem", "raptor", "hipporag2", *ABLATIONS])
def test_loader_requests_local_models_and_built_indexes(tmp_path, monkeypatch, method):
    script = Path(__file__).resolve().parents[1] / "scripts/retrieve_memories.py"
    module = runpy.run_path(str(script))
    load = module["load_retriever"]
    rows, model = corpus(), tmp_path / "local_model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    weights = b"fixture E5 weights"
    (model / "model.safetensors").write_bytes(weights)
    monkeypatch.setitem(load.__globals__, "E5_WEIGHTS", hashlib.sha256(weights).hexdigest())
    args = SimpleNamespace(method=method, embedding_path=model, embedding_revision=module["E5_REVISION"], qa_tokenizer=model,
                           tokenizer_revision="qwen-pin", llm_base_url=None, llm_revision=None, llm_name=None,
                           index=None if method == "densex-sentences" else tmp_path / "index")

    def model_loader(path, **kwargs):
        assert str(model) == path and kwargs.pop("local_files_only") is True
        assert kwargs.get("device", "cpu") == "cpu"
        if "device" in kwargs:
            assert kwargs["model_kwargs"] == {"use_safetensors": True}
        else:
            assert kwargs["use_fast"] is True
        return "local-model"

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=model_loader))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=model_loader)))
    if method in ("raptor", "raptor-leaf"):
        args.index.write_bytes(b"simulated RAPTOR index")
        tree = SimpleNamespace(provenance={
            "embedding_revision": args.embedding_revision, "tokenizer_revision": args.tokenizer_revision,
            "embedding_files_sha256": module["_model_files"](model, weights=True),
            "tokenizer_files_sha256": module["_model_files"](model, weights=False),
        })

        def raptor(corpus, built_tree, encoder, tokenizer, *, leaf_only=False):
            assert corpus == rows and built_tree is tree and encoder == tokenizer == "local-model"
            assert leaf_only == (method == "raptor-leaf")
            return method

        monkeypatch.setitem(sys.modules, "sgmem.raptor_memory", SimpleNamespace(
            RaptorTree=SimpleNamespace(load=lambda _: tree), RaptorRetriever=raptor))
    elif method == "densex-sentences":
        from sgmem import propositions

        def sentences(corpus, records, encoder):
            assert corpus == rows and encoder == "local-model"
            assert [record["units"] for record in records] == [[row["content"]] for row in rows]
            return method

        monkeypatch.setattr(propositions, "PropositionRetriever", sentences)
    else:
        args.index.mkdir()
        settings = {"embedding_device": "cpu", "embedding_path": str(model), "embedding_revision": args.embedding_revision}
        manifest = {"status": "complete", "protocol": {"embedding_revision": "e5-pin"},
                    "identity": {"settings": settings}}
        name = "sgmem-hipporag.json" if method in ("hipporag2", "hipporag2-dpr") else "manifest.json"
        (args.index / name).write_text(json.dumps(manifest))
        if method in ("hipporag2", "hipporag2-dpr"):
            def hippo(corpus, cache, **config):
                assert config.pop("dense_only", False) == (method == "hipporag2-dpr")
                assert corpus == rows and cache == args.index and config == settings

                def build(*, require_existing):
                    assert require_existing
                    return method

                return SimpleNamespace(build=build)

            monkeypatch.setitem(sys.modules, "sgmem.hipporag_memory", SimpleNamespace(HippoRAGMemory=hippo))
        elif method in ("lightmem", "lightmem-pre"):
            def lightmem(index, domain, **kwargs):
                assert index == args.index and domain == "Finance" and kwargs["device"] == "cpu"
                assert kwargs["stage"] == ("pre" if method == "lightmem-pre" else "post")
                return SimpleNamespace(messages={row["id"]: row for row in rows}, name=method)

            monkeypatch.setitem(sys.modules, "sgmem.lightmem_memory", SimpleNamespace(
                LightMemIndex=SimpleNamespace(load=lightmem)))
        else:
            from sgmem import propositions

            (args.index / "propositions.jsonl").write_text('{"domain":"Other"}\n{"domain":"Finance"}\n')

            def densex(corpus, records, encoder):
                assert corpus == rows and records == [{"domain": "Finance"}] and encoder == "local-model"
                return "densex"

            monkeypatch.setattr(propositions, "PropositionRetriever", densex)
    backend, index, models = load(args, public_sources(rows))
    assert getattr(backend, "name", backend) == method
    if method == "densex-sentences":
        assert index == {"units": "deterministic_sentences", "unit_count": len(rows), "model_extraction_calls": 0}
    else:
        assert index["path"] == str(args.index)
    if method in ("raptor", "raptor-leaf"):
        assert set(models["tokenizer"]["files_sha256"]) == {"config.json", "tokenizer.json"}


@pytest.mark.parametrize("method", ["densex", "raptor"])
@pytest.mark.parametrize("change", ["revision", "weights"])
def test_reject_nonfrozen_e5_before_encoder_loading(tmp_path, monkeypatch, method, change):
    script = Path(__file__).resolve().parents[1] / "scripts/retrieve_memories.py"
    module = runpy.run_path(str(script))
    load = module["load_retriever"]
    (tmp_path / "config.json").write_text("{}")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"original")
    monkeypatch.setitem(load.__globals__, "E5_WEIGHTS", hashlib.sha256(weight.read_bytes()).hexdigest())
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda *_a, **_kw: pytest.fail("must not load the encoder")))
    revision = module["E5_REVISION"]
    if change == "weights":
        weight.write_bytes(b"modified")
    else:
        revision = "different"
    args = SimpleNamespace(method=method, embedding_path=tmp_path, embedding_revision=revision, index=tmp_path)
    with pytest.raises(ValueError, match="differs from the frozen E5-small-v2"):
        load(args, public_sources(corpus()))


@pytest.mark.parametrize("change", ["embedding", "tokenizer"])
def test_raptor_rejects_same_revision_different_model_files(tmp_path, monkeypatch, change):
    script = Path(__file__).resolve().parents[1] / "scripts/retrieve_memories.py"
    module = runpy.run_path(str(script))
    load = module["load_retriever"]
    embedding, tokenizer = tmp_path / "e5", tmp_path / "qwen"
    embedding.mkdir()
    tokenizer.mkdir()
    for path in (embedding, tokenizer):
        (path / "config.json").write_text("{}")
    weights = embedding / "model.safetensors"
    weights.write_bytes(b"frozen E5 fixture")
    (tokenizer / "tokenizer.json").write_text('{"a":1}')
    monkeypatch.setitem(load.__globals__, "E5_WEIGHTS", hashlib.sha256(weights.read_bytes()).hexdigest())
    tree = SimpleNamespace(provenance={
        "embedding_revision": module["E5_REVISION"], "tokenizer_revision": "qwen-pin",
        "embedding_files_sha256": module["_model_files"](embedding, weights=True),
        "tokenizer_files_sha256": module["_model_files"](tokenizer, weights=False),
    })
    index = tmp_path / "raptor.npz"
    index.write_bytes(b"built index")
    def forbidden(*_args, **_kwargs):
        pytest.fail("fingerprint mismatch must fail before loading a model")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=forbidden))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=forbidden)))
    monkeypatch.setitem(sys.modules, "sgmem.raptor_memory", SimpleNamespace(
        RaptorTree=SimpleNamespace(load=lambda _path: tree), RaptorRetriever=forbidden))
    if change == "embedding":
        (embedding / "config.json").write_text('{"a":1}')
    else:
        (tokenizer / "tokenizer.json").write_text('{"b":1}')
    args = SimpleNamespace(method="raptor", embedding_path=embedding, embedding_revision=module["E5_REVISION"],
                           qa_tokenizer=tokenizer, tokenizer_revision="qwen-pin", index=index)
    with pytest.raises(ValueError, match=f"different or unspecified {change}_files_sha256"):
        load(args, public_sources(corpus()))
