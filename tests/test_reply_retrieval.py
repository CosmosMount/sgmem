import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sgmem.retrieval import BM25ReplyRetriever, BM25Retriever, build_retriever
from sgmem.cli import main
from test_baselines import _write_benchmark


def messages(count):
    return [{"id": str(i), "group": "A", "content": f"word{i}", "reply_to": None}
            for i in range(count)]


def ids(hits):
    return [r["message"]["id"] for r in hits]


def test_two_hops_recover_descendant_and_sibling_but_not_cross_group_or_third_hop():
    rows = messages(17)
    for child, parent in [(10, 0), (11, 10), (1, 12), (13, 12), (14, 11), (15, 0)]:
        rows[child]["reply_to"] = str(parent)
    rows[15]["group"] = "B"
    rows[16]["reply_to"] = "not-visible"
    values = np.array([20] * 8 + [10, 9, 1, 8, 1, 7, 8.9, 8.9, 8.9])
    retriever = BM25ReplyRetriever(rows)
    base = list(range(10))
    direct, _ = retriever.expand("What did they suggest?", values, base, hops=1)
    hits, pool = retriever.expand("What did they suggest?", values, base)
    assert ids(direct) == [str(i) for i in base]
    assert ids(hits) == [str(i) for i in range(8)] + ["11", "13"]
    assert [h["reply_path"] for h in hits[-2:]] == [["0", "10", "11"], ["1", "12", "13"]]
    assert {"14", "15", "16"}.isdisjoint(pool)


def test_linked_tail_counts_once_and_a_single_new_candidate_keeps_the_best_filler():
    rows = messages(12)
    rows[9]["reply_to"] = "0"
    rows[10]["reply_to"] = "1"
    retriever = BM25ReplyRetriever(rows)
    values = np.array([20] * 8 + [11, 10, 9, 0])
    base = list(range(10))
    hits, _ = retriever.expand("What SHOULD we do?", values, base)
    assert ids(hits) == [str(i) for i in range(8)] + ["9", "10"]
    rows[9]["reply_to"] = None
    hits, _ = BM25ReplyRetriever(rows).expand("What SHOULD we do?", values, base)
    assert ids(hits) == [str(i) for i in range(9)] + ["10"]


def test_query_gate_weak_replies_and_small_corpora_preserve_content_ranking():
    rows = messages(12)
    rows[10]["reply_to"] = "0"
    retriever = BM25ReplyRetriever(rows)
    base = list(range(10))
    values = np.array([20] * 8 + [11, 10, 4, 1])
    for question in ("What should we do?", "What is the current value?"):
        hits, _ = retriever.expand(question, values, base)
        assert ids(hits) == [str(i) for i in base]
    values[10] = 9
    hits, pool = retriever.expand("What is the current value?", values, base)
    assert pool == ids(hits) == [str(i) for i in base]
    tiny = build_retriever("bm25-reply", rows[:3])
    assert tiny.retrieve("What should word1 do?", 10) == BM25Retriever(rows[:3]).retrieve("What should word1 do?", 10)
    assert tiny.retrieve("!!!", 10) == []
    with pytest.raises(ValueError, match="positive"):
        tiny.retrieve("word1", 0)


def test_busy_parent_caps_edges_and_ties_prefer_an_existing_hit():
    rows = messages(30)
    for i in range(1, 30):
        rows[i]["reply_to"] = "0"
    rows[0]["reply_to"] = "0"
    retriever = BM25ReplyRetriever(rows)
    assert retriever.links[0] == [1, 2, 3, 4, 26, 27, 28, 29]
    base = [0, 2, 3, 4, 5, 6, 7, 8, 26, 27]
    values = np.ones(30)
    values[base[:8]] = 2
    hits, pool = retriever.expand("discussion", values, base)
    assert ids(hits) == [str(i) for i in base]
    assert "18" not in pool and len(pool) == len(set(pool)) <= 20


def test_duplicate_message_ids_are_rejected():
    rows = messages(12)
    rows[-1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="unique message IDs"):
        build_retriever("bm25-reply", rows)


def test_order_queries_rerank_only_eligible_replies_and_keep_the_same_pool(monkeypatch):
    rows = messages(16)
    for i in range(10, 15):
        rows[i]["reply_to"] = "0"
    rows[14]["group"] = "B"  # Hidden across this edge.
    values = np.array([20] * 8 + [10, 9, 8, 7, 6, 1, 8, 0])
    base = list(range(10))
    calls = []

    class Encoder:
        def __init__(self, name, device, revision):
            assert (name, device, revision) == ("fixture", "cpu", "tag")

        def encode(self, texts, **options):
            calls.append(texts)
            assert options["normalize_embeddings"] and options["batch_size"] == 4
            return np.array([[1, 0], [0.8, 0.6], [0, 1], [1, 0]])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    r = build_retriever("bm25-reply-order", rows, "fixture", "cuda", 4, "tag")
    monkeypatch.setattr(r, "rank_content", lambda query, k: (values, base))
    original, pool = r.expand("What should happen before launch?", values, base)
    assert ids(r.retrieve("What should happen?", 10)) == ids(original)
    assert ids(r.retrieve("Who spoke before launch?", 10)) == [str(i) for i in base]
    assert not calls
    ordered = r.retrieve("What should happen BEFORE launch?", 10)
    assert calls == [["query: What should happen BEFORE launch?",
                      "passage: word10\nParent: word0", "passage: word11\nParent: word0", "passage: word12\nParent: word0"]]
    assert ids(original)[-2:] == ["10", "11"]
    assert ids(ordered) == [str(i) for i in range(8)] + ["12", "10"]
    assert set(ids(ordered)) <= set(pool) and ordered[-2]["reply_path"] == ["0", "12"]
    assert ordered[-2]["e5_score"] == 1.0


def test_cli_supplies_the_actual_reply_to_qa_without_labels_or_extra_passages(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark"
    _write_benchmark(benchmark)
    messages_path = benchmark / "data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json"
    entries = [{"msg_node": f"m{i:02}", "content": "rollout gate" if i < 10 else f"unrelated{i}"}
               for i in range(40)]
    entries[10].update(content="rollout gate amber", reply_to="m00")
    messages_path.write_text(json.dumps({"Project": entries}), encoding="utf-8")
    question = {"id": "q", "question": "Which rollout gate should we use?", "answer": "PRIVATE_REFERENCE"}
    (benchmark / "questions/Finance/multi_hop.jsonl").write_text(json.dumps(question), encoding="utf-8")
    prompts = []

    class Client:
        last_usage = None
        def __init__(self, *args, **kwargs):
            pass
        def complete(self, system, prompt, max_tokens):
            if system == "judge":
                return "Final: Correct"
            prompts.append(prompt)
            return "Final: amber"

    monkeypatch.setattr("sgmem.runner.LocalClient", Client)
    for method in ("bm25", "bm25-reply"):
        output = tmp_path / method
        monkeypatch.setattr(sys, "argv", ["sgmem", "run", "--benchmark", str(benchmark), "--domains", "Finance",
                            "--categories", "multi_hop", "--method", method, "--answer-model", "fixture",
                            "--output", str(output)])
        main()
        record = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
        assert record["status"] == "ok" and len(set(ids(record["retrieval"]))) == 10
    assert "amber" not in prompts[0] and "amber" in prompts[1]
    assert all("PRIVATE_REFERENCE" not in p for p in prompts)
    assert record["retrieval"][-1]["reply_path"] == ["m00", "m10"]
