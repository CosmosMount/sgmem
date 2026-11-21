import pytest

from sgmem.query_metadata import build_catalog, query_hints
from sgmem.retrieval import BM25Retriever, BM25ReplyRetriever, STRUCTURED_METHODS, build_retriever
from sgmem.structured import StructuredRetriever, reciprocal_rank_fusion


def messages():
    # Background documents keep the meaningful query terms rare, as in the full corpus.
    rows = [{"id": str(i), "content": f"unrelated filler{i}", "group": "Elsewhere"} for i in range(80)]
    rows[:5] = [
        {"id": "0", "content": "sensor risk alarm", "group": "Factory", "phase_name": "Sensor Requirements",
         "speaker": "User_1", "role": "Manager", "timestamp": "2025-01-01"},
        {"id": "1", "content": "sensor risk alarm", "group": "Factory", "phase_name": "Sensor Requirements",
         "speaker": "User_2", "role": "Manager", "timestamp": "2025-02-01", "reply_to": "0"},
        {"id": "2", "content": "sensor risk alarm", "group": "Factory", "phase_name": "Sensor Requirements",
         "speaker": "User_1", "role": "Manager", "timestamp": "2025-03-01"},
        {"id": "3", "content": "sensor risk alarm", "group": "Other", "phase_name": "Sensor Requirements",
         "speaker": "User_1", "role": "Manager", "timestamp": "2025-04-01"},
        {"id": "4", "content": "sensor risk alarm", "group": "Other", "phase_name": "Requirements",
         "speaker": "User_3", "role": "Engineer"},
    ]
    return rows


def test_longest_names_preserve_separate_mentions_and_old_diagnostic():
    catalog = build_catalog(messages())
    assert len(query_hints("Sensor Requirements", catalog)["matched_names"]) == 2
    result = query_hints("Sensor Requirements", catalog, prefer_longest=True)
    assert [m["name"] for m in result["matched_names"]] == ["Sensor Requirements"]
    result = query_hints("Compare Sensor Requirements with Requirements", catalog, prefer_longest=True)
    assert {m["name"] for m in result["matched_names"]} == {"Sensor Requirements", "Requirements"}


def test_asker_alone_is_not_a_target_and_personal_state_is():
    r = StructuredRetriever(messages(), "bm25-person")
    assert r.query_plan("Can I ask about Sensor Requirements?", "User_1")["speakers"] == []
    assert r.query_plan("Which risk am I asking about?", "User_1")["speakers"] == []
    assert r.query_plan("Which risk i'm monitoring?", "User_1")["speakers"] == ["User_1"]
    assert r.query_plan("What is my open item?", "User_1")["speakers"] == ["User_1"]
    assert r.query_plan("What does User_2 monitor?", "User_1")["speakers"] == ["User_2"]
    assert r.query_plan("What does User_20 monitor?", "User_1")["speakers"] == []
    assert r.query_plan("What do I need to tell User_2?", "User_1")["speakers"] == ["User_1"]
    assert r.query_plan("What is the status of the component driven by User_2?", "User_1")["speakers"] == []
    assert r.query_plan("What did User_2 suggest after I proposed narrowing the scope?", "User_1")["speakers"] == ["User_2"]
    assert r.query_plan("What is my risk if the approver is User_3?", "User_1")["speakers"] == ["User_1"]


def test_role_subject_differs_from_recipient():
    r = StructuredRetriever(messages(), "bm25-person")
    assert r.query_plan("What does the Manager want the Engineer to do?")["roles"] == ["Manager"]
    assert r.query_plan("What should I ask the Manager?")["roles"] == []


def test_scope_and_person_are_constraints_on_an_extra_branch():
    r = StructuredRetriever(messages(), "bm25-person")
    orders, _, _ = r.candidate_orders("What risk am I monitoring in Factory Sensor Requirements?", 10,
                                     asking_user_id="User_1")
    assert set(orders["scope"]) == {0, 2}
    orders, _, _ = r.candidate_orders("What risk I am monitoring in Factory Sensor Requirements?", 10,
                                     asking_user_id="User_1")
    assert set(orders["scope"]) == {0, 2}
    assert len(orders["content"]) == 50


def test_time_is_conditional_and_cannot_promote_another_speaker():
    r = StructuredRetriever(messages(), "bm25-time")
    orders, _, _ = r.candidate_orders("What is my current risk in Factory Sensor Requirements?", 10,
                                     asking_user_id="User_1")
    assert orders["time"] == [2, 0]
    orders, _, _ = r.candidate_orders("What is my risk and current status in Factory Sensor Requirements?", 10,
                                     asking_user_id="User_1")
    assert orders["time"] == [2, 0]
    orders, _, _ = r.candidate_orders("What is my risk in Factory Sensor Requirements?", 10, asking_user_id="User_1")
    assert "time" not in orders
    orders, _, _ = r.candidate_orders("What is the latest alarm?", 10)
    assert "time" not in orders


def test_scope_disambiguates_reused_phase_with_explicit_group():
    r = StructuredRetriever(messages(), "bm25-scope")
    orders, _, _ = r.candidate_orders("Sensor Requirements risk in Factory", 10)
    assert set(orders["scope"]) == {0, 1, 2}
    orders, _, _ = r.candidate_orders("Sensor Requirements risk", 10)
    assert set(orders["scope"]) == {0, 1, 2, 3}
    orders, _, _ = r.candidate_orders("Compare Factory alarm with Other's Requirements", 10)
    assert set(orders["scope"]) == {0, 1, 2, 3, 4}


def test_fusion_is_deterministic_and_not_score_scale_dependent():
    order, scores = reciprocal_rank_fusion({"a": [2, 0], "b": [0, 2], "c": [1]})
    assert order == [0, 2, 1]
    assert scores[0] == scores[2] == 1 / 61 + 1 / 62


@pytest.mark.parametrize("method", [m for m in STRUCTURED_METHODS if not m.startswith("hybrid")])
def test_simple_methods_keep_raw_messages_budget_and_ignore_labels(method):
    rows = messages()
    r = build_retriever(method, rows)
    query = "What should User_1 do about Sensor Requirements risk?"
    hits = r.retrieve(query, 10, asking_user_id="User_2")
    assert len(hits) == len({h["message"]["id"] for h in hits}) == 10
    assert all(h["message"] is rows[int(h["message"]["id"])] for h in hits)
    poisoned = [{**m, "answer": "secret", "gold": "0", "target_group": "Other"} for m in rows]
    other = build_retriever(method, poisoned).retrieve(query, 10, asking_user_id="User_2")
    assert [h["message"]["id"] for h in hits] == [h["message"]["id"] for h in other]
    assert r.retrieve("!!!", 10) == []
    with pytest.raises(ValueError, match="positive"):
        r.retrieve(query, 0)


def test_old_query_protocol_is_preserved():
    for cls in (BM25Retriever, BM25ReplyRetriever):
        r = cls(messages())
        assert r.retrieve("risk", 10, asking_user_id="User_1") == r.retrieve("User_1 risk", 10)


def test_field_index_keeps_the_baseline_query_prefix():
    rows = messages()
    rows[70]["content"] = "User_1 unusual exception"
    r = StructuredRetriever(rows, "bm25-fields")
    orders, _, _ = r.candidate_orders("exception", 10, asking_user_id="User_1")
    assert orders["fields"] == r.fields.rank_content("User_1 exception", 50)[1]
    assert orders["fields"][0] == 70


def test_reply_extension_uses_scoped_seeds_and_keeps_the_pool(monkeypatch):
    rows = messages()
    scoped = StructuredRetriever(rows, "bm25-scope").retrieve("What should I do about Factory risk?", 10)
    base = [int(h["message"]["id"]) for h in scoped]
    r = StructuredRetriever(rows, "bm25-scope-reply")
    def expand(query, scores, seeds):
        assert seeds == base
        assert 78 not in base and 79 not in base
        chosen = seeds[:8] + [78, 79]
        return ([{"message": rows[i], "reply_path": [str(seeds[0]), str(i)]} if i >= 78
                 else {"message": rows[i]} for i in chosen], [str(i) for i in seeds + [77, 78, 79]])
    monkeypatch.setattr(r, "expand", expand)
    hits, trace = r.retrieve_with_trace("What should I do about Factory risk?", 10)
    assert [h["message"]["id"] for h in hits[:8]] == [h["message"]["id"] for h in scoped[:8]]
    assert [h["message"]["id"] for h in hits[-2:]] == ["78", "79"]
    assert "77" in trace["reply_pool"] and all(h["message"]["id"] != "77" for h in hits)
    assert "reply" not in trace["candidate_lists"]  # No extra vote for the existing eight content hits.


def test_hybrid_requires_dense_and_combines_complementary_candidates():
    rows = messages()
    class Dense:
        def retrieve(self, query, top_k, *, asking_user_id):
            assert query == "sensor risk" and asking_user_id == "User_1"
            return [{"message": m} for m in reversed(rows[-top_k:])]

    with pytest.raises(ValueError, match="dense"):
        StructuredRetriever(rows, "hybrid")
    r = StructuredRetriever(rows, "hybrid", dense=Dense())
    orders, _, _ = r.candidate_orders("sensor risk", 10, asking_user_id="User_1")
    assert len(set(orders["content"]) | set(orders["dense"])) > 50
    hits = r.retrieve("sensor risk", 10, asking_user_id="User_1")
    assert len(hits) == 10 and any("dense" in h["retrieval_sources"] for h in hits)
