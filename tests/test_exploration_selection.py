import sys

import pytest

from test_frozen_test_analysis import make_questions, mutate, write, write_rows
import select_exploration as selection


SOURCE = "a" * 40
BENCHMARK = "b" * 40
REPLY = "bm25-reply"
ORDERED = "bm25-reply-order"
HYBRID_META = "hybrid-meta"
FIELDS = "bm25-fields"


@pytest.fixture
def evidence_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "implementation_signature", lambda root, commit: {})
    evidence = tmp_path / "part12"
    evidence.mkdir()
    dev, test = make_questions("dev", 179), make_questions("test", 536)
    write(evidence / "split.json", {"dev": dev, "test": test})
    split = selection.digest(evidence / "split.json")
    models = {"qa_revision": "qa", "embedding_revision": "e5"}
    write(evidence / "provenance.json", {"benchmark_commit": BENCHMARK, "split_sha256": split, "dev": 179, "test": 536,
                                         "protocol": {"top_k": 10, "model_provenance": models}})
    methods = selection.STRUCTURED_METHODS
    comparison, probe, plan = (tmp_path / name for name in ("comparison.json", "probe.json", "plan.json"))
    write(plan, {"source_commit": SOURCE, "methods": methods, "top_k": 10, "dev_questions": 179, "test_questions": 536,
                 **models, "selection": "Select by dev count, grounded repairs, cost and fixed order."})
    top10 = [f"message-{i}" for i in range(10)]
    target0, target1 = "target-0", "excluded-target-1"
    ranks = {"benchmark_commit": BENCHMARK, "split_sha256": split, "top_k": 10, "dev_questions": 179}
    write(probe, {**ranks, "project_git": {"commit": SOURCE, "status": ""}, "methods": dict.fromkeys(methods, {}),
                  "questions": [{**q, "methods": {m: {"final_ids": top10[:-1] + [target0] if i == 0 and m == HYBRID_META
                                                      else top10[:-1] + [target1] if i == 1 and m == FIELDS else top10}
                                                for m in methods}} for i, q in enumerate(dev)]})
    write_rows(evidence / "outcomes.jsonl", [{**q, "partition": "dev", "method": "bm25", "retrieved_ids": top10,
                                              "official_correct": i == 178} for i, q in enumerate(dev)]
               + [{**test[0], "partition": "test", "method": "bm25"}])
    write_rows(evidence / "failure-annotations.jsonl", [
        {**dev[0], "partition": "dev", "method": "bm25", "primary_failure_type": "F1", "target_message_ids": [target0]},
        {**dev[1], "partition": "dev", "method": "bm25", "primary_failure_type": "F7", "target_message_ids": [target1]},
        {**test[0], "partition": "test", "method": "bm25", "primary_failure_type": "F1", "target_message_ids": [target0]},
    ])
    write(tmp_path / "bounded-reply-dev.json", {**ranks, "changed_questions": [{**dev[0], "final_ids": top10[:-1] + [target0]}]})
    write(tmp_path / "ordered-reply-dev.json", {**ranks, "changed_questions": [{**dev[0], "final_ids": top10[1:] + [target1]}],
                                               "changed_from_reply": [{**dev[0], "method": ORDERED, "final_ids": top10[:-1] + [target0]}]})

    correct = {"hybrid": 179, FIELDS: 2, HYBRID_META: 2, ORDERED: 2}
    hours = {FIELDS: 0.1, HYBRID_META: 0.5, ORDERED: 1.0}
    flags = [{**q, "bm25": i == 178, "dense": i == 176,
              **{m: i < correct.get(m, 1) for m in (*methods, REPLY, ORDERED)}} for i, q in enumerate(dev)]
    metrics = {m: {"overall": {"correct": sum(r[m] for r in flags), "total": 179, "completed": 179, "errors": 0}}
               for m in (*methods, REPLY, ORDERED, "bm25", "dense")}

    def qa(method):
        return {"project_commit": SOURCE, "benchmark_commit": BENCHMARK, "source_split_sha256": split,
                "raw_sha256": {"retrieval_probe": selection.digest(probe)}, "questions": flags,
                "cost": {"gpu_hours": hours.get(method, 1.0)},
                "metrics": {m: metrics[m] for m in (method, "bm25", "dense")}}

    reply_hashes = {}
    for method, filename in selection.REPLY_REPORTS.items():
        path = tmp_path / filename
        write(path, qa(method))
        reply_hashes[method] = selection.digest(path)
    write(comparison, {"complete_matrix": True, "runs": {m: qa(m) for m in methods},
                       "metrics": metrics, "reply_control_sha256": reply_hashes})
    return comparison, probe, plan, evidence


def test_selects_grounded_then_lower_cost_at_equal_accuracy(evidence_paths):
    freeze = selection.select(*evidence_paths)
    assert (freeze["selected_method"], freeze["control_method"]) == (HYBRID_META, "hybrid")
    scores = freeze["selection_table"]
    assert scores[FIELDS] == {"correct": 2, "grounded_repairs": 0, "gpu_hours": 0.1}
    assert scores[REPLY]["grounded_repairs"] == scores[ORDERED]["grounded_repairs"] == 1
    assert "hybrid" not in scores and freeze["dev_selection"]["control_correct"] == 179
    assert freeze["frozen_at_utc"].endswith("Z")


@pytest.mark.parametrize(("cost", "winner", "control"), [(0.5, HYBRID_META, "hybrid"), (0.3, ORDERED, "bm25")])
def test_old_reply_can_win_and_exact_ties_follow_plan_order(evidence_paths, cost, winner, control):
    comparison, _, _, evidence = evidence_paths
    reply = evidence.parent / selection.REPLY_REPORTS[ORDERED]
    mutate(reply, lambda r: r["cost"].update(gpu_hours=cost))
    mutate(comparison, lambda r: r["reply_control_sha256"].update({ORDERED: selection.digest(reply)}))
    freeze = selection.select(*evidence_paths)
    assert (freeze["selected_method"], freeze["control_method"]) == (winner, control)


def test_rejects_partial_matrix_or_different_probe(evidence_paths):
    comparison, probe, *_ = evidence_paths
    mutate(comparison, lambda r: r.update(complete_matrix=False))
    with pytest.raises(ValueError, match="cover all eight methods"):
        selection.select(*evidence_paths)
    mutate(comparison, lambda r: r.update(complete_matrix=True))
    mutate(probe, lambda r: r.update(purpose="Different bytes"))
    with pytest.raises(ValueError, match="on this retrieval probe"):
        selection.select(*evidence_paths)


def test_cli_creates_freeze_once(evidence_paths, tmp_path, monkeypatch):
    comparison, probe, plan, evidence = evidence_paths
    output = tmp_path / "freeze.json"
    monkeypatch.setattr(sys, "argv", ["select_exploration.py", "--comparison", str(comparison), "--probe", str(probe),
                                      "--plan", str(plan), "--output", str(output), "--evidence", str(evidence)])
    selection.main()
    saved = output.read_bytes()
    with pytest.raises(SystemExit) as failure:
        selection.main()
    assert failure.value.code == 1
    assert output.read_bytes() == saved
