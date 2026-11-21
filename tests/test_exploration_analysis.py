import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_exploration as exploration
import analyze_reply_qa as analysis


def test_runner_compatibility_allows_dispatch_but_rejects_changed_qa(monkeypatch):
    current = (ROOT / "src/sgmem/runner.py").read_text(encoding="utf-8")
    timer = "                            retrieval_started = time.perf_counter()\n"
    legacy = current.replace(timer, '                            query = f"{asker} {q[\'question\']}" if asker else q["question"]\n' + timer)
    legacy = legacy.replace('retriever.retrieve(q["question"], args.top_k, asking_user_id=asker)',
                            'retriever.retrieve(query, args.top_k)')
    sources = {"before": legacy, "after": current}
    monkeypatch.setattr(analysis.subprocess, "check_output", lambda args, **kwargs: sources[args[-1].split(":")[0]])
    analysis.check_query_dispatch(ROOT, "after", "before")
    sources["after"] = current.replace("evaluate_question(q,", 'evaluate_question({**q, "question": "wrong"},')
    with pytest.raises(ValueError, match="beyond"):
        analysis.check_query_dispatch(ROOT, "after", "before")


def test_matrix_reports_partial_coverage_paired_changes_and_cost_once(monkeypatch, tmp_path):
    # Synthetic audit outputs isolate aggregation; this does not simulate a real QA result.
    methods = ("bm25-fields", "bm25-person")
    controls = ("bm25", "dense", "bm25-reply", "bm25-reply-order")
    rows = [{"domain": "D", "category": "C", "question_id": str(i), **dict.fromkeys(controls, i == 0)}
            for i in range(2)]
    metrics = {m: {"overall": {"correct": 1, "total": 2}} for m in controls}
    def read(path):
        if path.name == "run-config.json":
            return {"arguments": {"methods": [path.parent.name]}}
        return {"questions": rows, "source_split_sha256": "split", "benchmark_commit": "benchmark", "metrics": metrics}
    def compare(run, audit, evidence, method, probe):
        q = copy.deepcopy(rows)
        for row in q:
            row[method] = row["question_id"] == "1"
        return {"questions": q, "source_split_sha256": "split", "benchmark_commit": "benchmark", "metrics": {**metrics, method: metrics["bm25"]},
                "cost": {"gpu_hours": 0.5}}
    monkeypatch.setattr(exploration, "read", read)
    monkeypatch.setattr(exploration, "compare", compare)
    monkeypatch.setattr(exploration, "digest", lambda path: "control-hash")
    monkeypatch.setattr(exploration, "read_rows", lambda path: [{"answer_finish_reason": "length", "judge_finish_reason": "stop"}])
    runs = [tmp_path / m for m in methods]
    result = exploration.analyze(runs, tmp_path / "probe", tmp_path / "audit", tmp_path / "evidence")
    assert result["complete_matrix"] is False and result["total_qa_gpu_hours"] == 1.0
    assert result["pairwise"]["bm25-fields"]["bm25"] == {"questions": 2, "gained": 1, "lost": 1, "net": 0}
    assert result["runs"]["bm25-person"]["truncated"] == {"answer": 1, "judge": 0}
    with pytest.raises(ValueError, match="Duplicate"):
        exploration.analyze([runs[0], runs[0]], tmp_path / "probe", tmp_path / "audit", tmp_path / "evidence")
