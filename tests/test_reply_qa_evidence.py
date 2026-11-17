import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_reply_qa import audit_calls, read
from analyze_full_baselines import key, read_rows
from sgmem.qa import official_correct


def test_published_reply_qa_reconstructs_from_individual_dev_questions():
    evidence = ROOT / "docs/evidence"
    result = read(evidence / "bounded-reply-qa-dev.json")
    baselines = read_rows(evidence / "part12-full-20261105/outcomes.jsonl")
    dev = {key(r) for r in read(evidence / "part12-full-20261105/split.json")["dev"]}
    old = {(r["method"], key(r)): r for r in baselines if r["partition"] == "dev"}
    changed = {key(r) for r in read(evidence / "bounded-reply-dev.json")["changed_questions"]}
    rows = result["questions"]
    assert len(rows) == len(dev) and {key(r) for r in rows} == dev
    assert result["calls"] == 2 * len(dev)
    for r in rows:
        k = key(r)
        assert r["changed_context"] == (k in changed)
        assert r["bm25-reply"] == official_correct(r["judge_answer"])
        assert r["same_answer_as_bm25"] == (r["answer"] == old["bm25", k]["answer"])
        for method in ("bm25", "dense"):
            assert r[method] == old[method, k]["official_correct"]
    for method, metrics in result["metrics"].items():
        for field, column in (("by_category", "category"), ("by_domain", "domain")):
            assert set(metrics[field]) == {r[column] for r in rows}
            for name, cell in metrics[field].items():
                group = [r for r in rows if r[column] == name]
                assert (cell["correct"], cell["total"]) == (sum(r[method] for r in group), len(group))
        assert metrics["overall"]["correct"] == sum(r[method] for r in rows)
    for name, group in [("all", rows), ("changed", [r for r in rows if r["changed_context"]]),
                        ("unchanged", [r for r in rows if not r["changed_context"]])]:
        comparison = result["vs_bm25"][name]
        assert comparison["gained"] == sum(r["bm25-reply"] and not r["bm25"] for r in group)
        assert comparison["lost"] == sum(r["bm25"] and not r["bm25-reply"] for r in group)
        assert comparison["net"] == sum(int(r["bm25-reply"]) - int(r["bm25"]) for r in group)


def test_call_audit_refuses_duplicates_and_held_out_calls(tmp_path):
    audit = tmp_path / "calls.jsonl"
    q = {"domain": "Finance", "category": "multi_hop", "question_id": "dev", "method": "bm25-reply"}
    calls = [{"question": q, "stage": s} for s in ("answer", "judge")]
    dev = {key(q)}
    for extra in (calls[0], {"question": {**q, "question_id": "held-out"}, "stage": "answer"}):
        audit.write_text("\n".join(json.dumps(r) for r in calls + [extra]), encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected or duplicate call"):
            audit_calls(audit, "bm25-reply", dev)
