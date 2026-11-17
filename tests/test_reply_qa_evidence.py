import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_reply_qa import audit_calls, read
from analyze_full_baselines import key, read_rows
from sgmem.qa import official_correct


@pytest.mark.parametrize("run_method,stem", [("bm25-reply", "bounded-reply"), ("bm25-reply-order", "ordered-reply")])
def test_published_reply_qa_reconstructs_from_individual_dev_questions(run_method, stem):
    evidence = ROOT / "docs/evidence"
    result = read(evidence / f"{stem}-qa-dev.json")
    baselines = read_rows(evidence / "part12-full-20261105/outcomes.jsonl")
    dev = {key(r) for r in read(evidence / "part12-full-20261105/split.json")["dev"]}
    old = {(r["method"], key(r)): r for r in baselines if r["partition"] == "dev"}
    probe = read(evidence / f"{stem}-dev.json")
    expected = {key(r): r["final_ids"] for r in probe["changed_questions"]}
    if run_method == "bm25-reply-order":
        vs_reply = {key(r): r["final_ids"] for r in probe["changed_from_reply"] if r["method"] == run_method}
        expected.update(vs_reply)
        reference = {key(r): r for r in read(evidence / "bounded-reply-qa-dev.json")["questions"]}
    rows = result["questions"]
    assert len(rows) == len(dev) and {key(r) for r in rows} == dev
    assert result["calls"] == 2 * len(dev)
    for r in rows:
        k = key(r)
        original = old["bm25", k]
        original_ids = original["retrieved_ids"]
        assert r["changed_context"] == (expected.get(k, original_ids) != original_ids)
        assert r[run_method] == official_correct(r["judge_answer"])
        assert r["same_answer_as_bm25"] == (r["answer"] == original["answer"])
        for method in ("bm25", "dense"):
            assert r[method] == old[method, k]["official_correct"]
        if run_method == "bm25-reply-order":
            assert r["bm25-reply"] == reference[k]["bm25-reply"]
            assert r["changed_from_reply"] == (k in vs_reply)
            assert r["changed_from_reply"] == (r["answer_prompt_sha256"] != reference[k]["answer_prompt_sha256"])
            assert r["same_answer_as_reply"] == (r["answer"] == reference[k]["answer"])
    for method, metrics in result["metrics"].items():
        for field, column in (("by_category", "category"), ("by_domain", "domain")):
            assert set(metrics[field]) == {r[column] for r in rows}
            for name, cell in metrics[field].items():
                group = [r for r in rows if r[column] == name]
                assert (cell["correct"], cell["total"]) == (sum(r[method] for r in group), len(group))
        assert metrics["overall"]["correct"] == sum(r[method] for r in rows)
        assert metrics["overall"]["total"] == len(rows)
    assert result["vs_dense"]["gained"] == sum(r[run_method] and not r["dense"] for r in rows)
    assert result["vs_dense"]["lost"] == sum(r["dense"] and not r[run_method] for r in rows)
    controls = [("bm25", "vs_bm25", "changed_context")]
    if run_method == "bm25-reply-order":
        controls.append(("bm25-reply", "vs_bm25_reply", "changed_from_reply"))
    for control, pair_key, field in controls:
        for name, group in [("all", rows), ("changed", [r for r in rows if r[field]]),
                            ("unchanged", [r for r in rows if not r[field]])]:
            comparison = result[pair_key][name]
            assert comparison["questions"] == len(group)
            assert comparison["gained"] == sum(r[run_method] and not r[control] for r in group)
            assert comparison["lost"] == sum(r[control] and not r[run_method] for r in group)
            assert comparison["net"] == sum(int(r[run_method]) - int(r[control]) for r in group)


def test_call_audit_refuses_duplicates_and_held_out_calls(tmp_path):
    audit = tmp_path / "calls.jsonl"
    q = {"domain": "Finance", "category": "multi_hop", "question_id": "dev", "method": "bm25-reply"}
    calls = [{"question": q, "stage": s} for s in ("answer", "judge")]
    dev = {key(q)}
    for extra in (calls[0], {"question": {**q, "question_id": "held-out"}, "stage": "answer"}):
        audit.write_text("\n".join(json.dumps(r) for r in calls + [extra]), encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected or duplicate call"):
            audit_calls(audit, "bm25-reply", dev)
