"""Compare a full reply-retrieval dev run to archived controls and the matching retrieval probe."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_full_baselines import ROOT, key, read_rows
from sgmem.qa import official_correct, summarize
from sgmem.shards import implementation_signature, partition_map


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def audit_calls(path, method, dev, *, allow_other_partitions=False, attempts=None):
    """Match each saved prediction to its successful attempt, retaining the full audit on disk."""
    calls = {}
    for row in read_rows(path):
        q = row["question"]
        qid = key(q)
        if allow_other_partitions and qid not in dev:
            continue
        call = qid, row["stage"]
        check(q["method"] == method and qid in dev and row["stage"] in ("answer", "judge"),
              f"Unexpected or duplicate call: {call}")
        expected_attempt = attempts[qid] if attempts is not None else 1
        attempt = q.get("attempt", 1)
        check(isinstance(attempt, int) and 1 <= attempt <= expected_attempt, f"Unexpected attempt: {call}")
        if attempt < expected_attempt:
            continue
        check(call not in calls, f"Unexpected or duplicate call: {call}")
        calls[call] = row
    check(set(calls) == {(qid, stage) for qid in dev for stage in ("answer", "judge")}, "Incomplete call audit")
    return calls


def paired(rows, method, control):
    gains = sum(r[method] and not r[control] for r in rows)
    losses = sum(r[control] and not r[method] for r in rows)
    return {"questions": len(rows), "gained": gains, "lost": losses, "net": gains - losses}


def paired_by_context(rows, method, control, field):
    return {"all": paired(rows, method, control),
            "changed": paired([r for r in rows if r[field]], method, control),
            "unchanged": paired([r for r in rows if not r[field]], method, control)}


def compare(run, bm25_audit, evidence, method="bm25-reply"):
    folder = run / method
    reference = read(evidence / "provenance.json")
    ordered = method == "bm25-reply-order"
    probe = read(evidence.parent / ("ordered-reply-dev.json" if ordered else "bounded-reply-dev.json"))
    manifest, runtime = read(folder / "manifest.json"), read(run / "run-config.json")
    cfg = manifest["config"]
    dev = {k for k, p in partition_map(read(evidence / "split.json")).items() if p == "dev"}
    check(runtime["finished"] and manifest["status"] == "complete" and cfg["partition"] == "dev"
          and cfg["method"] == method, f"Expected a completed {method} development run")
    check(partition_map(read(run / "split.json")) == dict.fromkeys(dev, "dev")
          and len(dev) == probe["dev_questions"] == manifest["expected_questions"], "Development split mismatch")
    check(digest(evidence / "split.json") == runtime["source_split_sha256"] == reference["split_sha256"]
          == probe["split_sha256"], "Published split hash mismatch")
    check(not manifest["project_git"]["status"] and not manifest["benchmark_git"]["status"]
          and manifest["benchmark_git"]["commit"] == reference["benchmark_commit"] == probe["benchmark_commit"],
          "Code or benchmark provenance mismatch")
    check(all(runtime[k] == v for k, v in reference["runtime"].items()), "Model, packages or decoding changed")
    protocol = reference["protocol"]
    # Weight revisions and file identities are checked independently of the host's cache paths.
    separate = {"inputs", "partition", "benchmark_commit", "answer_model", "embedding_model"}
    check(all(cfg[k] == v for k, v in protocol.items() if k not in separate),
          "QA protocol differs from the controls")
    check({k: v for k, v in manifest["input_sha256"].items() if k != "split"} == protocol["inputs"],
          "Benchmark messages, questions or prompt templates changed")
    signature = implementation_signature(ROOT, manifest["project_git"]["commit"])
    # Python versions serialize ASTs differently. Read both immutable commits with this interpreter.
    baseline_commit = next(p["commit"] for p in reference["raw_parts"] if p["method"] == "bm25")
    baseline_signature = implementation_signature(ROOT, baseline_commit)
    check(all(signature[k] == v for k, v in baseline_signature.items() if k != "src/sgmem/retrieval.py"),
          "Data, QA, runner or generation code differs")

    baseline = [r for r in read_rows(evidence / "outcomes.jsonl") if r["partition"] == "dev"]
    records = {m: [r for r in baseline if r["method"] == m] for m in ("bm25", "dense")}
    records[method] = read_rows(folder / "predictions.jsonl")
    indexed, metrics = {}, {}
    for m, group in records.items():
        indexed[m] = {key(r): r for r in group}
        check(set(indexed[m]) == dev and len(group) == len(dev), f"Missing, duplicate or extra questions in {m}")
        check(all(r["partition"] == "dev" and r["method"] == m and r["status"] == "ok"
                  and r["official_correct"] == official_correct(r["judge_answer"]) for r in group), f"Invalid {m} record")
        result = summarize(group)
        metrics[m] = {k: result[k] for k in ("overall", "by_category", "by_domain")}
    reported = read(folder / "metrics.json")
    check(reported["complete"] and all(reported[k] == v for k, v in metrics[method].items()), "Run metrics differ")

    old = audit_calls(bm25_audit, "bm25", dev, allow_other_partitions=True)
    attempts = {qid: r.get("attempt", 1) for qid, r in indexed[method].items()}
    new = audit_calls(run / "generation-audit.jsonl", method, dev, attempts=attempts)
    changes = {key(r): r["final_ids"] for r in probe["changed_questions"]}
    check(all(qid in dev and ids != indexed["bm25"][qid]["retrieved_ids"] for qid, ids in changes.items()),
          "Retrieval probe lists a question outside dev or an unchanged baseline ranking")
    if ordered:
        check(probe["model"]["revision"] == cfg["model_provenance"]["embedding_revision"],
              "E5 revision differs from retrieval probe")
        reply_changes = {key(r): r["final_ids"] for r in probe["changed_from_reply"] if r["method"] == method}
        changes.update(reply_changes)
    questions = []
    for qid in sorted(dev):
        a, b = indexed[method][qid], indexed["bm25"][qid]
        ids = [h["message"]["id"] for h in a["retrieval"]]
        changed = ids != b["retrieved_ids"]
        check(ids == changes.get(qid, b["retrieved_ids"]), f"Retrieval differs: {qid}")
        for stage in ("answer", "judge"):
            call = new[qid, stage]
            check(call["usage"] == a[stage + "_usage"] and call["finish_reason"] == a[stage + "_finish_reason"],
                  f"Generation trace differs: {qid}/{stage}")
        prompt = new[qid, "answer"]["prompt_sha256"]
        check(changed == (prompt != old[qid, "answer"]["prompt_sha256"]), f"Answer prompt mismatch: {qid}")
        same_answer = a["answer"] == b["answer"]
        check(same_answer == (new[qid, "judge"]["prompt_sha256"] == old[qid, "judge"]["prompt_sha256"]),
              f"Judge prompt mismatch: {qid}")
        questions.append({"domain": qid[0], "category": qid[1], "question_id": qid[2],
                          **{m: indexed[m][qid]["official_correct"] for m in records},
                          "changed_context": changed, "same_answer_as_bm25": same_answer,
                          "answer": a["answer"], "judge_answer": a["judge_answer"], "answer_prompt_sha256": prompt})

    costs = {k: runtime[k] for k in ("gpu_name", "wall_seconds", "gpu_hours", "peak_allocated_gpu_bytes")}
    costs["index_seconds_by_domain"] = manifest["index_seconds_by_domain"]
    for stage in ("answer", "judge"):
        costs[stage] = {k: sum(r[stage + "_usage"][k] for r in records[method]) for k in ("prompt_tokens", "completion_tokens")}
    costs["stage_seconds"] = {k: sum(r[k] for r in records[method]) for k in ("retrieval_seconds", "answer_seconds", "judge_seconds")}
    report = {
        "run": run.name, "project_commit": manifest["project_git"]["commit"],
        "benchmark_commit": reference["benchmark_commit"], "source_split_sha256": runtime["source_split_sha256"],
        "raw_sha256": {"predictions": digest(folder / "predictions.jsonl"), "run_config": digest(run / "run-config.json"),
                       "calls": digest(run / "generation-audit.jsonl"), "bm25_calls": digest(bm25_audit)},
        "calls": len(new), "metrics": metrics, "cost": costs,
        "vs_bm25": paired_by_context(questions, method, "bm25", "changed_context"),
        "vs_dense": paired(questions, method, "dense"), "questions": questions,
    }
    retries = [r for r in read_rows(run / "generation-audit.jsonl")
               if r["question"].get("attempt", 1) < attempts[key(r["question"])]]
    if retries:
        report["retry_calls"] = len(retries)
        costs["retries"] = {
            "seconds": sum(r["seconds"] for r in retries),
            **{k: sum(r["usage"][k] for r in retries) for k in ("prompt_tokens", "completion_tokens")},
        }
    if ordered:
        reference_path = evidence.parent / "bounded-reply-qa-dev.json"
        previous = read(reference_path)
        previous_rows = {key(r): r for r in previous["questions"]}
        check(set(previous_rows) == dev and len(previous["questions"]) == len(dev)
              and previous["source_split_sha256"] == report["source_split_sha256"]
              and previous["benchmark_commit"] == report["benchmark_commit"], "Reply control differs")
        for r in questions:
            qid = key(r)
            old_row = previous_rows[qid]
            r["bm25-reply"] = old_row["bm25-reply"]
            r["changed_from_reply"] = r["answer_prompt_sha256"] != old_row["answer_prompt_sha256"]
            r["same_answer_as_reply"] = r["answer"] == old_row["answer"]
            check(r["changed_from_reply"] == (qid in reply_changes), f"Reply-control prompt differs: {qid}")
        report["metrics"]["bm25-reply"] = previous["metrics"]["bm25-reply"]
        report["vs_bm25_reply"] = paired_by_context(questions, method, "bm25-reply", "changed_from_reply")
        report["raw_sha256"]["reply_control_evidence"] = digest(reference_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bm25-audit", type=Path, required=True, help="Part 1 BM25 generation-audit.jsonl")
    parser.add_argument("--method", choices=("bm25-reply", "bm25-reply-order"), default="bm25-reply")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/part12-full-20261105")
    args = parser.parse_args()
    print(json.dumps(compare(args.run, args.bm25_audit, args.evidence, args.method), ensure_ascii=True, indent=2))
