"""Compare an actual bm25-reply dev run to both archived controls and the retrieval probe."""

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


def audit_calls(path, method, dev, *, allow_other_partitions=False):
    calls = {}
    for row in read_rows(path):
        q = row["question"]
        qid = key(q)
        if allow_other_partitions and qid not in dev:
            continue
        call = qid, row["stage"]
        check(q["method"] == method and qid in dev and call not in calls, f"Unexpected or duplicate call: {call}")
        calls[call] = row
    check(set(calls) == {(qid, stage) for qid in dev for stage in ("answer", "judge")}, "Incomplete call audit")
    return calls


def paired(rows, control):
    gains = sum(r["bm25-reply"] and not r[control] for r in rows)
    losses = sum(r[control] and not r["bm25-reply"] for r in rows)
    return {"questions": len(rows), "gained": gains, "lost": losses, "net": gains - losses}


def compare(run, bm25_audit, evidence):
    method = "bm25-reply"
    folder = run / method
    reference = read(evidence / "provenance.json")
    probe = read(evidence.parent / "bounded-reply-dev.json")
    manifest, runtime = read(folder / "manifest.json"), read(run / "run-config.json")
    cfg = manifest["config"]
    dev = {k for k, p in partition_map(read(evidence / "split.json")).items() if p == "dev"}
    check(runtime["finished"] and manifest["status"] == "complete" and cfg["partition"] == "dev"
          and cfg["method"] == method, "Expected a completed bm25-reply development run")
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
    new = audit_calls(run / "generation-audit.jsonl", method, dev)
    changes = {key(r): r["final_ids"] for r in probe["changed_questions"]}
    questions = []
    for qid in sorted(dev):
        a, b = indexed[method][qid], indexed["bm25"][qid]
        ids = [h["message"]["id"] for h in a["retrieval"]]
        changed = ids != b["retrieved_ids"]
        check(ids == changes.get(qid, b["retrieved_ids"]) and changed == (qid in changes), f"Retrieval differs: {qid}")
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
    return {"run": run.name, "project_commit": manifest["project_git"]["commit"],
            "benchmark_commit": reference["benchmark_commit"], "source_split_sha256": runtime["source_split_sha256"],
            "raw_sha256": {"predictions": digest(folder / "predictions.jsonl"), "run_config": digest(run / "run-config.json"),
                           "calls": digest(run / "generation-audit.jsonl"), "bm25_calls": digest(bm25_audit)},
            "calls": len(new), "metrics": metrics, "cost": costs,
            "vs_bm25": {scope: paired(group, "bm25") for scope, group in
                        (("all", questions), ("changed", [r for r in questions if r["changed_context"]]),
                         ("unchanged", [r for r in questions if not r["changed_context"]]))},
            "vs_dense": paired(questions, "dense"), "questions": questions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bm25-audit", type=Path, required=True, help="Part 1 BM25 generation-audit.jsonl")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/part12-full-20261105")
    args = parser.parse_args()
    print(json.dumps(compare(args.run, args.bm25_audit, args.evidence), ensure_ascii=True, indent=2))
