"""Preserve one hard-stopped first session before the unchanged QA runner resumes it."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from analyze_reply_qa import check, digest, key, read


def read_rows(path):
    # Messages can contain Unicode separators, which are not physical JSONL newlines.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def prefix(path):
    data = path.read_bytes()
    check(not data or data.endswith(b"\n"), f"Incomplete physical JSONL write: {path}")
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def prepare(run, ledger):
    runtime = read(run / "run-config.json")
    check(not runtime["finished"] and runtime["sessions"] == [] and len(runtime["arguments"]["methods"]) == 1,
          "Only an interrupted first session with no recorded resource totals is supported")
    check(not (run / "interruption.json").exists() and not (run / "interrupted-run-config.json").exists(),
          "Interruption already recorded; inspect it instead of applying recovery twice")
    method = runtime["arguments"]["methods"][0]
    folder = run / method
    manifest = read(folder / "manifest.json")
    check(manifest["status"] == "running" and manifest["project_git"] == runtime["project_git"],
          "Expected the original running manifest")
    stages = [s for s in read_rows(ledger) if "--output" in s["command"]
              and Path(s["command"][s["command"].index("--output") + 1]).resolve() == run.resolve()]
    check(len(stages) == 1 and stages[0]["exit_code"] == 124 and stages[0]["wall_seconds"] > 0,
          "Require one recorded hard budget timeout for this exact run")
    partition = runtime["arguments"]["partition"]
    expected = {key(q) for q in read(run / "split.json")[partition]}
    attempts_path = folder / "attempts.jsonl"
    rows = read_rows(attempts_path)
    completed = {key(row): row for row in rows}
    check(len(completed) == len(rows) and set(completed) <= expected
          and all(r["status"] == "ok" and r["attempt"] == 1 and r["method"] == method for r in rows),
          "Expected unique successful checkpoints from the first session")
    calls, orphan = {}, {}
    for call in read_rows(run / "generation-audit.jsonl"):
        qid, stage = key(call["question"]), call["stage"]
        check(qid in expected and call["question"]["method"] == method and call["question"]["attempt"] == 1
              and stage in ("answer", "judge") and (qid, stage) not in calls, "Unexpected or duplicate generation call")
        check(stage != "judge" or (qid, "answer") in calls, "Judge call precedes answer")
        calls[qid, stage] = call
        if qid in completed:
            check(call["usage"] == completed[qid][stage + "_usage"]
                  and call["finish_reason"] == completed[qid][stage + "_finish_reason"], "Checkpoint and call disagree")
        else:
            orphan.setdefault(qid, []).append(call)
    check(all((qid, stage) in calls for qid in completed for stage in ("answer", "judge"))
          and len(orphan) <= 1, "Missing checkpoint calls or multiple uncommitted questions")
    interrupted = None
    for qid, captured in orphan.items():
        interrupted = dict(zip(("domain", "category", "question_id"), qid))
        interrupted.update(method=method, partition=partition, attempt=1, status="error",
                           error="Budget timeout before the question checkpoint; captured call usage retained, generated text unavailable.")
        for call in captured:
            stage = call["stage"]
            interrupted.update({stage + "_usage": call["usage"], stage + "_finish_reason": call["finish_reason"],
                                stage + "_seconds": call["seconds"]})
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "method": method,
            "runtime_sha256": digest(run / "run-config.json"), "budget_ledger": os.path.relpath(ledger, run),
            "budget_ledger_sha256": digest(ledger), "budget_stage": stages[0],
            "original_attempts": prefix(attempts_path), "original_calls": prefix(run / "generation-audit.jsonl"),
            "error_attempt": interrupted,
            "resource_note": "Interrupted wall time comes from the outer budget ledger. Original CPU time, GPU peaks and index times were not saved. Stage timings cover recorded work only; the interrupted call has no complete question timing."}


def verify(run):
    receipt = read(run / "interruption.json")
    check(digest(run / "interrupted-run-config.json") == receipt["runtime_sha256"], "Interrupted runtime snapshot changed")
    snapshot = read(run / "interrupted-run-config.json")
    check(snapshot["finished"] is False and snapshot["sessions"] == []
          and snapshot["arguments"]["methods"] == [receipt["method"]], "Interrupted session already had resource totals")
    ledger = run / receipt["budget_ledger"]
    check(digest(ledger) == receipt["budget_ledger_sha256"] and receipt["budget_stage"] in read_rows(ledger)
          and receipt["budget_stage"]["exit_code"] == 124 and receipt["budget_stage"]["wall_seconds"] > 0,
          "Original budget evidence changed")
    attempts_path = run / receipt["method"] / "attempts.jsonl"
    for path, field in ((attempts_path, "original_attempts"), (run / "generation-audit.jsonl", "original_calls")):
        original = receipt[field]
        check(hashlib.sha256(path.read_bytes()[:original["bytes"]]).hexdigest() == original["sha256"],
              f"Original evidence prefix changed: {path}")
    if receipt["error_attempt"] is not None:
        tail = attempts_path.read_bytes()[receipt["original_attempts"]["bytes"]:]
        check(json.loads(tail.split(b"\n", 1)[0]) == receipt["error_attempt"], "Interrupted attempt missing or changed")
    return receipt


def apply(run, receipt):
    # Preserve the original session and append only the call that lacked a question checkpoint.
    with (run / "interrupted-run-config.json").open("xb") as handle:
        handle.write((run / "run-config.json").read_bytes())
    with (run / "interruption.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    if receipt["error_attempt"] is not None:
        path = run / receipt["method"] / "attempts.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt["error_attempt"]) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    verify(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Otherwise print the recovery record without changing files")
    args = parser.parse_args()
    receipt = prepare(args.run, args.ledger)
    if args.apply:
        apply(args.run, receipt)
    print(json.dumps(receipt, indent=2))
