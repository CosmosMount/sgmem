"""Audit and compare the two completed test runs named by a development-only freeze."""

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess

from analyze_reply_qa import ROOT, check, digest, key, paired, read
from sgmem.qa import extract_final, format_message, official_correct, parse_judgment, summarize
from sgmem.retrieval import STRUCTURED_METHODS
from sgmem.shards import PUBLISHED_SPLIT, implementation_signature, partition_map


REPLY_REPORTS = {"bm25-reply": "bounded-reply-qa-dev.json", "bm25-reply-order": "ordered-reply-qa-dev.json"}
STAGES = ("answer", "judge")
USAGE = ("prompt_tokens", "completion_tokens", "total_tokens")


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    check(parsed.utcoffset() == timedelta(0), f"Expected a UTC timestamp: {value}")
    return parsed


def index_rows(rows, expected, label):
    indexed = {key(row): row for row in rows}
    check(set(indexed) == expected and len(indexed) == len(rows), f"Missing, duplicate or extra question in {label}")
    return indexed


def read_jsonl(path):
    # Text iteration splits on physical newlines and preserves Unicode separators inside JSON strings.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def check_signature(commit, source, signature):
    check(re.fullmatch(r"[0-9a-f]{40}", commit), f"Missing immutable project commit: {commit}")
    check(commit == source or implementation_signature(ROOT, commit) == signature,
          f"Implementation differs from the freeze: {commit}")


def audit_development(report, freeze, evidence, dev):
    methods = {"bm25", "dense", *REPLY_REPORTS, *STRUCTURED_METHODS}
    check(report["complete_matrix"] is True and set(report["runs"]) == set(STRUCTURED_METHODS)
          and set(report["metrics"]) == methods and set(report["reply_control_sha256"]) == set(REPLY_REPORTS),
          "Freeze requires the complete development matrix")
    metrics = report["metrics"]

    def audit_rows(run, method, reference_controls=None):
        check(run["source_split_sha256"] == freeze["split_sha256"]
              and run["benchmark_commit"] == freeze["benchmark_commit"], f"Development inputs changed for {method}")
        indexed = index_rows(run["questions"], dev, f"{method} dev")
        for name in (method, "bm25", "dense"):
            flags = [row[name] for row in indexed.values()]
            overall = metrics[name]["overall"]
            check(all(type(flag) is bool for flag in flags) and sum(flags) == overall["correct"]
                  and overall["total"] == overall["completed"] == len(dev) and overall["errors"] == 0
                  and run["metrics"][name] == metrics[name], f"Development metrics changed for {name} in {method}")
        controls = {qid: (row["bm25"], row["dense"]) for qid, row in indexed.items()}
        check(reference_controls is None or controls == reference_controls, f"Paired development controls changed for {method}")
        return controls

    controls = None
    for method, run in report["runs"].items():
        controls = audit_rows(run, method, controls)
    for method, filename in REPLY_REPORTS.items():
        path = evidence.parent / filename
        check(digest(path) == report["reply_control_sha256"][method], f"Development archive hash changed for {method}")
        audit_rows(read(path), method, controls)

    selection = freeze["dev_selection"]
    selected, control = freeze["selected_method"], freeze["control_method"]
    check(selection["total"] == len(dev) and selection["selected_correct"] == metrics[selected]["overall"]["correct"]
          and selection["control_correct"] == metrics[control]["overall"]["correct"]
          and isinstance(selection["reason"], str) and selection["reason"].strip(),
          "Freeze disagrees with the development scores or omits the reason")


def audit_attempts(path, indexed, method):
    attempts = {}
    for row in read_jsonl(path):
        qid = key(row)
        check(qid in indexed and row["partition"] == "test" and row["method"] == method, f"Unexpected attempt: {qid}")
        previous = attempts.get(qid, [])
        number = row["attempt"]
        check(type(number) is int and number == len(previous) + 1 and number <= indexed[qid]["attempt"],
              f"Invalid attempt sequence: {qid}")
        if number == indexed[qid]["attempt"]:
            check(row == indexed[qid], f"Final attempt differs from prediction: {qid}")
        else:
            check(row["status"] == "error" and row["error"], f"Invalid earlier attempt: {qid}/{number}")
        attempts.setdefault(qid, []).append(row)
    check(set(attempts) == indexed.keys() and all(len(rows) == indexed[qid]["attempt"] for qid, rows in attempts.items()),
          f"Incomplete attempt log for {method}")
    return attempts


def audit_generation(path, attempts, method, generation):
    calls = {}
    for call in read_jsonl(path):
        question, stage = call["question"], call["stage"]
        qid, number = key(question), question["attempt"]
        label = qid, number, stage
        check(qid in attempts and question["method"] == method and stage in STAGES and label not in calls
              and type(number) is int and 1 <= number <= len(attempts[qid]), f"Unexpected or duplicate call: {label}")
        usage, limit, finish = call["usage"], generation[stage + "_max_tokens"], call["finish_reason"]
        check(set(usage) == set(USAGE) and all(type(usage[k]) is int and usage[k] > 0 for k in USAGE)
              and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
              and usage["completion_tokens"] == call["output_tokens"] <= call["max_tokens"] == limit
              and finish in ("stop", "length") and (finish == "stop" or call["output_tokens"] == limit)
              and re.fullmatch(r"[0-9a-f]{64}", call["prompt_sha256"])
              and math.isfinite(call["seconds"]) and call["seconds"] >= 0, f"Invalid generation call: {label}")
        row = attempts[qid][number - 1]
        check(stage != "judge" or (qid, number, "answer") in calls, f"Judge before answer: {label}")
        check(usage == row[stage + "_usage"] and finish == row[stage + "_finish_reason"],
              f"Generation trace differs from attempt: {label}")
        calls[label] = call
    check(all((qid, row["attempt"], stage) in calls
              for qid, rows in attempts.items() for row in rows for stage in STAGES
              if row["status"] == "ok" or row.get(stage + "_usage") is not None),
          f"Incomplete generation audit for {method}")
    return calls


def audit_test(run, method, reference, freeze, frozen_at, signature, split, test):
    folder = run / method
    runtime, manifest = read(run / "run-config.json"), read(folder / "manifest.json")
    args, cfg = runtime["arguments"], manifest["config"]
    check(runtime["finished"] is True and manifest["status"] == "complete" and args["methods"] == [method]
          and args["full"] is True and args["partition"] == cfg["partition"] == "test"
          and args["shards"] == 1 and args["shard_index"] == 0 and cfg["method"] == method
          and manifest["expected_questions"] == len(test), f"Expected a complete single-method test run: {run}")
    check(frozen_at < utc(runtime["started_at_utc"]) and frozen_at < utc(manifest["started_at_utc"]),
          f"Run or its original attempt started before the freeze: {run}")
    check(runtime["project_git"] == manifest["project_git"] and manifest["project_git"]["status"] == ""
          and runtime["benchmark_git"] == manifest["benchmark_git"] == {"commit": freeze["benchmark_commit"], "status": ""},
          f"Dirty source or benchmark mismatch: {run}")
    commit = manifest["project_git"]["commit"]
    check_signature(commit, freeze["source_commit"], signature)
    launcher = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:scripts/run_local_baselines.py"])
    check(runtime["script_sha256"] == sha256(launcher).hexdigest()
          and runtime["tracked_patch_sha256"] == sha256(b"").hexdigest(), f"Run launcher or tracked patch differs: {run}")

    scope = {**split, "dev": []}
    check(runtime["source_split_sha256"] == freeze["split_sha256"] and read(run / "full-split.json") == split
          and read(run / "split.json") == scope, f"Saved test split differs from the frozen scope: {run}")
    protocol = reference["protocol"]
    variable = {"inputs", "partition", "benchmark_commit", "answer_model", "embedding_model"}
    check(all(cfg[k] == v for k, v in protocol.items() if k not in variable)
          and all(runtime[k] == v for k, v in reference["runtime"].items())
          and all(args[k] == cfg[k] for k in ("domains", "categories", "device"))
          and args["seed"] == cfg["generation_config"]["seed_per_call"]
          and args["model_revision"] == cfg["model_provenance"]["qa_revision"]
          and args["embedding_revision"] == cfg["model_provenance"]["embedding_revision"],
          f"Data, model or QA protocol changed: {run}")
    check(manifest["input_sha256"] == {**protocol["inputs"], "split": digest(run / "split.json")},
          f"Benchmark inputs or split file changed: {run}")
    index_rows(manifest["question_ids"], test, f"{method} manifest")
    rows = read_jsonl(folder / "predictions.jsonl")
    indexed = index_rows(rows, test, f"{method} predictions")
    for qid, row in indexed.items():
        messages = [hit["message"] for hit in row["retrieval"]]
        check(len(messages) == len({message["id"] for message in messages}) == cfg["top_k"]
              and row["retrieved_docs"] == [format_message(message) for message in messages],
              f"Missing, duplicate or reformatted retrieved passage: {method}/{qid}")
        check(row["status"] == "ok" and row["partition"] == "test" and row["method"] == method
              and type(row["attempt"]) is int and row["attempt"] >= 1
              and row["prediction"] == row["answer"] == extract_final(row["agent_output"])
              and row["judge_answer"] == extract_final(row["judge_output"])
              and row["official_correct"] is official_correct(row["judge_answer"])
              and row["correct"] is parse_judgment(row["judge_answer"]), f"Invalid prediction: {method}/{qid}")
    metrics = summarize(rows)
    complete = {**metrics, "complete": True, "expected_questions": len(test)}
    check(read(folder / "metrics.json") == complete and read(run / "benchmark-summary.json") == {method: complete},
          f"Archived metrics differ from recomputed metrics: {run}")

    attempts = audit_attempts(folder / "attempts.jsonl", indexed, method)
    calls = audit_generation(run / "generation-audit.jsonl", attempts, method, cfg["generation_config"])
    cost = {k: runtime[k] for k in ("gpu_name", "wall_seconds", "cpu_seconds", "gpu_hours",
                                    "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes")}
    check(math.isclose(cost["gpu_hours"], cost["wall_seconds"] / 3600) and cost["wall_seconds"] > 0
          and 0 < cost["peak_allocated_gpu_bytes"] <= cost["peak_reserved_gpu_bytes"], f"Invalid resource totals: {run}")
    cost["index_seconds_by_domain"] = manifest["index_seconds_by_domain"]
    cost["stage_seconds"] = {field: sum(row.get(field, 0) for group in attempts.values() for row in group)
                             for field in ("retrieval_seconds", "answer_seconds", "judge_seconds")}
    cost["generation"] = {}
    for stage in STAGES:
        by_stage = [call for (_, _, name), call in calls.items() if name == stage]
        cost["generation"][stage] = {"calls": len(by_stage), "seconds": sum(c["seconds"] for c in by_stage),
                                     **{k: sum(c["usage"][k] for c in by_stage) for k in USAGE}}
    cost["failed_attempts"] = sum(len(group) - 1 for group in attempts.values())
    cost["retry_calls"] = len(calls) - len(rows) * len(STAGES)
    raw_files = ("run-config.json", "generation-audit.jsonl", "split.json", "full-split.json", "benchmark-summary.json",
                 *(f"{method}/{name}" for name in ("manifest.json", "predictions.jsonl", "attempts.jsonl", "metrics.json")))
    return indexed, calls, runtime, {"run": run.name, "source_commit": commit,
                                    "started_at_utc": runtime["started_at_utc"],
                                    "method_started_at_utc": manifest["started_at_utc"], "metrics": metrics,
                                    "cost": cost, "raw_sha256": {name: digest(run / name) for name in raw_files}}


def analyze(selected_run, control_run, freeze_path, dev_comparison, selection_plan, evidence):
    freeze = read(freeze_path)
    reference, split = read(evidence / "provenance.json"), read(evidence / "split.json")
    partitions = partition_map(split)
    dev = {qid for qid, part in partitions.items() if part == "dev"}
    test = set(partitions) - dev
    selected, control = freeze["selected_method"], freeze["control_method"]
    check(selected in {*STRUCTURED_METHODS, *REPLY_REPORTS} - {"hybrid"}
          and control == ("hybrid" if selected == "hybrid-meta" else "bm25"), "Invalid frozen methods")
    check(len(dev) == reference["dev"] == 179 and len(test) == reference["test"] == 536
          and digest(evidence / "split.json") == reference["split_sha256"] == freeze["split_sha256"]
          and reference["benchmark_commit"] == freeze["benchmark_commit"], "Published benchmark or split mismatch")
    check(digest(dev_comparison) == freeze["dev_comparison_sha256"]
          and digest(selection_plan) == freeze["selection_plan_sha256"], "Development comparison or selection plan hash mismatch")
    check(isinstance(freeze["selection_rule"], str) and freeze["selection_rule"].strip()
          and re.fullmatch(r"[0-9a-f]{40}", freeze["source_commit"]), "Missing frozen selection rule or immutable source commit")
    frozen_at = utc(freeze["frozen_at_utc"])
    dev_report = read(dev_comparison)
    audit_development(dev_report, freeze, evidence, dev)
    source = freeze["source_commit"]
    signature = implementation_signature(ROOT, source)
    for commit in {run["project_commit"] for run in dev_report["runs"].values()}:
        check_signature(commit, source, signature)
    left, left_calls, left_runtime, left_report = audit_test(selected_run, selected, reference, freeze, frozen_at, signature, split, test)
    right, right_calls, right_runtime, right_report = audit_test(control_run, control, reference, freeze, frozen_at, signature, split, test)
    check(all(left_runtime["arguments"][k] == right_runtime["arguments"][k] for k in ("model", "embedding_model")),
          "Paired runs used different model paths")
    if control == "hybrid":
        embedding = read(evidence.parent / "ordered-reply-dev.json")["model"]
        check(left_runtime["embedding_weight_sha256"] == right_runtime["embedding_weight_sha256"] == embedding["weight_sha256"]
              and left_runtime["arguments"]["embedding_revision"] == embedding["revision"], "Frozen hybrid E5 identity differs")

    questions = []
    for qid in sorted(test):
        a, b = left[qid], right[qid]
        check(all(a[k] == b[k] for k in ("question", "asking_user_id", "reference_answer")), f"Paired question metadata changed: {qid}")
        same_answer = a["answer"] == b["answer"]
        changed_context = a["retrieved_docs"] != b["retrieved_docs"]
        for stage, same_input in (("answer", not changed_context), ("judge", same_answer)):
            a_hash = left_calls[qid, a["attempt"], stage]["prompt_sha256"]
            b_hash = right_calls[qid, b["attempt"], stage]["prompt_sha256"]
            check((a_hash == b_hash) == same_input, f"Paired {stage} prompt hashes disagree: {qid}")
        gain, loss = a["official_correct"] and not b["official_correct"], b["official_correct"] and not a["official_correct"]
        questions.append({"domain": qid[0], "category": qid[1], "question_id": qid[2],
                          selected: a["official_correct"], control: b["official_correct"],
                          "outcome": "gain" if gain else "loss" if loss else "tie",
                          "changed_context": changed_context, "same_answer": same_answer,
                          "truncated": {method: {stage: row[stage + "_finish_reason"] == "length" for stage in STAGES}
                                        for method, row in ((selected, a), (control, b))}})
    overall = paired(questions, selected, control)
    discordant = overall["gained"] + overall["lost"]
    tail = sum(math.comb(discordant, j) for j in range(min(overall["gained"], overall["lost"]) + 1))
    overall["accuracy_delta"] = overall["net"] / len(test)
    overall["mcnemar_exact_two_sided_p"] = min(1.0, 2 * tail / 2 ** discordant)
    return {
        "purpose": "Evaluation of the frozen selection on the fixed test set. Method selection used development data only.",
        "freeze": freeze, "freeze_sha256": digest(freeze_path), "implementation": signature,
        "qa_protocol": {k: reference["protocol"][k] for k in ("generation_config", "model_provenance", "top_k")},
        "runs": {selected: left_report, control: right_report},
        "total_test_gpu_hours": left_report["cost"]["gpu_hours"] + right_report["cost"]["gpu_hours"],
        "paired": {
            "overall": overall,
            "by_category": {c: paired([r for r in questions if r["category"] == c], selected, control)
                            for c in reference["protocol"]["categories"]},
            "by_domain": {d: paired([r for r in questions if r["domain"] == d], selected, control)
                          for d in reference["protocol"]["domains"]},
            "caveat": "Question-level p-value. Questions may share dialogues or question families.",
        },
        "questions": questions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selected-run", "control-run", "freeze", "dev-comparison", "selection-plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=(ROOT / PUBLISHED_SPLIT).parent)
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(args.selected_run, args.control_run, args.freeze, args.dev_comparison,
                                 args.selection_plan, args.evidence), ensure_ascii=False, indent=2, allow_nan=False))
    except (KeyError, OSError, ValueError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Frozen test audit failed: {exc}\n")
