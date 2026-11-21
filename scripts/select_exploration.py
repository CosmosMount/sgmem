"""Freeze one Part 3 method using only the fixed development evidence."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess

from analyze_frozen_test import REPLY_REPORTS, audit_development, check_signature, index_rows
from analyze_reply_qa import ROOT, check, digest, key, read
from sgmem.retrieval import STRUCTURED_METHODS
from sgmem.shards import PUBLISHED_SPLIT, implementation_signature, partition_map


def bm25_dev_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [row for row in map(json.loads, handle) if row["partition"] == "dev" and row["method"] == "bm25"]


def reply_rankings(path, method, baseline, freeze):
    probe = read(path)
    check(probe["benchmark_commit"] == freeze["benchmark_commit"] and probe["split_sha256"] == freeze["split_sha256"]
          and probe["dev_questions"] == len(baseline) and probe["top_k"] == 10, f"Reply probe inputs changed: {path}")
    ranks = {qid: row["retrieved_ids"] for qid, row in baseline.items()}
    overrides = list(probe["changed_questions"])
    if method == "bm25-reply-order":
        overrides.extend(row for row in probe["changed_from_reply"] if row["method"] == method)
    for row in overrides:
        check(key(row) in ranks, f"Reply probe references a question outside dev: {path}")
        ranks[key(row)] = row["final_ids"]
    return ranks


def select(comparison_path, probe_path, plan_path, evidence):
    report, probe, plan = (read(path) for path in (comparison_path, probe_path, plan_path))
    split_path = evidence / "split.json"
    reference = read(evidence / "provenance.json")
    partitions = partition_map(read(split_path))
    dev = {qid for qid, part in partitions.items() if part == "dev"}
    source = plan["source_commit"]
    check(re.fullmatch(r"[0-9a-f]{40}", source), "Selection plan requires an immutable source commit")
    check(set(plan["methods"]) == set(STRUCTURED_METHODS) and len(plan["methods"]) == len(STRUCTURED_METHODS)
          and len(dev) == reference["dev"] == plan["dev_questions"] == 179
          and len(partitions) - len(dev) == reference["test"] == plan["test_questions"] == 536
          and plan["top_k"] == reference["protocol"]["top_k"] == 10
          and all(plan[k] == reference["protocol"]["model_provenance"][k] for k in ("qa_revision", "embedding_revision"))
          and isinstance(plan["selection"], str) and plan["selection"].strip(), "Selection plan disagrees with the fixed protocol")
    freeze = {"source_commit": source, "benchmark_commit": reference["benchmark_commit"],
              "split_sha256": digest(split_path), "dev_comparison_sha256": digest(comparison_path),
              "selection_plan_sha256": digest(plan_path), "selection_rule": plan["selection"],
              "retrieval_probe_sha256": digest(probe_path)}
    check(probe["project_git"]["status"] == "" and probe["benchmark_commit"] == freeze["benchmark_commit"]
          and probe["split_sha256"] == freeze["split_sha256"] == reference["split_sha256"]
          and probe["dev_questions"] == len(dev) and probe["top_k"] == 10
          and set(probe["methods"]) == set(STRUCTURED_METHODS), "Retrieval probe inputs or provenance changed")
    signature = implementation_signature(ROOT, source)
    for commit in {probe["project_git"]["commit"], *(run["project_commit"] for run in report["runs"].values())}:
        check_signature(commit, source, signature)
    check(report["complete_matrix"] is True and set(report["runs"]) == set(STRUCTURED_METHODS)
          and all(run["raw_sha256"]["retrieval_probe"] == freeze["retrieval_probe_sha256"]
                  for run in report["runs"].values()), "Development comparison must cover all eight methods on this retrieval probe")

    probe_rows = index_rows(probe["questions"], dev, "retrieval probe")
    baseline = index_rows(bm25_dev_rows(evidence / "outcomes.jsonl"), dev, "archived BM25 dev")
    targets = {}
    for row in bm25_dev_rows(evidence / "failure-annotations.jsonl"):
        if row["primary_failure_type"] in {"F1", "F2", "F3", "F4", "F5", "F6"} and row["target_message_ids"]:
            check(key(row) in dev, "Failure annotation references a question outside dev")
            targets.setdefault(key(row), set()).update(row["target_message_ids"])

    candidates = [m for m in plan["methods"] if m != "hybrid"] + list(REPLY_REPORTS)
    table = {}
    for method in candidates:
        if method in REPLY_REPORTS:
            qa = read(evidence.parent / REPLY_REPORTS[method])
            ranks = reply_rankings(evidence.parent / REPLY_REPORTS[method].replace("-qa-", "-"), method, baseline, freeze)
        else:
            qa = report["runs"][method]
            ranks = {qid: row["methods"][method]["final_ids"] for qid, row in probe_rows.items()}
        questions = index_rows(qa["questions"], dev, f"{method} dev")
        hours = qa["cost"]["gpu_hours"]
        check(math.isfinite(hours) and hours > 0 and all(len(ids) == len(set(ids)) == 10 for ids in ranks.values())
              and all(questions[qid]["bm25"] is baseline[qid]["official_correct"] for qid in dev),
              f"Invalid cost, top-10 ranks or BM25 outcomes: {method}")
        repairs = sum(not questions[qid]["bm25"] and questions[qid][method] and bool(ids.intersection(ranks[qid]))
                      for qid, ids in targets.items())
        table[method] = {"correct": report["metrics"][method]["overall"]["correct"],
                         "grounded_repairs": repairs, "gpu_hours": hours}

    def order(method):
        item = table[method]
        return -item["correct"], -item["grounded_repairs"], item["gpu_hours"]

    selected = min(candidates, key=order)
    control = "hybrid" if selected == "hybrid-meta" else "bm25"
    result = table[selected]
    freeze.update(selected_method=selected, control_method=control, selection_table=table, tie_break_order=candidates,
                  dev_selection={"selected_correct": result["correct"], "total": len(dev),
                                 "control_correct": report["metrics"][control]["overall"]["correct"],
                                 "reason": f"{selected} ranked first by official dev correctness ({result['correct']}/{len(dev)}), "
                                           f"grounded repairs ({result['grounded_repairs']}), measured QA GPU-hours "
                                           f"({result['gpu_hours']:.6f}) and the fixed method order."})
    audit_development(report, freeze, evidence, dev)
    freeze["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "probe", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=(ROOT / PUBLISHED_SPLIT).parent)
    args = parser.parse_args()
    try:
        freeze = select(args.comparison, args.probe, args.plan, args.evidence)
        payload = json.dumps(freeze, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except (KeyError, OSError, ValueError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Exploration selection failed: {exc}\n")


if __name__ == "__main__":
    main()
