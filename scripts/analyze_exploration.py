"""Audit separate full-dev runs and compare every completed Part 3 configuration together."""

import argparse
import json
from pathlib import Path

from analyze_reply_qa import ROOT, check, compare, digest, key, paired, read, read_rows
from sgmem.retrieval import STRUCTURED_METHODS


def analyze(runs, probe, bm25_audit, evidence):
    reports = {}
    for run in runs:
        methods = read(run / "run-config.json")["arguments"]["methods"]
        check(len(methods) == 1 and methods[0] in STRUCTURED_METHODS,
              "Use one method per run so that GPU cost is attributable")
        method = methods[0]
        check(method not in reports, f"Duplicate run for {method}")
        report = compare(run, bm25_audit, evidence, method, probe)
        rows = read_rows(run / method / "predictions.jsonl")
        report["truncated"] = {stage: sum(r[stage + "_finish_reason"] == "length" for r in rows)
                               for stage in ("answer", "judge")}
        reports[method] = report
    metrics = {m: reports[next(iter(reports))]["metrics"][m] for m in ("bm25", "dense")}
    questions = {key(r): {m: r[m] for m in ("bm25", "dense")}
                 for r in next(iter(reports.values()))["questions"]}
    control_hashes = {}
    first = next(iter(reports.values()))
    for method, name in (("bm25-reply", "bounded-reply-qa-dev.json"),
                         ("bm25-reply-order", "ordered-reply-qa-dev.json")):
        control = read(evidence.parent / name)
        check(len(control["questions"]) == len(questions) and {key(r) for r in control["questions"]} == set(questions),
              "Reply control coverage changed")
        check(control["source_split_sha256"] == first["source_split_sha256"]
              and control["benchmark_commit"] == first["benchmark_commit"], "Reply control source changed")
        control_hashes[method] = digest(evidence.parent / name)
        metrics[method] = control["metrics"][method]
        for row in control["questions"]:
            questions[key(row)][method] = row[method]
    for method, report in reports.items():
        metrics[method] = report["metrics"][method]
        for row in report["questions"]:
            questions[key(row)][method] = row[method]
    return {
        "purpose": "Development selection only; test remains frozen",
        "probe": str(probe), "complete_matrix": set(reports) == set(STRUCTURED_METHODS),
        "metrics": metrics,
        "pairwise": {m: {control: paired(list(questions.values()), m, control) for control in metrics if control != m}
                     for m in reports},
        "total_qa_gpu_hours": sum(report["cost"]["gpu_hours"] for report in reports.values()),
        "cost_scope": "Formal QA runs only; CUDA retrieval probes and preflights must be added separately",
        "reply_control_sha256": control_hashes,
        "runs": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--bm25-audit", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/part12-full-20261105")
    args = parser.parse_args()
    print(json.dumps(analyze(args.runs, args.probe, args.bm25_audit, args.evidence), ensure_ascii=False, indent=2))
