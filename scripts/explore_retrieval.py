"""Replay the fixed Part 3 retrieval matrix on dev; gold is attached only after retrieval."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sgmem.data import load_messages, load_questions
from sgmem.retrieval import BM25Retriever, STRUCTURED_METHODS, build_retriever
from sgmem.runner import _git_info, question_file
from sgmem.shards import partition_map, row_key
from sgmem.structured import DEPTH


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explore(benchmark, evidence, methods, *, embedding_model=None, embedding_revision=None, device="cpu"):
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(STRUCTURED_METHODS):
        raise ValueError("Select distinct methods from the fixed exploration matrix")
    if any(m.startswith("hybrid") for m in methods) and not embedding_model:
        raise ValueError("Hybrid runs require the existing embedding model and model-run authorization")
    split_path = evidence / "split.json"
    dev = {k for k, part in partition_map(json.loads(split_path.read_text(encoding="utf-8"))).items() if part == "dev"}
    reference = json.loads((evidence / "provenance.json").read_text(encoding="utf-8"))
    embedding_sha = None
    if any(m.startswith("hybrid") for m in methods):
        provenance = reference["protocol"]["model_provenance"]
        if embedding_revision != provenance["embedding_revision"] or not Path(embedding_model).is_dir():
            raise ValueError("Use the existing local E5 model with its frozen revision")
        for name, expected in provenance["files"]["embedding"]["configuration_sha256"].items():
            if digest(Path(embedding_model) / name) != expected:
                raise ValueError(f"Embedding configuration changed: {name}")
        frozen_model = json.loads((evidence.parent / "ordered-reply-dev.json").read_text(encoding="utf-8"))["model"]
        embedding_sha = digest(Path(embedding_model) / "model.safetensors")
        if embedding_sha != frozen_model["weight_sha256"]:
            raise ValueError("Embedding weights differ from the verified E5 control")
    benchmark_git = _git_info(benchmark)
    if benchmark_git["commit"] != reference["benchmark_commit"] or benchmark_git["status"]:
        raise ValueError("Use the clean, pinned benchmark")
    if digest(split_path) != reference["split_sha256"]:
        raise ValueError("The fixed question split changed")
    # Python's text loaders normalize Windows checkouts to the same LF text used on Linux.
    normalized_inputs = []
    for name, expected in reference["protocol"]["inputs"].items():
        if name != "split" and digest(benchmark / name) != expected:
            normalized = (benchmark / name).read_text(encoding="utf-8").encode("utf-8")
            if hashlib.sha256(normalized).hexdigest() != expected:
                raise ValueError(f"Benchmark input changed: {name}")
            normalized_inputs.append(name)
    controls = {method: {row_key(r): r for r in read_rows(evidence / "outcomes.jsonl")
                        if r["method"] == method and r["partition"] == "dev"} for method in ("bm25", "dense")}
    if any(set(rows) != dev for rows in controls.values()):
        raise ValueError("Archived controls do not cover the fixed dev set")
    results, timings = {}, {m: {"index_seconds": {}, "retrieval_seconds": 0.0} for m in methods}
    started = time.perf_counter()
    project_git = _git_info(ROOT)
    for domain in sorted({k[0] for k in dev}):
        messages = load_messages(benchmark / f"data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json")
        questions = [q for category in sorted({k[1] for k in dev if k[0] == domain})
                     for q in load_questions(question_file(benchmark, domain, category), domain, category)
                     if row_key(q) in dev]
        baseline = BM25Retriever(messages)
        for q in questions:
            qid = row_key(q)
            hits = baseline.retrieve(q["question"], 10, asking_user_id=q["asking_user_id"])
            if ([h["message"]["id"] for h in hits] != controls["bm25"][qid]["retrieved_ids"] or
                    [h["score"] for h in hits] != controls["bm25"][qid]["retrieval_scores"]):
                raise ValueError(f"BM25 control changed: {qid}")
            results[qid] = {"domain": domain, "category": q["category"], "question_id": q["id"], "methods": {}}
        for method in methods:
            start = time.perf_counter()
            retriever = build_retriever(method, messages, model_name=embedding_model, revision=embedding_revision,
                                        device=device, batch_size=64)
            timings[method]["index_seconds"][domain] = time.perf_counter() - start
            for q in questions:
                qid = row_key(q)
                start = time.perf_counter()
                hits, trace = retriever.retrieve_with_trace(q["question"], 10, asking_user_id=q["asking_user_id"])
                timings[method]["retrieval_seconds"] += time.perf_counter() - start
                lists = trace["candidate_lists"]
                ids = [h["message"]["id"] for h in hits]
                if len(ids) != 10 or len(set(ids)) != 10:
                    raise ValueError(f"Invalid top-10 at {qid}/{method}")
                if any(len(order) > DEPTH or len(set(order)) != len(order) for order in lists.values()):
                    raise ValueError(f"Invalid candidate list at {qid}/{method}")
                if "dense" in lists and lists["dense"][:10] != controls["dense"][qid]["retrieved_ids"]:
                    raise ValueError(f"Dense control changed: {qid}")
                if len(trace["reply_pool"]) > 20 or len(set(trace["reply_pool"])) != len(trace["reply_pool"]):
                    raise ValueError(f"Invalid reply pool at {qid}/{method}")
                results[qid]["query_plan"] = trace["query_plan"]
                results[qid]["methods"][method] = {
                    "final_ids": ids, "scores": [h["score"] for h in hits],
                    "candidate_lists": lists, "reply_pool": trace["reply_pool"],
                    "reply_paths": {h["message"]["id"]: h["reply_path"] for h in hits if "reply_path" in h},
                }
            print(f"{domain}/{method}: {len(questions)} dev retrievals verified", file=sys.stderr, flush=True)
            del retriever
    if set(results) != dev:
        raise ValueError("Incomplete development coverage")
    # Targets enter only here, after every ranking has been produced.
    annotations = [r for r in read_rows(evidence / "failure-annotations.jsonl")
                   if r["partition"] == "dev" and row_key(r) in dev and r["target_message_ids"]]
    targets = defaultdict(set)
    for row in annotations:
        targets[row_key(row)].update(row["target_message_ids"])
    bm25_targets = {row_key(r): set(r["target_message_ids"]) for r in annotations if r["method"] == "bm25"}
    stats = {m: Counter(order_changed=0, set_changed=0, support_in_pool=0, support_in_top10=0,
                        support_gained=0, support_lost=0, max_pool=0, bm25_failure_support_hits=0) for m in methods}
    triggers = {m: Counter() for m in methods}
    for qid, row in sorted(results.items()):
        target = targets[qid]
        old = controls["bm25"][qid]["retrieved_ids"]
        if target:
            row["diagnostic_targets"] = sorted(target)
            row["control_support_ranks"] = {
                m: next((i for i, mid in enumerate(records[qid]["retrieved_ids"], 1) if mid in target), None)
                for m, records in controls.items()}
        for m, result in row["methods"].items():
            ids, lists = result["final_ids"], result["candidate_lists"]
            pool = {mid for items in lists.values() for mid in items} | set(result["reply_pool"])
            triggers[m].update(lists.keys())
            if result["reply_paths"]:
                triggers[m].update(["reply_added"])
            stats[m].update(order_changed=int(ids != old), set_changed=int(set(ids) != set(old)))
            stats[m]["max_pool"] = max(stats[m]["max_pool"], len(pool))
            if target:
                found, was_found = bool(target.intersection(ids)), bool(target.intersection(old))
                stats[m].update(support_in_pool=int(bool(target.intersection(pool))), support_in_top10=int(found),
                                support_gained=int(found and not was_found), support_lost=int(was_found and not found),
                                bm25_failure_support_hits=int(bool(bm25_targets.get(qid, set()).intersection(ids))))
                result["support_rank"] = next((i for i, mid in enumerate(ids, 1) if mid in target), None)
    return {
        "purpose": "Fixed-dev retrieval only; support annotations are a biased diagnostic sample, not QA accuracy",
        "project_git": project_git, "benchmark_commit": benchmark_git["commit"], "split_sha256": digest(split_path),
        "newline_normalized_inputs": normalized_inputs,
        "dev_questions": len(dev), "top_k": 10, "candidate_depth": DEPTH, "rrf_constant": 60,
        "embedding": {"model": embedding_model, "revision": embedding_revision, "device": device,
                      "weight_sha256": embedding_sha},
        "reviewed_questions_with_targets": sum(bool(t) for t in targets.values()),
        "bm25_failures_with_targets": len(bm25_targets),
        "control_support_hits": {m: sum(bool(target.intersection(rows[qid]["retrieved_ids"]))
                                        for qid, target in targets.items()) for m, rows in controls.items()},
        "methods": stats, "branches": triggers, "timings": timings,
        "wall_seconds": time.perf_counter() - started,
        "questions": [results[k] for k in sorted(results)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "external/GroupMemBench")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/part12-full-20261105")
    parser.add_argument("--methods", nargs="+", choices=STRUCTURED_METHODS,
                        default=[m for m in STRUCTURED_METHODS if not m.startswith("hybrid")])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; keep prior evidence")
    report = explore(args.benchmark, args.evidence, args.methods, embedding_model=args.embedding_model,
                     embedding_revision=args.embedding_revision, device=args.device)
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("questions", "timings")}, ensure_ascii=False, indent=2))
