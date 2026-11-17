"""Evaluate bounded replies and their ablations on the published BM25 development questions."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

from analyze_metadata_hints import ROOT, read_rows
from sgmem.data import load_messages, load_questions
from sgmem.retrieval import BM25ReplyRetriever
from sgmem.runner import _git_info, question_file
from sgmem.shards import partition_map, row_key

K = 10
VARIANTS = {"one_hop": {"hops": 1}, "one_slot": {"slots": 1},
            "ungated": {"require_cue": False}, "bm25-reply": {}}
E5_VARIANTS = ("e5-reply", "e5-reply-parent", "bm25-reply-order")


def probe(evidence, benchmark, e5_model=None):
    split_path = evidence / "split.json"
    dev = {key for key, part in partition_map(json.loads(split_path.read_text(encoding="utf-8"))).items()
           if part == "dev"}
    provenance = json.loads((evidence / "provenance.json").read_text(encoding="utf-8"))
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != provenance["split_sha256"] or _git_info(benchmark)["commit"] != provenance["benchmark_commit"]:
        raise ValueError("Benchmark or split differs from the published baseline")
    saved = {name: {} for name in ("bm25", "dense")}
    for r in read_rows(evidence / "outcomes.jsonl"):
        if r["partition"] == "dev":
            saved[r["method"]][row_key(r)] = r
    if any(set(rows) != dev for rows in saved.values()):
        raise ValueError("Both published methods must cover the frozen development set")

    encoder = None
    if e5_model is not None:
        from sentence_transformers import SentenceTransformer

        published = json.loads((evidence.parent / "bounded-reply-dev.json").read_text(encoding="utf-8"))
        frozen = {row_key(r): r["final_ids"] for r in published["changed_questions"]}
        config_hashes = {p.relative_to(e5_model).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in e5_model.rglob("*.json")}
        if config_hashes != provenance["runtime"]["model_files"]["embedding"]["configuration_sha256"]:
            raise ValueError("E5 configuration differs from the Part 1 control")
        weights_hash = hashlib.sha256((e5_model / "model.safetensors").read_bytes()).hexdigest()
        encoder = SentenceTransformer(str(e5_model), device="cpu", local_files_only=True)

    results, changes = {}, []
    for domain in sorted({k[0] for k in dev}):
        messages = load_messages(benchmark / f"data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json")
        retriever = BM25ReplyRetriever(messages, encoder)
        variants = dict(VARIANTS)
        if e5_model is not None:
            variants["e5-reply"] = {"rank_replies": partial(retriever.e5_scores, include_parent=False)}
            variants["e5-reply-parent"] = {"rank_replies": retriever.e5_scores}
        for category in sorted({k[1] for k in dev if k[0] == domain}):
            for q in load_questions(question_file(benchmark, domain, category), domain, category):
                qid = row_key(q)
                if qid not in dev:
                    continue
                query = f"{q['asking_user_id']} {q['question']}" if q["asking_user_id"] else q["question"]
                scores, base = retriever.rank_content(query, K)
                old = [messages[i]["id"] for i in base]
                if old != saved["bm25"][qid]["retrieved_ids"] or [scores[i] for i in base] != saved["bm25"][qid]["retrieval_scores"]:
                    raise ValueError(f"Baseline replay differs at {qid}")
                results[qid] = {}
                for name, options in variants.items():
                    hits, pool = retriever.expand(query, scores, base, **options)
                    final = [h["message"]["id"] for h in hits]
                    if (len(final) != K or len(set(final)) != K or len(pool) != len(set(pool)) or
                            len(pool) > 2 * K or not set(final) <= set(pool) or final[:8] != old[:8]):
                        raise ValueError(f"Invalid candidate set for {name} at {qid}")
                    results[qid][name] = (pool, final)
                    if name in E5_VARIANTS:
                        control_pool, control_final = results[qid]["bm25-reply"]
                        if pool != control_pool or control_final != frozen.get(qid, old):
                            raise ValueError(f"Candidate pool or frozen reply control changed at {qid}")
                    if name == "bm25-reply" and final != old:
                        changes.append({"domain": domain, "category": category, "question_id": q["id"],
                                        "final_ids": final, "removed": [mid for mid in old if mid not in final],
                                        "added": [{"id": h["message"]["id"], "score": h["score"], "path": h["reply_path"]}
                                                  for h in hits if h["message"]["id"] not in old]})
                if encoder is not None:
                    # Exercise the QA runner's public retrieval entry point for the selected rule.
                    final = [h["message"]["id"] for h in retriever.retrieve(query, K)]
                    pool, lexical = results[qid]["bm25-reply"]
                    if final not in (lexical, results[qid]["e5-reply-parent"][1]):
                        raise ValueError(f"Ordered replies changed more than candidate ranking at {qid}")
                    results[qid]["bm25-reply-order"] = (pool, final)
    if set(results) != dev:
        raise ValueError("Questions do not cover the published development set")

    # Evidence annotations enter only after selection. Compare all methods on the same reviewed questions.
    targets = {row_key(a): set(a["target_message_ids"]) for a in read_rows(evidence / "failure-annotations.jsonl")
               if a["partition"] == "dev" and a["method"] == "bm25" and a["target_message_ids"]}
    cases, summary = [], {}
    methods = ("bm25", "dense", *VARIANTS, *(E5_VARIANTS if e5_model is not None else ()))
    for name in methods:
        changed = gained = lost = pool_hits = final_hits = 0
        for qid in sorted(dev):
            old = saved["bm25"][qid]["retrieved_ids"]
            if name in saved:
                final = pool = saved[name][qid]["retrieved_ids"]
            else:
                pool, final = results[qid][name]
            changed += set(final) != set(old)
            if qid in targets:
                target = targets[qid]
                was_hit, hit = bool(target.intersection(old)), bool(target.intersection(final))
                pool_hits += bool(target.intersection(pool))
                final_hits += hit
                gained += hit and not was_hit
                lost += was_hit and not hit
                if name == "bm25-reply":
                    cases.append({"domain": qid[0], "category": qid[1], "question_id": qid[2],
                                  "targets": sorted(target), "baseline_rank": best_rank(old, target),
                                  "in_pool": bool(target.intersection(pool)), "final_rank": best_rank(final, target)})
        summary[name] = {"changed_top10_sets": changed, "support_in_pool": pool_hits, "support_in_top10": final_hits,
                         "gained_vs_bm25": gained, "lost_vs_bm25": lost}
    report = {"benchmark_commit": provenance["benchmark_commit"], "split_sha256": split_hash,
              "dev_questions": len(dev), "reviewed_bm25_failures_with_support": len(targets), "top_k": K,
              "methods": summary, "reviewed_cases": cases, "changed_questions": changes}
    if e5_model is not None:
        report["model"] = {"revision": provenance["protocol"]["model_provenance"]["embedding_revision"],
                           "weight_sha256": weights_hash, "device": "cpu"}
        report["changed_from_reply"] = [
            {"domain": q[0], "category": q[1], "question_id": q[2], "method": name, "final_ids": results[q][name][1]}
            for name in E5_VARIANTS for q in sorted(dev) if results[q][name][1] != results[q]["bm25-reply"][1]]
        for case in cases:
            q = row_key(case)
            for name in E5_VARIANTS:
                case[name + "_rank"] = best_rank(results[q][name][1], targets[q])
        for name in E5_VARIANTS:
            rank = name + "_rank"
            summary[name].update(
                changed_prompts_vs_reply=sum(r["method"] == name for r in report["changed_from_reply"]),
                gained_vs_reply=sum(c[rank] is not None and c["final_rank"] is None for c in cases),
                lost_vs_reply=sum(c[rank] is None and c["final_rank"] is not None for c in cases))
    return report


def best_rank(ids, target):
    return next((rank for rank, mid in enumerate(ids, 1) if mid in target), None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "external/GroupMemBench")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/part12-full-20261105")
    parser.add_argument("--e5-model", type=Path, help="Compare reply-only and reply-plus-parent scores using local Part 1 E5 weights")
    args = parser.parse_args()
    print(json.dumps(probe(args.evidence, args.benchmark, args.e5_model), indent=2))
