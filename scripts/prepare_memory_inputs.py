"""Export label-free inputs for Dense X, LightMem, RAPTOR and HippoRAG 2."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgmem.data import load_messages, load_questions
from sgmem.runner import CATEGORIES, DOMAINS, _git_info, question_file, write_json
from sgmem.shards import PUBLISHED_SPLIT, load_frozen_split, row_key


def write_rows(path, rows):
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
            digest.update(raw)
            handle.write(raw)
    return digest.hexdigest()


def prepare(output, benchmark, split_path, partition="dev", audit_per_domain=50):
    """Keep original domain/ID identity; labels never enter exported messages/queries."""
    output, benchmark, split_path = map(Path, (output, benchmark, split_path))
    if partition not in ("dev", "test") or audit_per_domain < 1:
        raise ValueError("Use dev/test and a positive source audit sample size")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    split = load_frozen_split(split_path, benchmark, DOMAINS, CATEGORIES)
    selected = {row_key(row) for row in split[partition]}
    queries = []
    for domain in DOMAINS:
        for category in CATEGORIES:
            questions = load_questions(question_file(benchmark, domain, category), domain, category)
            for q in questions:
                if row_key(q) in selected:
                    # Category and ID identify the record, not features for retrieval.
                    queries.append({key: q[key] for key in (
                        "domain", "category", "id", "question", "asking_user_id")})
    if {row_key(row) for row in queries} != selected:
        raise ValueError("Public query export does not cover the frozen selection")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "purpose": "Label-free memory pipeline inputs; no inference or scored results",
        "status": "incomplete",
        "benchmark_git": _git_info(benchmark),
        "project_git": _git_info(ROOT),
        "partition": partition,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "question_count": len(queries),
        "source_audit": {"seed": 42, "per_domain": audit_per_domain,
                         "selection": "uniform source messages, independent of questions and answers"},
        "identity": "(domain, id); never collapse different source IDs by identical text",
        "visibility": "complete domain corpus, matching the published GroupMemBench baseline",
        "files": {},
        "domains": {},
    }
    write_json(output / "manifest.json", manifest)
    audit = []
    for domain in DOMAINS:
        path = benchmark / f"data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json"
        messages = [{"domain": domain, **row} for row in load_messages(path)]
        if not messages:
            raise ValueError(f"Empty corpus: {domain}")
        filename = f"messages-{domain}.jsonl"
        manifest["files"][filename] = write_rows(output / filename, messages)
        bodies = [html.unescape(row["content"]).strip() for row in messages]
        counts = Counter(bodies)
        manifest["domains"][domain] = {
            "messages": len(messages),
            "groups": len({row["group"] for row in messages}),
            "distinct_bodies": len(counts),
            "duplicate_body_occurrences": len(messages) - len(counts),
            "largest_identical_body_group": max(counts.values()),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        # Reset per domain so adding another domain does not change an existing sample.
        indices = random.Random(42).sample(range(len(messages)), min(audit_per_domain, len(messages)))
        audit.extend(messages[i] for i in sorted(indices))
    filename = f"queries-{partition}.jsonl"
    manifest["files"][filename] = write_rows(output / filename, queries)
    manifest["files"]["extraction-audit.jsonl"] = write_rows(output / "extraction-audit.jsonl", audit)
    manifest["source_audit"]["total"] = len(audit)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New input directory; never overwritten")
    parser.add_argument("--benchmark", default=str(ROOT / "external/GroupMemBench"))
    parser.add_argument("--split", default=str(ROOT / PUBLISHED_SPLIT))
    parser.add_argument("--partition", choices=("dev", "test"), default="dev")
    parser.add_argument("--audit-per-domain", type=int, default=50)
    args = parser.parse_args()
    result = prepare(args.output, args.benchmark, args.split, args.partition, args.audit_per_domain)
    print(json.dumps({"output": str(Path(args.output).resolve()), "queries": result["question_count"],
                      "messages": sum(d["messages"] for d in result["domains"].values()),
                      "inference_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
