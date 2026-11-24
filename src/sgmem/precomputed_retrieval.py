"""Validate a completed memory ranking and resolve its hits to benchmark messages."""

import hashlib
import json
from math import isfinite
from pathlib import Path

from sgmem.data import load_messages, load_questions
from sgmem.memory_retrieval import ABLATIONS, CANDIDATE_DEPTH, CONTEXT_BUDGET, CONTEXT_METHODS, METHODS, public_sources
from sgmem.retrieval import _content
from sgmem.runner import CATEGORIES, question_file


PRECOMPUTED_METHODS = tuple(name for method in METHODS for name in (method, f"{method}-meta")) + ABLATIONS + CONTEXT_METHODS
KEY_FIELDS = ("domain", "category", "question_id")
RANKING_FIELDS = {*KEY_FIELDS, "question", "asking_user_id", "method", "retrieval", "query_trace"}


def context_protocol(method):
    if method in CONTEXT_METHODS:
        view = "raw-budget" if method.endswith("-raw-budget") else "native"
        return {"view": view, "budget_tokens": CONTEXT_BUDGET, "candidate_depth": CANDIDATE_DEPTH,
                "count": "rendered_prompt_delta"}
    return None


def _question_key(row, domain):
    if (not isinstance(row, dict) or row.get("domain") != domain
            or row.get("category") not in CATEGORIES or not isinstance(row.get("question_id"), str)
            or not row["question_id"]):
        raise ValueError(f"Invalid ranking question identity for {domain}")
    return tuple(row[field] for field in KEY_FIELDS)


def _source_digest(messages, domain):
    # Match prepare_memory_inputs.write_rows for the full ordered domain pack.
    digest = hashlib.sha256()
    for message in messages:
        text = json.dumps({"domain": domain, **message}, ensure_ascii=False, sort_keys=True)
        digest.update((text + "\n").encode("utf-8"))
    return digest.hexdigest()


def _resolve_hits(ranking, originals, top_k):
    if not isinstance(ranking, list) or len(ranking) != min(top_k, len(originals)):
        raise ValueError("Precomputed ranking has the wrong number of raw source IDs")
    hits, seen = [], set()
    for hit in ranking:
        if not isinstance(hit, dict) or set(hit) != {"message_id", "score", "trace"}:
            raise ValueError("Precomputed hits must contain a raw source ID, score and trace")
        message_id, score = hit["message_id"], hit["score"]
        if not isinstance(message_id, str) or message_id not in originals or message_id in seen:
            raise ValueError(f"Unknown or repeated precomputed source ID: {message_id!r}")
        if type(score) not in (int, float) or not isfinite(score) or not isinstance(hit["trace"], dict):
            raise ValueError(f"Invalid precomputed score or trace for {message_id}")
        seen.add(message_id)
        hits.append({"message": originals[message_id], "score": score, "trace": hit["trace"]})
    return hits


def _validate_units(units, originals, *, raw):
    if not isinstance(units, list) or not 1 <= len(units) <= CANDIDATE_DEPTH:
        raise ValueError("Invalid precomputed context candidate count")
    if raw and len(units) != min(CANDIDATE_DEPTH, len(originals)):
        raise ValueError("Raw budget ranking does not cover its candidate depth")
    seen = set()
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != {"unit_id", "text", "score", "source_ids"}:
            raise ValueError("Invalid precomputed context unit fields")
        identifier, text, score, ids = (unit[field] for field in ("unit_id", "text", "score", "source_ids"))
        if not ((isinstance(identifier, str) and identifier.strip()) or (type(identifier) is int and identifier >= 0)):
            raise ValueError("Invalid precomputed context unit ID")
        if identifier in seen or not isinstance(text, str) or not text.strip() or type(score) not in (int, float) or not isfinite(score):
            raise ValueError(f"Repeated or invalid precomputed context unit: {identifier!r}")
        if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or item not in originals for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f"Unknown or invalid precomputed context source lineage: {identifier!r}")
        if raw and (ids != [identifier] or text != _content(originals[ids[0]])):
            raise ValueError(f"Raw budget unit differs from its original source: {identifier!r}")
        seen.add(identifier)
    return units


def load_rankings(directories, method, benchmark, expected, top_k):
    """Verify artifacts before QA and return hits keyed by immutable question identity."""
    if method not in PRECOMPUTED_METHODS:
        raise ValueError(f"Precomputed rankings are unsupported for {method}")
    needed_domains = {key[0] for key in expected}
    if not needed_domains:
        raise ValueError("No QA-selected questions for precomputed retrieval")
    context = context_protocol(method)
    records, provenance, sources_by_domain = {}, {}, {}
    for directory in directories:
        directory = Path(directory).resolve()
        manifest_path = directory / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if (not isinstance(manifest, dict) or not isinstance(manifest.get("methods"), list)
                or not isinstance(manifest.get("domain"), str)):
            raise ValueError(f"Invalid ranking manifest: {manifest_path}")
        domain = manifest.get("domain")
        if method not in manifest["methods"] or domain not in needed_domains:
            continue
        if domain in provenance:
            raise ValueError(f"Multiple precomputed {method} artifacts for {domain}")
        if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise ValueError(f"Precomputed ranking manifest is incomplete or unsupported: {manifest_path}")
        filename = f"{method}.jsonl"
        if not isinstance(manifest.get("files"), dict) or manifest["files"].get(method) != filename:
            raise ValueError(f"Precomputed ranking filename mismatch: {manifest_path}")
        if context is None and manifest.get("top_k") != top_k:
            raise ValueError(f"Precomputed ranking top-k mismatch: {manifest_path}")
        if context is not None and ("top_k" not in manifest or manifest["top_k"] is not None
                                    or manifest.get("context_view") != context["view"]
                                    or manifest.get("context_budget") != CONTEXT_BUDGET
                                    or manifest.get("candidate_depth") != CANDIDATE_DEPTH):
            raise ValueError(f"Precomputed context protocol mismatch: {manifest_path}")
        membership = manifest.get("expected_queries")
        if not isinstance(membership, list) or manifest.get("completed_queries") != len(membership):
            raise ValueError(f"Incomplete precomputed membership: {manifest_path}")
        declared = {_question_key(row, domain) for row in membership}
        if not declared or len(declared) != len(membership):
            raise ValueError(f"Duplicate or empty precomputed membership: {manifest_path}")

        messages = load_messages(benchmark / "data/final" / domain / f"synthetic_domain_channels_rolevariants_{domain}.json")
        if not messages or _source_digest(messages, domain) != manifest.get("messages_sha256"):
            raise ValueError(f"Precomputed source digest differs from the benchmark corpus for {domain}")
        originals = {message["id"]: message for message in messages}
        if context is not None:
            sources_by_domain[domain] = public_sources({"domain": domain, **message} for message in messages)
        official = {(domain, category, row["id"]): row for category in {key[1] for key in declared}
                    for row in load_questions(question_file(benchmark, domain, category), domain, category)}
        ranking_path = directory / filename
        ranking_bytes = ranking_path.read_bytes()
        seen = set()
        for line in ranking_bytes.splitlines():
            row = json.loads(line)
            key = _question_key(row, domain)
            if key not in declared or key in seen or key not in official:
                raise ValueError(f"Undeclared, repeated or unknown precomputed question: {key}")
            selected = expected.get(key, official[key])
            if (set(row) != RANKING_FIELDS or row["method"] != method or not isinstance(row["query_trace"], dict)
                    or row["question"] != selected["question"] or row["asking_user_id"] != selected["asking_user_id"]):
                raise ValueError(f"Precomputed question wording, asker, method or fields differ: {key}")
            if context is None:
                hits = _resolve_hits(row["retrieval"], originals, top_k)
            else:
                hits = _validate_units(row["retrieval"], originals, raw=context["view"] == "raw-budget")
            if key in expected:
                records[key] = hits, row["query_trace"]
            seen.add(key)
        if seen != declared:
            raise ValueError(f"Precomputed rows do not match their manifest membership: {ranking_path}")
        provenance[domain] = {"manifest": str(manifest_path), "ranking": str(ranking_path),
                              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                              "ranking_sha256": hashlib.sha256(ranking_bytes).hexdigest(),
                              "source_pack_sha256": manifest["messages_sha256"]}
    if records.keys() != expected.keys():
        missing = sorted(expected.keys() - records.keys())
        raise ValueError(f"Precomputed rankings omit {len(missing)} QA-selected questions, first: {missing[:3]}")
    return records, provenance, sources_by_domain
