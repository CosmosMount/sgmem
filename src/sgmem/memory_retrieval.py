"""Export raw-message rankings and before-budget native memory candidates."""

from contextlib import ExitStack
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import re

from sgmem.propositions import source_digest, source_messages
from sgmem.query_metadata import build_catalog
from sgmem.retrieval import _content
from sgmem.runner import write_json
from sgmem.structured import StructuredRetriever, reciprocal_rank_fusion


METHODS = ("densex", "lightmem", "raptor", "hipporag2")
ABLATIONS = ("densex-sentences", "lightmem-pre", "raptor-leaf", "hipporag2-dpr")
CONTEXT_BASES = ("densex", "lightmem", "raptor")
CONTEXT_VIEWS = ("native", "raw-budget")
CONTEXT_METHODS = tuple(f"{method}-{view}" for method in CONTEXT_BASES for view in CONTEXT_VIEWS)
CONTEXT_BUDGET = 3500
CANDIDATE_DEPTH, TOP_K = 50, 10
SOURCE_FIELDS = {"domain", "id", "content", "speaker", "group", "role", "timestamp", "reply_to", "phase_name", "topic"}
QUERY_FIELDS = {"domain", "category", "id", "question"}
SENTENCE_END = re.compile(r"[.!?]+[\"”'’)\]]*(?=\s|$)")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "fig", "dept", "e.g", "i.e"}


def sentence_records(messages):
    """Produce deterministic sentence units with the same parent IDs as Dense X."""
    records = []
    for message in messages:
        text = _content(message)
        start, units = 0, []
        for boundary in SENTENCE_END.finditer(text):
            stem = re.search(r"[a-zA-Z]+(?:\.[a-zA-Z]+)*$", text[start:boundary.start()])
            word = stem[0] if stem else ""
            if boundary[0] == "." and (word.lower() in ABBREVIATIONS or re.fullmatch(r"(?:[A-Z]\.)*[A-Z]", word)):
                continue
            units.append(text[start:boundary.end()].strip())
            start = boundary.end()
        units.append(text[start:].strip())
        units = list(dict.fromkeys(unit for unit in units if unit))
        records.append({"domain": message["domain"], "message_id": message["id"],
                        "source_sha256": source_digest(message), "units": units, "status": "ok", "reason": None,
                        "raw_output": json.dumps(units, ensure_ascii=False),
                        "input_tokens": 0, "tokenized_input_tokens": 0, "output_tokens": 0, "seconds": 0})
    return records


def public_sources(messages):
    sources = source_messages(messages)
    if len({domain for domain, _ in sources}) != 1:
        raise ValueError("The message pack must belong to one domain")
    for message in sources.values():
        if (set(message) - SOURCE_FIELDS or not isinstance(message.get("group"), str)
                or any(value is not None and not isinstance(value, str) for value in message.values())):
            raise ValueError("Expected normalized public messages without additional or label fields")
    return sources


def public_questions(questions, domain):
    seen, selected = set(), []
    for question in questions:
        if (not isinstance(question, dict) or set(question) - (QUERY_FIELDS | {"asking_user_id"})
                or not all(isinstance(question.get(field), str) and question[field].strip() for field in QUERY_FIELDS)
                or (question.get("asking_user_id") is not None and not isinstance(question["asking_user_id"], str))):
            raise ValueError("Expected public question fields only, without answers or support labels")
        key = question["domain"], question["category"], question["id"]
        if key in seen:
            raise ValueError(f"Duplicate public question identity: {key}")
        seen.add(key)
        if key[0] == domain:
            selected.append(question)
    if not selected:
        raise ValueError(f"No public questions for source domain {domain}")
    return selected


class PairedMemoryRetriever:
    # The public query resolver needs only this catalog and these names, not a BM25 index.
    query_plan = StructuredRetriever.query_plan

    def __init__(self, retriever, sources, *, compare_metadata=True):
        self.domain = next(iter(sources))[0]
        self.retriever = retriever
        self.sources = {message["id"]: dict(message) for message in sources.values()}
        self.compare_metadata = compare_metadata
        if compare_metadata:
            self.catalog = build_catalog(self.sources.values())
            self.speakers = sorted({message["speaker"] for message in self.sources.values() if message.get("speaker")})
            self.roles = sorted({message["role"] for message in self.sources.values() if message.get("role")})

    def raw_candidates(self, question, asking_user_id=None):
        depth = min(CANDIDATE_DEPTH, len(self.sources))
        if hasattr(self.retriever, "retrieve_with_trace"):
            hits, native_trace = self.retriever.retrieve_with_trace(question, depth, asking_user_id=asking_user_id)
        else:
            hits, native_trace = self.retriever.retrieve(question, depth, asking_user_id=asking_user_id), {}
        if len(hits) != depth:
            raise ValueError(f"Expected {depth} raw candidates, received {len(hits)}")
        raw = {}
        for hit in hits:
            source = hit.get("message") if isinstance(hit, dict) else None
            message_id = source.get("id") if isinstance(source, dict) else None
            if (not isinstance(message_id, str) or message_id in raw or message_id not in self.sources
                    or source != self.sources[message_id]):
                raise ValueError(f"Wrong-domain, duplicate, unknown or modified original source: {message_id!r}")
            score = hit.get("score")
            if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score):
                raise ValueError(f"Invalid retrieval score for {message_id}")
            trace = {**hit.get("trace", {}), **{key: value for key, value in hit.items()
                                              if key not in {"message", "score", "trace"}}}
            raw[message_id] = {"message_id": message_id, "score": float(score), "trace": trace}

        ids = list(raw)
        base_trace = {"base_order": ids, "base_scores": {key: hit["score"] for key, hit in raw.items()},
                      "retriever": native_trace}
        return raw, base_trace

    def rank(self, question, asking_user_id=None):
        raw, base_trace = self.raw_candidates(question, asking_user_id)
        base = list(raw.values())[:TOP_K], base_trace
        if not self.compare_metadata:
            return (base,)

        plan = self.query_plan(question, asking_user_id)
        groups = {item["name"] for item in plan["matched_names"] if item["field"] == "group"}
        named_fields = [(item["field"], item["name"]) for item in plan["matched_names"] if item["field"] != "group"]
        speakers, roles = set(plan["speakers"]), set(plan["roles"])
        matched = []
        if groups or named_fields or speakers or roles:
            for position, message_id in enumerate(raw):
                message = self.sources[message_id]
                scope = (not groups or message.get("group") in groups) and (
                    len(groups) > 1 or not named_fields or any(message.get(field) == name for field, name in named_fields))
                actor = (message.get("speaker") in speakers if speakers else
                         not roles or message.get("role") in roles)
                if scope and actor:
                    matched.append(position)

        ids = list(raw)
        if matched:
            order, scores = reciprocal_rank_fusion({"base": range(len(ids)), "metadata": matched})
            meta_scores = {ids[pos]: scores[pos] for pos in order}
        else:
            meta_scores = {message_id: hit["score"] for message_id, hit in raw.items()}
        meta_trace = {**base_trace, "metadata": {"triggered": bool(matched), "plan": plan,
                      "matched_ids": [ids[pos] for pos in matched], "scores": meta_scores}}
        meta = [dict(raw[key], score=meta_scores[key]) for key in list(meta_scores)[:TOP_K]]
        return base, (meta, meta_trace)


def _native_candidates(retriever, method, sources, domain, question, asking_user_id):
    if method == "raptor":
        hits = retriever.retrieve_nodes(question, CANDIDATE_DEPTH, asking_user_id=asking_user_id,
                                        max_context_tokens=None)
        id_field = "node_id"
    else:
        hits = retriever.retrieve_native(question, CANDIDATE_DEPTH, asking_user_id=asking_user_id)
        id_field = "id" if method == "lightmem" else "unit_id"
    if not 0 < len(hits) <= CANDIDATE_DEPTH:
        raise ValueError("Expected a nonempty native candidate pool of at most 50 units")
    seen, units, details = set(), [], []
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError("Expected a native memory unit object")
        identifier, text, score, source_ids = (hit.get(key) for key in (id_field, "text", "score", "source_ids"))
        if not ((isinstance(identifier, str) and identifier.strip()) or (type(identifier) is int and identifier >= 0)):
            raise ValueError("Invalid native unit ID")
        if identifier in seen or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Duplicate native ID or invalid text: {identifier!r}")
        if (isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score)
                or (units and score > units[-1]["score"])):
            raise ValueError(f"Invalid native score or ranking order for unit {identifier!r}")
        if (not isinstance(source_ids, list) or not source_ids
                or any(not isinstance(source, str) or (domain, source) not in sources for source in source_ids)
                or len(set(source_ids)) != len(source_ids)):
            raise ValueError(f"Unknown, duplicate or wrong-domain source lineage for native unit {identifier!r}")
        seen.add(identifier)
        units.append({"unit_id": identifier, "text": text, "score": float(score), "source_ids": list(source_ids)})
        extras = {key: hit[key] for key in ("layer", "children", "anchor_dia_id") if key in hit}
        if extras:
            details.append({"unit_id": identifier, **extras})
    return units, {"unit_metadata": details}


def export_rankings(retriever, sources, questions, method, output, provenance, *, context_view=None):
    if method not in (*METHODS, *ABLATIONS):
        raise ValueError(f"Unknown memory method: {method}")
    domain = next(iter(sources))[0]
    if context_view is None:
        paired = PairedMemoryRetriever(retriever, sources, compare_metadata=method in METHODS)
        names = (method, f"{method}-meta") if paired.compare_metadata else (method,)
    else:
        if f"{method}-{context_view}" not in CONTEXT_METHODS:
            raise ValueError(f"Unsupported context view {context_view!r} for {method}")
        messages_sha = provenance.get("messages_sha256")
        if not isinstance(messages_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", messages_sha):
            raise ValueError("Context export requires the public source pack SHA-256")
        paired = PairedMemoryRetriever(retriever, sources, compare_metadata=False) if context_view == "raw-budget" else None
        names = (f"{method}-{context_view}",)
    questions = public_questions(questions, domain)
    output = Path(output)
    output.mkdir(parents=True)
    path = output / "manifest.json"
    keys = [{"domain": row["domain"], "category": row["category"], "question_id": row["id"]} for row in questions]
    manifest = {**provenance, "schema_version": 1, "status": "incomplete", "domain": domain,
                "methods": list(names), "files": {name: f"{name}.jsonl" for name in names},
                "expected_queries": keys, "completed_queries": 0, "candidate_depth": CANDIDATE_DEPTH, "top_k": TOP_K}
    if context_view is not None:
        manifest.update(top_k=None, context_budget=CONTEXT_BUDGET, context_view=context_view)
    write_json(path, manifest)
    try:
        with ExitStack() as stack:
            handles = [stack.enter_context((output / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n"))
                       for name in names]
            for question, key in zip(questions, keys, strict=True):
                query, asker = question["question"], question.get("asking_user_id")
                if context_view == "native":
                    pairs = (_native_candidates(retriever, method, sources, domain, query, asker),)
                elif context_view == "raw-budget":
                    raw, trace = paired.raw_candidates(query, asker)
                    units = [{"unit_id": source, "text": _content(paired.sources[source]), "score": hit["score"],
                              "source_ids": [source]} for source, hit in raw.items()]
                    pairs = ((units, trace),)
                else:
                    pairs = paired.rank(query, asker)
                payloads = [json.dumps({**key, "question": question["question"],
                            "asking_user_id": question.get("asking_user_id"), "method": name,
                            "retrieval": hits, "query_trace": trace}, ensure_ascii=False, allow_nan=False) + "\n"
                            for name, (hits, trace) in zip(names, pairs, strict=True)]
                for handle, payload in zip(handles, payloads, strict=True):
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                manifest["completed_queries"] += 1
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["error"] = str(exc)
        if isinstance(getattr(exc, "trace", None), dict):
            manifest["error_trace"] = exc.trace
        raise
    finally:
        write_json(path, manifest)
    return manifest
