"""Human-led failure annotation and descriptive summaries."""

from collections import Counter
from typing import Any, Iterable


FAILURE_TYPES = {
    "F1": "speaker_confusion",
    "F2": "group_confusion",
    "F3": "belief_attribution",
    "F4": "reply_context_loss",
    "F5": "temporal_update",
    "F6": "cross_participant_reasoning",
    "F7": "other_retrieval_error",
    "F8": "qa_reasoning_error",
}
_KEY_FIELDS = ("domain", "category", "question_id", "method")


def _key(record: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(record.get(field) for field in _KEY_FIELDS)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"Record must have non-empty string fields {_KEY_FIELDS}: {values}")
    return values


def _failed_records(records: Iterable[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    failed: dict[tuple[str, ...], dict[str, Any]] = {}
    seen: set[tuple[str, ...]] = set()
    for record in records:
        key = _key(record)
        if key in seen:
            raise ValueError(f"Duplicate question and method: {key}")
        seen.add(key)
        correct = record.get("correct")
        official_correct = record.get("official_correct")
        if correct is not True and correct is not False and correct is not None:
            raise ValueError(f"correct must be true, false, or null: {key}")
        if official_correct is not True and official_correct is not False and official_correct is not None:
            raise ValueError(f"official_correct must be true, false, or null: {key}")
        if correct is not None and official_correct is None:
            raise ValueError(f"A parsed verdict needs official_correct: {key}")
        if correct is not None and official_correct is False:
            failed[key] = record
    return failed


def build_annotation_template(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create unlabelled rows for judged failures; do not infer labels or evidence."""
    failed = _failed_records(records)
    return [
        {
            **{field: record[field] for field in _KEY_FIELDS},
            "question": record.get("question"),
            "prediction": record.get("prediction"),
            "reference_answer": record.get("reference_answer"),
            "retrieval": record.get("retrieval", []),
            "annotator": "",
            "failure_types": [],
            "primary_failure_type": None,
            "error_stage": None,
            "supporting_message_ids": [],
            "metadata_could_help": None,
            "notes": "",
        }
        for record in failed.values()
    ]


def summarize_annotations(
    records: Iterable[dict[str, Any]], annotations: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize independent annotations over judged failures only.

    Counts of secondary labels are multi-label counts, so their sum may exceed
    annotated examples. Evidence coverage is measured only for rows where a
    human has supplied at least one supporting message ID.
    """
    records = list(records)
    failed = _failed_records(records)
    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in annotations:
        key = _key(row)
        if key in rows:
            raise ValueError(f"Duplicate annotation: {key}")
        if key not in failed:
            raise ValueError(f"Annotation is not a judged failure in the supplied records: {key}")
        rows[key] = row

    primary = Counter()
    labels = Counter()
    stages = Counter()
    by_category: dict[str, Counter] = {}
    by_method: dict[str, dict[str, Any]] = {}
    metadata_recoverable = Counter()
    evidence_annotated = 0
    evidence_all_retrieved = 0
    evidence_any_retrieved = 0
    examples = []
    annotated = 0
    for key, record in failed.items():
        method_summary = by_method.setdefault(record["method"], {"judged_failures": 0, "annotated_failures": 0, "primary_failure_counts": Counter()})
        method_summary["judged_failures"] += 1
        row = rows.get(key)
        if row is None:
            continue
        failure_types = row.get("failure_types", [])
        if not isinstance(failure_types, list) or any(
            not isinstance(label, str) or label not in FAILURE_TYPES for label in failure_types
        ):
            raise ValueError(f"Invalid failure_types for {key}; expected labels F1–F8")
        if len(set(failure_types)) != len(failure_types):
            raise ValueError(f"Duplicate failure label for {key}")
        chosen = row.get("primary_failure_type")
        annotator = row.get("annotator", "")
        evidence = row.get("supporting_message_ids", [])
        stage = row.get("error_stage")
        could_help = row.get("metadata_could_help")
        if not isinstance(annotator, str) or not isinstance(evidence, list) or any(
            not isinstance(item, str) or not item for item in evidence
        ):
            raise ValueError(f"Invalid annotator or supporting_message_ids for {key}")
        if stage not in (None, "retrieval", "qa", "mixed", "unknown"):
            raise ValueError(f"Invalid error_stage for {key}")
        if could_help is not None and type(could_help) is not bool:
            raise ValueError(f"metadata_could_help must be true, false, or null: {key}")
        if not failure_types:
            if chosen is not None or stage is not None or annotator.strip() or evidence or could_help is not None or row.get("notes"):
                raise ValueError(f"Partial annotation without failure_types for {key}")
            continue
        if not annotator.strip() or chosen not in failure_types:
            raise ValueError(f"Annotated row needs an annotator and primary label in failure_types: {key}")
        annotated += 1
        method_summary["annotated_failures"] += 1
        method_summary["primary_failure_counts"][chosen] += 1
        metadata_recoverable["unknown" if could_help is None else ("yes" if could_help else "no")] += 1
        primary[chosen] += 1
        labels.update(failure_types)
        stages[stage or "unknown"] += 1
        by_category.setdefault(record["category"], Counter())[chosen] += 1
        if evidence:
            retrieved_ids = {
                item.get("message", {}).get("id")
                for item in record.get("retrieval", [])
            }
            evidence_set = set(evidence)
            evidence_annotated += 1
            evidence_any_retrieved += bool(evidence_set & retrieved_ids)
            evidence_all_retrieved += evidence_set <= retrieved_ids
        if row.get("notes"):
            examples.append({**{field: record[field] for field in _KEY_FIELDS},
                             "question": record.get("question"),
                             "prediction": record.get("prediction"),
                             "reference_answer": record.get("reference_answer"),
                             "retrieval": record.get("retrieval", []),
                             "failure_types": failure_types,
                             "primary_failure_type": chosen,
                             "supporting_message_ids": evidence,
                             "notes": row["notes"]})

    return {
        "taxonomy": FAILURE_TYPES,
        "total_records": len(records),
        "judged_correct": sum(record.get("correct") is not None and record.get("official_correct") is True for record in records),
        "unjudged": sum(record.get("correct") is None for record in records),
        "judged_failures": len(failed),
        "annotated_failures": annotated,
        "unannotated_failures": len(failed) - annotated,
        "primary_failure_counts": dict(sorted(primary.items())),
        "multi_label_counts": dict(sorted(labels.items())),
        "primary_counts_by_category": {category: dict(sorted(counts.items())) for category, counts in sorted(by_category.items())},
        "error_stage_counts": dict(sorted(stages.items())),
        "metadata_could_help_counts": dict(sorted(metadata_recoverable.items())),
        "by_method": {method: {**value, "unannotated_failures": value["judged_failures"] - value["annotated_failures"], "primary_failure_counts": dict(sorted(value["primary_failure_counts"].items()))} for method, value in sorted(by_method.items())},
        "manually_annotated_evidence": {
            "examples_with_evidence": evidence_annotated,
            "examples_with_any_evidence_retrieved": evidence_any_retrieved,
            "examples_with_all_evidence_retrieved": evidence_all_retrieved,
        },
        "examples_with_notes": examples,
    }
