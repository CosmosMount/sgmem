"""Describe public query metadata on frozen dev questions; never run QA or rerank."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sgmem.data import load_messages, load_questions


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_catalog(messages):
    """Use names and their groups from visible messages, without answer labels."""
    names = {}
    for message in messages:
        for field in ("group", "phase_name", "topic"):
            name = message.get(field)
            if name and normalize(name):
                names.setdefault((field, name), set()).add(message["group"])
    return [{"field": field, "name": name, "groups": sorted(groups)}
            for (field, name), groups in sorted(names.items())]


def query_hints(question, catalog):
    """A hint is unique only when every matched name belongs to the same group."""
    query = " " + normalize(question) + " "
    matches = [entry for entry in catalog if " " + normalize(entry["name"]) + " " in query]
    groups = sorted({group for entry in matches for group in entry["groups"]})
    return {
        "matched_names": matches,
        "group_hint": groups[0] if len(groups) == 1 else None,
        "status": "none" if not matches else ("unique" if len(groups) == 1 else "ambiguous"),
        "first_person_mention": bool(re.search(r"\b(i|my|mine)\b", question, re.I)),
    }


def key(row):
    return row["domain"], row["category"], row.get("question_id", row.get("id"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(evidence, benchmark):
    split_path = evidence / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dev_ids = {key(row) for row in split["dev"]}
    questions = {}
    messages = {}
    for domain in sorted({k[0] for k in dev_ids}):
        rows = load_messages(benchmark / f"data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json")
        messages[domain] = {row["id"]: row for row in rows}
        catalog = build_catalog(rows)
        for category in sorted({k[1] for k in dev_ids if k[0] == domain}):
            for question in load_questions(benchmark / f"questions/{domain}/{category}.jsonl", domain, category):
                if key(question) in dev_ids:
                    questions[key(question)] = {
                        "domain": domain, "category": category, "question_id": question["id"],
                        "question": question["question"], "asking_user_id": question["asking_user_id"],
                        **query_hints(question["question"], catalog),
                    }
    if set(questions) != dev_ids:
        raise ValueError("Question files do not cover the frozen development split")

    # Reference-derived target IDs enter only this separate post-hoc comparison.
    outcomes = {(*key(row), row["method"]): row for row in read_rows(evidence / "outcomes.jsonl")
                if row["partition"] == "dev"}
    support_rows = []
    for annotation in read_rows(evidence / "failure-annotations.jsonl"):
        targets = annotation["target_message_ids"]
        if not targets:
            continue
        if annotation["partition"] != "dev":
            raise ValueError("Support diagnostics must use development annotations only")
        question = questions[key(annotation)]
        by_id = messages[annotation["domain"]]
        target_groups = sorted({by_id[mid]["group"] for mid in targets})
        outcome = outcomes[(*key(annotation), annotation["method"])]
        retrieved = [by_id[mid] for mid in outcome["retrieved_ids"]]
        hint = question["group_hint"]
        support_rows.append({
            **{field: annotation[field] for field in ("case_id", "domain", "category", "question_id", "method")},
            "group_hint": hint, "target_groups": target_groups,
            "hint_agrees_with_support_group": hint in target_groups if hint else None,
            "top10_in_support_group": sum(m["group"] in target_groups for m in retrieved),
            "first_person_mention": question["first_person_mention"],
            "all_target_speakers_equal_asker": all(by_id[mid]["speaker"] == question["asking_user_id"] for mid in targets),
        })

    question_rows = [questions[k] for k in sorted(questions)]
    return {
        "scope": "Descriptive dev-only metadata hints; no QA run, no changed retrieval, no accuracy claim.",
        "rule": "Full names after lowercase/punctuation/space normalization; union of matched groups must be a singleton.",
        "query_hint_inputs": ["question text", "visible message group/phase_name/topic"],
        "support_comparison": "Reviewer target IDs are used only after query hints are computed, not to derive hints.",
        "first_person_caveat": "An I/my/mine mention alone does not prove the asker is the target speaker.",
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "dev_questions": len(question_rows),
        "group_hint_counts": dict(sorted(Counter(q["status"] for q in question_rows).items())),
        "matched_field_question_counts": dict(sorted(Counter(
            field for q in question_rows for field in {x["field"] for x in q["matched_names"]}).items())),
        "first_person_questions": sum(q["first_person_mention"] for q in question_rows),
        "by_domain": {domain: dict(sorted(Counter(q["status"] for q in question_rows if q["domain"] == domain).items()))
                      for domain in sorted(messages)},
        "positive_support": {
            "records": len(support_rows),
            "group_hint_available": sum(r["group_hint"] is not None for r in support_rows),
            "group_hint_agrees": sum(r["hint_agrees_with_support_group"] is True for r in support_rows),
            "group_hint_conflicts": sum(r["hint_agrees_with_support_group"] is False for r in support_rows),
            "all_top10_already_in_support_group": sum(r["top10_in_support_group"] == 10 for r in support_rows),
            "hint_available_and_some_wrong_group_candidates": sum(
                r["group_hint"] is not None and r["top10_in_support_group"] < 10 for r in support_rows),
        },
        "questions": question_rows,
        "positive_support_rows": support_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "external/GroupMemBench")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/part12-full-20261105")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic JSON; existing files are preserved")
    args = parser.parse_args()
    result = analyze(args.evidence, args.benchmark)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("questions", "positive_support_rows")}, indent=2))
