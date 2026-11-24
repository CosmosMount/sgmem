"""Select a ranked prefix under the evidence budget of the actual QA prompt."""

from math import isfinite
from numbers import Real

from sgmem.qa import build_answer_prompt, format_message


def _native_document(text, originals):
    tags = ["Native memory"]
    for field, label in (("speaker", "source_user"), ("group", "source_channel")):
        value = originals[0].get(field)
        if isinstance(value, str) and value.strip() and all(source.get(field) == value for source in originals):
            tags.append(f"{label}={value}")
    return f"[{' / '.join(tags)}]\n{text}"


def select_units(question, units, tokenizer, agent_system, budget=3500, *, sources, raw=False):
    """Return formatted docs and trace for a prefix of ranked units with original-source lineage.

    The source pack is the mapping from public_sources, keyed by (domain, message ID).
    Every unit supplies unit_id, text, score and an ordered source_ids list.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("Evidence budget must be a nonnegative integer")
    domains = {domain for domain, _ in sources}
    if len(domains) != 1:
        raise ValueError("Native context requires one domain in its public source pack")
    domain = next(iter(domains))
    if question.get("domain", domain) != domain:
        raise ValueError("Question domain differs from the public source pack")

    seen, documents, lineage = set(), [], []
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Expected a ranked memory unit object")
        identifier, text = unit.get("unit_id"), unit.get("text")
        score, source_ids = unit.get("score"), unit.get("source_ids")
        if not ((isinstance(identifier, str) and identifier.strip()) or (type(identifier) is int and identifier >= 0)):
            raise ValueError("Invalid memory unit ID")
        if identifier in seen:
            raise ValueError(f"Duplicate memory unit ID {identifier!r}")
        if not isinstance(text, str) or not text.strip() or isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score):
            raise ValueError(f"Invalid text or score for memory unit {identifier!r}")
        if (not isinstance(source_ids, list) or not source_ids
                or any(not isinstance(value, str) or not value.strip() for value in source_ids)
                or len(set(source_ids)) != len(source_ids) or (raw and len(source_ids) != 1)):
            raise ValueError(f"Invalid source lineage for memory unit {identifier!r}")
        originals = []
        for source_id in source_ids:
            source = sources.get((domain, source_id))
            if not isinstance(source, dict) or (source.get("domain"), source.get("id")) != (domain, source_id):
                raise ValueError(f"Unknown or wrong-domain source {source_id!r} for memory unit {identifier!r}")
            originals.append(source)
        seen.add(identifier)
        lineage.append({"unit_id": identifier, "source_ids": list(source_ids), "score": float(score)})
        documents.append(format_message(originals[0]) if raw else _native_document(text, originals))

    def prompt_tokens(docs):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": agent_system}, {"role": "user", "content": build_answer_prompt(question, docs)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])

    empty = total = prompt_tokens([])
    kept, rejected = 0, None
    for count in range(1, len(documents) + 1):
        candidate = prompt_tokens(documents[:count])
        if candidate - empty > budget:
            rejected = candidate
            break
        kept, total = count, candidate
    return documents[:kept], {"kept": lineage[:kept], "dropped": lineage[kept:], "budget_tokens": budget,
                              "empty_prompt_tokens": empty, "prompt_tokens": total, "evidence_tokens": total - empty,
                              "first_rejected_prompt_tokens": rejected}
