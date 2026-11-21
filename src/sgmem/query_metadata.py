"""Match public conversation names without using answers or reviewer labels."""

import re


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_catalog(messages):
    names = {}
    for message in messages:
        for field in ("group", "phase_name", "topic"):
            name = message.get(field)
            if name and normalize(name):
                names.setdefault((field, name), set()).add(message["group"])
    return [{"field": field, "name": name, "groups": sorted(groups)}
            for (field, name), groups in sorted(names.items())]


def query_hints(question, catalog, *, prefer_longest=False):
    query = " " + normalize(question) + " "
    matches = [entry for entry in catalog if " " + normalize(entry["name"]) + " " in query]
    if prefer_longest:
        spans = [(match.start(), match.end(), entry)
                 for entry in matches
                 for match in re.finditer(r"\b" + re.escape(normalize(entry["name"])) + r"\b", query)]
        keep = {(entry["field"], entry["name"]) for start, end, entry in spans
                if not any(a <= start and end <= b and b - a > end - start for a, b, _ in spans)}
        matches = [entry for entry in matches if (entry["field"], entry["name"]) in keep]
    groups = sorted({group for entry in matches for group in entry["groups"]})
    return {
        "matched_names": matches,
        "group_hint": groups[0] if len(groups) == 1 else None,
        "status": "none" if not matches else ("unique" if len(groups) == 1 else "ambiguous"),
        "first_person_mention": bool(re.search(r"\b(i|my|mine)\b", question, re.I)),
    }
