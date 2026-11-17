from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_metadata_hints import build_catalog, query_hints
from probe_metadata_retrieval import candidate_orders


def test_full_name_boundaries_and_punctuation_normalization():
    catalog = build_catalog([{"group": "Project A", "phase_name": "Risk: Feed Delay", "topic": None}])
    result = query_hints("What changed in RISK - FEED DELAY?", catalog)
    assert result["group_hint"] == "Project A"
    assert query_hints("Is this Project Alpha?", catalog)["status"] == "none"
    assert query_hints("What changed?", catalog)["status"] == "none"


def test_shared_or_conflicting_names_do_not_invent_a_unique_group():
    catalog = build_catalog([
        {"group": "Project A", "phase_name": "Architecture Design", "topic": None},
        {"group": "Project B", "phase_name": "Architecture Design", "topic": None},
    ])
    assert query_hints("Architecture Design status?", catalog)["status"] == "ambiguous"
    assert query_hints("Compare Project A with Project B", catalog)["group_hint"] is None


def test_catalog_ignores_labels_and_first_person_is_only_a_mention():
    message = {"group": "Project A", "phase_name": None, "topic": "Feed Review"}
    catalog = build_catalog([message])
    poisoned = build_catalog([{**message, "answer": "Project B", "gold": True, "target_group": "Project B"}])
    assert catalog == poisoned
    result = query_hints("Can I ask when Feed Review ends?", catalog)
    assert result["first_person_mention"] is True
    assert result["group_hint"] == "Project A"
    assert "target_speaker" not in result


def test_scope_can_recover_a_lexical_hit_outside_twenty_and_preserves_budget():
    hits = [{'message': {'group': 'B', 'topic': None}, 'score': 1.0} for _ in range(24)]
    for index in (13, 20):
        hits[index]['message'] = {'group': 'A', 'topic': 'Payroll'}
    hits[21] = {'message': {'group': 'B', 'topic': 'Payroll'}, 'score': 1.0}
    hits[22] = {'message': {'group': 'A', 'topic': 'Payroll'}, 'score': 0.0}
    hints = {'group_hint': 'A', 'matched_names': [{'field': 'topic', 'name': 'Payroll'}]}
    result = candidate_orders(hits, hints)
    short_pool, short_top = result['rank_boost']
    scoped_pool, scoped_top = result['scope_fusion']
    assert short_pool == list(range(20)) and short_top == list(range(9)) + [13]
    assert set(scoped_pool) == set(range(10)) | {13, 20}
    assert 20 in scoped_top and len(scoped_top) == len(set(scoped_top)) == 10

    hints['group_hint'] = None
    for _, chosen in candidate_orders(hits, hints).values():
        assert chosen == list(range(10))
