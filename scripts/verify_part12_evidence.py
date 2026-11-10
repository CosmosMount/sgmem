"""Recompute published Part 1/2 tables from compact records, without a model or GPU."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sgmem.analysis import summarize_annotations
from sgmem.data import load_messages
from sgmem.qa import official_correct, summarize
from analyze_full_baselines import key, paired_summary, read_rows


def verify(folder, benchmark=None):
    records = read_rows(folder / 'outcomes.jsonl')
    expected_metrics = json.loads((folder / 'metrics.json').read_text())
    methods = {}
    for method in ('bm25', 'dense'):
        rows = [r for r in records if r['method'] == method]
        assert len(rows) == len({key(r) for r in rows}) == 715
        assert all(r['status'] == 'ok' and r['official_correct'] == official_correct(r['judge_answer']) for r in rows)
        assert all(len(r['retrieved_ids']) == len(set(r['retrieved_ids'])) == len(r['retrieval_scores']) == 10 for r in rows)
        assert sum(r['partition'] == 'dev' for r in rows) == 179
        assert sum(r['partition'] == 'test' for r in rows) == 536
        assert all(expected_metrics[method][k] == value for k, value in summarize(rows).items())
        methods[method] = {key(r): r for r in rows}
    assert len(records) == 1430 and set(methods['bm25']) == set(methods['dense'])
    assert all(methods['bm25'][k]['partition'] == methods['dense'][k]['partition'] for k in methods['bm25'])
    comparisons = json.loads((folder / 'comparison.json').read_text())
    for partition in ('all', 'dev', 'test'):
        subsets = [{k: r for k, r in methods[m].items() if partition == 'all' or r['partition'] == partition}
                   for m in ('bm25', 'dense')]
        assert paired_summary(*subsets) == comparisons[partition]
    sample = json.loads((folder / 'failure-sample.json').read_text())
    selected = {key(r) for r in sample['questions']}
    cells = {}
    for k, r in methods['bm25'].items():
        if r['partition'] == 'dev' and not (r['official_correct'] and methods['dense'][k]['official_correct']):
            cells.setdefault(k[:2], []).append(k)
    reproduced = {min(keys, key=lambda k: hashlib.sha256((sample['salt'] + '|' + '|'.join(k)).encode()).hexdigest())
                  for keys in cells.values()}
    assert reproduced == selected and len(selected) == 24
    annotations = read_rows(folder / 'failure-annotations.jsonl')
    annotation_keys = {(key(r), r['method']) for r in annotations}
    failed_sample = {(key(r), r['method']) for r in records if key(r) in selected and not r['official_correct']}
    assert len(annotations) == len(annotation_keys) == 38 and annotation_keys == failed_sample
    sources = read_rows(folder / 'source-references.jsonl')
    source_map = {(s['domain'], s['id']): s for s in sources}
    for a in annotations:
        assert a['partition'] == 'dev' and 'AI-assisted' in a['annotator']
        assert a['notes'] and a['supporting_message_ids']
        assert all((a['domain'], mid) in source_map for mid in a['supporting_message_ids'])
        assert set(a['target_message_ids']) <= set(a['supporting_message_ids'])
    dev_records = []
    for r in records:
        if r['partition'] == 'dev':
            dev_records.append({**r, 'prediction': r['answer'],
                                'retrieval': [{'message': {'id': mid}} for mid in r['retrieved_ids']]})
    failure_summary = summarize_annotations(dev_records, annotations)
    failure_summary.pop('examples_with_notes')
    saved_failure = json.loads((folder / 'failure-analysis.json').read_text())
    assert all(saved_failure[k] == value for k, value in failure_summary.items())
    diagnostics = json.loads((folder / 'metadata-diagnostics.json').read_text(encoding='utf-8'))
    diagnostic_rows = diagnostics['rows']
    positive_annotations = {(key(a), a['method']): a for a in annotations if a['target_message_ids']}
    diagnostic_keys = {(key(d), d['method']) for d in diagnostic_rows}
    assert len(diagnostic_rows) == len(diagnostic_keys) and diagnostic_keys == set(positive_annotations)
    for d in diagnostic_rows:
        annotation = positive_annotations[key(d), d['method']]
        assert d['case_id'] == annotation['case_id']
        assert sorted(d['targets']) == sorted(annotation['target_message_ids'])
        retrieved_ids = methods[d['method']][key(d)]['retrieved_ids']
        assert d['target_in_top10'] is bool(set(d['targets']).intersection(retrieved_ids))
    assert diagnostics['positive_support_failure_records'] == len(diagnostic_rows)
    for field in ('target_in_top10', 'target_within_one_reply_hop', 'target_within_two_reply_hops'):
        assert diagnostics[field] == sum(d[field] for d in diagnostic_rows)
    for field in ('one_hop_candidates', 'two_hop_candidates'):
        assert diagnostics['median_' + field] == median(d[field] for d in diagnostic_rows)
    provenance = json.loads((folder / 'provenance.json').read_text())
    split_path = folder / 'split.json'
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == provenance['split_sha256']
    split = json.loads(split_path.read_text())
    frozen = {key(row): partition for partition in ('dev', 'test') for row in split[partition]}
    assert all(frozen[key(r)] == r['partition'] for r in records)
    assert abs(sum(x['gpu_hours'] for x in provenance['resource_runs'].values()) - provenance['gpu_hours']) < 1e-10
    if benchmark:
        for domain in sorted({a['domain'] for a in annotations}):
            messages = load_messages(benchmark / f'data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json')
            by_id = {m['id']: m for m in messages}
            for (d, mid), source in source_map.items():
                if d == domain:
                    assert {k: v for k, v in source.items() if k != 'domain'} == by_id[mid]
            adjacency = {mid: set() for mid in by_id}
            for m in messages:
                if m['reply_to']:
                    adjacency[m['id']].add(m['reply_to'])
                    adjacency[m['reply_to']].add(m['id'])
            for d in [d for d in diagnostic_rows if d['domain'] == domain]:
                targets = set(d['targets'])
                seeds = set(methods[d['method']][key(d)]['retrieved_ids'])
                one = seeds | {n for mid in seeds for n in adjacency[mid]}
                two = one | {n for mid in one for n in adjacency[mid]}
                assert d['target_within_one_reply_hop'] == bool(targets & one)
                assert d['target_within_two_reply_hops'] == bool(targets & two)
                assert d['one_hop_candidates'] == len(one) and d['two_hop_candidates'] == len(two)
    query_diagnostics = folder / 'query-metadata-diagnostics.json'
    query_hints_checked = benchmark is not None and query_diagnostics.exists()
    if query_hints_checked:
        from analyze_metadata_hints import analyze
        assert json.loads(query_diagnostics.read_text(encoding='utf-8')) == analyze(folder, benchmark)
    return {'verified': True, 'records': len(records), 'questions_per_method': 715,
            'dev_per_method': 179, 'test_per_method': 536, 'reviewed_dev_questions': 24,
            'reviewed_failures': len(annotations), 'source_messages': len(sources),
            'source_content_checked_against_benchmark': benchmark is not None,
            'query_metadata_checked_against_benchmark': query_hints_checked,
            'gpu_hours': provenance['gpu_hours']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'docs/evidence/part12-full-20261105')
    parser.add_argument('--benchmark', type=Path, help='Optional pinned GroupMemBench checkout for source-content verification')
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence, args.benchmark), indent=2))
