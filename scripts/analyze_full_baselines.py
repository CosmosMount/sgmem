"""Audit completed local baselines and export compact, reproducible course evidence.

This script only reads saved predictions. It never calls an LLM or tunes a retriever.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sgmem.analysis import summarize_annotations
from sgmem.data import load_messages
from sgmem.qa import official_correct, summarize


def key(row):
    return row['domain'], row['category'], row.get('question_id', row.get('id'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def paired_summary(left, right):
    ids = sorted(left)
    assert set(left) == set(right)
    differences = np.array([int(left[k]['official_correct']) - int(right[k]['official_correct']) for k in ids])
    counts = Counter(differences.tolist())
    discordant = counts[1] + counts[-1]
    p_value = min(1.0, 2 * sum(math.comb(discordant, j) for j in range(min(counts[1], counts[-1]) + 1))
                  / 2 ** discordant) if discordant else 1.0
    # Paired question bootstrap; shared conversations may violate independent-question assumptions.
    rng = np.random.default_rng(42)
    draws = rng.multinomial(len(ids), [counts[-1] / len(ids), counts[0] / len(ids), counts[1] / len(ids)], 20000)
    delta = (draws[:, 2] - draws[:, 0]) / len(ids)
    return {'questions': len(ids), 'both_correct': sum(left[k]['official_correct'] and right[k]['official_correct'] for k in ids),
            'bm25_only_correct': counts[1], 'dense_only_correct': counts[-1],
            'both_incorrect': sum(not left[k]['official_correct'] and not right[k]['official_correct'] for k in ids),
            'bm25_minus_dense': float(differences.mean()),
            'paired_bootstrap_95_ci': np.quantile(delta, [.025, .975]).tolist(),
            'mcnemar_exact_two_sided_p': p_value, 'bootstrap_draws': 20000, 'seed': 42,
            'caveat': 'Question-level descriptive uncertainty; shared dialogue and related questions are not independent.'}


def plot_tables(output, metrics, failures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    categories = ['multi_hop', 'knowledge_update', 'temporal', 'user_implicit', 'term_ambiguity', 'abstention']
    labels = ['Overall\n(n=715)', 'Multi-hop\n(n=182)', 'Knowledge\nupdate (n=107)', 'Temporal\n(n=162)',
              'User-implicit\n(n=49)', 'Term ambiguity\n(n=106)', 'Abstention\n(n=109)']
    fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
    for method, offset, color in [('bm25', -.19, '#337AA6'), ('dense', .19, '#C8852E')]:
        values = [metrics[method]['overall']['accuracy']] + [metrics[method]['by_category'][c]['accuracy'] for c in categories]
        bars = ax.bar(np.arange(7) + offset, np.array(values) * 100, .36, label=method.upper(), color=color)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    ax.set(xticks=np.arange(7), xticklabels=labels, ylim=(0, 88), ylabel='Accuracy (%)',
           title='GroupMemBench: full 715-question baseline results')
    ax.legend(frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.savefig(output / 'baseline-accuracy.png', dpi=180)
    plt.close(fig)
    names = ['Speaker confusion (F1)', 'Group confusion (F2)', 'Belief attribution (F3)', 'Reply context loss (F4)',
             'Temporal update (F5)', 'Cross-participant reasoning (F6)', 'Other retrieval error (F7)', 'QA reasoning error (F8)']
    counts = [failures['task_primary_failure_counts'].get('F' + str(i), 0) for i in range(1, 9)]
    fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
    bars = ax.barh(names, counts, color='#337AA6', height=.65)
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.invert_yaxis()
    ax.set(xlim=(0, max(counts) + 2), xlabel='Reviewed failure records (one primary label each)',
           title='Primary causes in 34 reviewed task failures\n24 dev questions; 4 evaluation disputes excluded')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    fig.savefig(output / 'failure-taxonomy.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Extracted full evidence archive')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--archive', type=Path, help='Optional raw evidence archive to identify by SHA256')
    parser.add_argument('--plots', action='store_true', help='Also render report figures; requires matplotlib')
    args = parser.parse_args()
    parent = args.bundle / 'part1-full-20261105'
    parallel = args.bundle / 'part1-full-20261105-parallel'
    combined = parallel / 'combined'
    assert json.loads((parallel / 'completion.json').read_text())['complete']
    assert (parallel / 'controller.exit').read_text().strip() == '0'
    split = json.loads((combined / 'split.json').read_text())
    partitions = {key(row): p for p in ('dev', 'test') for row in split[p]}
    assert len(partitions) == sum(len(split[p]) for p in ('dev', 'test')) == 715
    maps, metrics, all_records, profiles = {}, {}, [], {}
    compact = []
    for method in ('bm25', 'dense'):
        rows = read_rows(combined / method / 'predictions.jsonl')
        raw_rows = read_rows(parent / 'bm25/predictions.jsonl') if method == 'bm25' else (
            read_rows(parallel / 'dense-0/dense/predictions.jsonl') + read_rows(parallel / 'dense-1/dense/predictions.jsonl'))
        assert {key(r): r for r in rows} == {key(r): r for r in raw_rows}
        assert len(rows) == len({key(r) for r in rows}) == 715
        assert {key(r) for r in rows} == set(partitions)
        for row in rows:
            assert row['status'] == 'ok' and row['method'] == method
            assert row['official_correct'] == official_correct(row['judge_answer'])
            assert row['partition'] == partitions[key(row)]
            assert len(row['retrieval']) == len({r['message']['id'] for r in row['retrieval']}) == 10
        expected = json.loads((combined / method / 'metrics.json').read_text())
        recomputed = summarize(rows)
        assert all(expected[k] == value for k, value in recomputed.items())
        assert expected['complete'] and expected['expected_questions'] == 715
        maps[method], metrics[method] = {key(r): r for r in rows}, expected
        all_records.extend(rows)
        profiles[method] = {}
        for field in ('retrieval_seconds', 'answer_seconds', 'judge_seconds'):
            values = np.array([r[field] for r in rows])
            profiles[method][field] = {'sum': float(values.sum()), 'mean': float(values.mean()),
                                      'median': float(np.median(values)), 'p95': float(np.quantile(values, .95))}
        for stage in ('answer', 'judge'):
            profiles[method][stage + '_tokens'] = {
                name: sum(r[stage + '_usage'][name] for r in rows)
                for name in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            profiles[method][stage + '_max_prompt_tokens'] = max(r[stage + '_usage']['prompt_tokens'] for r in rows)
        for r in rows:
            fields = ('domain', 'category', 'question_id', 'method', 'partition', 'status', 'correct', 'official_correct',
                      'answer', 'judge_answer', 'answer_finish_reason', 'judge_finish_reason',
                      'answer_usage', 'judge_usage', 'retrieval_seconds', 'answer_seconds', 'judge_seconds')
            compact.append({**{f: r[f] for f in fields},
                            'retrieved_ids': [t['message']['id'] for t in r['retrieval']],
                            'retrieval_scores': [t['score'] for t in r['retrieval']]})
    comparisons = {}
    for name in ('all', 'dev', 'test'):
        selected = {m: {k: r for k, r in rows.items() if name == 'all' or r['partition'] == name}
                    for m, rows in maps.items()}
        comparisons[name] = paired_summary(selected['bm25'], selected['dense'])
    cells = defaultdict(list)
    for k, row in maps['bm25'].items():
        if row['partition'] == 'dev' and not (row['official_correct'] and maps['dense'][k]['official_correct']):
            cells[k[:2]].append(k)
    salt = 'sgmem-part2-20261105'
    sample_ids = [min(ids, key=lambda k: hashlib.sha256((salt + '|' + '|'.join(k)).encode()).hexdigest())
                  for _, ids in sorted(cells.items())]
    sample = {'selection': 'One development question with at least one failed method per nonempty domain/category cell, chosen by smallest salted SHA256.',
              'salt': salt, 'test_questions_inspected_for_part2': 0,
              'questions': [{'domain': k[0], 'category': k[1], 'id': k[2],
                             'bm25_correct': maps['bm25'][k]['official_correct'],
                             'dense_correct': maps['dense'][k]['official_correct']} for k in sample_ids],
              'eligible_questions_by_cell': {'/'.join(cell): len(ids) for cell, ids in sorted(cells.items())},
              'distribution_scope': 'Quota sample; percentages are not population estimates for all benchmark failures.'}
    aggregation = json.loads((combined / 'aggregation.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'split.json').write_bytes((combined / 'split.json').read_bytes())
    save(args.output / 'metrics.json', metrics)
    save(args.output / 'comparison.json', comparisons)
    save(args.output / 'performance.json', profiles)
    save(args.output / 'failure-sample.json', sample)
    (args.output / 'outcomes.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in compact), encoding='utf-8')
    run_roots = [parent, parallel / 'dense-0', parallel / 'dense-1']
    provenance = {'benchmark': 'GroupMemBench', 'benchmark_commit': aggregation['protocol']['benchmark_commit'],
                  'method_core': aggregation['implementation'], 'protocol': aggregation['protocol'],
                  'runtime': aggregation['runtime'], 'gpu_hours': aggregation['gpu_hours'],
                  'completion': json.loads((parallel / 'completion.json').read_text()),
                  'raw_parts': aggregation['parts'], 'resource_runs': aggregation['resource_runs'],
                  'question_count': 715, 'dev': 179, 'test': 536,
                  'split_sha256': hashlib.sha256((combined / 'split.json').read_bytes()).hexdigest(),
                  'index_seconds': {}, 'peak_allocated_gpu_bytes': {}, 'generation_calls': {}}
    if args.archive:
        provenance['raw_archive'] = {'name': args.archive.name, 'bytes': args.archive.stat().st_size,
                                     'sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest()}
    call_ids = set()
    for r in run_roots:
        run_config = json.loads((r / 'run-config.json').read_text())
        provenance['peak_allocated_gpu_bytes'][r.name] = run_config['peak_allocated_gpu_bytes']
        calls = read_rows(r / 'generation-audit.jsonl')
        provenance['generation_calls'][r.name] = dict(Counter(c['stage'] for c in calls))
        for call in calls:
            q = call['question']
            identity = (*key(q), q['method'], call['stage'])
            assert identity not in call_ids
            call_ids.add(identity)
            assert call['max_tokens'] == {'answer': 512, 'judge': 256}[call['stage']]
        for method in ('bm25', 'dense'):
            p = r / method / 'manifest.json'
            if p.exists():
                manifest = json.loads(p.read_text())
                provenance['index_seconds'][r.name + '/' + method] = manifest.get('index_seconds_by_domain', {})
    assert call_ids == {(*key(r), r['method'], stage) for r in all_records for stage in ('answer', 'judge')}
    save(args.output / 'provenance.json', provenance)
    if args.annotations:
        annotations = read_rows(args.annotations)
        selected_failures = {(r['domain'], r['category'], r['question_id'], r['method'])
                             for r in all_records if key(r) in sample_ids and not r['official_correct']}
        assert selected_failures == {(r['domain'], r['category'], r['question_id'], r['method']) for r in annotations}
        assert all(r.get('annotator') and r.get('notes') and r.get('supporting_message_ids') for r in annotations)
        dev = [r for r in all_records if r['partition'] == 'dev']
        analysis = summarize_annotations(dev, annotations)
        analysis.pop('examples_with_notes')  # Full traces stay in the raw archive; annotations reference source IDs.
        analysis.update(sample_questions=len(sample_ids), sample_records=2 * len(sample_ids),
                        sample_failed_records=len(selected_failures), annotation_provenance='AI-assisted source review; not human gold labels')
        save(args.output / 'failure-analysis.json', analysis)
        evidence, diagnostics = {}, []
        for domain in sorted({r['domain'] for r in annotations}):
            messages = load_messages(ROOT / f'external/GroupMemBench/data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json')
            by_id = {m['id']: m for m in messages}
            adjacent = defaultdict(set)
            for m in messages:
                if m['reply_to']:
                    assert m['reply_to'] in by_id
                    adjacent[m['id']].add(m['reply_to'])
                    adjacent[m['reply_to']].add(m['id'])
            for a in [a for a in annotations if a['domain'] == domain]:
                for mid in a['supporting_message_ids']:
                    assert mid in by_id
                    evidence[(domain, mid)] = {'domain': domain, **by_id[mid]}
                targets = set(a['target_message_ids'])
                assert targets <= set(a['supporting_message_ids'])
                if not targets:
                    continue  # Abstention/evaluation disputes have no positive answer-support target.
                record = maps[a['method']][key(a)]
                seeds = {x['message']['id'] for x in record['retrieval']}
                one_hop = seeds | {neighbor for mid in seeds for neighbor in adjacent[mid]}
                two_hop = one_hop | {neighbor for mid in one_hop for neighbor in adjacent[mid]}
                diagnostics.append({**{f: a[f] for f in ('domain', 'category', 'question_id', 'method', 'case_id')},
                                    'targets': sorted(targets), 'target_in_top10': bool(targets & seeds),
                                    'target_within_one_reply_hop': bool(targets & one_hop),
                                    'target_within_two_reply_hops': bool(targets & two_hop),
                                    'one_hop_candidates': len(one_hop), 'two_hop_candidates': len(two_hop),
                                    'public_asking_user_id': record['asking_user_id'],
                                    'target_speakers': sorted({by_id[mid]['speaker'] for mid in targets}),
                                    'retrieved_group_count': len({x['message']['group'] for x in record['retrieval']}),
                                    'target_group_already_retrieved': any(by_id[mid]['group'] in {x['message']['group'] for x in record['retrieval']} for mid in targets)})
        save(args.output / 'metadata-diagnostics.json', {
            'scope': 'Post-hoc reachability of reviewer-identified support; not official evidence-recall labels or a top-10 repair experiment.',
            'uses_reference_for_diagnosis_only': True, 'qa_rerun': False,
            'positive_support_failure_records': len(diagnostics),
            'target_in_top10': sum(d['target_in_top10'] for d in diagnostics),
            'target_within_one_reply_hop': sum(d['target_within_one_reply_hop'] for d in diagnostics),
            'target_within_two_reply_hops': sum(d['target_within_two_reply_hops'] for d in diagnostics),
            'median_one_hop_candidates': float(np.median([d['one_hop_candidates'] for d in diagnostics])),
            'median_two_hop_candidates': float(np.median([d['two_hop_candidates'] for d in diagnostics])),
            'rows': diagnostics,
        })
        (args.output / 'source-references.jsonl').write_text(
            ''.join(json.dumps(evidence[k], ensure_ascii=False) + '\n' for k in sorted(evidence)), encoding='utf-8')
        if args.plots:
            plot_tables(args.output, metrics, analysis)
    print(json.dumps({'questions_per_method':715, 'sample_questions':len(sample_ids),
                      'sample_failures':sum(not maps[m][k]['official_correct'] for k in sample_ids for m in maps),
                      'comparisons':comparisons, 'gpu_hours':aggregation['gpu_hours']}, indent=2))


if __name__ == '__main__':
    main()
