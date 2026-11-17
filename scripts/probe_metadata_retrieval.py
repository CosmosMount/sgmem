"""Replay two metadata rules on the frozen BM25 development set, without QA."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_metadata_hints import ROOT, build_catalog, key, query_hints, read_rows
from sgmem.data import load_messages, load_questions
from sgmem.retrieval import BM25Retriever
from sgmem.runner import question_file
from sgmem.shards import partition_map

K = 10


def candidate_orders(hits, hints):
    """Indices into the content ranking. Each rule returns both its pool and its final K."""
    global_top = list(range(min(K, len(hits))))
    shortlist = list(range(min(2 * K, len(hits))))
    group = hints['group_hint']
    if group is None:
        return {'rank_boost': (shortlist, global_top), 'scope_fusion': (global_top, global_top)}

    names = {(entry['field'], entry['name']) for entry in hints['matched_names']
             if entry['field'] in ('phase_name', 'topic')}

    def matches_detail(message):
        return any((field, message.get(field)) in names for field in ('phase_name', 'topic'))

    def boosted_rank(index):
        message = hits[index]['message']
        if message['group'] != group:
            return index
        return index - (5 if matches_detail(message) else 2)

    boosted = sorted(shortlist, key=boosted_rank)[:K]
    # The second list contains lexical hits in the named phase/topic, or just the named group.
    scoped = []
    for i, hit in enumerate(hits):
        message = hit['message']
        if hit['score'] > 0 and message['group'] == group and (not names or matches_detail(message)):
            scoped.append(i)
            if len(scoped) == K:
                break
    scores = {i: 1 / (K + rank) for rank, i in enumerate(global_top, 1)}
    for rank, i in enumerate(scoped, 1):
        scores[i] = scores.get(i, 0) + 1 / (K + rank)
    fused = sorted(scores, key=lambda i: (-scores[i], i))[:K]
    return {'rank_boost': (shortlist, boosted), 'scope_fusion': (list(scores), fused)}


def probe(evidence, benchmark):
    split = json.loads((evidence / 'split.json').read_text(encoding='utf-8'))
    dev = {k for k, part in partition_map(split).items() if part == 'dev'}
    baseline = {key(row): row for row in read_rows(evidence / 'outcomes.jsonl')
                if row['partition'] == 'dev' and row['method'] == 'bm25'}
    if set(baseline) != dev:
        raise ValueError('Saved BM25 records do not cover the frozen development questions')
    results = {}
    unique = 0
    for domain in sorted({k[0] for k in dev}):
        messages = load_messages(benchmark / f'data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json')
        catalog = build_catalog(messages)
        retriever = BM25Retriever(messages)
        for category in sorted({k[1] for k in dev if k[0] == domain}):
            for q in load_questions(question_file(benchmark, domain, category), domain, category):
                qid = key(q)
                if qid not in dev:
                    continue
                text, asker = q['question'], q['asking_user_id']
                hints = query_hints(text, catalog)
                unique += int(hints['group_hint'] is not None)
                pool_size = len(messages) if hints['group_hint'] is not None else 2 * K
                hits = retriever.retrieve(f'{asker} {text}' if asker else text, pool_size)
                ids = [hit['message']['id'] for hit in hits]
                if (ids[:K] != baseline[qid]['retrieved_ids'] or
                        [hit['score'] for hit in hits[:K]] != baseline[qid]['retrieval_scores']):
                    raise ValueError(f'Baseline top-{K} IDs or scores differ at {qid}')
                results[qid] = {name: ([ids[i] for i in pool], [ids[i] for i in chosen])
                                for name, (pool, chosen) in candidate_orders(hits, hints).items()}
    if set(results) != dev:
        raise ValueError('Benchmark questions do not cover the frozen development split')

    # Attach reviewed targets only after all retrievals have finished.
    targets = {key(row): set(row['target_message_ids']) for row in read_rows(evidence / 'failure-annotations.jsonl')
               if row['partition'] == 'dev' and row['method'] == 'bm25' and row['target_message_ids']}
    stats = {name: Counter(order_changed=0, set_changed=0, support_in_pool=0, support_in_top10=0,
                           support_gained=0, support_lost=0) for name in ('rank_boost', 'scope_fusion')}
    support_rows = []
    for qid, methods in sorted(results.items()):
        original = baseline[qid]['retrieved_ids']
        target = targets.get(qid, set())
        old_rank = next((rank for rank, mid in enumerate(original, 1) if mid in target), None)
        case = {'domain': qid[0], 'category': qid[1], 'id': qid[2],
                'targets': sorted(target), 'baseline_support_rank': old_rank}
        for name, (pool, chosen) in methods.items():
            if len(chosen) != K or len(set(chosen)) != K or len(pool) > 2 * K:
                raise ValueError(f'Invalid retrieval budget at {qid} for {name}')
            stats[name].update(order_changed=int(chosen != original), set_changed=int(set(chosen) != set(original)))
            if target:
                new_rank = next((rank for rank, mid in enumerate(chosen, 1) if mid in target), None)
                in_pool = bool(target.intersection(pool))
                stats[name].update(support_in_pool=int(in_pool), support_in_top10=int(new_rank is not None),
                                   support_gained=int(old_rank is None and new_rank is not None),
                                   support_lost=int(old_rank is not None and new_rank is None))
                case[name] = {'in_pool': in_pool, 'support_rank': new_rank}
        if target:
            support_rows.append(case)
    return {'dev_questions': len(dev), 'unique_group_hints': unique, 'reviewed_bm25_failures_with_targets': len(targets),
            'baseline_support_hits': sum(row['baseline_support_rank'] is not None for row in support_rows),
            'methods': stats, 'support_cases': support_rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'external/GroupMemBench')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'docs/evidence/part12-full-20261105')
    args = parser.parse_args()
    print(json.dumps(probe(args.evidence, args.benchmark), indent=2))
