"""Parallel execution must not change coverage, partitions, or experiment settings."""

import copy
import json

import pytest

from sgmem import shards
from sgmem.runner import write_json


@pytest.fixture
def split():
    rows = [{'domain': 'Finance', 'category': 'multi_hop', 'id': f'q{i}'} for i in range(7)]
    return {'seed': 42, 'dev_fraction': .25, 'unit': 'question', 'dev': rows[:2], 'test': rows[2:]}


def test_shards_are_disjoint_exhaustive_and_keep_frozen_partitions(split):
    original = copy.deepcopy(split)
    maps = [shards.partition_map(shards.shard_split(split, i, 3)) for i in range(3)]
    assert sorted(map(len, maps)) == [2, 2, 3]
    assert not any(set(a) & set(b) for i, a in enumerate(maps) for b in maps[i + 1:])
    assert {k: v for m in maps for k, v in m.items()} == shards.partition_map(split)
    assert split == original
    assert shards.shard_split(split, 0, 1) == split


def test_sharding_rejects_overlapping_reference_split(split):
    split['test'].append(split['dev'][0])
    with pytest.raises(ValueError, match='overlapping'):
        shards.shard_split(split, 0, 2)


@pytest.fixture
def parts(tmp_path, monkeypatch, split):
    monkeypatch.setattr(shards, 'implementation_signature', lambda *args: {'core': 'same'})
    split_path = tmp_path / 'full-split.json'
    write_json(split_path, split)
    result = []
    for name, method, selected in [('bm25', 'bm25', split),
                                    ('dense0', 'dense', shards.shard_split(split, 0, 2)),
                                    ('dense1', 'dense', shards.shard_split(split, 1, 2))]:
        parent = tmp_path / name
        part = parent / method
        part.mkdir(parents=True)
        write_json(parent / 'split.json', selected)
        runtime = {k: {} for k in ('packages', 'model_files')}
        runtime.update(enable_thinking=False, temperature=.2, top_p=1, top_k=0,
                       answer_max_tokens=512, judge_max_tokens=256, seed_per_call=42,
                       gpu_hours=.1, wall_seconds=360, CUDA_VISIBLE_DEVICES='2')
        write_json(parent / 'run-config.json', runtime)
        rows = [{'domain': key[0], 'category': key[1], 'question_id': key[2],
                 'partition': partition, 'method': method, 'status': 'ok', 'judge_answer': 'Correct'}
                for key, partition in shards.partition_map(selected).items()]
        write_json(part / 'manifest.json', {
            'status': 'complete', 'expected_questions': len(rows),
            'config': {'method': method, 'partition': 'all', 'limit': None, 'top_k': 10},
            'benchmark_git': {'commit': 'data-commit', 'status': ''},
            'project_git': {'commit': 'code-commit', 'status': ''},
            'input_sha256': {'prompt': 'same', 'split': 'varies'},
        })
        (part / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        write_json(part / 'metrics.json', {})
        result.append(part)
    return result, split_path


def test_merge_has_exact_coverage_and_resources(parts, tmp_path):
    paths, split_path = parts
    output = tmp_path / 'merged'
    summary = shards.merge_runs(paths, split_path, output, tmp_path)
    assert all(m['complete'] and m['overall']['total'] == 7 for m in summary.values())
    assert all(m['by_partition']['dev']['overall']['total'] == 2 for m in summary.values())
    assert json.loads((output / 'aggregation.json').read_text())['gpu_hours'] == pytest.approx(.3)


@pytest.mark.parametrize('fault,match', [('duplicate', 'Duplicate result'), ('missing', 'entire frozen'),
                                       ('partition', 'incorrect partition'), ('protocol', 'differ in protocol'),
                                       ('failed', 'Failed record')])
def test_merge_refuses_invalid_shards(parts, tmp_path, fault, match):
    paths, split_path = parts
    if fault == 'duplicate':
        paths.append(paths[-1])
    elif fault == 'missing':
        paths.pop()
    elif fault == 'protocol':
        target = paths[-1] / 'manifest.json'
        manifest = json.loads(target.read_text())
        manifest['config']['top_k'] = 9
        write_json(target, manifest)
    else:
        target = paths[-1] / 'predictions.jsonl'
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        rows[0]['partition' if fault == 'partition' else 'status'] = 'invalid'
        target.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=match):
        shards.merge_runs(paths, split_path, tmp_path / 'merged', tmp_path)
    assert not (tmp_path / 'merged').exists()
