"""Disjoint question selection and strict aggregation of local baseline runs."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

from sgmem.qa import summarize
from sgmem.runner import write_json


def row_key(row):
    return row['domain'], row['category'], row.get('question_id', row.get('id'))


def partition_map(split):
    result = {}
    for partition in ('dev', 'test'):
        for row in split[partition]:
            key = row_key(row)
            if key in result:
                raise ValueError('Duplicate question or overlapping dev/test split')
            result[key] = partition
    return result


def shard_split(split, index, count):
    """Filter a frozen split; never reshuffle or reassign dev/test membership."""
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid shard index/count')
    keys = sorted(partition_map(split))
    selected = set(keys[index::count])
    return {**split, **{p: [r for r in split[p] if row_key(r) in selected] for p in ('dev', 'test')}}


def implementation_signature(root, commit):
    """Allow orchestration-only commits while checking the executed core code."""
    def content(path):
        return subprocess.check_output(['git', '-C', str(root), 'show', f'{commit}:{path}'])

    signature = {path: hashlib.sha256(content(path)).hexdigest() for path in (
        'src/sgmem/data.py', 'src/sgmem/retrieval.py', 'src/sgmem/qa.py', 'src/sgmem/runner.py')}
    tree = ast.parse(content('scripts/run_local_baselines.py'))
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef) and node.name == 'DirectClient']
    # The model/tokenizer construction also fixes dtype and attention implementation.
    nodes += [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id in ('model', 'tokenizer', 'dtype', 'eos_ids')
                      for target in node.targets)]
    if not any(isinstance(node, ast.ClassDef) for node in nodes):
        raise ValueError('Cannot verify local generation implementation')
    signature['local_generation_ast'] = hashlib.sha256(
        '\n'.join(ast.dump(node, include_attributes=False) for node in nodes).encode()).hexdigest()
    return signature


def merge_runs(parts, split_path, output, project_root):
    """Merge completed method directories, refusing missing IDs or mixed protocols."""
    output, split_path = Path(output), Path(split_path)
    if output.exists():
        raise ValueError('Aggregate output already exists')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    expected = partition_map(split)
    if not expected:
        raise ValueError('Empty reference split')
    records = {'bm25': {}, 'dense': {}}
    provenance, resource_runs = [], {}
    reference_protocol = reference_runtime = reference_implementation = None
    for raw_part in parts:
        part = Path(raw_part).resolve()
        manifest = json.loads((part / 'manifest.json').read_text(encoding='utf-8'))
        config = manifest['config']
        method = config['method']
        if method not in records or manifest['status'] != 'complete':
            raise ValueError(f'Incomplete or unknown method: {part}')
        if config['partition'] != 'all' or config['limit'] is not None:
            raise ValueError('Only full benchmark shards can be aggregated')
        protocol = {k: v for k, v in config.items()
                    if k not in {'method', 'output', 'project_root', 'benchmark', 'split', 'resume'}}
        protocol['benchmark_commit'] = manifest['benchmark_git']['commit']
        protocol['inputs'] = {k: v for k, v in manifest['input_sha256'].items() if k != 'split'}
        run = json.loads((part.parent / 'run-config.json').read_text(encoding='utf-8'))
        runtime = {k: run[k] for k in ('packages', 'model_files', 'enable_thinking', 'temperature',
                                      'top_p', 'top_k', 'answer_max_tokens', 'judge_max_tokens', 'seed_per_call')}
        if manifest['project_git']['status'] or manifest['benchmark_git']['status']:
            raise ValueError('Cannot merge uncommitted source')
        signature = implementation_signature(project_root, manifest['project_git']['commit'])
        if reference_protocol is None:
            reference_protocol, reference_runtime, reference_implementation = protocol, runtime, signature
        elif protocol != reference_protocol or runtime != reference_runtime or signature != reference_implementation:
            raise ValueError('Shards differ in protocol, runtime, model, input, or core implementation')
        subset = partition_map(json.loads((part.parent / 'split.json').read_text(encoding='utf-8')))
        if any(key not in expected or expected[key] != value for key, value in subset.items()):
            raise ValueError('Shard changes the frozen dev/test membership')
        part_records = [json.loads(line) for line in (part / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(part_records) != manifest['expected_questions'] or len(part_records) != len(subset):
            raise ValueError('Shard record count disagrees with manifest or split')
        seen = set()
        for record in part_records:
            key = row_key(record)
            if record['status'] != 'ok' or record['method'] != method:
                raise ValueError('Failed record or wrong method')
            if key not in subset or record.get('partition') != expected.get(key):
                raise ValueError('Question missing from split or incorrect partition')
            if key in seen or key in records[method]:
                raise ValueError('Duplicate result across shards')
            seen.add(key)
            records[method][key] = record
        if seen != set(subset):
            raise ValueError('Incomplete shard coverage')
        provenance.append({'path': str(part), 'commit': manifest['project_git']['commit'],
                           'method': method, 'questions': len(part_records),
                           'sha256': {name: hashlib.sha256((part / name).read_bytes()).hexdigest()
                                      for name in ('manifest.json', 'predictions.jsonl', 'metrics.json')}})
        resource_runs[str(part.parent)] = {'gpu_hours': run.get('gpu_hours'),
                                         'wall_seconds': run.get('wall_seconds'),
                                         'gpu': run.get('CUDA_VISIBLE_DEVICES')}
    if any(set(rows) != set(expected) for rows in records.values()):
        raise ValueError('BM25/Dense do not both cover the entire frozen benchmark')
    output.mkdir(parents=True)
    summary = {}
    for method, rows in records.items():
        method_output = output / method
        method_output.mkdir()
        ordered = [rows[key] for key in sorted(expected)]
        (method_output / 'predictions.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in ordered), encoding='utf-8')
        metrics = summarize(ordered)
        metrics.update(complete=True, expected_questions=len(expected))
        write_json(method_output / 'metrics.json', metrics)
        summary[method] = metrics
    write_json(output / 'split.json', split)
    write_json(output / 'benchmark-summary.json', summary)
    costs = [r['gpu_hours'] for r in resource_runs.values()]
    write_json(output / 'aggregation.json', {
        'complete': True, 'parts': provenance, 'implementation': reference_implementation,
        'protocol': reference_protocol, 'runtime': reference_runtime, 'resource_runs': resource_runs,
        'gpu_hours': sum(costs) if all(cost is not None for cost in costs) else None,
        'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
    })
    return summary
