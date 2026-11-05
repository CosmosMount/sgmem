"""Checkpoint and coverage checks that do not load a language model."""

import json
from argparse import Namespace

import pytest

from sgmem.runner import make_split, run, split_keys
from test_baselines import _write_benchmark


def _args(tmp_path):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    _write_benchmark(benchmark)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(make_split(benchmark, ['Finance'], 42, .25)))
    return Namespace(benchmark=str(benchmark), project_root=str(tmp_path), output=str(tmp_path / 'run'),
                     split=str(split), partition='all', domains=['Finance'], categories=['multi_hop'],
                     method='bm25', base_url='in-process', answer_model='local', judge_model=None,
                     embedding_model='unused', embedding_revision=None, device='cpu', batch_size=2,
                     top_k=2, timeout=1, limit=None, resume=False)


class Client:
    last_usage = None

    def __init__(self, *args, **kwargs):
        pass

    def complete(self, system, prompt, max_tokens):
        return 'Final: Correct' if system == 'judge' else 'Final: 10'


def test_resume_retries_failed_call_without_repeating_success(tmp_path):
    args = _args(tmp_path)
    seen = []

    class InterruptClient(Client):
        def complete(self, system, prompt, max_tokens):
            if system == 'agent':
                seen.append(prompt)
                if len(seen) == 2:
                    raise RuntimeError('temporary model failure')
            return super().complete(system, prompt, max_tokens)

    with pytest.raises(RuntimeError, match='Model call failed'):
        run(args, client_factory=InterruptClient)
    first_success = seen[0]
    args.resume = True
    resumed = []

    class ResumedClient(Client):
        def complete(self, system, prompt, max_tokens):
            if system == 'agent':
                resumed.append(prompt)
            return super().complete(system, prompt, max_tokens)

    out = run(args, client_factory=ResumedClient)
    assert first_success not in resumed and len(resumed) == 3
    records = [json.loads(s) for s in (out / 'predictions.jsonl').read_text().splitlines()]
    assert len(records) == 4 and len({r['question_id'] for r in records}) == 4
    assert [r['attempt'] for r in records] == [1, 2, 1, 1]
    assert len((out / 'attempts.jsonl').read_text().splitlines()) == 5
    metrics = json.loads((out / 'metrics.json').read_text())
    assert metrics['complete'] and metrics['overall']['accuracy'] == 1
    assert metrics['by_partition']['dev']['overall']['total'] == 1
    assert metrics['by_partition']['test']['overall']['total'] == 3
    assert json.loads((out / 'manifest.json').read_text())['status'] == 'complete'
    assert len(split_keys(tmp_path / 'split.json', 'all')) == 24


@pytest.mark.parametrize('change', ['config', 'question', 'prompt'])
def test_resume_rejects_experiment_drift_before_model_use(tmp_path, change):
    args = _args(tmp_path)
    run(args, client_factory=Client)
    args.resume = True
    if change == 'config':
        args.top_k = 1
    else:
        path = tmp_path / 'benchmark' / ('prompts/hipporag_agent_system.txt' if change == 'prompt'
                                        else 'questions/Finance/multi_hop.jsonl')
        path.write_text(path.read_text().replace('agent', 'changed agent') if change == 'prompt'
                        else path.read_text().replace('Budget question', 'Changed question'))

    def forbidden(*args, **kwargs):
        pytest.fail('must not initialize model after source/config/input changes')

    with pytest.raises(ValueError, match='Resume rejected'):
        run(args, client_factory=forbidden)


def test_resume_preserves_partial_tail_and_does_not_duplicate(tmp_path):
    args = _args(tmp_path)
    out = run(args, client_factory=Client)
    with (out / 'attempts.jsonl').open('ab') as handle:
        handle.write(b'{"partial')
    args.resume = True
    run(args, client_factory=Client)
    assert list(out.glob('interrupted-tail-*.bin'))[0].read_bytes() == b'{"partial'
    assert len((out / 'attempts.jsonl').read_text().splitlines()) == 4
    assert len((out / 'predictions.jsonl').read_text().splitlines()) == 4


def test_runner_requires_split_and_partition_together(tmp_path):
    args = _args(tmp_path)
    args.partition = None
    with pytest.raises(ValueError, match='provided together'):
        run(args, client_factory=Client)
