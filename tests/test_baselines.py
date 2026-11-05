import json
import sys
import types
from argparse import Namespace

import numpy as np
import pytest

from sgmem.analysis import build_annotation_template, summarize_annotations
from sgmem.data import load_messages, load_questions
from sgmem.qa import evaluate_question, summarize
from sgmem.retrieval import BM25Retriever, DenseRetriever
from sgmem.runner import CATEGORIES, make_split, run, split_keys


def _write_benchmark(root):
    (root / 'data/final/Finance').mkdir(parents=True)
    (root / 'questions/Finance').mkdir(parents=True)
    (root / 'prompts').mkdir()
    (root / 'prompts/hipporag_agent_system.txt').write_text('agent', encoding='utf-8')
    (root / 'prompts/hipporag_judge_system.txt').write_text('judge', encoding='utf-8')
    messages = {
        'Project': [
            {'msg_node': 'Msg_2', 'content': 'Budget &amp; schedule', 'author': 'User_2', 'timestamp': '2025-02-02', 'reply_to': 'Msg_1'},
            {'msg_node': 'Msg_1', 'content': 'Budget 10', 'author': 'User_1', 'timestamp': '2025-02-01', 'role': 'Manager'},
            {'msg_node': 'Msg_3', 'content': 'Meeting notes', 'author': 'User_3', 'timestamp': '2025-02-03'},
        ],
    }
    (root / 'data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json').write_text(json.dumps(messages), encoding='utf-8')
    for category in CATEGORIES:
        questions = [
            {'id': f'{category}_{n}', 'question': f'Budget question {n}?', 'answer': 'SECRET_GOLD', 'asking_user_id': 'User_1'}
            for n in range(1, 5)
        ]
        (root / 'questions/Finance' / f'{category}.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in questions), encoding='utf-8')


def test_upstream_message_sort_and_bm25(tmp_path):
    _write_benchmark(tmp_path)
    messages = load_messages(tmp_path / 'data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json')
    assert [m['id'] for m in messages] == ['Msg_1', 'Msg_2', 'Msg_3']
    assert messages[0]['role'] == 'Manager'
    assert messages[1]['reply_to'] == 'Msg_1'
    retriever = BM25Retriever(messages)
    assert retriever.retrieve('!!!', 10) == []
    result = retriever.retrieve('budget schedule', 2)
    assert [item['message']['id'] for item in result] == ['Msg_2', 'Msg_1']


def test_e5_prefix_normalization_and_ranking(monkeypatch):
    class FakeModel:
        def __init__(self, model, device=None, revision=None):
            assert revision == 'fixed-revision'

        def encode(self, texts, **kwargs):
            assert kwargs['normalize_embeddings'] is True
            assert all(text.startswith(('query: ', 'passage: ')) for text in texts)
            return np.array([[1.0, 0.0] if 'alpha' in text else [0.0, 1.0] for text in texts])

    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=FakeModel))
    messages = [{'id': 'a', 'content': 'alpha'}, {'id': 'b', 'content': 'beta'}]
    dense = DenseRetriever(messages, 'e5', 'cpu', 2, revision='fixed-revision')
    assert [r['message']['id'] for r in dense.retrieve('beta', 2)] == ['b', 'a']


def test_qa_separates_gold_from_answer():
    class Client:
        last_usage = None
        def __init__(self, result):
            self.result = result
            self.prompts = []
        def complete(self, system, prompt, max_tokens):
            self.prompts.append(prompt)
            return self.result

    answer = Client('Reasoning\nFinal: 10')
    judge = Client('Final: Correct')
    question = {'question': 'What budget?', 'answer': 'SECRET_GOLD', 'asking_user_id': 'User_1'}
    result = evaluate_question(question, [{'message': {'id': 'M1', 'content': '10', 'speaker': 'User_1'}, 'score': 0.9}], answer, judge, 'agent', 'judge')
    assert 'SECRET_GOLD' not in answer.prompts[0]
    assert 'SECRET_GOLD' in judge.prompts[0]
    assert result['correct'] is True and result['official_correct'] is True
    metrics = summarize([{'category': 'multi_hop', 'status': 'ok', 'judge_answer': 'Unknown'}])
    assert metrics['overall']['accuracy'] == 0 and metrics['overall']['unclear'] == 1
    assert summarize([{'category': 'multi_hop', 'status': 'error'}])['overall']['accuracy'] is None


def test_split_disjoint_and_stratified(tmp_path):
    _write_benchmark(tmp_path)
    split = make_split(tmp_path, ['Finance'], 42, 0.25)
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(split), encoding='utf-8')
    dev, test = split_keys(path, 'dev'), split_keys(path, 'test')
    assert len(dev) == 6 and len(test) == 18 and not (dev & test)
    assert split == make_split(tmp_path, ['Finance'], 42, 0.25)
    split['test'].append(split['dev'][0])
    path.write_text(json.dumps(split), encoding='utf-8')
    with pytest.raises(ValueError, match='overlap'):
        split_keys(path, 'dev')


@pytest.mark.parametrize('use_factory', [False, True])
def test_full_runner_and_human_analysis(tmp_path, monkeypatch, use_factory):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    _write_benchmark(benchmark)
    prompts = []
    class Client:
        last_usage = None
        def __init__(self, *args, **kwargs):
            pass
        def complete(self, system, prompt, max_tokens):
            prompts.append((system, prompt))
            return 'Final: Incorrect' if system == 'judge' else 'Final: 5'
    if not use_factory:
        monkeypatch.setattr('sgmem.runner.LocalClient', Client)
    args = Namespace(benchmark=str(benchmark), project_root=str(tmp_path), output=str(tmp_path / 'run'), split=None, partition=None,
                     domains=['Finance'], categories=['multi_hop'], method='bm25', base_url='http://localhost/v1',
                     answer_model='local', judge_model=None, embedding_model='intfloat/e5-base-v2',
                     embedding_revision=None, device='cpu', batch_size=2, top_k=2, timeout=1, limit=1)
    out = run(args, client_factory=Client) if use_factory else run(args)
    records = [json.loads(line) for line in (out / 'predictions.jsonl').read_text().splitlines()]
    assert records[0]['status'] == 'ok' and records[0]['correct'] is False
    assert json.loads((out / 'metrics.json').read_text())['overall']['accuracy'] == 0
    assert 'SECRET_GOLD' not in next(p for system, p in prompts if system == 'agent')
    assert 'SECRET_GOLD' in next(p for system, p in prompts if system == 'judge')
    template = build_annotation_template(records)
    assert len(template) == 1 and template[0]['failure_types'] == []
    template[0].update(annotator='reviewer', failure_types=['F1', 'F4'], primary_failure_type='F1',
                       supporting_message_ids=['Msg_1'], notes='Speaker misattributed')
    summary = summarize_annotations(records, template)
    assert summary['annotated_failures'] == 1
    assert summary['primary_failure_counts'] == {'F1': 1}
    assert summary['multi_label_counts'] == {'F1': 1, 'F4': 1}
    assert summary['manually_annotated_evidence']['examples_with_any_evidence_retrieved'] == 1


def test_http_payload_and_usage(monkeypatch):
    from sgmem.qa import LocalClient

    captured = {}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, *args):
            return json.dumps({'choices': [{'message': {'content': 'Final: answer'}, 'finish_reason': 'length'}],
                               'usage': {'total_tokens': 42}}).encode()

    def fake_urlopen(request, timeout):
        captured['url'] = request.full_url
        captured['payload'] = json.loads(request.data)
        captured['timeout'] = timeout
        return Response()

    monkeypatch.setattr('sgmem.qa.urlopen', fake_urlopen)
    client = LocalClient('http://127.0.0.1:8000/v1', 'model', timeout=3)
    assert client.complete('system', 'user', 10) == 'Final: answer'
    assert captured['url'] == 'http://127.0.0.1:8000/v1/chat/completions'
    assert captured['payload']['messages'] == [{'role': 'system', 'content': 'system'},
                                                {'role': 'user', 'content': 'user'}]
    assert client.last_usage == {'total_tokens': 42}
    assert client.last_finish_reason == 'length'


def test_invalid_split_fails_before_output_and_model_use(tmp_path, monkeypatch):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    _write_benchmark(benchmark)
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({'dev': [{'domain': 'Finance', 'category': 'multi_hop', 'id': 'missing'}], 'test': []}))
    args = Namespace(benchmark=str(benchmark), project_root=str(tmp_path), output=str(tmp_path / 'run'),
                     split=str(path), partition='dev', domains=['Finance'], categories=['multi_hop'],
                     method='bm25', base_url='http://localhost/v1', answer_model='local', judge_model=None,
                     embedding_model='intfloat/e5-base-v2', embedding_revision=None, device='cpu',
                     batch_size=2, top_k=2, timeout=1, limit=1)
    monkeypatch.setattr('sgmem.runner.build_retriever', lambda *a: pytest.fail('must not build index'))
    with pytest.raises(ValueError, match='absent from benchmark'):
        run(args)
    assert not (tmp_path / 'run').exists()


def test_judge_failure_preserves_answer():
    from sgmem.qa import JudgeFailure
    class Client:
        last_usage = None
        def __init__(self, fail=False):
            self.fail = fail
        def complete(self, *args, **kwargs):
            if self.fail:
                raise RuntimeError('judge unavailable')
            return 'Final: retained answer'
    question = {'question': 'What?', 'answer': 'reference'}
    with pytest.raises(JudgeFailure) as error:
        evaluate_question(question, [], Client(), Client(fail=True), 'agent', 'judge')
    assert error.value.partial['answer'] == 'retained answer'
    assert error.value.partial['agent_output'] == 'Final: retained answer'


def test_shared_client_finish_reason_is_snapshotted():
    class Client:
        last_usage = None
        last_finish_reason = None

        def complete(self, system, prompt, max_tokens):
            self.last_finish_reason = 'length' if system == 'agent' else 'stop'
            return 'Final: answer' if system == 'agent' else 'Final: Incorrect'

    client = Client()
    result = evaluate_question({'question': 'What?', 'answer': 'gold'}, [], client, client, 'agent', 'judge')
    assert result['answer_finish_reason'] == 'length'
    assert result['judge_finish_reason'] == 'stop'
    metrics = summarize([{**result, 'category': 'multi_hop', 'status': 'ok'}])
    assert metrics['overall']['truncated_answers'] == 1
    assert metrics['overall']['truncated_judgments'] == 0
    assert metrics['overall']['accuracy'] == 0


def test_judge_review_does_not_rewrite_scores_or_mix_task_errors():
    records = [{'domain': 'Finance', 'category': 'multi_hop', 'question_id': 'q1',
                'method': 'bm25', 'correct': False, 'official_correct': False}]
    rows = build_annotation_template(records)
    rows[0].update(annotator='reviewer', failure_types=['F9'], primary_failure_type='F9',
                   error_stage='evaluation', notes='Evidence agrees with answer; recheck judge.')
    summary = summarize_annotations(records, rows)
    assert summary['judged_correct'] == 0 and summary['judged_failures'] == 1
    assert summary['evaluation_review_cases'] == 1
    assert summary['annotated_task_failures'] == 0
    assert summary['task_primary_failure_counts'] == {}
    rows[0]['failure_types'].append('F8')
    with pytest.raises(ValueError, match='Judge disagreements'):
        summarize_annotations(records, rows)
