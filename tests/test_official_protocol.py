"""Compare the adapter against the pinned upstream using synthetic model outputs."""

import ast
import html
import json
import os
from pathlib import Path
import re

import pytest

from sgmem.data import load_messages
from sgmem.qa import evaluate_question, extract_final, format_message, official_correct, parse_judgment
from sgmem.retrieval import BM25Retriever
from test_baselines import _write_benchmark


UPSTREAM = Path(__file__).resolve().parents[1] / 'external/GroupMemBench'


def upstream_functions(relative, names, extra=None):
    path = UPSTREAM / relative
    if not path.exists():
        pytest.skip('Initialize the pinned GroupMemBench submodule for upstream parity tests')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names) + 1
    namespace = dict(html=html, json=json, os=os, re=re)
    namespace.update(extra or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def test_official_load_format_bm25_and_empty_query(tmp_path):
    from rank_bm25 import BM25Okapi

    _write_benchmark(tmp_path)
    path = tmp_path / 'data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json'
    upstream = upstream_functions('baselines/rag_common/eval_lib.py',
                                  ['load_conversation_messages', 'format_retrieved_message', 'message_index_text'])
    tokenize = upstream_functions('baselines/bm25/eval_benchmark.py', ['tokenize'],
                                  {'_TOKEN_RE': re.compile(r'[A-Za-z0-9_]+')})['tokenize']
    raw = upstream['load_conversation_messages'](str(path))
    adapted = load_messages(path)
    assert [m['id'] for m in adapted] == [m['msg_node'] for m in raw]
    assert [format_message(m) for m in adapted] == [upstream['format_retrieved_message'](m) for m in raw]
    official = BM25Okapi([tokenize(upstream['message_index_text'](m)) for m in raw])
    retriever = BM25Retriever(adapted)
    for query in ['budget schedule', 'unknownword', 'User_1 budget', '!!!']:
        tokens = tokenize(query)
        scores = official.get_scores(tokens)
        ids = sorted(range(len(raw)), key=lambda i: scores[i], reverse=True)[:2] if tokens else []
        found = retriever.retrieve(query, 2)
        assert [r['message']['id'] for r in found] == [raw[i]['msg_node'] for i in ids]
        assert [r['score'] for r in found] == [float(scores[i]) for i in ids]


def test_official_qa_and_judge_prompts_match_byte_for_byte(tmp_path):
    class Progress:
        def __init__(self, rows, **kwargs): self.rows = rows
        def __iter__(self): return iter(self.rows)
        def set_postfix_str(self, *args): pass

    calls = []
    def chat(client, model, system, user, max_tokens):
        calls.append((system, user, max_tokens))
        return 'because\nFinal answer: budget 10' if system == 'agent' else 'reason\nFinal: Correct'

    names = ['run_qa', 'format_retrieved_message', 'extract_final', 'split_reasoning_and_final', 'parse_judgment']
    upstream = upstream_functions('baselines/rag_common/eval_lib.py', names, {'tqdm': Progress, 'call_chat': chat})
    question = {'id': 'q', 'question': 'What is my budget?', 'asking_user_id': 'User_1', 'answer': 'SECRET_GOLD'}
    message = {'msg_node': 'M1', '_channel': 'G', 'author': 'User_2', 'content': 'Budget &amp; 10'}
    upstream['run_qa'](questions=[question], messages=[message], retrieve=lambda q, k: [0],
                       top_k=10, client=None, agent_model='local', judge_model='local', agent_system='agent',
                       judge_system='judge', output_jsonl=str(tmp_path / 'official.jsonl'))
    expected = calls.copy()
    calls.clear()
    class Client:
        last_usage = None
        def complete(self, system, user, max_tokens): return chat(None, 'local', system, user, max_tokens)
    adapted = {'id': 'M1', 'group': 'G', 'speaker': 'User_2', 'content': 'Budget &amp; 10'}
    result = evaluate_question(question, [{'message': adapted}], Client(), Client(), 'agent', 'judge')
    assert calls == expected
    raw = json.loads((tmp_path / 'official.jsonl').read_text())
    assert result['answer'] == raw['agent_answer']
    assert result['judge_answer'] == raw['judge_answer']
    assert 'SECRET_GOLD' not in calls[0][1] and 'SECRET_GOLD' in calls[1][1]


@pytest.mark.parametrize('text', ['', 'Final: Correct', 'Final answer: Incorrect', 'reason\nCorrect.',
                                  'Final: not correct', 'Final: Correct because it matches', 'UNKNOWN'])
def test_official_final_extraction_and_strict_scoring(text):
    upstream = upstream_functions('baselines/rag_common/eval_lib.py', ['extract_final', 'parse_judgment'])
    strict = upstream_functions('task_synthesis/summarize_typed_eval.py', ['_is_correct'],
                                {'_NORM_RE': re.compile(r'[^a-zA-Z]+')})['_is_correct']
    final = extract_final(text)
    assert final == upstream['extract_final'](text)
    assert parse_judgment(text) == upstream['parse_judgment'](text)
    assert official_correct(final) == strict(final)
