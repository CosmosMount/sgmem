"""Published results must recompute, and accidental evidence edits must be rejected."""

import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_part12_evidence import verify

EVIDENCE = ROOT / 'docs/evidence/part12-full-20261105'


def test_published_tables_recompute_without_models():
    assert verify(EVIDENCE)['verified']


@pytest.fixture
def evidence_copy(tmp_path):
    for path in EVIDENCE.iterdir():
        if path.suffix in ('.json', '.jsonl'):
            shutil.copy2(path, tmp_path / path.name)
    return tmp_path


@pytest.mark.parametrize('change', ['verdict', 'missing_record', 'annotation_partition', 'frozen_split'])
def test_verifier_rejects_inconsistent_evidence(evidence_copy, change):
    if change in ('verdict', 'missing_record'):
        path = evidence_copy / 'outcomes.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if change == 'verdict':
            rows[0]['official_correct'] = not rows[0]['official_correct']
        else:
            rows.pop()
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    elif change == 'annotation_partition':
        path = evidence_copy / 'failure-annotations.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['partition'] = 'test'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    else:
        path = evidence_copy / 'split.json'
        split = json.loads(path.read_text(encoding='utf-8'))
        split['dev'][0], split['test'][0] = split['test'][0], split['dev'][0]
        path.write_text(json.dumps(split), encoding='utf-8')
    with pytest.raises(AssertionError):
        verify(evidence_copy)


@pytest.mark.parametrize('change', ['aggregate', 'median', 'dropped_row', 'duplicate_row', 'targets', 'top10'])
def test_verifier_rejects_inconsistent_metadata_diagnostics(evidence_copy, change):
    path = evidence_copy / 'metadata-diagnostics.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['rows']
    if change == 'aggregate':
        data['target_within_two_reply_hops'] += 1
    elif change == 'median':
        data['median_two_hop_candidates'] += 1
    elif change == 'dropped_row':
        rows.pop()
    elif change == 'duplicate_row':
        rows[1] = rows[0]
    elif change == 'targets':
        rows[0]['targets'] = ['unrelated-message']
    else:
        rows[0]['target_in_top10'] = not rows[0]['target_in_top10']
        data['target_in_top10'] = sum(row['target_in_top10'] for row in rows)
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(AssertionError):
        verify(evidence_copy)
