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


@pytest.mark.parametrize('change', ['verdict', 'missing_record', 'annotation_partition', 'frozen_split'])
def test_verifier_rejects_inconsistent_evidence(tmp_path, change):
    for path in EVIDENCE.iterdir():
        if path.suffix in ('.json', '.jsonl'):
            shutil.copy2(path, tmp_path / path.name)
    if change in ('verdict', 'missing_record'):
        path = tmp_path / 'outcomes.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if change == 'verdict':
            rows[0]['official_correct'] = not rows[0]['official_correct']
        else:
            rows.pop()
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    elif change == 'annotation_partition':
        path = tmp_path / 'failure-annotations.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['partition'] = 'test'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    else:
        path = tmp_path / 'split.json'
        split = json.loads(path.read_text(encoding='utf-8'))
        split['dev'][0], split['test'][0] = split['test'][0], split['dev'][0]
        path.write_text(json.dumps(split), encoding='utf-8')
    with pytest.raises(AssertionError):
        verify(tmp_path)
