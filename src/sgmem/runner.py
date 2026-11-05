"""Run content-only retrieval baselines with the GroupMemBench QA protocol."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

from sgmem.data import load_messages, load_questions
from sgmem.qa import JudgeFailure, LocalClient, evaluate_question, summarize
from sgmem.retrieval import build_retriever

CATEGORIES = ("multi_hop", "knowledge_update", "temporal", "user_implicit", "term_ambiguity", "abstention")
DOMAINS = ("Finance", "Technology", "Healthcare", "Manufacturing")


def question_file(benchmark: Path, domain: str, category: str) -> Path:
    return benchmark / "questions" / domain / f"{category}.jsonl"


def make_split(benchmark: Path, domains: list[str], seed: int, dev_fraction: float) -> dict:
    """Deterministic question-level stratification; does not claim dialogue independence."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev fraction must be between 0 and 1")
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"dev": [], "test": []}
    for domain in domains:
        for category in CATEGORIES:
            questions = load_questions(question_file(benchmark, domain, category), domain, category)
            if len(questions) < 2:
                raise ValueError(f"Need at least two questions for {domain}/{category}")
            ids = sorted(question["id"] for question in questions)
            rng.shuffle(ids)
            ndev = max(1, min(len(ids) - 1, round(len(ids) * dev_fraction)))
            for partition, selected in (("dev", ids[:ndev]), ("test", ids[ndev:])):
                splits[partition].extend({"domain": domain, "category": category, "id": qid} for qid in selected)
    return {"seed": seed, "dev_fraction": dev_fraction, "unit": "question", **splits}


def split_keys(path: Path, partition: str) -> set[tuple[str, str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sets = {}
    for name in ("dev", "test"):
        rows = data[name]
        keys = [(row["domain"], row["category"], row["id"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate question in split {name}")
        sets[name] = set(keys)
    if sets["dev"] & sets["test"]:
        raise ValueError("Development and test splits overlap")
    return sets['dev'] | sets['test'] if partition == 'all' else sets[partition]


def _git_info(path: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def write_json(path: Path, value: object) -> None:
    """Replace a small snapshot atomically; the JSONL attempt log is append-only."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _checkpoint(path: Path) -> list[dict]:
    """Recover only an incomplete final write, preserving the discarded bytes."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records = []
    offset = 0
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'Corrupt checkpoint at line {i + 1}: {path}')
            tail = path.with_name(f'interrupted-tail-{time.time_ns()}.bin')
            tail.write_bytes(line)
            with path.open('r+b') as handle:
                handle.truncate(offset)
            break
        offset += len(line)
    else:
        if raw and not raw.endswith(b'\n'):
            with path.open('ab') as handle:
                handle.write(b'\n')
    return records


def run(args, client_factory=None) -> Path:
    benchmark = Path(args.benchmark).resolve()
    output = Path(args.output).resolve()
    resume = getattr(args, 'resume', False)
    if bool(args.split) != bool(args.partition):
        raise ValueError('split and partition must be provided together')
    selected = split_keys(Path(args.split), args.partition) if args.split else None
    if len(set(args.domains)) != len(args.domains) or len(set(args.categories)) != len(args.categories):
        raise ValueError("Domains and categories must not be repeated")
    if args.top_k <= 0 or args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("top-k, batch-size and limit must be positive")
    if selected is not None:
        allowed = {(domain, category) for domain in args.domains for category in args.categories}
        selected = {key for key in selected if key[:2] in allowed}
        if not selected:
            raise ValueError("No split questions match the selected domains and categories")
    questions_by_cell = {
        (domain, category): load_questions(question_file(benchmark, domain, category), domain, category)
        for domain in args.domains for category in args.categories
    }
    if selected is not None:
        available = {(domain, category, question["id"])
                     for (domain, category), questions in questions_by_cell.items() for question in questions}
        missing = selected - available
        if missing:
            raise ValueError(f"Split contains {len(missing)} question IDs absent from benchmark")

    for (domain, category), questions in questions_by_cell.items():
        if selected is not None:
            questions = [q for q in questions if (domain, category, q['id']) in selected]
        questions_by_cell[domain, category] = questions[:args.limit] if args.limit is not None else questions
    expected = {(d, c, q['id']): q for (d, c), qs in questions_by_cell.items() for q in qs}
    if not expected:
        raise ValueError('No questions selected')
    partition_by_id = {}
    if args.split:
        for partition in ('dev', 'test'):
            partition_by_id.update({key: partition for key in split_keys(Path(args.split), partition)})
    agent_path = benchmark / 'prompts/hipporag_agent_system.txt'
    judge_path = benchmark / 'prompts/hipporag_judge_system.txt'
    agent_system = agent_path.read_text(encoding='utf-8').strip()
    judge_system = judge_path.read_text(encoding='utf-8').strip()
    inputs = [agent_path, judge_path]
    inputs += [benchmark / f'data/final/{d}/synthetic_domain_channels_rolevariants_{d}.json' for d in args.domains]
    inputs += [question_file(benchmark, d, c) for d, c in questions_by_cell]
    fingerprints = {str(p.relative_to(benchmark)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if args.split:
        fingerprints['split'] = hashlib.sha256(Path(args.split).read_bytes()).hexdigest()
    config = {key: value for key, value in vars(args).items() if key not in {'handler', 'resume'}}
    config["benchmark"] = str(benchmark)
    manifest = {
        'schema_version': 2,
        "config": config,
        "project_git": _git_info(Path(args.project_root).resolve()),
        "benchmark_git": _git_info(benchmark),
        'input_sha256': fingerprints,
        "evaluation": "GroupMemBench prompt and typed-summary protocol with local models",
    }
    index_times = {}
    records_by_id = {}
    if resume:
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if any(previous.get(key) != value for key, value in manifest.items()):
            raise ValueError('Resume rejected: configuration, source, or input fingerprint changed')
        index_times = previous.get('index_seconds_by_domain', {})
        for record in _checkpoint(output / 'attempts.jsonl'):
            qid = (record['domain'], record['category'], record['question_id'])
            if qid not in expected or record['method'] != args.method:
                raise ValueError('Checkpoint contains a question outside this run')
            prior = records_by_id.get(qid)
            if prior and prior['status'] == 'ok':
                raise ValueError('Checkpoint repeats a completed question')
            records_by_id[qid] = record
        manifest = previous
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest['started_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest.update(status='running', expected_questions=len(expected))
    write_json(output / 'manifest.json', manifest)
    key = os.environ.get("SGMEM_API_KEY")
    client_factory = client_factory or LocalClient
    answer_client = client_factory(args.base_url, args.answer_model, timeout=args.timeout, api_key=key)
    judge_client = client_factory(args.base_url, args.judge_model or args.answer_model, timeout=args.timeout, api_key=key)
    started = time.perf_counter()
    try:
        with (output / 'attempts.jsonl').open('a' if resume else 'x', encoding='utf-8') as handle:
            for domain in args.domains:
                pending = {qid for qid in expected if qid[0] == domain
                           and records_by_id.get(qid, {}).get('status') != 'ok'}
                if not pending:
                    continue
                message_path = benchmark / "data/final" / domain / f"synthetic_domain_channels_rolevariants_{domain}.json"
                messages = load_messages(message_path)
                index_started = time.perf_counter()
                retriever = build_retriever(args.method, messages, args.embedding_model, args.device,
                                            args.batch_size, args.embedding_revision)
                index_times[domain] = index_times.get(domain, 0) + time.perf_counter() - index_started
                for category in args.categories:
                    questions = questions_by_cell[(domain, category)]
                    for q in questions:
                        qid = (domain, category, q['id'])
                        if qid not in pending:
                            continue
                        record = {"domain": domain, "category": category, "question_id": q["id"],
                                  "method": args.method, "question": q["question"],
                                  "asking_user_id": q.get("asking_user_id"), "reference_answer": q["answer"]}
                        if partition_by_id:
                            record['partition'] = partition_by_id[qid]
                        record['attempt'] = records_by_id.get(qid, {}).get('attempt', 0) + 1
                        for client in (answer_client, judge_client):
                            if hasattr(client, 'context'):
                                client.context = {'domain': domain, 'category': category, 'question_id': q['id'],
                                                  'method': args.method, 'attempt': record['attempt']}
                        try:
                            asker = q.get("asking_user_id") or ""
                            query = f"{asker} {q['question']}" if asker else q["question"]
                            retrieval_started = time.perf_counter()
                            retrieved = retriever.retrieve(query, args.top_k)
                            record["retrieval_seconds"] = time.perf_counter() - retrieval_started
                            record["retrieval"] = retrieved
                            result = evaluate_question(q, retrieved, answer_client, judge_client, agent_system, judge_system)
                            record.update(result)
                            record["prediction"] = result["answer"]
                            record["status"] = "ok"
                        except JudgeFailure as exc:
                            record.update(exc.partial)
                            record["prediction"] = exc.partial["answer"]
                            record["status"] = "error"
                            record["error"] = str(exc)
                        except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
                            record["status"] = "error"
                            record["error"] = str(exc)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        records_by_id[qid] = record
                        write_json(output / 'progress.json', {
                            'expected': len(expected), 'recorded': len(records_by_id),
                            'completed': sum(r['status'] == 'ok' for r in records_by_id.values()),
                            'last_question': list(qid),
                            'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        })
                        print(f"{domain}/{category}/{q['id']}: {record['status']} {record.get('judge_answer', '')}", flush=True)
                        if record['status'] == 'error':
                            raise RuntimeError(f'Model call failed for {qid}; fix the cause, then resume')
                del retriever
    finally:
        records = [records_by_id[qid] for qid in expected if qid in records_by_id]
        manifest['status'] = 'complete' if len(records) == len(expected) and all(r['status'] == 'ok' for r in records) else 'incomplete'
        manifest['index_seconds_by_domain'] = index_times
        manifest['wall_seconds'] = manifest.get('wall_seconds', 0) + time.perf_counter() - started
        manifest['question_ids'] = [{'domain': r['domain'], 'category': r['category'], 'id': r['question_id']} for r in records]
        temporary = output / 'predictions.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        temporary.replace(output / 'predictions.jsonl')
        metrics = summarize(records)
        metrics['expected_questions'] = len(expected)
        metrics['complete'] = manifest['status'] == 'complete'
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'manifest.json', manifest)
    return output
