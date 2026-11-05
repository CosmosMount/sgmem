"""Run content-only retrieval baselines with the GroupMemBench QA protocol."""

from __future__ import annotations

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
    return sets[partition]


def _git_info(path: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def run(args) -> Path:
    benchmark = Path(args.benchmark).resolve()
    output = Path(args.output).resolve()
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

    agent_system = (benchmark / "prompts/hipporag_agent_system.txt").read_text(encoding="utf-8").strip()
    judge_system = (benchmark / "prompts/hipporag_judge_system.txt").read_text(encoding="utf-8").strip()
    output.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in vars(args).items() if key not in {"handler"}}
    config["benchmark"] = str(benchmark)
    manifest = {
        "config": config,
        "project_git": _git_info(Path(args.project_root).resolve()),
        "benchmark_git": _git_info(benchmark),
        "evaluation": "GroupMemBench prompt and typed-summary protocol with local models",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    key = os.environ.get("SGMEM_API_KEY")
    answer_client = LocalClient(args.base_url, args.answer_model, timeout=args.timeout, api_key=key)
    judge_client = LocalClient(args.base_url, args.judge_model or args.answer_model, timeout=args.timeout, api_key=key)
    records = []
    index_times = {}
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for domain in args.domains:
            message_path = benchmark / "data/final" / domain / f"synthetic_domain_channels_rolevariants_{domain}.json"
            messages = load_messages(message_path)
            started = time.perf_counter()
            retriever = build_retriever(args.method, messages, args.embedding_model, args.device,
                                        args.batch_size, args.embedding_revision)
            index_times[domain] = time.perf_counter() - started
            for category in args.categories:
                questions = questions_by_cell[(domain, category)]
                if selected is not None:
                    questions = [q for q in questions if (domain, category, q["id"]) in selected]
                if args.limit is not None:
                    questions = questions[:args.limit]
                for q in questions:
                    record = {"domain": domain, "category": category, "question_id": q["id"],
                              "method": args.method, "question": q["question"],
                              "asking_user_id": q.get("asking_user_id"), "reference_answer": q["answer"]}
                    try:
                        asker = q.get("asking_user_id") or ""
                        query = f"{asker} {q['question']}" if asker else q["question"]
                        started = time.perf_counter()
                        retrieved = retriever.retrieve(query, args.top_k)
                        record["retrieval_seconds"] = time.perf_counter() - started
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
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"{domain}/{category}/{q['id']}: {record['status']} {record.get('judge_answer', '')}", flush=True)
    if not records:
        raise ValueError("No questions selected")
    if selected is not None and args.limit is None:
        seen = {(r["domain"], r["category"], r["question_id"]) for r in records}
        if seen != selected:
            raise ValueError(f"Split contains {len(selected - seen)} question IDs absent from benchmark")
    manifest["index_seconds_by_domain"] = index_times
    manifest["question_ids"] = [{"domain": r["domain"], "category": r["category"], "id": r["question_id"]} for r in records]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(summarize(records), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
