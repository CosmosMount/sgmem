"""GroupMemBench QA and judge protocol using a local OpenAI-compatible server."""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class JudgeFailure(RuntimeError):
    """A judge call failed after an answer was produced."""

    def __init__(self, message: str, partial: dict):
        super().__init__(message)
        self.partial = partial


class LocalClient:
    """Minimal client for a locally hosted chat-completions endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 120,
                 temperature: float = 0.2, api_key: str | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.api_key = api_key
        self.last_usage = None
        self.last_finish_reason = None

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
        self.last_usage = None
        self.last_finish_reason = None
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Chat completion failed with HTTP status {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Chat completion connection failed: {exc.reason}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Chat completion response has no message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chat completion response has empty message content")
        self.last_usage = data.get("usage")
        self.last_finish_reason = data["choices"][0].get("finish_reason")
        return content


def format_message(message: dict) -> str:
    """Match the baseline's passage header for the project message schema."""
    tags = [f"user={message.get('speaker') or '?'}"]
    for key, label in (("role", "speaker_role"), ("group", "channel"),
                       ("phase_name", "phase_name"), ("topic", "topic"),
                       ("timestamp", "timestamp"), ("reply_to", "reply_to"),
                       ("id", "msg_node")):
        value = message.get(key)
        if value not in (None, ""):
            tags.append(f"{label}={value}")
    body = message.get("content", "")
    body = html.unescape(body).strip() if isinstance(body, str) else ""
    return f"[{' / '.join(tags)}]\n{body}".strip()


def extract_final(text: str) -> str:
    """Use the official baseline's final-line extraction."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith(("final:", "final answer:")):
            return line.split(":", 1)[1].strip()
    return lines[-1] if lines else text.strip()


def parse_judgment(text: str) -> bool | None:
    """Interpret the baseline judge's Correct/Incorrect judgment."""
    final = extract_final(text).lower()
    if "incorrect" in final or "wrong" in final or "not correct" in final:
        return False
    if "correct" in final:
        return True
    return None


def official_correct(judge_answer: str) -> bool:
    """Match the benchmark's category summary script, including strict matching."""
    return re.sub(r"[^a-zA-Z]+", "", judge_answer or "").lower() == "correct"


def evaluate_question(question: dict, retrieved: list[dict], answer_client: LocalClient,
                      judge_client: LocalClient, agent_system: str, judge_system: str) -> dict:
    """Apply the official prompt layout and judge to one already-retrieved question."""
    docs = [format_message(item["message"]) for item in retrieved]
    passages = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(docs, start=1))
    asker = question.get("asking_user_id") or ""
    asker_line = f"Asking user: {asker}\n\n" if asker else ""
    answer_prompt = (
        f"{asker_line}Question:\n{question['question']}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        "Answer the question using the retrieved passages."
    )
    start = perf_counter()
    agent_output = answer_client.complete(agent_system, answer_prompt, max_tokens=512)
    answer_seconds = perf_counter() - start
    answer_usage = answer_client.last_usage
    answer_finish_reason = getattr(answer_client, "last_finish_reason", None)
    answer = extract_final(agent_output)
    judge_prompt = (
        f"Question:\n{question['question']}\n\n"
        f"Gold Answer:\n{question['answer']}\n\n"
        f"Agent Answer:\n{answer}\n"
    )
    start = perf_counter()
    try:
        judge_output = judge_client.complete(judge_system, judge_prompt, max_tokens=256)
    except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
        raise JudgeFailure(str(exc), {"answer": answer, "agent_output": agent_output,
                                     "retrieved_docs": docs, "answer_seconds": answer_seconds,
                                     "answer_usage": answer_usage, "answer_finish_reason": answer_finish_reason}) from exc
    judge_seconds = perf_counter() - start
    judge_answer = extract_final(judge_output)
    return {
        "answer": answer,
        "agent_output": agent_output,
        "judge_output": judge_output,
        "judge_answer": judge_answer,
        "correct": parse_judgment(judge_answer),
        "official_correct": official_correct(judge_answer),
        "retrieved_docs": docs,
        "answer_seconds": answer_seconds,
        "judge_seconds": judge_seconds,
        "answer_usage": answer_usage,
        "judge_usage": judge_client.last_usage,
        "answer_finish_reason": answer_finish_reason,
        "judge_finish_reason": getattr(judge_client, "last_finish_reason", None),
    }


def summarize(records: list[dict]) -> dict:
    """Count unclear judgments as misses, as in the official typed summary.

    API failures are reported separately; accuracy is unavailable until they are
    resolved so missing model calls cannot silently become wrong answers.
    """
    def aggregate(items: list[dict]) -> dict:
        failed = sum(item.get("status") == "error" for item in items)
        completed = [item for item in items if item.get("status") != "error"]
        unclear = sum(parse_judgment(item.get("judge_answer", "")) is None
                      for item in completed)
        correct = sum(official_correct(item.get("judge_answer", "")) for item in completed)
        total = len(items)
        return {
            "total": total,
            "completed": len(completed),
            "errors": failed,
            "unclear": unclear,
            "correct": correct,
            "truncated_answers": sum(item.get("answer_finish_reason") == "length" for item in items),
            "truncated_judgments": sum(item.get("judge_finish_reason") == "length" for item in items),
            "accuracy": correct / total if total and not failed else None,
        }

    groups = defaultdict(list)
    domains = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)
        if "domain" in record:
            domains[record["domain"]].append(record)
    return {"overall": aggregate(records),
            "by_category": {name: aggregate(items) for name, items in groups.items()},
            "by_domain": {name: aggregate(items) for name, items in domains.items()},
            "by_domain_and_category": {
                domain: {category: aggregate([r for r in items if r["category"] == category])
                         for category in sorted({r["category"] for r in items})}
                for domain, items in domains.items()
            }}
