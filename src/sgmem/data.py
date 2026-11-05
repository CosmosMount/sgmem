"""Load GroupMemBench messages and question files without exposing labels to retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_messages(path: str | Path) -> list[dict[str, Any]]:
    """Read one domain's channels in the order used by the official baselines."""
    with Path(path).open(encoding="utf-8") as file:
        channels = json.load(file)
    if not isinstance(channels, dict):
        raise ValueError(f"Expected a channel-to-messages object in {path}")

    messages = []
    ids: set[str] = set()
    for channel, entries in channels.items():
        if not isinstance(entries, list):
            raise ValueError(f"Expected a list for channel {channel!r}")
        for message in entries:
            if not isinstance(message, dict):
                raise ValueError(f"Expected a message object in channel {channel!r}")
            message_id = message.get("msg_node")
            if not isinstance(message_id, str) or not message_id:
                raise ValueError(f"Missing message ID in channel {channel!r}")
            if message_id in ids:
                raise ValueError(f"Duplicate message ID {message_id!r} in {path}")
            ids.add(message_id)
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"Missing or invalid content for message {message_id!r}")
            messages.append({
                "id": message_id,
                "content": content,
                "speaker": message.get("author"),
                "group": channel,
                "role": message.get("role"),
                "timestamp": message.get("timestamp"),
                "reply_to": message.get("reply_to"),
                "phase_name": message.get("phase_name"),
                "topic": message.get("topic"),
            })
    messages.sort(key=lambda m: (m["group"], m["timestamp"] or "", m["id"]))
    return messages


def load_questions(path: str | Path, domain: str, category: str) -> list[dict[str, Any]]:
    """Read questions and reference answers; only text is passed to the retriever.

    Question IDs are unique within a domain/category file, not across domains.
    Callers must identify a question with (domain, category, id).
    """
    questions = []
    ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            question = json.loads(line)
            if not isinstance(question, dict):
                raise ValueError(f"Expected a question object at {path}:{line_number}")
            question_id = question.get("id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"Missing question ID at {path}:{line_number}")
            if question_id in ids:
                raise ValueError(f"Duplicate question ID {question_id!r} in {path}")
            if not isinstance(question.get("question"), str) or "answer" not in question:
                raise ValueError(f"Missing question text or answer at {path}:{line_number}")
            ids.add(question_id)
            questions.append({
                "id": question_id,
                "question": question["question"],
                "asking_user_id": question.get("asking_user_id"),
                "answer": question["answer"],
                "domain": domain,
                "category": category,
            })
    return questions
