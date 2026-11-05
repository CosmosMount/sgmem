"""Content-only BM25 and E5 retrieval for GroupMemBench."""

from __future__ import annotations

import html
import re
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _content(message: dict[str, Any]) -> str:
    return html.unescape(message["content"]).strip()


class BM25Retriever:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.messages = messages
        self.index = BM25Okapi([_tokenize(_content(message)) for message in messages])

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        # Python's stable sort preserves upstream message order when scores tie.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"message": self.messages[i], "score": float(scores[i])} for i in order]


class DenseRetriever:
    def __init__(self, messages: list[dict[str, Any]], model_name: str,
                 device: str | None, batch_size: int, revision: str | None) -> None:
        from sentence_transformers import SentenceTransformer

        self.messages = messages
        self.model = SentenceTransformer(model_name, device=device, revision=revision)
        self.batch_size = batch_size
        self.vectors = np.asarray(self.model.encode(
            ["passage: " + _content(message) for message in messages],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ), dtype=np.float32)

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = np.asarray(self.model.encode(
            ["query: " + query],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ), dtype=np.float32)[0]
        scores = self.vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [{"message": self.messages[int(i)], "score": float(scores[i])} for i in order]


def build_retriever(method: str, messages: list[dict[str, Any]],
                    model_name: str = "intfloat/e5-base-v2",
                    device: str | None = None, batch_size: int = 32,
                    revision: str | None = None) -> BM25Retriever | DenseRetriever:
    """Build a content-only retriever. Answers and other labels are never inputs."""
    if not messages:
        raise ValueError("Cannot build a retriever without messages")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if method == "bm25":
        return BM25Retriever(messages)
    if method == "dense":
        return DenseRetriever(messages, model_name, device, batch_size, revision)
    raise ValueError(f"Unknown retrieval method: {method!r}")
