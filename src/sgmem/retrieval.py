"""BM25, E5 and bounded reply retrieval for GroupMemBench."""

from __future__ import annotations

import html
import re
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_REPLY_CUE = re.compile(r"\b(?:should|recommend\w*|suggest\w*|propos\w*|discussion|feedback|"
                        r"repl(?:y|ies|ied)|respond\w*|follow[ -]?up)\b", re.IGNORECASE)
_ORDER_CUE = re.compile(r"\b(?:before|after|first|next)\b", re.IGNORECASE)
METHODS = ("bm25", "dense", "bm25-reply", "bm25-reply-order")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _content(message: dict[str, Any]) -> str:
    return html.unescape(message["content"]).strip()


class BM25Retriever:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.messages = messages
        self.index = BM25Okapi([_tokenize(_content(message)) for message in messages])

    def rank_content(self, query: str, top_k: int) -> tuple[np.ndarray, list[int]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        tokens = _tokenize(query)
        if not tokens:
            return np.array([]), []
        scores = self.index.get_scores(tokens)
        # Python's stable sort preserves upstream message order when scores tie.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return scores, order

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scores, order = self.rank_content(query, top_k)
        return [{"message": self.messages[i], "score": float(scores[i])} for i in order]


class BM25ReplyRetriever(BM25Retriever):
    """Keep the leading content hits and let nearby replies compete for the last slots."""

    def __init__(self, messages, encoder=None, batch_size=32):
        super().__init__(messages)
        self.encoder = encoder
        self.batch_size = batch_size
        by_id = {m["id"]: i for i, m in enumerate(messages)}
        if len(by_id) != len(messages):
            raise ValueError("Reply retrieval requires unique message IDs")
        parents = [None] * len(messages)
        children = [[] for _ in messages]
        for i, m in enumerate(messages):
            parent = by_id.get(m.get("reply_to"))
            if parent is None or parent == i:
                continue
            if m.get("group") is not None and m["group"] == messages[parent].get("group"):
                parents[i] = parent
                children[parent].append(i)
        self.parents = parents
        self.links = []
        for parent, replies in zip(parents, children):
            # load_messages orders each group by timestamp and ID. Keep both ends of busy threads.
            bounded = replies if len(replies) <= 8 else replies[:4] + replies[-4:]
            self.links.append(([parent] if parent is not None else []) + bounded)

    def e5_scores(self, query, indices, *, include_parent=True):
        texts = ["query: " + query]
        for i in indices:
            passage = _content(self.messages[i])
            parent = self.parents[i]
            if include_parent and parent is not None:
                passage += "\nParent: " + _content(self.messages[parent])
            texts.append("passage: " + passage)
        vectors = self.encoder.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        return dict(zip(indices, vectors[1:] @ vectors[0]))

    def expand(self, query, scores, base, *, hops=2, slots=2, require_cue=True, rank_replies=None):
        """Return final hits and finalist IDs. Optional scores only order existing eligible replies."""
        if hops not in (1, 2) or slots not in (0, 1, 2):
            raise ValueError("Use one or two hops and at most two reply slots")
        if not base:
            return [], []
        slots = min(slots, len(base) // 5)
        cut = len(base) - slots
        protected = set(base[:cut])
        paths = {}

        def consider(path):
            target = path[-1]
            if target in protected or scores[target] <= 0:
                return
            if target not in paths or len(path) < len(paths[target]):
                paths[target] = path

        if slots and (not require_cue or _REPLY_CUE.search(query)):
            for seed in base[:min(cut, 8)]:
                for via in self.links[seed]:
                    consider((seed, via))
                    if hops == 2:
                        for target in self.links[via]:
                            consider((seed, via, target))

        base_set = set(base)
        linked = sorted(paths, key=lambda i: (-scores[i], i not in base_set, len(paths[i]), i))
        extras = [i for i in linked if i not in base_set][:len(base)]
        # Weak replies leave their slots to content hits. A reply already in the tail consumes just one slot.
        pool = base + extras
        pool_set = set(pool)
        eligible = [i for i in linked if i in pool_set and scores[i] >= scores[base[-1]] / 2]
        rerank_scores = {}
        if rank_replies is not None and len(eligible) > 1:
            rerank_scores = rank_replies(query, eligible)
            eligible.sort(key=rerank_scores.__getitem__, reverse=True)  # Ties keep the lexical order.
        chosen = eligible[:slots]
        fillers = [i for i in base[cut:] if i not in chosen][:slots - len(chosen)]
        keep = protected.union(chosen, fillers)
        final = [i for i in base if i in keep] + [i for i in chosen if i not in base_set]
        hits = []
        for i in final:
            hit = {"message": self.messages[i], "score": float(scores[i])}
            if i not in base_set:
                hit["reply_path"] = [self.messages[node]["id"] for node in paths[i]]
            if i in rerank_scores:
                hit["e5_score"] = float(rerank_scores[i])
            hits.append(hit)
        return hits, [self.messages[i]["id"] for i in pool]

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scores, base = self.rank_content(query, top_k)
        rank_replies = self.e5_scores if self.encoder is not None and _ORDER_CUE.search(query) else None
        return self.expand(query, scores, base, rank_replies=rank_replies)[0]


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
    """Build a retriever from the visible messages."""
    if not messages:
        raise ValueError("Cannot build a retriever without messages")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if method == "bm25":
        return BM25Retriever(messages)
    if method == "bm25-reply":
        return BM25ReplyRetriever(messages)
    if method == "bm25-reply-order":
        from sentence_transformers import SentenceTransformer

        # Only a short list of replies is encoded; keep QA's GPU available for generation.
        encoder = SentenceTransformer(model_name, device="cpu", revision=revision)
        return BM25ReplyRetriever(messages, encoder, batch_size)
    if method == "dense":
        return DenseRetriever(messages, model_name, device, batch_size, revision)
    raise ValueError(f"Unknown retrieval method: {method!r}")
