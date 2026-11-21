"""Small, fixed retrieval comparisons using public conversation metadata."""

import re

import numpy as np

from sgmem.query_metadata import build_catalog, normalize, query_hints
from sgmem.retrieval import BM25ReplyRetriever, BM25Retriever, STRUCTURED_METHODS, _content, _tokenize

DEPTH = 50
FIELDS = ("role", "group", "phase_name", "topic")
PERSON_METHODS = ("bm25-person", "bm25-time", "hybrid-meta")
PERSON_CUE = re.compile(
    r"\b(?:(?:i (?:am )?|am i )(?:still )?(?:need|monitor|monitoring|watch|watching|prefer|proposed|suggested)"
    r"|my (?:(?:current|latest|open) )?(?:open item|item|risk|concern|preference|proposal|suggestion))\b"
)
CURRENT_CUE = re.compile(r"\b(?:current|updated|latest|most recent|still need)\b")
BACKGROUND_CUE = re.compile(r"\b(?:when incorporating|in light of|given that)\b", re.I)


def reciprocal_rank_fusion(orders):
    scores = {}
    for order in orders.values():
        for rank, index in enumerate(order, 1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (60 + rank)
    return sorted(scores, key=lambda index: (-scores[index], index)), scores


class StructuredRetriever(BM25ReplyRetriever):
    """Ablations share candidate generation and rank fusion; final passages stay raw."""

    def __init__(self, messages, method, *, dense=None):
        if method not in STRUCTURED_METHODS or method.startswith("hybrid") != (dense is not None):
            raise ValueError("Select a structured method, with a dense retriever only for hybrid methods")
        super().__init__(messages)
        self.method = method
        self.dense = dense
        self.by_id = {m["id"]: i for i, m in enumerate(messages)}
        self.catalog = build_catalog(messages)
        self.values = {field: np.array([m.get(field) or "" for m in messages], dtype=object)
                       for field in (*FIELDS, "speaker", "timestamp")}
        self.speakers = sorted(set(self.values["speaker"]) - {""})
        self.roles = sorted(set(self.values["role"]) - {""})
        if method == "bm25-fields":
            # Speaker IDs need an explicit target. An asker's ID alone must not boost their messages.
            indexed = [{**m, "content": _content(m) + "\n" + " ".join(
                normalize(m.get(field) or "") for field in FIELDS)} for m in messages]
            self.fields = BM25Retriever(indexed)

    def query_plan(self, question, asking_user_id=None):
        hints = query_hints(question, self.catalog, prefer_longest=True)
        text = normalize(re.sub(r"\bi['’]m\b", "I am", question, flags=re.I))
        subject_text = re.split(r"\b(?:after|before|while|if|because|although)\b", text, maxsplit=1)[0]

        def is_subject(name):
            name = re.escape(normalize(name))
            return re.search(r"\b(?:does|did|has|had|is|was|should) (?:the )?" + name + r"\b", subject_text)

        personal = asking_user_id in self.speakers and PERSON_CUE.search(subject_text)
        speakers = [s for s in self.speakers if is_subject(s)]
        if not speakers and personal:
            speakers = [asking_user_id]
        roles = [role for role in self.roles if is_subject(role)]
        content_query = text
        for entry in hints["matched_names"]:
            content_query = re.sub(r"\b" + re.escape(normalize(entry["name"])) + r"\b", " ", content_query)
        return {"matched_names": hints["matched_names"], "speakers": speakers, "roles": roles,
                "content_query": " ".join(content_query.split()),
                "current": bool(CURRENT_CUE.search(text))}

    def candidate_orders(self, question, top_k, *, asking_user_id=None):
        depth = max(DEPTH, top_k)
        original_query = f"{asking_user_id} {question}" if asking_user_id else question
        content_scores, content_order = self.rank_content(original_query, depth)
        plan = self.query_plan(question, asking_user_id)
        if not content_order:
            return {}, plan, content_scores
        if self.method == "bm25-fields":
            _, order = self.fields.rank_content(original_query, depth)
            return {"fields": order}, plan, content_scores
        orders = {"content": content_order}
        if self.dense is not None:
            hits = self.dense.retrieve(question, depth, asking_user_id=asking_user_id)
            orders["dense"] = [self.by_id[h["message"]["id"]] for h in hits]
        if self.method == "hybrid":
            return orders, plan, content_scores

        names = [e for e in plan["matched_names"] if e["field"] != "group"] or plan["matched_names"]
        scope = np.zeros(len(self.messages), dtype=bool)
        for entry in names:
            scope |= self.values[entry["field"]] == entry["name"]
        groups = [e["name"] for e in plan["matched_names"] if e["field"] == "group"]
        if len(groups) == 1:
            scope &= np.isin(self.values["group"], groups)
        elif groups:
            # Do not attach one clause's phase to every group in a comparison.
            scope = np.isin(self.values["group"], groups)
        actor = np.zeros(len(self.messages), dtype=bool)
        if self.method in PERSON_METHODS:
            actor = np.isin(self.values["speaker"], plan["speakers"]) if plan["speakers"] else np.isin(
                self.values["role"], plan["roles"])
        mask = scope & actor if scope.any() and actor.any() else scope | actor
        residual = plan["content_query"]
        if mask.any() and residual:
            scores = self.index.get_scores(_tokenize(residual))
            candidates = np.flatnonzero(mask & (scores > 0)).tolist()
            scoped = sorted(candidates, key=lambda i: (-scores[i], i))[:depth]
            if scoped:
                orders["scope"] = scoped
                if self.method == "bm25-time" and plan["current"]:
                    recent = [i for i in scoped if self.values["timestamp"][i] and scores[i] >= scores[scoped[0]] / 2]
                    if recent:
                        orders["time"] = sorted(recent, key=lambda i: self.values["timestamp"][i], reverse=True)
        if self.method == "bm25-focus" and names:
            focus = BACKGROUND_CUE.split(question, maxsplit=1)[0].strip()
            if focus != question.strip():
                focus = normalize(focus)
                for entry in names:
                    focus = re.sub(r"\b" + re.escape(normalize(entry["name"])) + r"\b", " ", focus)
                scores = self.index.get_scores(_tokenize(focus))
                candidates = np.flatnonzero(scope & (scores > 0)).tolist()
                if candidates:
                    orders["focus"] = sorted(candidates, key=lambda i: (-scores[i], i))[:depth]

        return orders, plan, content_scores

    def retrieve_with_trace(self, query, top_k, *, asking_user_id=None):
        """Return the actual ranking and its candidate lists without repeating an encoder call."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        orders, plan, content_scores = self.candidate_orders(query, top_k, asking_user_id=asking_user_id)
        ranked, scores = reciprocal_rank_fusion(orders)
        selected, paths, reply_pool = ranked[:top_k], {}, []
        if self.method == "bm25-scope-reply":
            original_query = f"{asking_user_id} {query}" if asking_user_id else query
            expanded, reply_pool = self.expand(original_query, content_scores, selected)
            selected = [self.by_id[h["message"]["id"]] for h in expanded]
            paths = {self.by_id[h["message"]["id"]]: h["reply_path"] for h in expanded if "reply_path" in h}
        positions = {name: {index: rank for rank, index in enumerate(order, 1)} for name, order in orders.items()}
        hits = []
        for index in selected:
            # score is the pre-expansion RRF score; a new reply can have zero votes.
            hit = {"message": self.messages[index], "score": scores.get(index, 0.0),
                   "retrieval_sources": {name: ranks[index] for name, ranks in positions.items() if index in ranks}}
            if index in paths:
                hit["reply_path"] = paths[index]
            hits.append(hit)
        return hits, {
            "query_plan": plan,
            "candidate_lists": {name: [self.messages[i]["id"] for i in order] for name, order in orders.items()},
            "reply_pool": reply_pool,
        }

    def retrieve(self, query, top_k, *, asking_user_id=None):
        # Keep the runner's ordinary hit-list interface; the probe also consumes the trace.
        return self.retrieve_with_trace(query, top_k, asking_user_id=asking_user_id)[0]
