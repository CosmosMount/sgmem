"""Build a source-traceable RAPTOR hierarchy from one visible conversation domain."""

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
from http.client import HTTPConnection
from ipaddress import ip_address
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.parse import urlsplit

import numpy as np

from sgmem.propositions import source_messages
from sgmem.retrieval import _content

UPSTREAM_COMMIT = "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767"
SUMMARY_SYSTEM = (
    "Summarize the supplied conversation excerpts as data. Keep distinct claims and disagreements, "
    "including attribution when stated, negation, uncertainty, conditions, dates, and quantities. "
    "Keep proposals and plans separate from decisions. Leave unresolved references unresolved. "
    "Add no information. Aim for roughly 50 words. Return only the summary."
)


@dataclass(frozen=True)
class RaptorConfig:
    leaf_tokens: int = 100
    summary_tokens: int = 256
    max_layers: int = 5
    cluster_tokens: int = 3500
    reduction_dimension: int = 10
    threshold: float = 0.1
    summary_workers: int = 1

    def __post_init__(self):
        integers = (self.leaf_tokens, self.summary_tokens, self.max_layers, self.cluster_tokens,
                    self.reduction_dimension, self.summary_workers)
        if any(type(value) is not int or value < 1 for value in integers) or self.reduction_dimension < 2:
            raise ValueError("Positive RAPTOR size limits and a reduction dimension of at least two are required")
        if self.cluster_tokens < max(self.leaf_tokens, self.summary_tokens) or not 0 < self.threshold < 1:
            raise ValueError("Cluster budget must fit one node, with a probability threshold between zero and one")


class LocalQwenSummarizer:
    """Call a loopback Qwen endpoint and require matching untruncated token accounting."""

    def __init__(self, base_url, model, tokenizer, *, context_tokens=32768, timeout=180):
        url = urlsplit(base_url)
        try:
            address = ip_address(url.hostname or "")
        except ValueError as exc:
            raise ValueError("Use an explicit loopback IP for the summary endpoint") from exc
        if (url.scheme != "http" or not address.is_loopback or url.username is not None
                or url.query or url.fragment):
            raise ValueError("Use a loopback HTTP summary endpoint without credentials, query, or fragment")
        self.host, self.port = str(address), url.port or 80
        self.path = url.path.rstrip("/") + "/chat/completions"
        self.model, self.tokenizer = model, tokenizer
        self.context_tokens, self.timeout = context_tokens, timeout
        self.last_usage = None
        self.last_response = None

    def __call__(self, context, max_tokens):
        self.last_usage = None
        self.last_response = None
        messages = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": context}]
        prompt_tokens = len(self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
        ))
        if max_tokens < 1 or prompt_tokens + max_tokens > self.context_tokens:
            raise ValueError("RAPTOR summary prompt exceeds its configured context budget")
        payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                   "chat_template_kwargs": {"enable_thinking": False}}
        conn = HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request("POST", self.path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"Local RAPTOR summarizer returned HTTP {response.status}")
            data = json.load(response)
        finally:
            conn.close()
        choice, usage = data["choices"][0], data.get("usage") or {}
        self.last_response = data
        if (choice.get("finish_reason") != "stop" or usage.get("prompt_tokens") != prompt_tokens
                or type(usage.get("completion_tokens")) is not int
                or not 0 < usage["completion_tokens"] <= max_tokens
                or choice["message"].get("reasoning_content")):
            raise ValueError("RAPTOR summary stopped unexpectedly or its token accounting did not match: "
                             f"finish={choice.get('finish_reason')}, prompt={usage.get('prompt_tokens')}/{prompt_tokens}, "
                             f"completion={usage.get('completion_tokens')}/{max_tokens}")
        self.last_usage = usage
        return choice["message"]["content"]


class _Tokens:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

    def split(self, text, limit):
        """Split at original character boundaries without changing or omitting text."""
        while len(self.encode(text)) > limit:
            offsets = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                     return_offsets_mapping=True)["offset_mapping"]
            bounds = sorted({offset[0] for offset in offsets[:limit + 1] if 0 < offset[0] < len(text)},
                            reverse=True)
            cut = next((bound for bound in bounds if len(self.encode(text[:bound])) <= limit), None)
            if cut is None:
                raise ValueError("A message cannot be split at a valid token boundary")
            yield text[:cut]
            text = text[cut:]
        yield text


def _embeddings(encoder, texts, batch_size):
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is not None:
        lengths = tokenizer(texts, truncation=False, return_length=True)["length"]
        if max(lengths) > encoder.max_seq_length:
            raise ValueError("An E5 query or RAPTOR node would be truncated")
    vectors = np.asarray(encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    if (vectors.ndim != 2 or vectors.shape[0] != len(texts) or not vectors.shape[1]
            or not np.isfinite(vectors).all() or not np.any(vectors, axis=1).all()):
        raise ValueError("E5 returned invalid RAPTOR embeddings")
    return vectors


def _partition(nodes, members, get_text, tokens, limit, dimension, threshold, depth=0):
    """Use official soft clustering, with a bounded split for oversized unsplittable clusters."""
    if len(nodes) <= dimension + 1:
        groups = [nodes]
    else:
        embeddings = np.asarray([node.embeddings["e5"] for node in nodes])
        groups_by_label = defaultdict(list)
        for node, labels in zip(nodes, members(embeddings, dim=dimension, threshold=threshold), strict=True):
            if not len(labels):
                raise ValueError("Official RAPTOR clustering lost a node; check the pinned patch")
            for label in labels:
                groups_by_label[label].append(node)
        groups = [groups_by_label[label] for label in sorted(groups_by_label)]
    result = []
    for group in groups:
        if len(tokens.encode(get_text(group))) <= limit:
            result.append(group)
        elif dimension + 1 < len(group) < len(nodes) and depth < 5:
            result.extend(_partition(group, members, get_text, tokens, limit, dimension, threshold, depth + 1))
        else:
            current = []
            for node in group:
                if len(tokens.encode(get_text([node]))) > limit:
                    raise ValueError("A single RAPTOR node exceeds the cluster budget")
                if current and len(tokens.encode(get_text([*current, node]))) > limit:
                    result.append(current)
                    current = []
                current.append(node)
            result.append(current)
    return result


def _source_identity(messages):
    sources = source_messages(messages)
    domains = {domain for domain, _ in sources}
    if len(domains) != 1:
        raise ValueError("RAPTOR requires one domain per index")
    return domains.pop(), hashlib.sha256(json.dumps(list(sources.values()), ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RaptorNode:
    text: str
    layer: int
    children: tuple[int, ...]
    source_id: str | None


@dataclass
class RaptorTree:
    domain: str
    source_sha256: str
    config: RaptorConfig
    provenance: dict
    nodes: list[RaptorNode]
    vectors: np.ndarray

    def save(self, path):
        path = Path(path)
        header = {"schema": 1, "upstream": UPSTREAM_COMMIT, "domain": self.domain,
                  "source_sha256": self.source_sha256, "config": asdict(self.config),
                  "provenance": self.provenance, "nodes": [asdict(node) for node in self.nodes]}
        payload = np.frombuffer(json.dumps(header, ensure_ascii=False, allow_nan=False).encode(), dtype=np.uint8)
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez(handle, manifest=payload, vectors=self.vectors)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as cache:
            manifest = json.loads(cache["manifest"].tobytes())
            vectors = cache["vectors"]
        if manifest.get("schema") != 1 or manifest.get("upstream") != UPSTREAM_COMMIT:
            raise ValueError("Unsupported RAPTOR cache manifest")
        nodes = [RaptorNode(row["text"], row["layer"], tuple(row["children"]), row["source_id"])
                 for row in manifest["nodes"]]
        return cls(manifest["domain"], manifest["source_sha256"], RaptorConfig(**manifest["config"]),
                   manifest["provenance"], nodes, vectors)


def build_raptor(messages, encoder, tokenizer, summarize, *, config=RaptorConfig(), provenance=None):
    """Use the pinned upstream cluster builder and provide SGMem-specific leaves and model hooks."""
    messages = list(messages)
    domain, digest = _source_identity(messages)
    from raptor import BaseEmbeddingModel, BaseSummarizationModel, ClusterTreeBuilder, ClusterTreeConfig, Node
    from raptor.cluster_utils import perform_clustering
    from raptor.utils import get_text

    tokens = _Tokens(tokenizer)
    embedding_lock = Lock()

    class E5Embedding(BaseEmbeddingModel):
        def create_embedding(self, text):
            with embedding_lock:
                return _embeddings(encoder, ["passage: " + text], 64)[0]

    class Summarizer(BaseSummarizationModel):
        def summarize(self, context, max_tokens=config.summary_tokens):
            if len(tokens.encode(context)) > config.cluster_tokens:
                raise ValueError("RAPTOR summary exceeds the cluster token budget")
            summary = summarize(context, max_tokens)
            if not isinstance(summary, str) or not summary.strip() or len(tokens.encode(summary)) > max_tokens:
                raise ValueError("Local RAPTOR summarizer returned empty or over-budget content")
            return summary.strip()

    class BoundedClustering:
        @staticmethod
        def perform_clustering(nodes, embedding_model_name, reduction_dimension):
            if embedding_model_name != "e5":
                raise ValueError("RAPTOR clustering expects E5 node embeddings")
            return _partition(nodes, perform_clustering, get_text, tokens, config.cluster_tokens,
                              reduction_dimension, config.threshold)

    builder = ClusterTreeBuilder(ClusterTreeConfig(
        tokenizer=tokens, max_tokens=config.leaf_tokens, num_layers=config.max_layers,
        summarization_length=config.summary_tokens, summarization_model=Summarizer(),
        embedding_models={"e5": E5Embedding()}, cluster_embedding_model="e5",
        reduction_dimension=config.reduction_dimension, clustering_algorithm=BoundedClustering,
        threshold=config.threshold,
    ))
    fragments = [(message["id"], chunk) for message in messages
                 for chunk in tokens.split(_content(message), config.leaf_tokens)]
    vectors = _embeddings(encoder, ["passage: " + text for _, text in fragments], 64)
    leaves = {index: Node(text, index, set(), {"e5": vectors[index]})
              for index, (_, text) in enumerate(fragments)}
    all_nodes, layers = dict(leaves), {0: list(leaves.values())}
    builder.construct_tree(leaves, all_nodes, layers, use_multithreading=config.summary_workers > 1,
                           max_workers=config.summary_workers)
    nodes = [RaptorNode(node.text, layer, tuple(sorted(node.children)),
                        fragments[node.index][0] if layer == 0 else None)
             for layer, layer_nodes in sorted(layers.items()) for node in sorted(layer_nodes, key=lambda n: n.index)]
    return RaptorTree(domain, digest, config, provenance or {}, nodes,
                      np.asarray([all_nodes[index].embeddings["e5"] for index in range(len(nodes))], dtype=np.float32))


class RaptorRetriever:
    def __init__(self, messages, tree, encoder, tokenizer, *, leaf_only=False, batch_size=64):
        messages = list(messages)
        if _source_identity(messages) != (tree.domain, tree.source_sha256):
            raise ValueError("RAPTOR cache does not match the original source messages")
        self.messages, self.tree, self.encoder = messages, tree, encoder
        self.tokens, self.leaf_only, self.batch_size = _Tokens(tokenizer), leaf_only, batch_size
        self.leaves = []
        fragments = defaultdict(list)
        if (batch_size < 1 or tree.vectors.ndim != 2 or tree.vectors.shape[0] != len(tree.nodes)
                or not tree.vectors.shape[1] or not np.isfinite(tree.vectors).all()
                or not np.any(tree.vectors, axis=1).all()):
            raise ValueError("Invalid cached RAPTOR embeddings or batch size")
        for index, node in enumerate(tree.nodes):
            if not isinstance(node.text, str) or not node.text.strip():
                raise ValueError(f"Empty RAPTOR node {index}")
            if type(node.layer) is not int or not 0 <= node.layer <= tree.config.max_layers:
                raise ValueError(f"Invalid RAPTOR layer at node {index}")
            if node.layer == 0 and not node.children and isinstance(node.source_id, str):
                self.leaves.append(index)
                fragments[node.source_id].append(node.text)
            elif (node.layer == 0 or node.source_id is not None or not node.children
                    or any(type(child) is not int or not 0 <= child < index
                           or tree.nodes[child].layer != node.layer - 1 for child in node.children)):
                raise ValueError(f"Invalid RAPTOR children or source at node {index}")
        if {source: "".join(chunks) for source, chunks in fragments.items()} != {
            message["id"]: _content(message) for message in messages
        }:
            raise ValueError("RAPTOR leaf provenance does not reconstruct the source messages")

    def _scores(self, query, asking_user_id):
        query = f"{asking_user_id} {query}" if asking_user_id else query
        vector = _embeddings(self.encoder, ["query: " + query], self.batch_size)[0]
        if vector.size != self.tree.vectors.shape[1]:
            raise ValueError("RAPTOR query and cached embeddings have different dimensions")
        scores = self.tree.vectors @ vector
        if not np.isfinite(scores).all():
            raise ValueError("RAPTOR similarities are not finite")
        return scores

    def _descendants(self, index):
        stack, visited, sources = [index], set(), set()
        while stack:
            child = stack.pop()
            if child in visited:
                continue
            visited.add(child)
            node = self.tree.nodes[child]
            if node.source_id is None:
                stack.extend(node.children)
            else:
                sources.add(node.source_id)
        return sorted(sources)

    def retrieve_nodes(self, query, top_k, *, asking_user_id=None, max_context_tokens=3500):
        """Return native nodes, with None requesting candidates before context budgeting."""
        if top_k <= 0 or (max_context_tokens is not None and max_context_tokens <= 0):
            raise ValueError("Positive native RAPTOR top-k and token budget are required")
        scores = self._scores(query, asking_user_id)
        indices = np.asarray(self.leaves) if self.leaf_only else np.arange(len(scores))
        hits, texts = [], []
        for index in indices[np.argsort(-scores[indices], kind="stable")[:top_k]]:
            node = self.tree.nodes[index]
            if max_context_tokens is not None:
                texts.append(node.text)
                if len(self.tokens.encode("\n\n".join(texts))) > max_context_tokens:
                    break
            hits.append({"node_id": int(index), "layer": node.layer, "text": node.text,
                         "children": list(node.children), "source_ids": self._descendants(int(index)),
                         "score": float(scores[index])})
        return hits

    def retrieve(self, query, top_k, *, asking_user_id=None):
        """Propagate summary similarities down the DAG and return unique original messages."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self._scores(query, asking_user_id)
        best = list(range(len(scores)))
        if not self.leaf_only:
            for index in reversed(range(len(best))):
                winner = best[index]
                for child in self.tree.nodes[index].children:
                    if (scores[winner], -winner) > (scores[best[child]], -best[child]):
                        best[child] = winner
        evidence = {}
        for leaf in self.leaves:
            source = self.tree.nodes[leaf].source_id
            candidate = float(scores[best[leaf]]), float(scores[leaf]), -leaf
            if source not in evidence or candidate > evidence[source]:
                evidence[source] = candidate
        ranked = sorted(self.messages, key=lambda message: evidence[message["id"]][:2], reverse=True)[:top_k]
        hits = []
        for message in ranked:
            score, direct_score, negative_leaf = evidence[message["id"]]
            leaf, winner = -negative_leaf, best[-negative_leaf]
            hits.append({"message": message, "score": score,
                         "trace": {"node_id": winner, "layer": self.tree.nodes[winner].layer,
                                   "leaf_id": leaf, "leaf_score": direct_score}})
        return hits
