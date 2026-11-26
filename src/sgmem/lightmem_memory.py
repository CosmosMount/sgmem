"""GroupMemBench adapter for the pinned, patched official LightMem pipeline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np


UPSTREAM_COMMIT = "80211869f55e9c2276a7e7f564356d6cb03b7d8c"
E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
E5_WEIGHTS = "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DIMENSIONS = 384
E5_MAX_TOKENS = 512
E5_LENGTH_POLICY = "lossless_passage_chunks_token_weighted_mean_512"
SLEEP_TIME_POLICY = "strictly_later_source_instant_utc"
SLEEP_RESPONSE_POLICY = "strict_json_schema_action_and_memory_max_16000_chars"
PATCH = Path(__file__).resolve().parents[2] / "patches" / "lightmem.patch"
PATCHED_SOURCE = {
    "src/lightmem/factory/memory_buffer/sensory_memory.py":
        "afda81858408c2de117ec472b09174c1a96b8628c633e21d42ae9ce26fe18a39",
    "src/lightmem/factory/memory_manager/openai.py":
        "b37ae2264fd5125d4588999c7098bc2197724161c71c50c2bdcfbf98ee14d6a0",
    "src/lightmem/factory/pre_compressor/llmlingua_2.py":
        "82dff55fd1a62db2920e7e4497710cdd9c288ad1e90975b2ddf310b8e69abb88",
    "src/lightmem/factory/retriever/embeddingretriever/qdrant.py":
        "8e255fd66f9fced6378c5b4460a64dbacbde1061d21e0fae7410e7d87ed00eb7",
    "src/lightmem/memory/lightmem.py":
        "711da50d9f95b8fe2e0b3ffaa978920783bad1cd7f643841f5cf225f8ab1a5a7",
    "src/lightmem/memory/prompts.py":
        "48e46d94838589728fbc45ad3f8fe9e7d499b5a864e5af204831c67c821b5f13",
    "src/lightmem/memory/utils.py":
        "f112351acc31618a1031f3b98732e0a5f3f52fb8754d8f5d9ffc89afae0c370a",
}
SOURCE_FIELDS = {
    "domain", "group", "id", "content", "speaker", "timestamp", "role",
    "reply_to", "phase_name", "topic",
}
STAGES = ("pre", "post")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _instant(timestamp):
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    instant = datetime.fromisoformat(timestamp)
    return instant.replace(tzinfo=timezone.utc) if instant.tzinfo is None else instant.astimezone(timezone.utc)


def _normalize(messages):
    rows = []
    seen = set()
    for row in messages:
        if not isinstance(row, dict) or set(row) != SOURCE_FIELDS:
            raise ValueError("LightMem input must be a normalized, label-free message pack")
        for key in ("domain", "group", "id", "content", "timestamp"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Missing LightMem message field: {key}")
        try:
            _instant(row["timestamp"])
        except ValueError as exc:
            raise ValueError(f"Invalid source timestamp for {row['id']}") from exc
        if any(value is not None and not isinstance(value, str) for key, value in row.items()
               if key in {"speaker", "role", "reply_to", "phase_name", "topic"}):
            raise ValueError(f"Invalid source metadata for {row['id']}")
        key = row["domain"], row["id"]
        if key in seen:
            raise ValueError(f"Duplicate source ID: {key}")
        seen.add(key)
        rows.append(dict(row))
    if not rows:
        raise ValueError("LightMem requires source messages")
    return sorted(rows, key=lambda row: (row["domain"], row["group"], _instant(row["timestamp"]), row["id"]))


def _turn(row):
    content = html.unescape(row["content"]).strip()
    if not content:
        raise ValueError(f"Empty source message after normalization: {row['id']}")
    timestamp = _instant(row["timestamp"]).isoformat()
    timing = {"time_stamp": timestamp, "source_time_stamp": timestamp}
    return [
        {
            "role": "user", "content": content, "dia_id": row["id"],
            "speaker_id": row["speaker"], "speaker_name": row["speaker"],
            **timing,
        },
        {"role": "assistant", "content": "", **timing},
    ]


def _git(upstream, *args):
    result = subprocess.run(
        ["git", "-C", str(upstream), *args], capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"LightMem source check failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _verify_upstream(upstream, patch=PATCH):
    upstream = Path(upstream).resolve()
    if _git(upstream, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError(f"Use official LightMem revision {UPSTREAM_COMMIT}")
    patch_bytes = Path(patch).read_bytes()
    changed = set(_git(upstream, "diff", "HEAD", "--name-only").splitlines())
    if changed != PATCHED_SOURCE.keys():
        raise ValueError("LightMem worktree must contain the required tracked patch")
    if _git(upstream, "ls-files", "--others", "--exclude-standard", "--", "src"):
        raise ValueError("LightMem checkout has unexpected untracked source files")
    _git(upstream, "apply", "--reverse", "--check", "--", str(Path(patch).resolve()))
    for relative_path, digest in PATCHED_SOURCE.items():
        source = (upstream / relative_path).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError(f"LightMem patched source differs: {relative_path}")
    return upstream, hashlib.sha256(patch_bytes).hexdigest()


def _model_path(path):
    path = Path(path).resolve()
    if not (path / "config.json").is_file():
        raise ValueError(f"Expected a local model directory: {path}")
    return path


def _file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _verify_e5(path):
    path = _model_path(path)
    if _file_digest(path / "model.safetensors") != E5_WEIGHTS:
        raise ValueError("LightMem E5 weights differ from the frozen E5-small-v2 comparator")
    return path


def _endpoint(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.port is None or parsed.path.rstrip("/") != "/v1"
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use a loopback OpenAI-compatible http://host:port/v1 endpoint")
    if os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("Unset OPENROUTER_API_KEY for the local LightMem worker")
    return url.rstrip("/")


class E5Encoder:
    """Encode LightMem passages, symmetric turn pairs, and retrieval queries."""

    def __init__(self, model):
        self.model = model
        self.calls = Counter()
        self.length_stats = Counter()
        if model.get_sentence_embedding_dimension() != DIMENSIONS:
            raise ValueError(f"LightMem requires the {DIMENSIONS}-dimensional E5-small-v2")
        self.tokenizer = model.tokenizer
        if model.max_seq_length != E5_MAX_TOKENS or not self.tokenizer.is_fast:
            raise ValueError(f"LightMem requires the fast E5 tokenizer with a {E5_MAX_TOKENS}-token limit")
        self._lock = Lock()

    def _token_count(self, text, *, special=True):
        return len(self.tokenizer.encode(text, add_special_tokens=special, truncation=False))

    def _split_passage(self, prefix, text):
        chunks = []
        while self._token_count(prefix + text) > E5_MAX_TOKENS:
            offsets = self.tokenizer(prefix + text, add_special_tokens=True, truncation=False,
                                     return_offsets_mapping=True)["offset_mapping"]
            for _, end in reversed(offsets[:E5_MAX_TOKENS]):
                cut = end - len(prefix)
                if 0 < cut < len(text) and self._token_count(prefix + text[:cut]) <= E5_MAX_TOKENS:
                    break
            else:
                raise ValueError("E5 could not split an oversized LightMem passage")
            chunks.append(text[:cut])
            text = text[cut:]
        return [*chunks, text]

    def _encode(self, prefix, text, use):
        with self._lock:
            prefixed = prefix + text
            length = self._token_count(prefixed)
            chunks = None
            if length > E5_MAX_TOKENS:
                if use != "passage":
                    raise ValueError(f"E5 LightMem {use} has {length} tokens and exceeds {E5_MAX_TOKENS}")
                chunks = self._split_passage(prefix, text)
                prefixed = [prefix + chunk for chunk in chunks]
            self.calls[use] += 1
            vectors = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
            array = np.asarray(vectors, dtype=np.float32)
            if chunks is not None:
                if array.shape != (len(chunks), DIMENSIONS) or not np.isfinite(array).all():
                    raise ValueError("Invalid E5 LightMem passage chunk embeddings")
                weights = [self._token_count(chunk, special=False) for chunk in chunks]
                average = np.average(array, axis=0, weights=weights)
                norm = np.linalg.norm(average)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("Invalid E5 LightMem pooled passage embedding")
                array = (average / norm).astype(np.float32)
                self.length_stats["long_passages"] += 1
                self.length_stats["long_passage_chunks"] += len(chunks)
        if array.shape != (DIMENSIONS,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid E5 {use} embedding")
        return array

    def embed(self, text):
        return self._encode("passage: ", text, "passage").tolist()

    def embed_turn(self, text):
        return self._encode("query: ", text, "turn").tolist()

    def query(self, text):
        return self._encode("query: ", text, "query")

    def get_stats(self):
        return {"total_calls": sum(self.calls.values()), "total_tokens": 0}


def _config(collection, *, endpoint, model, embedding_model, compressor_model, device, seed):
    return {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": str(compressor_model), "device_map": device, "use_llmlingua2": True,
                },
                "compress_config": {"instruction": "", "rate": 0.6, "target_token": -1},
            },
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {"model_name": "llmlingua-2", "configs": {}},
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "extraction_mode": "flat",
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": model, "api_key": os.getenv("SGMEM_API_KEY") or "EMPTY",
                "openai_base_url": endpoint, "seed": seed, "max_tokens": 4096,
                "temperature": 0.2, "top_p": 1.0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        },
        "index_strategy": "embedding",
        "retrieve_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": str(embedding_model), "embedding_dims": DIMENSIONS,
                "model_kwargs": {"device": device, "local_files_only": True},
            },
        },
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {"collection_name": collection, "path": ":memory:",
                        "embedding_model_dims": DIMENSIONS, "on_disk": True},
        },
        "update": "offline",
    }


def _complete_response(create, params):
    """Retry a truncated response once; count both attempts in token usage."""
    totals = Counter()
    for attempt in range(2):
        response = create(**dict(params, max_tokens=params["max_tokens"] * (attempt + 1)))
        if not response.choices or response.usage is None:
            raise ValueError("LightMem response has no choice or token usage, "
                             f"request_messages={_json(params.get('messages'))}")
        choice = response.choices[0]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[key] += getattr(response.usage, key)
        if choice.finish_reason == "stop":
            for key, value in totals.items():
                setattr(response.usage, key, value)
            return response
        if choice.finish_reason != "length" or attempt:
            raise ValueError(
                f"Incomplete LightMem generation: finish_reason={choice.finish_reason}, "
                f"attempts={attempt + 1}, usage={dict(totals)}, "
                f"response_tail={(choice.message.content or '')[-300:]!r}, "
                f"request_messages={_json(params.get('messages'))}"
            )


class _NativeWorker:
    """Reuse loaded models while giving each channel its own memory buffers and store."""

    def __init__(self, upstream, options):
        source = str(upstream / "src")
        sys.path.insert(0, source)
        from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant, QdrantConfig
        from lightmem.memory import lightmem as implementation

        if Path(implementation.__file__).resolve() != Path(source) / "lightmem/memory/lightmem.py":
            raise ValueError("The imported LightMem module is outside the pinned checkout")
        self.implementation = implementation
        self.qdrant = Qdrant
        self.qdrant_config = QdrantConfig
        self.memory = None
        self.options = options

    def new_group(self, name):
        if self.memory is None:
            self.memory = self.implementation.LightMemory.from_config(_config(name, **self.options))
            completions = self.memory.manager.client.chat.completions
            create = completions.create
            completions.create = lambda **params: _complete_response(create, params)
            compressor = self.memory.compressor._compressor
            if compressor.model.config.model_type != "bert":
                raise ValueError("This LightMem profile requires the pinned LLMLingua-2 BERT compressor")
            # LLMLingua 0.2.2 selects WordPiece rules by name, even for a local snapshot.
            compressor.model_name = "bert-base-multilingual-cased"
            self.memory.text_embedder = E5Encoder(self.memory.text_embedder.model)
        else:
            self.memory.embedding_retriever.client.close()
            self.memory.embedding_retriever = self.qdrant(self.qdrant_config(
                collection_name=name, path=":memory:", embedding_model_dims=DIMENSIONS, on_disk=True,
            ))
            self.memory._sgmem_insert_index = 0
            self.memory._sgmem_topic_index = 0
        return self.memory


def _stem(domain, group):
    return hashlib.sha256(_json([domain, group]).encode()).hexdigest()[:24]


def _point_sources(point, channel):
    identity, payload = point.get("id"), point.get("payload", {})
    anchor = payload.get("anchor_dia_id")
    sources = payload.get("source_ids")
    if (not isinstance(identity, str) or not identity or not isinstance(anchor, str)
            or not isinstance(sources, list) or not sources or sources[0] != anchor
            or any(not isinstance(source, str) or source not in channel for source in sources)
            or len(sources) != len(set(sources))):
        raise ValueError(f"Invalid LightMem source lineage for entry {identity}")
    memory = payload.get("memory")
    if not isinstance(memory, str) or not memory.strip():
        raise ValueError(f"Missing factual content in LightMem entry {identity}")
    return sources


def _snapshot(memory, channel):
    points = sorted(memory.embedding_retriever.get_all(), key=lambda entry: entry["id"])
    entries = []
    vectors = []
    for point in points:
        sources = _point_sources(point, channel)
        payload = point["payload"]
        anchor = channel[sources[0]]
        entries.append({
            "id": point["id"], "text": payload["memory"], "source_ids": sources,
            "anchor_dia_id": sources[0], "time_stamp": anchor["timestamp"],
            "speaker_id": anchor["speaker"], "group": anchor["group"], "domain": anchor["domain"],
            "topic_id": payload.get("topic_id"), "entry_type": payload.get("entry_type"),
        })
        vectors.append(point["vector"])
    matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, DIMENSIONS)
    if len(entries) != len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Invalid LightMem vector snapshot")
    if len(matrix) and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise ValueError("Expected normalized LightMem entry vectors")
    return entries, matrix


def _build_group(memory, rows, score_threshold):
    channel = {row["id"]: row for row in rows}
    compression = Counter()
    source_compress = memory.compressor.compress
    source_propose_cut = memory.segmenter.propose_cut
    source_cut_buffer = memory.senmem_buffer_manager.cut_with_segmenter
    source_extract = memory.manager.meta_text_extract
    extracted_ids = []

    def counted_compress(messages, tokenizer):
        before = [message["content"] for message in messages if message["role"] == "user"]
        compression["input_turns"] += len(before)
        compression["input_tokens"] += sum(len(tokenizer.encode(text)) for text in before)
        compact = source_compress(messages, tokenizer)
        after = [message for message in compact if message["role"] == "user"]
        compression["compressed_tokens"] += sum(len(tokenizer.encode(row["content"])) for row in after)
        compression["changed_turns"] += sum(old != current["content"] for old, current in zip(before, after))
        return compact

    def counted_propose_cut(texts):
        compression["segmentation_calls"] += 1
        cuts = source_propose_cut(texts)
        compression["coarse_boundaries"] += len(cuts)
        return cuts

    def counted_cut_buffer(*args, **kwargs):
        segments = source_cut_buffer(*args, **kwargs)
        compression["released_segments"] += len(segments)
        return segments

    def counted_extract(*, extract_list, **kwargs):
        extracted_ids.extend(message["dia_id"] for batch in extract_list for segment in batch
                             for message in segment if message["role"] == "user")
        return source_extract(extract_list=extract_list, **kwargs)

    counters = Counter()
    embedding_start = memory.text_embedder.calls.copy()
    length_start = memory.text_embedder.length_stats.copy()
    token_start = memory.get_token_statistics()["llm"]
    memory.compressor.compress = counted_compress
    memory.segmenter.propose_cut = counted_propose_cut
    memory.senmem_buffer_manager.cut_with_segmenter = counted_cut_buffer
    memory.manager.meta_text_extract = counted_extract
    started = perf_counter()
    try:
        for row in rows:
            counts = memory.add_memory(_turn(row))
            counters["extraction_calls"] += counts["api_call_nums"]
        counts = memory.add_memory([], force_segment=True, force_extract=True)
        counters["extraction_calls"] += counts["api_call_nums"]
    finally:
        memory.compressor.compress = source_compress
        memory.segmenter.propose_cut = source_propose_cut
        memory.senmem_buffer_manager.cut_with_segmenter = source_cut_buffer
        memory.manager.meta_text_extract = source_extract
    add_seconds = perf_counter() - started

    if (compression["input_turns"] != len(rows) or memory.senmem_buffer_manager.buffer
            or memory.senmem_buffer_manager.big_buffer or memory.shortmem_buffer_manager.buffer):
        raise RuntimeError("LightMem did not finish ingesting its complete channel")
    if extracted_ids != list(channel):
        raise RuntimeError("LightMem extraction did not receive every source turn in channel order")
    pre = _snapshot(memory, channel)
    events = []
    event_lock = Lock()
    update = memory.manager._call_update_llm

    def audited_update(prompt, target, candidates):
        target_sources = _point_sources(target, channel)
        candidate_sources = [_point_sources(point, channel) for point in candidates]
        result = update(prompt, target, candidates)
        event = {
            "target": target["id"], "source_ids": target_sources,
            "candidates": [point["id"] for point in candidates],
            "candidate_source_ids": candidate_sources, "action": result["action"],
            "usage": result["usage"],
        }
        with event_lock:
            events.append(event)
        return result

    started = perf_counter()
    memory.construct_update_queue_all_entries(top_k=20, keep_top_n=10)
    queue_seconds = perf_counter() - started

    memory.manager._call_update_llm = audited_update
    started = perf_counter()
    try:
        memory.offline_update_all_entries(score_threshold=score_threshold)
    finally:
        memory.manager._call_update_llm = update
    sleep_seconds = perf_counter() - started
    post = _snapshot(memory, channel)
    _check_updates(pre[0], post[0], events)
    token_end = memory.get_token_statistics()["llm"]
    tokens = {phase: {key: value - token_start[phase][key] for key, value in token_end[phase].items()}
              for phase in ("add_memory", "update")}
    metrics = {
        **dict(counters), **dict(compression), "rows": len(rows),
        "pre_entries": len(pre[0]), "post_entries": len(post[0]),
        "extraction_source_turns": len(extracted_ids),
        "source_coverage": {
            stage: {
                "anchors": len({entry["anchor_dia_id"] for entry in entries}),
                "lineage": len({source for entry in entries for source in entry["source_ids"]}),
            }
            for stage, entries in (("pre", pre[0]), ("post", post[0]))
        },
        "embedding_calls": dict(memory.text_embedder.calls - embedding_start),
        "embedding_length": {key: memory.text_embedder.length_stats[key] - length_start[key]
                             for key in ("long_passages", "long_passage_chunks")},
        "llm": tokens,
        "add_seconds": add_seconds, "queue_seconds": queue_seconds, "sleep_seconds": sleep_seconds,
        "update_actions": dict(Counter(event["action"] for event in events)),
    }
    return pre, post, sorted(events, key=lambda event: event["target"]), metrics


def _check_updates(pre, post, events):
    source = {row["id"]: row for row in pre}
    surviving = {row["id"]: row for row in post}
    by_target = {event["target"]: event for event in events}
    if (len(source) != len(pre) or len(surviving) != len(post) or len(by_target) != len(events)
            or not surviving.keys() <= source.keys() or not by_target.keys() <= source.keys()):
        raise RuntimeError("Inconsistent LightMem update IDs")
    for identity, event in by_target.items():
        candidates = event["candidates"]
        if (event["action"] not in ("update", "delete", "ignore")
                or event["source_ids"] != source[identity]["source_ids"]
                or not candidates or identity in candidates or len(candidates) != len(set(candidates))
                or any(candidate not in source for candidate in candidates)
                or event["candidate_source_ids"] != [source[candidate]["source_ids"] for candidate in candidates]):
            raise RuntimeError(f"Invalid LightMem update inputs for {identity}")
        target_time = _instant(source[identity]["time_stamp"])
        if any(_instant(source[candidate]["time_stamp"]) <= target_time for candidate in candidates):
            raise RuntimeError(f"LightMem sleep candidate is not strictly newer than {identity}")
    for identity, row in source.items():
        event = by_target.get(identity)
        actual = surviving.get(identity)
        if event is None or event["action"] == "ignore":
            if actual != row:
                raise RuntimeError(f"Unexpected change to LightMem entry {identity}")
            continue
        if event["action"] == "delete":
            if actual is not None:
                raise RuntimeError(f"LightMem deletion did not remove {identity}")
            continue
        expected = list(dict.fromkeys(
            item for items in [row["source_ids"], *event["candidate_source_ids"]] for item in items
        ))
        if actual is None or actual["source_ids"] != expected:
            raise RuntimeError(f"LightMem did not record update provenance for {identity}")


def _write_group(path, result, domain, group):
    pre, post, events, metrics = result
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(
            output, pre_vectors=pre[1], post_vectors=post[1],
            metadata=_json({"domain": domain, "group": group, "pre": pre[0], "post": post[0],
                            "updates": events, "metrics": metrics}),
        )
    temporary.replace(path)


def _read_group(path, domain, group):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        matrices = {stage: saved[f"{stage}_vectors"] for stage in STAGES}
    if (metadata["domain"], metadata["group"]) != (domain, group):
        raise ValueError(f"Wrong LightMem channel identity in {path}")
    for stage, matrix in matrices.items():
        if (matrix.shape != (len(metadata[stage]), DIMENSIONS)
                or not np.isfinite(matrix).all()):
            raise ValueError(f"Invalid LightMem {stage} array in {path}")
    _check_updates(metadata["pre"], metadata["post"], metadata["updates"])
    return metadata, matrices


def _write_manifest(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(data) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_index(messages, output, protocol, new_group, *, resume=False, score_threshold=0.9):
    rows = _normalize(messages)
    source_bytes = "".join(_json(row) + "\n" for row in rows).encode()
    groups = defaultdict(list)
    for row in rows:
        groups[row["domain"], row["group"]].append(row)
    output = Path(output)
    manifest_path = output / "manifest.json"
    source_path = output / "sources.jsonl"
    expected = {
        "protocol": {**protocol, "e5_length_policy": E5_LENGTH_POLICY, "sleep_time_policy": SLEEP_TIME_POLICY},
        "sources_sha256": hashlib.sha256(source_bytes).hexdigest(), "rows": len(rows),
        "groups": [{"domain": domain, "group": group, "file": _stem(domain, group) + ".npz"}
                   for domain, group in groups],
    }
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("LightMem resume inputs or configuration changed")
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected["sources_sha256"]:
            raise ValueError("LightMem source sidecar changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        with source_path.open("xb") as handle:
            handle.write(source_bytes)
        _write_manifest(manifest_path, {**expected, "status": "building", "metrics": []})
    metrics = []
    for item in expected["groups"]:
        domain, group = item["domain"], item["group"]
        path = output / item["file"]
        if path.exists():
            saved, _ = _read_group(path, domain, group)
            metrics.append(saved["metrics"])
            continue
        started = perf_counter()
        status = "failed"
        try:
            result = _build_group(new_group(path.stem), groups[domain, group], score_threshold)
            _write_group(path, result, domain, group)
            metrics.append(result[3])
            status = "complete"
        finally:
            with (output / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(_json({"domain": domain, "group": group, "status": status,
                                    "seconds": perf_counter() - started}) + "\n")
    complete = {**expected, "status": "complete", "metrics": metrics}
    _write_manifest(manifest_path, complete)
    return complete


def build_lightmem(messages, output, *, upstream, endpoint, model, embedding_model, compressor_model,
                   compressor_revision, device="cuda", seed=42, score_threshold=0.9, resume=False):
    """Build the full native LightMem memory twice, before and after its sleep pass."""
    if not isinstance(model, str) or not model.strip() or not compressor_revision.strip():
        raise ValueError("Record the served Qwen model and LLMLingua-2 revision")
    if not 0 <= score_threshold <= 1:
        raise ValueError("LightMem sleep cosine threshold must lie in [0, 1]")
    upstream, patch_hash = _verify_upstream(upstream)
    embedding_model, compressor_model = _verify_e5(embedding_model), _model_path(compressor_model)
    compressor_weights = compressor_model / "model.safetensors"
    if not compressor_weights.is_file():
        raise ValueError("Use the pinned LLMLingua-2 safetensors checkpoint")
    options = {
        "endpoint": _endpoint(endpoint), "model": model, "embedding_model": str(embedding_model),
        "compressor_model": str(compressor_model), "device": device, "seed": seed,
    }
    protocol = {
        "adapter_sha256": _file_digest(Path(__file__)),
        **options, "upstream_commit": UPSTREAM_COMMIT, "patch_sha256": patch_hash,
        "embedding_revision": E5_REVISION, "qa_revision": QWEN_REVISION,
        "embedding_weights_sha256": E5_WEIGHTS,
        "compressor_revision": compressor_revision,
        "compressor_config_sha256": _file_digest(compressor_model / "config.json"),
        "compressor_weights_sha256": _file_digest(compressor_weights),
        "packages": {name: version(name) for name in (
            "numpy", "torch", "transformers", "sentence-transformers", "llmlingua",
            "openai", "qdrant-client", "pydantic", "tiktoken", "httpx", "pyyaml",
        )},
        "score_threshold": score_threshold, "rate": 0.6, "stm_tokens": 512,
        "generation_policy": "4096_tokens_retry_length_once_at_8192",
        "sleep_response_policy": SLEEP_RESPONSE_POLICY,
        "update_top_k": 20, "update_keep_n": 10,
    }
    worker = _NativeWorker(upstream, options)
    return _build_index(messages, output, protocol, worker.new_group,
                        resume=resume, score_threshold=score_threshold)


class LightMemIndex:
    """Domain-wide cosine ranking over a complete LightMem memory snapshot."""

    def __init__(self, entries, vectors, messages, encoder):
        self.entries, self.vectors, self.messages, self.encoder = entries, vectors, messages, encoder
        if len({entry["id"] for entry in entries}) != len(entries):
            raise ValueError("Duplicate LightMem IDs in the domain snapshot")

    @classmethod
    def load(cls, path, domain, *, stage="post", encoder=None, device="cpu"):
        if stage not in STAGES:
            raise ValueError(f"Unknown LightMem snapshot: {stage}")
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("LightMem index is incomplete")
        policy = manifest["protocol"]
        if (policy.get("e5_length_policy") != E5_LENGTH_POLICY
                or policy.get("sleep_time_policy") != SLEEP_TIME_POLICY):
            raise ValueError("Rebuild this LightMem index with the current encoding and sleep time policies")
        source_bytes = (path / "sources.jsonl").read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != manifest["sources_sha256"]:
            raise ValueError("LightMem source sidecar changed")
        rows = [row for row in map(json.loads, source_bytes.splitlines()) if row["domain"] == domain]
        messages = {row["id"]: row for row in rows}
        entries = []
        vectors = []
        for item in manifest["groups"]:
            if item["domain"] != domain:
                continue
            group = item["group"]
            metadata, matrices = _read_group(path / item["file"], domain, group)
            for entry in metadata[stage]:
                parent = messages.get(entry["anchor_dia_id"])
                if (parent is None or parent["group"] != group or parent["timestamp"] != entry["time_stamp"]
                        or parent["speaker"] != entry["speaker_id"] or entry["domain"] != domain
                        or not entry["source_ids"] or entry["source_ids"][0] != parent["id"]
                        or len(set(entry["source_ids"])) != len(entry["source_ids"])
                        or any(messages.get(source, {}).get("group") != group for source in entry["source_ids"])):
                    raise ValueError(f"Invalid projected LightMem source for {entry['id']}")
            entries.extend(metadata[stage])
            vectors.append(matrices[stage])
        if not messages or not vectors:
            raise ValueError(f"LightMem index has no channel for {domain}")
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            model_path = _verify_e5(manifest["protocol"]["embedding_model"])
            encoder = E5Encoder(SentenceTransformer(str(model_path), device=device, local_files_only=True))
        return cls(entries, np.vstack(vectors), messages, encoder)

    def _rank(self, query, asking_user_id):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A nonempty LightMem query is required")
        text = f"{asking_user_id} {query}" if asking_user_id else query
        vector = np.asarray(self.encoder.query(text), dtype=np.float32)
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all():
            raise ValueError("Invalid E5 query embedding")
        scores = self.vectors @ vector
        order = sorted(range(len(scores)), key=lambda pos: (-float(scores[pos]), self.entries[pos]["id"]))
        return scores, order

    def retrieve(self, query, top_k, *, asking_user_id=None):
        """Rank distinct original messages by their best scoring memory entry."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores, order = self._rank(query, asking_user_id)
        hits = []
        seen = set()
        for pos in order:
            entry = self.entries[pos]
            for source in entry["source_ids"]:
                if source in seen:
                    continue
                hits.append({
                    "message": self.messages[source], "score": float(scores[pos]), "unit_id": entry["id"],
                })
                seen.add(source)
                if len(hits) == top_k:
                    return hits
        raise ValueError(f"LightMem found {len(hits)} unique original messages, fewer than top_k={top_k}")

    def retrieve_native(self, query, top_k, *, asking_user_id=None):
        """Return factual entries and their source IDs for the native context track."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores, order = self._rank(query, asking_user_id)
        return [dict(self.entries[pos], score=float(scores[pos])) for pos in order[:top_k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("messages", type=Path, help="Prepared sources JSONL, including domain")
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream", type=Path, required=True, help="Pinned, patched official LightMem checkout")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Qwen3-8B name served by the local endpoint")
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--compressor-model", type=Path, required=True)
    parser.add_argument("--compressor-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    options = vars(args).copy()
    messages, output = options.pop("messages"), options.pop("output")
    result = build_lightmem(_read_rows(messages), output, **options)
    print(_json({"output": str(output), "rows": result["rows"], "groups": len(result["groups"])}))


if __name__ == "__main__":
    main()
