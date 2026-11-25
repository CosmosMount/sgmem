"""Run the pinned HippoRAG 2 retriever over one GroupMemBench domain."""

import argparse
import ast
import hashlib
from importlib import metadata
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
from threading import Lock
from urllib.parse import urlsplit

import numpy as np

from sgmem.propositions import source_messages
from sgmem.retrieval import _content
from sgmem.runner import write_json


UPSTREAM_COMMIT = "d5c8329422e0a0b834a15874545cb6a74b4f9b26"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:18951/v1"
FILTER_FIELD = re.compile(r"^\[\[ ## (\w+) ## \]\]\s*$")
OPENIE_SCOPE = (
    "Extract only from the final passage. The examples illustrate the output format only. "
    "Do not copy their entities or facts. Preserve negations, conditions and uncertainty. "
    "Do not turn questions or requests into facts or guess who an unresolved pronoun refers to. "
    "Return distinct items only. Do not repeat a relation, and do not use a bare conjunction as a relation."
)


class RecognitionError(RuntimeError):
    def __init__(self, trace):
        super().__init__(trace["error"])
        self.trace = trace


class OpenIEError(RuntimeError):
    def __init__(self, trace):
        super().__init__(trace["error"])
        self.trace = trace


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_upstream(module):
    # Support an installed VCS checkout or an unmodified source checkout on PYTHONPATH.
    try:
        distribution = metadata.distribution("hipporag")
        installed = Path(distribution.locate_file("hipporag/__init__.py")).resolve()
        if installed == Path(module.__file__).resolve():
            origin = json.loads(distribution.read_text("direct_url.json") or "{}")
            if origin.get("vcs_info", {}).get("commit_id") == UPSTREAM_COMMIT:
                return
    except metadata.PackageNotFoundError:
        pass

    checkout = Path(module.__file__).resolve().parents[2]
    command = ["git", "-C", str(checkout)]
    try:
        head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        clean = subprocess.run([*command, "diff", "--quiet", "HEAD", "--", "src/hipporag"],
                               check=False, stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.CalledProcessError):
        head, clean = None, False
    if head != UPSTREAM_COMMIT or not clean:
        raise ValueError(f"Use an unmodified HippoRAG checkout at {UPSTREAM_COMMIT}")


def _parse_filter(response):
    """Validate the official filter output field before candidate matching."""
    sections = {}
    current = None
    for line in response.splitlines():
        match = FILTER_FIELD.fullmatch(line.strip())
        if match:
            current = match[1]
            if current in sections:
                raise ValueError(f"Duplicate recognition response field: {current}")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    if "fact_after_filter" not in sections:
        raise ValueError("Recognition response has no fact_after_filter field")
    payload = "\n".join(sections["fact_after_filter"]).strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = ast.literal_eval(payload)
    triples = value.get("fact") if isinstance(value, dict) else None
    if not isinstance(triples, list) or not all(
        isinstance(triple, list) and len(triple) == 3
        and all(isinstance(field, str) and field.strip() for field in triple)
        for triple in triples
    ):
        raise ValueError("Recognition response must contain a fact list of three-string triples")
    return triples


def _openie_profile():
    profile = {}
    for stage, field in (("ner", "named_entities"), ("triple_extraction", "triples")):
        limit = 64 if stage == "ner" else 32
        items = {"type": "string", "minLength": 1}
        structure = f'Return only a JSON object with "{field}" as an array of nonempty strings.'
        if stage == "triple_extraction":
            items = {"type": "array", "minItems": 3, "maxItems": 3, "items": items}
            structure = (
                'Return only a JSON object with "triples" as an array. Each triple has exactly '
                "three nonempty strings in subject, predicate, object order."
            )
        profile[stage] = {
            "instruction": f"{OPENIE_SCOPE} {structure} Return at most {limit} items, then stop.",
            "frequency_penalty": 0.15,
            "entity_grounding": "normalized_word_spans_in_current_passage" if stage == "ner" else None,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"hipporag_{stage}", "strict": True,
                    "schema": {
                        "type": "object", "required": [field], "additionalProperties": False,
                        "properties": {field: {"type": "array", "items": items, "maxItems": limit}},
                    },
                },
            },
        }
    return profile


def _validate_openie_json(stage, response):
    payload = json.loads(response)
    field = "named_entities" if stage == "ner" else "triples"
    if not isinstance(payload, dict) or set(payload) != {field}:
        raise ValueError(f"expected an OpenIE object with only {field}")
    values = payload[field]
    limit = 64 if stage == "ner" else 32
    if isinstance(values, list) and len(values) > limit:
        raise ValueError(f"OpenIE {field} exceeds its {limit}-item limit")
    if stage == "ner":
        if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values):
            raise ValueError("expected named_entities as a list of nonempty strings")
        return values, 0
    if not isinstance(values, list) or not all(
            isinstance(triple, list) and len(triple) == 3
            and all(isinstance(item, str) for item in triple) for triple in values):
        raise ValueError("expected triples as a list of three-string arrays")
    usable = [triple for triple in values if all(item.strip() for item in triple)]
    return usable, len(values) - len(usable)


def _entity_mentioned(entity, passage):
    normalize = lambda value: " ".join(re.findall(r"\w+", value.casefold()))
    name = normalize(entity)
    return bool(name) and f" {name} " in f" {normalize(passage)} "


def _guard_openie(engine):
    failure = None
    failure_lock = Lock()

    def checked(stage, infer):
        def extract(chunk_key, *args, **kwargs):
            nonlocal failure
            # The upstream executor can still dequeue work after the first invalid result.
            with failure_lock:
                if failure is not None:
                    raise OpenIEError(failure)
            row = infer(chunk_key, *args, **kwargs)
            response = getattr(row, "response", None)
            metadata = getattr(row, "metadata", None)
            try:
                if not isinstance(response, str) or not isinstance(metadata, dict):
                    raise ValueError("missing raw response or metadata")
                if metadata.get("error"):
                    raise ValueError(metadata["error"])
                if getattr(row, "chunk_id", None) != chunk_key:
                    raise ValueError(f"returned an unexpected chunk ID {getattr(row, 'chunk_id', None)!r}")
                if metadata.get("finish_reason") != "stop":
                    raise ValueError(f"finish reason {metadata.get('finish_reason')!r}")
                _validate_openie_json(stage, response)
            except ValueError as exc:
                source_ids = engine.chunk_metadata.get(chunk_key, {}).get("message_ids", [])
                with failure_lock:
                    if failure is None:
                        failure = {"stage": stage, "chunk_id": chunk_key, "message_ids": source_ids,
                                   "raw_output": response, "metadata": metadata,
                                   "error": f"OpenIE {stage} failed for chunk {chunk_key} (messages {source_ids}): {exc}"}
                raise OpenIEError(failure) from exc
            return row
        return extract

    engine.openie.ner = checked("ner", engine.openie.ner)
    engine.openie.triple_extraction = checked("triple_extraction", engine.openie.triple_extraction)


def _engine(settings, cache_dir, identity, rebuild):
    import hipporag
    from hipporag.embedding_model import BaseEmbeddingModel
    from hipporag.information_extraction.openie_openai import OpenIE
    from hipporag.llm import find_gateway
    from hipporag.rerank import DSPyFilter
    from hipporag.utils.config_utils import BaseConfig
    from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput, text_processing
    from sentence_transformers import SentenceTransformer

    _verify_upstream(hipporag)
    model_name = settings["llm_name"]
    if find_gateway(model_name) is not None or model_name.startswith(("Transformers/", "bedrock/", "bedrock-mantle/")):
        raise ValueError("Use a served Qwen model name compatible with the loopback endpoint")
    encoder = SentenceTransformer(settings["embedding_path"], device=settings["embedding_device"],
                                  local_files_only=True)
    trace_lock = Lock()

    class SchemaOpenIE(OpenIE):
        def _extract(self, stage, max_tokens, **template_values):
            response, usage = "", {}
            try:
                messages = self.prompt_template_manager.render(name=stage, **template_values)
                profile = _openie_profile()[stage]
                system = messages[0]
                system = {**system, "content": f"{system['content']}\n\n{profile['instruction']}"}
                response, metadata, cache_hit = self.llm_model.infer(
                    messages=[system, *messages[1:]], max_new_tokens=max_tokens,
                    response_format=profile["response_format"],
                    frequency_penalty=profile["frequency_penalty"],
                )
                usage = dict(metadata, cache_hit=cache_hit)
                if usage.get("finish_reason") == "stop":
                    values, dropped = _validate_openie_json(stage, response)
                    if dropped:
                        usage["dropped_blank_triples"] = dropped
                    return response, usage, values
            except Exception as exc:
                usage["error"] = str(exc)
            return response, usage, []

        def ner(self, chunk_key, passage):
            response, usage, entities = self._extract("ner", self.ner_max_tokens, passage=passage)
            accepted = list(dict.fromkeys(entity for entity in entities if _entity_mentioned(entity, passage)))
            unmatched = list(dict.fromkeys(entity for entity in entities if not _entity_mentioned(entity, passage)))
            if unmatched:
                record = {"chunk_id": chunk_key, "accepted": accepted, "unmatched": unmatched}
                with trace_lock, (Path(cache_dir) / "ner-grounding.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return NerRawOutput(chunk_id=chunk_key, response=response, metadata=usage,
                                unique_entities=accepted)

        def triple_extraction(self, chunk_key, passage, named_entities):
            response, usage, triples = self._extract(
                "triple_extraction", self.triple_max_tokens, passage=passage,
                named_entity_json=json.dumps({"named_entities": named_entities}),
            )
            if usage.get("dropped_blank_triples"):
                record = {"chunk_id": chunk_key, "raw_output": response,
                          "dropped_blank_triples": usage["dropped_blank_triples"]}
                with trace_lock, (Path(cache_dir) / "openie-blank-triples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            unique = dict.fromkeys(tuple(triple) for triple in triples)
            return TripleRawOutput(chunk_id=chunk_key, response=response, metadata=usage,
                                   triples=[list(triple) for triple in unique])

    class E5(BaseEmbeddingModel):
        query_instruction_mode = "shared"
        embedding_dim = 384

        def __init__(self, config, indexed_prefix):
            super().__init__(config)
            self.indexed_prefix = indexed_prefix

        def batch_encode(self, texts, *, instruction=None, norm=True):
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            prefix = "query: " if instruction is not None else self.indexed_prefix
            vectors = np.asarray(encoder.encode(
                [prefix + text for text in texts], batch_size=settings["batch_size"],
                normalize_embeddings=norm, convert_to_numpy=True, show_progress_bar=False,
            ), dtype=np.float32)
            if (vectors.shape != (len(texts), self.embedding_dim) or not np.isfinite(vectors).all()
                    or not np.any(vectors, axis=1).all()):
                raise ValueError("E5 returned invalid embeddings")
            return vectors

    class CheckedFilter(DSPyFilter):
        def __init__(self, engine):
            super().__init__(engine)
            self.infer = self.llm_infer_fn
            self.llm_infer_fn = self.checked_infer

        def checked_infer(self, *args, **kwargs):
            response, usage, cache_hit = self.infer(*args, **kwargs)
            self.trace.update(raw_output=response, usage=usage, cache_hit=cache_hit)
            if usage.get("finish_reason") != "stop":
                raise ValueError(f"Recognition finish reason: {usage.get('finish_reason')!r}")
            return response, usage, cache_hit

        def rerank(self, query, candidate_items, candidate_indices, len_after_rerank=None):
            self.trace = {"candidate_triples": candidate_items}
            try:
                fact_input = json.dumps({"fact": [list(item) for item in candidate_items]})
                triples = _parse_filter(self.llm_call(query, fact_input))
                self.trace["parsed_triples"] = triples
                # Indexed triples are already normalized using this same upstream function.
                candidates = {
                    tuple(item): (index, item)
                    for index, item in zip(candidate_indices, candidate_items, strict=True)
                }
                selected, seen = [], set()
                for triple in triples:
                    normalized = tuple(text_processing(triple))
                    if normalized not in candidates:
                        raise ValueError(f"Recognition returned a triple outside its candidates: {triple!r}")
                    if normalized not in seen:
                        selected.append(candidates[normalized])
                        seen.add(normalized)
                selected = selected[:len_after_rerank]
                return [index for index, _ in selected], [item for _, item in selected], {"confidence": None}
            except Exception as exc:
                raise RuntimeError(f"HippoRAG recognition failed: {exc}") from exc

    class TracedHippoRAG(hipporag.HippoRAG):
        def rerank_facts(self, query, query_fact_scores):
            self.rerank_filter.trace = {}
            indices, triples, trace = super().rerank_facts(query, query_fact_scores)
            if "error" in trace:
                raise RecognitionError({**self.rerank_filter.trace, "mode": "error",
                                        "error": f"HippoRAG fact selection failed: {trace['error']}"})
            before = trace["facts_before_rerank"]
            mode = "ppr" if triples else "dense_valid_empty" if before else "dense_no_facts"
            self.recognition = {**self.rerank_filter.trace, "mode": mode,
                                "candidate_triples": before, "selected_triples": triples}
            return indices, triples, trace

    config = BaseConfig(
        save_dir=str(cache_dir), openie_mode="online", llm_name=settings["llm_name"],
        llm_base_url=settings["llm_base_url"], openie_max_workers=settings["workers"],
        embedding_model_name="intfloat/e5-small-v2", embedding_batch_size=settings["batch_size"],
        embedding_max_seq_len=512, vector_store_type="parquet", temperature=0.0,
        linking_top_k=5, damping=0.5, passage_node_weight=0.05,
        synonymy_edge_sim_threshold=settings["synonym_threshold"],
        force_index_from_scratch=rebuild, save_openie=False,
    )
    engine = TracedHippoRAG(global_config=config, embedding_model=E5(config, "passage: "),
                           index_identity=identity)
    try:
        engine.entity_embedding_store.embedding_model = E5(config, "query: ")
        engine.rerank_filter = CheckedFilter(engine)
        openie = engine.openie
        engine.openie = SchemaOpenIE(
            openie.llm_model, max_workers=openie.max_workers, ner_max_tokens=openie.ner_max_tokens,
            triple_max_tokens=openie.triple_max_tokens,
        )
        _guard_openie(engine)
    except Exception:
        engine.close()
        raise
    return engine, hipporag.Chunk


class HippoRAGMemory:
    """Build or reopen the official graph and map ranked passages back to raw messages."""

    def __init__(self, messages, cache_dir, *, embedding_path, embedding_revision, llm_revision,
                 llm_name="Qwen/Qwen3-8B", llm_base_url=DEFAULT_LLM_BASE_URL,
                 embedding_device="cpu", batch_size=64, workers=8, synonym_threshold=0.8, dense_only=False):
        sources = source_messages(messages)
        domains = {key[0] for key in sources}
        if len(domains) != 1:
            raise ValueError("HippoRAG retrieval requires one domain per index")
        if min(batch_size, workers) < 1 or not -1 <= synonym_threshold <= 1:
            raise ValueError("Invalid batch size, worker count or synonym threshold")
        if not llm_revision or not embedding_revision:
            raise ValueError("Provide both pinned model revisions")
        endpoint = urlsplit(llm_base_url)
        if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}
                or endpoint.path.rstrip("/") != "/v1" or endpoint.username is not None
                or endpoint.query or endpoint.fragment):
            raise ValueError("Use a loopback OpenAI-compatible Qwen endpoint ending in /v1")

        self.domain = next(iter(domains))
        self.messages = {message_id: dict(message) for (_, message_id), message in sources.items()}
        self.source_order = {message_id: index for index, message_id in enumerate(self.messages)}
        self.passages = {}
        for message_id, message in self.messages.items():
            body = _content(message)
            self.passages.setdefault(body, []).append(message_id)
        self.cache_dir = Path(cache_dir).resolve()
        embedding_path = Path(embedding_path).resolve()
        if not (embedding_path / "config.json").is_file():
            raise ValueError(f"Expected a local E5 model directory: {embedding_path}")
        self.settings = {
            "llm_name": llm_name, "llm_base_url": llm_base_url.rstrip("/"), "llm_revision": llm_revision,
            "embedding_path": str(embedding_path), "embedding_revision": embedding_revision,
            "embedding_device": embedding_device, "batch_size": batch_size, "workers": workers,
            "synonym_threshold": synonym_threshold,
        }
        # Preserve source order, including the order of same-text source IDs.
        corpus = hashlib.sha256()
        for message in self.messages.values():
            corpus.update((json.dumps(message, sort_keys=True, ensure_ascii=False) + "\n").encode())
        openie_profile = _openie_profile()
        self.identity = {
            "upstream_commit": UPSTREAM_COMMIT, "adapter_sha256": _digest(Path(__file__)),
            "openie_profile": openie_profile,
            "openie_profile_sha256": hashlib.sha256(json.dumps(openie_profile, sort_keys=True).encode()).hexdigest(),
            "domain": self.domain, "corpus_sha256": corpus.hexdigest(), "settings": self.settings,
            "embedding_files": {
                item.relative_to(embedding_path).as_posix(): _digest(item)
                for item in sorted(embedding_path.rglob("*"))
                if item.is_file() and item.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"}
            },
        }
        self.engine = None
        self.dense_only = dense_only

    def build(self, *, require_existing=False):
        from filelock import FileLock

        self.close()
        if not require_existing:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        elif not self.cache_dir.is_dir():
            raise ValueError(f"A completed HippoRAG index is required: {self.cache_dir}")
        manifest_path = self.cache_dir / "sgmem-hipporag.json"
        with FileLock(str(self.cache_dir / ".sgmem-hipporag.lock")):
            previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
            if require_existing and (previous is None or previous.get("status") != "complete"):
                raise ValueError(f"A completed HippoRAG index is required: {self.cache_dir}")
            if previous is None and any(item.name != ".sgmem-hipporag.lock" for item in self.cache_dir.iterdir()):
                raise ValueError(f"Use a dedicated empty HippoRAG index directory: {self.cache_dir}")
            if previous is not None and (previous.get("identity") != self.identity
                                         or previous.get("status") not in {"building", "complete"}):
                raise ValueError("HippoRAG cache identity changed; use a fresh index directory")
            complete = previous is not None and previous["status"] == "complete"
            if previous is None:
                write_json(manifest_path, {"identity": self.identity, "status": "building"})
            identity = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
            engine, chunk = _engine(self.settings, self.cache_dir, identity,
                                    rebuild=previous is not None and not complete)
            try:
                if not complete:
                    engine.index([chunk(content=body, metadata={"domain": self.domain, "message_ids": list(ids)})
                                  for body, ids in self.passages.items()])
                engine.prepare_retrieval_objects()
                if len(engine.passage_node_keys) != len(self.passages):
                    raise ValueError("HippoRAG index passage count differs from the source corpus")
                for key in engine.passage_node_keys:
                    body = engine.chunk_embedding_store.get_row(key)["content"]
                    self._check_sources(body, engine.chunk_metadata.get(key, {}))
                if not complete:
                    write_json(manifest_path, {"identity": self.identity, "status": "complete",
                                               "messages": len(self.messages), "graph": engine.get_graph_info()})
            except BaseException as exc:
                try:
                    if isinstance(exc, OpenIEError):
                        with (self.cache_dir / "openie-errors.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(exc.trace, ensure_ascii=True, allow_nan=False) + "\n")
                finally:
                    engine.close()
                raise
            self.engine = engine
        return self

    def _check_sources(self, body, metadata):
        ids = self.passages.get(body)
        if ids is None or metadata.get("domain") != self.domain or metadata.get("message_ids") != ids:
            raise ValueError("HippoRAG passage provenance differs from the source corpus")
        return ids

    def retrieve_with_trace(self, query, top_k, *, asking_user_id=None):
        if self.engine is None:
            raise RuntimeError("Build or reopen the HippoRAG index before retrieval")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        search_query = f"{asking_user_id} {query}" if asking_user_id else query
        depth = len(self.passages)
        retrieve = self.engine.retrieve_dpr if self.dense_only else self.engine.retrieve
        result = retrieve([search_query], num_to_retrieve=depth)[0]
        if not len(result.docs) == len(result.doc_scores) == len(result.doc_metadata) == depth:
            raise ValueError("HippoRAG returned an incomplete passage ranking")
        trace_rows, expanded, seen = [], [], set()
        for rank, (body, score, metadata) in enumerate(zip(result.docs, result.doc_scores, result.doc_metadata,
                                                          strict=True), 1):
            score = float(score)
            if not isfinite(score) or (trace_rows and score > trace_rows[-1]["score"]):
                raise ValueError("HippoRAG passage ranking has invalid scores")
            if len(expanded) >= top_k and score < trace_rows[-1]["score"]:
                break
            ids = self._check_sources(body, metadata)
            if body in seen:
                raise ValueError("HippoRAG passage ranking repeats a passage")
            seen.add(body)
            trace_rows.append({"rank": rank, "score": score, "message_ids": list(ids)})
            expanded.extend((message_id, score, rank) for message_id in ids)
        # All originals inherit their passage score. Source order breaks exact score ties.
        expanded.sort(key=lambda row: (-row[1], self.source_order[row[0]]))
        hits = [{"message": self.messages[message_id], "score": score, "passage_rank": rank}
                for message_id, score, rank in expanded[:top_k]]
        if len(hits) != min(top_k, len(self.messages)):
            raise ValueError("HippoRAG result did not supply enough original messages")
        recognition = {"mode": "dense_control", "candidate_triples": [], "selected_triples": []} if self.dense_only else (
            self.engine.recognition)
        return hits, {"query": search_query, "passages": trace_rows, "passages_scored": depth,
                      "expanded_candidates": len(expanded), "recognition": recognition}

    def retrieve(self, query, top_k, *, asking_user_id=None):
        return self.retrieve_with_trace(query, top_k, asking_user_id=asking_user_id)[0]

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None


def _read_rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--llm-revision", required=True)
    parser.add_argument("--llm-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--synonym-threshold", type=float, default=0.8)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if bool(args.queries) != bool(args.output) or args.top_k < 1:
        parser.error("Use --queries and --output together, with positive --top-k")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    memory = HippoRAGMemory(
        _read_rows(args.messages), args.index, embedding_path=args.embedding_path,
        embedding_revision=args.embedding_revision, llm_revision=args.llm_revision,
        llm_name=args.llm_name, llm_base_url=args.llm_base_url, embedding_device=args.embedding_device,
        workers=args.workers, synonym_threshold=args.synonym_threshold,
    )
    try:
        queries = []
        if args.queries:
            queries = [row for row in _read_rows(args.queries) if row["domain"] == memory.domain]
            keys = {(row["category"], row["id"]) for row in queries}
            if not queries or len(keys) != len(queries):
                raise ValueError("Query input must contain unique questions for the index domain")
        memory.build()
        if queries:
            with args.output.open("x", encoding="utf-8") as handle:
                for question in queries:
                    row = {"domain": memory.domain, "category": question["category"],
                           "question_id": question["id"]}
                    try:
                        hits, trace = memory.retrieve_with_trace(question["question"], args.top_k,
                                                                  asking_user_id=question.get("asking_user_id"))
                    except RecognitionError as exc:
                        row.update(status="error", trace={"recognition": exc.trace})
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        raise
                    row.update(status="ok", hits=hits, trace=trace)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
