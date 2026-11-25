"""Load a built memory index and export rankings for one GroupMemBench domain."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgmem.memory_retrieval import (ABLATIONS, CONTEXT_BASES, CONTEXT_VIEWS, METHODS, export_rankings,
                                    public_questions, public_sources, sentence_records)
from sgmem.lightmem_memory import E5_REVISION, E5_WEIGHTS


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path):
    content = path.read_bytes()
    return [json.loads(line) for line in content.splitlines() if line.strip()], hashlib.sha256(content).hexdigest()


def _manifest(path):
    content = path.read_bytes()
    manifest = json.loads(content)
    if manifest.get("status") != "complete":
        raise ValueError(f"Index is incomplete: {path}")
    return manifest, hashlib.sha256(content).hexdigest()


def _local_model(path, revision, designation):
    if not path or not revision or not (path / "config.json").is_file():
        raise ValueError(f"Provide a local {designation} directory and its pinned revision")
    return path.resolve()


def _model_files(path, *, weights):
    suffixes = {".json", ".model", ".txt", ".safetensors", ".bin"} if weights else {
        ".json", ".model", ".txt", ".jinja",
    }
    return {item.relative_to(path).as_posix(): _digest(item)
            for item in sorted(path.rglob("*")) if item.is_file() and item.suffix in suffixes}


def _prune_synonyms(graph, threshold):
    """Raise the cutoff of an in-memory, source-aware HippoRAG graph."""
    if "hipporag_edge_schema" not in graph.attributes() or graph["hipporag_edge_schema"] != 1:
        raise ValueError("Synonym replay requires the pinned source-aware graph")
    removed, mixed = [], 0
    before = graph.ecount()
    for edge in graph.es.select(synonym_score_gt=0.0, synonym_score_lt=threshold):
        facts = edge["fact_source_counts"]
        passage = edge["passage_source"] is not None
        if not facts and not passage:
            removed.append(edge.index)
            continue
        weight = max(sum(facts.values()), float(passage))
        if edge["weight"] != weight:
            raise ValueError("Mixed graph edge has unexpected fact or passage weight")
        edge["synonym_score"] = 0.0
        kinds = ["fact"] if facts else []
        if passage:
            kinds.append("passage")
        edge["edge_kind"] = "+".join(kinds)
        mixed += 1
    graph.delete_edges(removed)
    return {"threshold": threshold, "edges_before": before, "edges_after": graph.ecount(),
            "removed_synonym_only_edges": len(removed), "pruned_mixed_edges": mixed}


def load_retriever(args, sources):
    domain = next(iter(sources))[0]
    messages = [dict(message) for message in sources.values()]
    index = args.index.resolve(strict=True) if args.index is not None else None
    origin = {"path": str(index)} if index is not None else {}

    if args.method in ("lightmem", "lightmem-pre"):
        from sgmem.lightmem_memory import LightMemIndex

        state, checksum = _manifest(index / "manifest.json")
        stage = "pre" if args.method == "lightmem-pre" else "post"
        retriever = LightMemIndex.load(index, domain, stage=stage, device="cpu")
        if retriever.messages != {message["id"]: message for message in messages}:
            raise ValueError("LightMem index source pack differs from the supplied domain corpus")
        return retriever, {**origin, "manifest_sha256": checksum, "protocol": state["protocol"], "stage": stage}, {}

    if args.method in ("hipporag2", "hipporag2-dpr"):
        from sgmem.hipporag_memory import HippoRAGMemory

        state, checksum = _manifest(index / "sgmem-hipporag.json")
        settings = dict(state["identity"]["settings"])
        for key in ("embedding_path", "embedding_revision", "llm_revision", "llm_base_url", "llm_name"):
            value = getattr(args, key)
            if value is not None:
                settings[key] = str(value)
        if settings["embedding_device"] != "cpu":
            raise ValueError("Rank export requires a HippoRAG index configured with CPU embeddings")
        dense_only = args.method == "hipporag2-dpr"
        if dense_only:
            settings["dense_only"] = True
        threshold = getattr(args, "synonym_threshold", None)
        if threshold is not None and (dense_only or not settings["synonym_threshold"] < threshold <= 1.0):
            raise ValueError("Use a higher synonym threshold on the completed HippoRAG graph")
        retriever = HippoRAGMemory(messages, index, **settings).build(require_existing=True)
        provenance = {**origin, "manifest_sha256": checksum, "identity": state["identity"],
                      "retrieval": "dpr" if dense_only else "graph"}
        if threshold is not None:
            def require_cached_recognition(*args, **kwargs):
                raise RuntimeError("Export the original graph rankings before replaying the synonym cutoff")

            try:
                provenance["synonym_replay"] = _prune_synonyms(retriever.engine.graph, threshold)
                retriever.engine.llm_model.openai_client.chat.completions.create = require_cached_recognition
                provenance["recognition_cache_only"] = True
            except Exception:
                retriever.close()
                raise
        return retriever, provenance, {}

    path = _local_model(args.embedding_path, args.embedding_revision, "E5")
    embedding_files = _model_files(path, weights=True)
    if args.embedding_revision != E5_REVISION or embedding_files.get("model.safetensors") != E5_WEIGHTS:
        raise ValueError("E5 revision or model.safetensors differs from the frozen E5-small-v2 comparator")
    models = {"embedding": {"path": str(path), "revision": args.embedding_revision, "files_sha256": embedding_files}}
    if args.method in ("densex", "densex-sentences"):
        from sgmem.propositions import PropositionRetriever
        from sentence_transformers import SentenceTransformer

        if args.method == "densex-sentences":
            local_records = sentence_records(messages)
            origin = {"units": "deterministic_sentences", "unit_count": sum(len(r["units"]) for r in local_records),
                      "model_extraction_calls": 0}
        else:
            state, checksum = _manifest(index / "manifest.json")
            records, rows_sha = _rows(index / "propositions.jsonl")
            local_records = [record for record in records if record["domain"] == domain]
            origin.update(manifest_sha256=checksum, rows_sha256=rows_sha, identity=state["identity"])
        encoder = SentenceTransformer(str(path), device="cpu", local_files_only=True,
                                      model_kwargs={"use_safetensors": True})
        walk = PropositionRetriever(messages, local_records, encoder)
        return walk, origin, models

    from sgmem.raptor_memory import RaptorRetriever, RaptorTree
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    tokenizer_path = _local_model(args.qa_tokenizer, args.tokenizer_revision, "Qwen tokenizer")
    tokenizer_files = _model_files(tokenizer_path, weights=False)
    if "tokenizer.json" not in tokenizer_files:
        raise ValueError(f"No Qwen fast tokenizer found: {tokenizer_path}")
    tree = RaptorTree.load(index)
    for name, value in (("embedding_revision", args.embedding_revision), ("tokenizer_revision", args.tokenizer_revision),
                        ("embedding_files_sha256", embedding_files), ("tokenizer_files_sha256", tokenizer_files)):
        if tree.provenance.get(name) != value:
            raise ValueError(f"RAPTOR cache was built with a different or unspecified {name}")
    encoder = SentenceTransformer(str(path), device="cpu", local_files_only=True,
                                  model_kwargs={"use_safetensors": True})
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
    models["tokenizer"] = {"path": str(tokenizer_path), "revision": args.tokenizer_revision,
                           "files_sha256": tokenizer_files}
    leaf_only = args.method == "raptor-leaf"
    retriever = RaptorRetriever(messages, tree, encoder, tokenizer, leaf_only=leaf_only)
    return retriever, {**origin, "sha256": _digest(index), "provenance": tree.provenance, "leaf_only": leaf_only}, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, *ABLATIONS), required=True)
    parser.add_argument("--context-view", choices=CONTEXT_VIEWS,
                        help="Export candidates for the shared native-context QA budget")
    for name in ("messages", "questions", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--index", type=Path, help="Completed index, except for the sentence-only control")
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--qa-tokenizer", type=Path)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-name")
    parser.add_argument("--llm-revision")
    parser.add_argument("--synonym-threshold", type=float,
                        help="Raise the HippoRAG cutoff and reuse cached recognition without model calls")
    args = parser.parse_args()
    if args.context_view is not None and args.method not in CONTEXT_BASES:
        parser.error("--context-view is supported for densex, lightmem and raptor")
    if args.synonym_threshold is not None and args.method != "hipporag2":
        parser.error("--synonym-threshold is supported for hipporag2")
    if (args.index is None) != (args.method == "densex-sentences"):
        parser.error("--index is required for all methods except densex-sentences")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    message_rows, messages_sha = _rows(args.messages)
    question_rows, questions_sha = _rows(args.questions)
    sources = public_sources(message_rows)
    questions = public_questions(question_rows, next(iter(sources))[0])
    retriever, index, models = load_retriever(args, sources)
    try:
        implementation = [Path(__file__).resolve(), ROOT / "src/sgmem/memory_retrieval.py",
                          ROOT / "src/sgmem/structured.py", ROOT / "src/sgmem/query_metadata.py"]
        if args.context_view is not None:
            adapter = {"densex": "propositions.py", "lightmem": "lightmem_memory.py", "raptor": "raptor_memory.py"}
            implementation.append(ROOT / "src" / "sgmem" / adapter[args.method])
        result = export_rankings(retriever, sources, questions, args.method, args.output, {
            "messages_sha256": messages_sha, "questions_sha256": questions_sha, "index": index, "models": models,
            "implementation_sha256": {str(path.relative_to(ROOT)): _digest(path) for path in implementation},
        }, context_view=args.context_view)
    finally:
        if args.method in ("hipporag2", "hipporag2-dpr"):
            retriever.close()
    print(json.dumps({"output": str(args.output.resolve()), "domain": result["domain"],
                      "questions": result["completed_queries"], "methods": result["methods"]}))


if __name__ == "__main__":
    main()
