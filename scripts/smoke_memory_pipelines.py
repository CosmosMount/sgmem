"""Exercise real indexing and retrieval on a declared public message window, without QA labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgmem.propositions import source_messages
from sgmem.runner import write_json

E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def check_hits(hits, sources):
    ids = [hit["message"]["id"] for hit in hits]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("The pilot must return ten distinct original messages")
    for hit in hits:
        message = hit["message"]
        if message != sources[message["domain"], message["id"]]:
            raise ValueError("Retrieval changed an original source message")
    return [{"message_id": hit["message"]["id"], "score": hit["score"]} for hit in hits]


def lightmem_coverage(manifest):
    metrics = manifest["metrics"]
    return {
        "changed_turns": sum(row.get("changed_turns", 0) for row in metrics),
        "segmentation_calls": sum(row.get("segmentation_calls", 0) for row in metrics),
        "turn_embedding_calls": sum(row["embedding_calls"].get("turn", 0) for row in metrics),
        "extraction_calls": sum(row["llm"]["add_memory"]["calls"] for row in metrics),
        "sleep_calls": sum(row["llm"]["update"]["calls"] for row in metrics),
        "sleep_mutations": sum(row["update_actions"].get(action, 0)
                               for row in metrics for action in ("update", "delete")),
    }


def run(args):
    raw = args.messages.read_bytes()
    messages = [json.loads(line) for line in raw.splitlines()]
    sources = source_messages(messages)
    if len({row["domain"] for row in messages}) != 1:
        raise ValueError("Use one public domain window")
    args.output.mkdir(parents=True, exist_ok=False)
    queries = [row["content"] for row in messages[:3]]
    summary = {"messages_sha256": hashlib.sha256(raw).hexdigest(), "source_count": len(messages),
               "qa_scoring": False, "queries": queries, "methods": {}}
    for method in args.methods:
        started = perf_counter()
        try:
            if method == "raptor":
                sys.path.insert(0, str(args.vendor / "raptor"))
                from sentence_transformers import SentenceTransformer
                from transformers import AutoTokenizer
                from sgmem.raptor_memory import build_raptor, RaptorTree, RaptorRetriever, LocalQwenSummarizer
                encoder = SentenceTransformer(str(args.embedding_model), device="cpu", local_files_only=True)
                tokenizer = AutoTokenizer.from_pretrained(args.qa_model, local_files_only=True)
                summarize = LocalQwenSummarizer(args.endpoint, "Qwen/Qwen3-8B", tokenizer, context_tokens=8192)
                tree = build_raptor(messages, encoder, tokenizer, summarize,
                                    provenance={"embedding_revision": E5_REVISION,
                                                "tokenizer_revision": QWEN_REVISION})
                tree.save(args.output / "raptor.npz")
                tree = RaptorTree.load(args.output / "raptor.npz")
                retriever = RaptorRetriever(messages, tree, encoder, tokenizer)
                result = {"nodes_by_layer": dict(Counter(node.layer for node in tree.nodes)),
                          "hits": [check_hits(retriever.retrieve(q, 10), sources) for q in queries],
                          "native": [retriever.retrieve_nodes(q, 10) for q in queries]}
                coverage = {"summary_nodes": sum(node.layer > 0 for node in tree.nodes)}
            elif method == "lightmem":
                from sgmem.lightmem_memory import build_lightmem, LightMemIndex
                directory = args.output / "lightmem"
                manifest = build_lightmem(messages, directory, upstream=args.vendor / "LightMem",
                    endpoint=args.endpoint, model="Qwen/Qwen3-8B", embedding_model=args.embedding_model,
                    compressor_model=args.compressor_model,
                    compressor_revision="5f0c82792b7ea14c6484e015b6a072009496b7f2", device="cpu")
                retriever = LightMemIndex.load(directory, messages[0]["domain"])
                result = {"manifest": manifest,
                          "hits": [check_hits(retriever.retrieve(q, 10), sources) for q in queries],
                          "native": [retriever.retrieve_native(q, 10) for q in queries]}
                coverage = lightmem_coverage(manifest)
            else:
                sys.path.insert(0, str(args.vendor / "HippoRAG/src"))
                from sgmem.hipporag_memory import HippoRAGMemory
                retriever = HippoRAGMemory(messages, args.output / "hipporag2",
                    embedding_path=args.embedding_model, embedding_revision=E5_REVISION,
                    llm_revision=QWEN_REVISION, llm_base_url=args.endpoint, embedding_device="cpu")
                try:
                    retriever.build()
                    result = {"graph": retriever.engine.get_graph_info(), "retrievals": []}
                    for query in queries:
                        hits, trace = retriever.retrieve_with_trace(query, 10)
                        result["retrievals"].append({"hits": check_hits(hits, sources), "trace": trace})
                    coverage = {"ppr_retrievals": sum(
                        row["trace"]["recognition"]["mode"] == "ppr" for row in result["retrievals"])}
                finally:
                    retriever.close()
            missing = [step for step, count in coverage.items() if step != "sleep_mutations" and not count]
            summary["methods"][method] = {
                "status": "incomplete_coverage" if missing else "complete",
                "seconds": perf_counter() - started, "coverage": coverage, "missing_coverage": missing,
                "result": result,
            }
        except Exception as exc:
            traceback.print_exc()
            summary["methods"][method] = {"status": "failed", "seconds": perf_counter() - started,
                                          "error": str(exc)}
        write_json(args.output / "summary.json", summary)
        print(method, summary["methods"][method]["status"], flush=True)
    return int(any(row["status"] != "complete" for row in summary["methods"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--qa-model", type=Path, required=True)
    parser.add_argument("--compressor-model", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18951/v1")
    parser.add_argument("--methods", nargs="+", choices=("raptor", "lightmem", "hipporag2"),
                        default=["raptor", "lightmem", "hipporag2"])
    raise SystemExit(run(parser.parse_args()))
