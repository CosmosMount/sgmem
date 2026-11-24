"""Build and audit a full RAPTOR tree for one public GroupMemBench domain."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgmem import raptor_memory as raptor
from sgmem.memory_retrieval import public_sources
from sgmem.runner import _git_info, write_json

E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
PATCH = ROOT / "patches" / "raptor.patch"


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(upstream, *args):
    return subprocess.check_output(["git", "-C", str(upstream), *args])


def verify_upstream(upstream):
    """Compare the importable sources to the exact patched pinned tree."""
    if _git(upstream, "rev-parse", "HEAD").decode().strip() != raptor.UPSTREAM_COMMIT:
        raise ValueError(f"Use RAPTOR checkout {raptor.UPSTREAM_COMMIT}")
    names = _git(upstream, "ls-tree", "-r", "--name-only", "HEAD", "--", "raptor").decode().splitlines()
    files = {p.relative_to(upstream).as_posix() for p in (upstream / "raptor").rglob("*.py")}
    if set(names) != files or "raptor/__init__.py" not in files:
        raise ValueError("RAPTOR source files differ from the pinned checkout")
    with TemporaryDirectory(prefix="sgmem-raptor-check-") as temp:
        expected = Path(temp)
        for name in names:
            path = expected / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_git(upstream, "show", "HEAD:" + name))
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(PATCH)], cwd=expected, check=True)
        hashes = {}
        for name in names:
            contents = (upstream / name).read_bytes().replace(b"\r\n", b"\n")
            if contents != (expected / name).read_bytes().replace(b"\r\n", b"\n"):
                raise ValueError(f"RAPTOR source differs from the exact project patch: {name}")
            hashes[name] = hashlib.sha256(contents).hexdigest()
    return {"commit": raptor.UPSTREAM_COMMIT, "path": str(upstream),
            "patch_sha256": _digest(PATCH), "source_sha256": hashes}


def _model_files(path, *, weights):
    if not (path / "config.json").is_file():
        raise ValueError(f"Expected a local model snapshot containing config.json: {path}")
    suffixes = {".json", ".model", ".txt", ".safetensors", ".bin"} if weights else {
        ".json", ".model", ".txt", ".jinja",
    }
    files = {p.relative_to(path).as_posix(): _digest(p)
             for p in sorted(path.rglob("*")) if p.is_file() and p.suffix in suffixes}
    if weights and not any(Path(name).suffix in {".safetensors", ".bin"} for name in files):
        raise ValueError(f"No local E5 weights found: {path}")
    if not weights and "tokenizer.json" not in files:
        raise ValueError(f"No Qwen fast tokenizer found: {path}")
    return files


def _versions():
    versions = {}
    for package in ("numpy", "torch", "transformers", "sentence-transformers", "tokenizers", "scikit-learn",
                    "umap-learn", "numba", "scipy", "tiktoken", "openai", "tenacity"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build(args):
    from filelock import FileLock

    config = raptor.RaptorConfig(summary_workers=getattr(args, "summary_workers", 1))
    if args.embedding_revision != E5_REVISION or args.tokenizer_revision != QWEN_REVISION:
        raise ValueError("Use the frozen SGMem E5 and Qwen revisions")
    output = args.output.absolute()
    if output.suffix != ".npz":
        raise ValueError("RAPTOR output must name an .npz file")
    raw = args.messages.read_bytes()
    sources = public_sources(json.loads(line) for line in raw.splitlines() if line.strip())
    messages = list(sources.values())
    upstream, embedding, tokenizer_path = (p.resolve() for p in (args.upstream, args.embedding_path, args.qa_tokenizer))
    manifest_path = output.with_suffix(".manifest.json")
    receipts_path = output.with_suffix(".summary-calls.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        for path in (output, manifest_path, receipts_path):
            if os.path.lexists(path):
                raise FileExistsError(f"RAPTOR output already exists: {path}")
        started = perf_counter()
        provenance = {
            "embedding_revision": E5_REVISION, "embedding_path": str(embedding),
            "embedding_files_sha256": _model_files(embedding, weights=True),
            "tokenizer_revision": QWEN_REVISION, "tokenizer_path": str(tokenizer_path),
            "tokenizer_files_sha256": _model_files(tokenizer_path, weights=False),
            "messages_sha256": hashlib.sha256(raw).hexdigest(), "upstream": verify_upstream(upstream),
            "project_commit": _git_info(ROOT)["commit"], "runtime_versions": _versions(),
            "implementation_sha256": {p.relative_to(ROOT).as_posix(): _digest(p) for p in (
                Path(__file__).resolve(), ROOT / "src/sgmem/raptor_memory.py",
                ROOT / "src/sgmem/propositions.py", ROOT / "src/sgmem/retrieval.py",
                ROOT / "src/sgmem/memory_retrieval.py",
            )},
            "summary": {"endpoint": args.endpoint, "model": "Qwen/Qwen3-8B", "declared_revision": QWEN_REVISION,
                        "context_tokens": 8192, "enable_thinking": False, "temperature": 0,
                        "max_tokens": config.summary_tokens, "workers": config.summary_workers,
                        "system_prompt": raptor.SUMMARY_SYSTEM},
        }
        manifest = {"schema": 1, "status": "building", "domain": messages[0]["domain"],
                    "source_count": len(messages), "provenance": provenance, "config": asdict(config),
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(manifest_path, manifest)
        calls = 0
        receipts_lock = Lock()
        sys.path.insert(0, str(upstream))
        try:
            import raptor as official
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer

            if Path(official.__file__).resolve() != upstream / "raptor" / "__init__.py":
                raise ValueError("Python imported RAPTOR outside the verified source checkout")
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
            encoder = SentenceTransformer(str(embedding), device="cpu", local_files_only=True)
            with receipts_path.open("x", encoding="utf-8", newline="\n") as receipts:
                def summarize(context, max_tokens):
                    nonlocal calls
                    call_start = perf_counter()
                    with receipts_lock:
                        calls += 1
                        row = {"call": calls, "context": context, "max_tokens": max_tokens}
                    client = None
                    try:
                        client = raptor.LocalQwenSummarizer(args.endpoint, "Qwen/Qwen3-8B", tokenizer, context_tokens=8192)
                        summary = client(context, max_tokens)
                        row.update(status="returned", raw_summary=summary, usage=client.last_usage)
                        return summary
                    except Exception as exc:
                        row.update(status="error", error=str(exc), usage=getattr(client, "last_usage", None),
                                   raw_response=getattr(client, "last_response", None))
                        raise
                    finally:
                        row["seconds"] = perf_counter() - call_start
                        with receipts_lock:
                            receipts.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            receipts.flush()

                tree = raptor.build_raptor(messages, encoder, tokenizer, summarize,
                                           config=config, provenance=provenance)
            raptor.RaptorRetriever(messages, tree, encoder, tokenizer)
            layers = Counter(node.layer for node in tree.nodes)
            if not any(layer > 0 for layer in layers):
                raise ValueError("RAPTOR build did not produce a summary layer")
            tree.save(output)
            manifest.update(status="complete", nodes_by_layer=dict(layers), source_sha256=tree.source_sha256,
                            tree_sha256=_digest(output), summary_calls_sha256=_digest(receipts_path))
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            sys.path.remove(str(upstream))
            manifest.update(summary_calls=calls, wall_seconds=perf_counter() - started,
                            ended_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("messages", "upstream", "embedding-path", "qa-tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--embedding-revision", default=E5_REVISION)
    parser.add_argument("--tokenizer-revision", default=QWEN_REVISION)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18951/v1")
    parser.add_argument("--summary-workers", type=int, default=1, help="Concurrent summary requests within a RAPTOR layer")
    result = build(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("domain", "status", "nodes_by_layer", "summary_calls")}))


if __name__ == "__main__":
    main()
