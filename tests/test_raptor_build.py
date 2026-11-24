import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from argparse import Namespace
from pathlib import Path
import runpy
import subprocess
import sys
from threading import Event, Lock
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest

from sgmem import raptor_memory as raptor


@pytest.fixture
def builder():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "build_raptor.py"))["build"]


@pytest.fixture
def pinned_source(tmp_path, monkeypatch, builder):
    root = tmp_path / "upstream"
    package = root / "raptor"
    package.mkdir(parents=True)
    original = b"# original RAPTOR source\nvalue = 1\n# EOF\n"
    patched = original.replace(b"value = 1", b"value = 2")
    init = package / "__init__.py"
    init.write_bytes(patched)
    patch = tmp_path / "raptor.patch"
    patch.write_bytes(b"diff --git a/raptor/__init__.py b/raptor/__init__.py\n"
                      b"--- a/raptor/__init__.py\n+++ b/raptor/__init__.py\n@@ -1,3 +1,3 @@\n"
                      b" # original RAPTOR source\n-value = 1\n+value = 2\n # EOF\n")

    def git(path, *args):
        assert path == root
        return {"rev-parse": raptor.UPSTREAM_COMMIT.encode(), "ls-tree": b"raptor/__init__.py\n",
                "show": original}[args[0]]

    monkeypatch.setitem(builder.__globals__, "_git", git)
    monkeypatch.setitem(builder.__globals__, "PATCH", patch)
    return root, init, patch, patched


def test_upstream_requires_exact_patch_and_no_extra_python(pinned_source, builder):
    root, init, patch, patched = pinned_source
    verify = builder.__globals__["verify_upstream"]
    result = verify(root)
    assert result["commit"] == raptor.UPSTREAM_COMMIT
    assert result["patch_sha256"] == hashlib.sha256(patch.read_bytes()).hexdigest()
    assert result["source_sha256"] == {"raptor/__init__.py": hashlib.sha256(patched).hexdigest()}
    init.write_bytes(patched + b"extra_code = 0\n")
    with pytest.raises(ValueError, match="exact project patch"):
        verify(root)
    init.write_bytes(patched)
    init.with_name("injected.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="source files differ"):
        verify(root)


@pytest.fixture
def build_pack(tmp_path, monkeypatch, builder, pinned_source):
    root, init, _, _ = pinned_source
    public_rows = [{"domain": "Finance", "id": "M1", "content": "Test shipment passed.", "group": "G1"},
                   {"domain": "Finance", "id": "M2", "content": "Shipment is still conditional.", "group": "G1"}]
    messages = tmp_path / "messages-Finance.jsonl"
    messages.write_text("".join(json.dumps(row) + "\n" for row in public_rows), encoding="utf-8")
    e5, tokenizer = tmp_path / "e5", tmp_path / "qwen"
    for path in (e5, tokenizer):
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")
    (e5 / "model.safetensors").write_bytes(b"fixed E5 weights")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer / "model.safetensors").write_bytes(b"unused Qwen weights")
    args = Namespace(messages=messages, upstream=root, embedding_path=e5, qa_tokenizer=tokenizer,
                     output=tmp_path / "finance.npz", endpoint="http://127.0.0.1:18951/v1",
                     embedding_revision=builder.__globals__["E5_REVISION"],
                     tokenizer_revision=builder.__globals__["QWEN_REVISION"], summary_workers=1)
    state = SimpleNamespace(calls=[], clients=[], loads=[], fail=False)

    def load_tokenizer(path, **kwargs):
        assert kwargs == {"local_files_only": True, "use_fast": True} and path == str(tokenizer)
        state.loads.append("tokenizer")
        return None

    def load_encoder(path, **kwargs):
        assert kwargs == {"local_files_only": True, "device": "cpu"} and path == str(e5)
        state.loads.append("encoder")
        return None

    class Summary:
        last_response = None
        def __init__(self, endpoint, model, tokenizer, *, context_tokens):
            assert endpoint == args.endpoint and model == "Qwen/Qwen3-8B" and context_tokens == 8192
            self.last_usage = None
            self.last_response = None
            state.clients.append(self)

        def __call__(self, context, max_tokens):
            state.calls.append((context, max_tokens))
            if state.fail:
                self.last_response = {"choices": [{"finish_reason": "length"}]}
                raise ValueError("mock summary token mismatch")
            prompt_tokens = 37 if args.summary_workers == 1 else int(context.split()[-1])
            self.last_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 12}
            return " The two messages differ about shipment. "

    def build_tree(messages, encoder, tokenizer, summarize, *, config, provenance):
        assert messages == public_rows and config == raptor.RaptorConfig(summary_workers=args.summary_workers)
        domain, digest = raptor._source_identity(messages)
        context = "Test shipment passed.\n\nShipment is still conditional.\n\n"
        if config.summary_workers == 1:
            summary = summarize(context, config.summary_tokens)
        else:
            with ThreadPoolExecutor(max_workers=config.summary_workers) as pool:
                results = list(pool.map(lambda i: summarize(f"{context} {i}", config.summary_tokens), range(51)))
            summary = results[0]
        nodes = [raptor.RaptorNode(row["content"], 0, (), row["id"]) for row in messages]
        nodes.append(raptor.RaptorNode(summary.strip(), 1, (0, 1), None))
        return raptor.RaptorTree(domain, digest, config, provenance, nodes,
                                 np.asarray([[1, 0]] * len(nodes), dtype=np.float32))

    monkeypatch.setitem(sys.modules, "raptor", SimpleNamespace(__file__=str(init)))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load_encoder))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
    ))
    monkeypatch.setattr(raptor, "LocalQwenSummarizer", Summary)
    monkeypatch.setattr(raptor, "build_raptor", build_tree)
    return args, state


def test_builder_writes_tree_hashes_provenance_and_summary_receipt(builder, build_pack):
    args, state = build_pack
    result = builder(args)
    assert result["status"] == "complete" and result["nodes_by_layer"] == {0: 2, 1: 1}
    assert result["summary_calls"] == 1 and result["wall_seconds"] > 0
    files = result["provenance"]
    assert result["config"]["summary_tokens"] == files["summary"]["max_tokens"] == 256
    assert files["embedding_files_sha256"]["model.safetensors"] == hashlib.sha256(b"fixed E5 weights").hexdigest()
    assert files["tokenizer_files_sha256"] == {name: hashlib.sha256(b"{}").hexdigest()
                                                for name in ("config.json", "tokenizer.json")}
    tree = raptor.RaptorTree.load(args.output)
    assert tree.provenance == files and tree.nodes[-1].children == (0, 1)
    receipts = args.output.with_suffix(".summary-calls.jsonl")
    row = json.loads(receipts.read_text(encoding="utf-8"))
    assert row["max_tokens"] == 256
    assert row["status"] == "returned" and (row["context"], row["max_tokens"]) == state.calls[0]
    assert row["usage"] == {"prompt_tokens": 37, "completion_tokens": 12}
    assert row["raw_summary"].startswith(" ")
    manifest = json.loads(args.output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["tree_sha256"] == hashlib.sha256(args.output.read_bytes()).hexdigest()
    assert manifest["summary_calls_sha256"] == hashlib.sha256(receipts.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="already exists"):
        builder(args)
    assert state.loads == ["tokenizer", "encoder"]


def test_failed_summary_preserves_receipt_without_tree(builder, build_pack):
    args, state = build_pack
    state.fail = True
    with pytest.raises(ValueError, match="mock summary token mismatch"):
        builder(args)
    assert not args.output.exists()
    manifest = json.loads(args.output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads(args.output.with_suffix(".summary-calls.jsonl").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["summary_calls"] == 1
    assert receipt["status"] == "error" and receipt["error"] == manifest["error"]
    assert receipt["raw_response"]["choices"][0]["finish_reason"] == "length"


def test_wrong_domain_and_concurrent_writer_stop_before_model_load(builder, build_pack):
    from filelock import FileLock, Timeout

    args, state = build_pack
    with FileLock(str(args.output) + ".lock"), pytest.raises(Timeout):
        builder(args)
    rows = [json.loads(line) for line in args.messages.read_text().splitlines()]
    rows[-1]["domain"] = "Technology"
    args.messages.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="one domain"):
        builder(args)
    assert not state.loads and not args.output.with_suffix(".manifest.json").exists()


def test_parallel_summary_receipts_have_independent_clients_and_complete_lines(builder, build_pack):
    args, state = build_pack
    args.summary_workers = 8
    manifest = builder(args)
    assert manifest["summary_calls"] == len(state.clients) == 51
    assert len({id(client) for client in state.clients}) == 51
    assert manifest["config"]["summary_workers"] == manifest["provenance"]["summary"]["workers"] == 8
    receipts = args.output.with_suffix(".summary-calls.jsonl")
    rows = [json.loads(line) for line in receipts.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["call"] for row in rows) == list(range(1, 52))
    assert all(int(row["context"].split()[-1]) == row["usage"]["prompt_tokens"] for row in rows)


def test_patched_official_parallel_builder_orders_layers_and_surfaces_worker_error(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    upstream = repo.parent / "raptor"
    if not (upstream / "raptor" / "cluster_tree_builder.py").is_file():
        pytest.skip("Stage the pinned sibling RAPTOR checkout to exercise its patched builder")
    path = tmp_path / "raptor" / "cluster_tree_builder.py"
    path.parent.mkdir()
    path.write_bytes(subprocess.check_output([
        "git", "-C", str(upstream), "show", f"{raptor.UPSTREAM_COMMIT}:raptor/cluster_tree_builder.py",
    ]))
    subprocess.run(["git", "apply", "--include=raptor/cluster_tree_builder.py", str(repo / "patches/raptor.patch")],
                   cwd=tmp_path, check=True)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ClusterTreeBuilder")
    construct = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "construct_tree")
    namespace = dict(Dict=Dict, List=List, Node=SimpleNamespace, Tree=lambda *args: None, Lock=Lock, logging=logging,
                     ThreadPoolExecutor=ThreadPoolExecutor, as_completed=as_completed,
                     get_node_list=lambda nodes: list(nodes.values()), get_text=lambda nodes: " ".join(n.text for n in nodes))
    exec(compile(ast.Module(body=[construct], type_ignores=[]), str(path), "exec"), namespace)

    def build_layers(parallel, fail=False):
        event, finished = Event(), []
        def summarize(*, context, max_tokens):
            if fail and context == "m2 m3":
                raise RuntimeError("summary worker failed")
            return f"({context})"

        def create_node(index, text, children):
            if parallel and index == 8 and not fail:
                assert event.wait(5), "second worker did not finish concurrently"
            finished.append(index)
            if index == 9:
                event.set()
            return index, SimpleNamespace(index=index, text=text, children=children)

        def groups(nodes, *_, **__):
            return [nodes[i:i + 2] for i in range(0, len(nodes), 2)]

        builder = SimpleNamespace(num_layers=5, reduction_dimension=2, cluster_embedding_model="e5",
                                  clustering_params={}, clustering_algorithm=SimpleNamespace(perform_clustering=groups),
                                  tokenizer=SimpleNamespace(encode=str.split), summarize=summarize, create_node=create_node,
                                  summarization_length=256)
        leaves = {i: SimpleNamespace(index=i, text=f"m{i}", children=set()) for i in range(8)}
        layers = {0: list(leaves.values())}
        all_nodes = dict(leaves)
        namespace["construct_tree"](builder, leaves, all_nodes, layers, use_multithreading=parallel, max_workers=2)
        return all_nodes, layers, finished

    serial, serial_layers, _ = build_layers(False)
    parallel, parallel_layers, finished = build_layers(True)
    assert finished.index(9) < finished.index(8)
    assert list(parallel) == list(serial) and parallel == serial and parallel_layers == serial_layers
    with pytest.raises(RuntimeError, match="summary worker failed"):
        build_layers(True, fail=True)
