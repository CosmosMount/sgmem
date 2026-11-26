"""Run patched LightMem sleep code on CPU with a pinned checkout and fake I/O."""

import ast
from collections import Counter
import concurrent.futures
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import runpy
import subprocess
import threading
from types import SimpleNamespace

import pytest

from sgmem.lightmem_memory import PATCH, UPSTREAM_COMMIT


def _load_method(path, class_name, method_name, **namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.fixture(scope="module")
def sleep_methods(tmp_path_factory):
    upstream = os.environ.get("SGMEM_LIGHTMEM_TEST_UPSTREAM")
    if not upstream:
        pytest.skip("Set SGMEM_LIGHTMEM_TEST_UPSTREAM to a checkout containing the pinned LightMem commit")
    paths = (
        "src/lightmem/memory/lightmem.py",
        "src/lightmem/factory/memory_manager/openai.py",
        "src/lightmem/memory/prompts.py",
    )
    work = tmp_path_factory.mktemp("lightmem")
    for relative in paths:
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "-C", upstream, "show", f"{UPSTREAM_COMMIT}:{relative}"]))
    subprocess.run(
        ["git", "-C", str(work), "apply", "--whitespace=nowarn", *(f"--include={p}" for p in paths), str(PATCH)],
        check=True,
    )
    prompts = runpy.run_path(str(work / paths[2]))
    prompt = prompts["UPDATE_PROMPT"]
    return (
        _load_method(work / paths[0], "LightMemory", "offline_update_all_entries",
                     datetime=datetime, concurrent=concurrent, threading=threading, UPDATE_PROMPT=prompt),
        _load_method(work / paths[1], "OpenaiManager", "_call_update_llm", json=json, SLEEP_REWRITE_PROMPT=prompts["SLEEP_REWRITE_PROMPT"]),
        prompt,
    )


def _point(identity, instant=0, score=None):
    queue = [] if score is None else [{"id": "target", "score": score}]
    return {"id": identity, "payload": {"memory": identity, "float_time_stamp": instant,
                                        "source_ids": ["shared", identity], "update_queue": queue}}


@pytest.mark.parametrize("cap", [None, 20])
def test_sleep_ranks_incoming_edges_and_records_only_selected_lineage(sleep_methods, cap):
    sleep, _, prompt = sleep_methods
    target = _point("target")
    sources = [
        _point("below-threshold", 15, 0.899), _point("threshold", 14, 0.9),
        _point("tie-b", 11, 0.92), _point("tie-older", 10, 0.92), _point("tie-a", 11, 0.92),
        *[_point(f"high-{i}", i, 0.98) for i in range(1, 9)],
    ]
    expected = [f"high-{i}" for i in range(8, 0, -1)] + ["tie-a", "tie-b"]
    if cap is not None:
        expected += ["tie-older", "threshold"]
    calls, changes = [], []

    def update_llm(system, point, candidates):
        calls.append((system, point["id"], [source["id"] for source in candidates]))
        return {"action": "update", "selected_candidate_index": 0,
                "new_memory": "Revised memory.", "usage": {"total_tokens": 4}}

    memory = SimpleNamespace(
        logger=logging.getLogger(__name__),
        embedding_retriever=SimpleNamespace(
            get_all=lambda: [*sources, target],
            update=lambda **change: changes.append(change),
        ),
        manager=SimpleNamespace(_call_update_llm=update_llm),
        text_embedder=SimpleNamespace(embed=lambda text: [0.75]),
        token_stats=Counter(),
    )
    sleep(memory, **({} if cap is None else {"max_candidate_sources": cap}))
    assert calls == [(prompt, "target", expected)]
    indexed, = changes
    assert indexed["vector_id"] == "target"
    assert indexed["payload"]["memory"] == "Revised memory."
    assert indexed["payload"]["source_ids"] == ["shared", "target", expected[0]]
    assert indexed["payload"]["float_time_stamp"] == 0
    assert indexed["vector"] == [0.75]

