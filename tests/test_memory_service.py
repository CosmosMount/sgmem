import json
from contextlib import nullcontext
from pathlib import Path
import runpy
import socket
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def service(tmp_path, monkeypatch):
    run = runpy.run_path(str(SCRIPTS / "run_memory_service.py"))["run"]
    state = SimpleNamespace(now=0, group_alive=True, leader_exit=None, stubborn=False, calls=[],
                            handlers={}, worker_exit=0, during_launch=False, during_worker=False)
    fake_signal = SimpleNamespace(SIGTERM=15, SIGINT=2, SIGKILL=9,
                                  signal=lambda signum, handler: state.handlers.update({signum: handler}))

    def killpg(group, signum):
        assert group == 12345
        if not state.group_alive:
            raise ProcessLookupError
        if signum != 0:
            state.calls.append(signum)
        if signum == fake_signal.SIGTERM:
            state.handlers[fake_signal.SIGTERM](fake_signal.SIGTERM, None)
            state.leader_exit = 0
            state.group_alive = state.stubborn
        if signum == fake_signal.SIGKILL:
            state.group_alive = False

    def wait(*, timeout):
        assert 0 < timeout <= 7
        assert state.leader_exit == 0
        return state.leader_exit

    process = SimpleNamespace(pid=12345, poll=lambda: state.leader_exit, wait=wait)

    def popen(command, **kwargs):
        assert kwargs["start_new_session"] and command[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        state.server_command = command
        if state.during_launch:
            state.handlers[fake_signal.SIGTERM](fake_signal.SIGTERM, None)
        return process

    def worker(command, *, env, check):
        assert command == ["worker"] and env["CUDA_VISIBLE_DEVICES"] == "" and not check
        if state.during_worker:
            state.handlers[fake_signal.SIGTERM](fake_signal.SIGTERM, None)
        return SimpleNamespace(returncode=state.worker_exit)

    def sleep(duration):
        state.now += duration

    state.socket_options = []
    probe = SimpleNamespace(bind=lambda *_: None, setsockopt=lambda *args: state.socket_options.append(args))
    props = {
        "version": lambda name: "test-runtime",
        "os": SimpleNamespace(killpg=killpg, environ={"CUDA_VISIBLE_DEVICES": "2"}),
        "signal": fake_signal,
        "socket": SimpleNamespace(socket=lambda: nullcontext(probe), SOL_SOCKET=socket.SOL_SOCKET,
                                  SO_REUSEADDR=socket.SO_REUSEADDR),
        "subprocess": SimpleNamespace(Popen=popen, run=worker, STDOUT=-2),
        "time": SimpleNamespace(monotonic=lambda: state.now, sleep=sleep),
        "urlopen": lambda *a, **kw: nullcontext(SimpleNamespace(status=200)),
    }
    for name, value in props.items():
        monkeypatch.setitem(run.__globals__, name, value)
    tokenizer = SimpleNamespace(chat_template="template", apply_chat_template=lambda *a, **kw: [1, 2, 3])
    auto = SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=auto))
    args = SimpleNamespace(command=["--", "worker"], output=tmp_path / "service", memory_fraction=0.38,
                           model=tmp_path / "model", port=18951, max_sequences=8, max_batched_tokens=4096,
                           cuda_graphs=False, context_length=8192)
    return run, args, state


@pytest.mark.parametrize(("worker_exit", "status"), [(0, "complete"), (17, "failed")])
def test_server_group_closes_after_worker(service, worker_exit, status):
    run, args, state = service
    state.worker_exit = worker_exit
    assert run(args) == worker_exit
    assert state.socket_options == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
    assert state.calls == [15] and not state.group_alive
    assert json.loads((args.output / "service.json").read_text())["status"] == status


def test_busy_port_does_not_leave_an_unusable_output_directory(service, monkeypatch):
    run, args, state = service

    def occupied(address):
        raise OSError("Address already in use")

    probe = SimpleNamespace(setsockopt=lambda *args: None, bind=occupied)
    monkeypatch.setattr(run.__globals__["socket"], "socket", lambda: nullcontext(probe))
    with pytest.raises(OSError, match="already in use"):
        run(args)
    assert not args.output.exists()
    assert not hasattr(state, "server_command")


def test_server_escalates_even_if_leader_exited(service):
    run, args, state = service
    state.stubborn = True
    assert run(args) == 0
    assert state.calls == [15, 9] and not state.group_alive
    assert state.now == 6


def test_full_card_configuration_is_recorded_and_forwarded(service):
    run, args, state = service
    args.memory_fraction, args.max_sequences, args.max_batched_tokens = 0.85, 32, 8192
    args.cuda_graphs = True
    args.context_length = 16384
    assert run(args) == 0
    command = state.server_command
    assert "--enforce-eager" not in command
    for flag, value in (("--gpu-memory-utilization", "0.85"), ("--max-num-seqs", "32"),
                        ("--max-num-batched-tokens", "8192"), ("--max-model-len", "16384")):
        assert command[command.index(flag) + 1] == value
    assert json.loads((args.output / "service.json").read_text())["server_command"] == command


@pytest.mark.parametrize("stage", ["launch", "worker"])
def test_sigterm_cleans_group_even_during_launch(service, stage):
    run, args, state = service
    setattr(state, f"during_{stage}", True)
    with pytest.raises(SystemExit) as exit_info:
        run(args)
    assert exit_info.value.code == 143
    assert state.calls == [15] and not state.group_alive
    report = json.loads((args.output / "service.json").read_text())
    assert report["status"] == "interrupted_or_failed" and report["ended_at_utc"]


def test_smoke_preserves_lightmem_and_hippo_results_when_core_path_is_missing(tmp_path, monkeypatch):
    run = runpy.run_path(str(SCRIPTS / "smoke_memory_pipelines.py"))["run"]
    messages = [{"domain": "Finance", "id": str(i), "content": f"Source message {i}"} for i in range(10)]
    path = tmp_path / "messages.jsonl"
    path.write_text("\n".join(json.dumps(message) for message in messages), encoding="utf-8")
    hits = [{"message": row, "score": 0.5} for row in messages]
    metrics = {"changed_turns": 4, "segmentation_calls": 2, "embedding_calls": {"turn": 9},
               "llm": {"add_memory": {"calls": 3}, "update": {"calls": 0}}, "update_actions": {}}
    lightmem = SimpleNamespace(
        build_lightmem=lambda *a, **kw: {"metrics": [metrics]},
        LightMemIndex=SimpleNamespace(load=lambda *a: SimpleNamespace(
            retrieve=lambda *a: hits, retrieve_native=lambda *a: [])),
    )
    hippo = SimpleNamespace(HippoRAGMemory=lambda *a, **kw: SimpleNamespace(
        build=lambda: None, close=lambda: None, engine=SimpleNamespace(get_graph_info=lambda: {}),
        retrieve_with_trace=lambda *a: (hits, {"recognition": {"mode": "dense_valid_empty"}}),
    ))
    monkeypatch.setitem(sys.modules, "sgmem.lightmem_memory", lightmem)
    monkeypatch.setitem(sys.modules, "sgmem.hipporag_memory", hippo)
    settings = dict(messages=path, vendor=tmp_path, endpoint="http://127.0.0.1:18951/v1",
                    embedding_model=tmp_path, compressor_model=tmp_path, methods=["lightmem", "hipporag2"])
    output = tmp_path / "incomplete"
    assert run(SimpleNamespace(output=output, **settings)) == 1
    results = json.loads((output / "summary.json").read_text())["methods"]
    assert results["lightmem"]["missing_coverage"] == ["sleep_calls"]
    assert results["hipporag2"]["missing_coverage"] == ["ppr_retrievals"]
    assert all(method["status"] == "incomplete_coverage" and method["result"] for method in results.values())

    metrics["llm"]["update"]["calls"] = 1
    complete = tmp_path / "lightmem-exercised"
    assert run(SimpleNamespace(output=complete, **dict(settings, methods=["lightmem"]))) == 0
    stage = json.loads((complete / "summary.json").read_text())["methods"]["lightmem"]
    assert stage["status"] == "complete" and stage["coverage"]["sleep_mutations"] == 0
