import json
import runpy
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
from io import StringIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from sgmem import propositions
from sgmem.propositions import LocalPropositionizer, PropositionRetriever, _parse_propositions, source_digest


def source(id="Msg_1", domain="Finance"):
    return {"domain": domain, "id": id, "content": "Do not proceed unless the test passes &amp; is signed.",
            "speaker": "User_2", "group": "Project A", "timestamp": "2026-01-01",
            "reply_to": "Msg_0", "answer": "GOLD-DO-NOT-READ"}


def record(message, *units):
    return {"domain": message["domain"], "message_id": message["id"], "units": list(units),
            "source_sha256": source_digest(message),
            "status": "ok", "reason": None, "raw_output": json.dumps(units), "input_tokens": 2,
            "tokenized_input_tokens": 2,
            "output_tokens": 4, "seconds": 0.1}


@pytest.mark.parametrize(("text", "expected"), [
    ('["P", " P ", "Q"]', (["P", "Q"], None)),
    ('[]', ([], "empty")),
    ('["P", 2]', ([], "malformed")),
    ('["  "]', ([], "malformed")),
    ('{"text":"P"}', ([], "malformed")),
    ('```json\n["P"]\n```', ([], "malformed")),
    ('["P"', ([], "malformed")),
])
def test_strict_proposition_parsing(text, expected):
    assert _parse_propositions(text) == expected


@pytest.mark.parametrize("backend", ["flan-t5", "qwen"])
def test_local_backends_use_content_only_and_decode_generation(monkeypatch, backend):
    causal = backend == "qwen"
    prefix = [10, 11, 12] if causal else [0]
    body = "Do not proceed unless the test passes & is signed."
    state = SimpleNamespace(generated=[7, 8, 1], calls=0)

    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        model_max_length = 16

        def __call__(self, text, **kwargs):
            assert not causal and text == f"Title: . Section: . Content: {body}"
            assert kwargs["truncation"] is False
            return Batch(input_ids=np.array([[10, 11, 12]]))

        def apply_chat_template(self, messages, **kwargs):
            assert causal and kwargs["enable_thinking"] is False
            assert kwargs["truncation"] is False
            assert messages[1] == {"role": "user", "content": body}
            assert "GOLD-DO-NOT-READ" not in str(messages)
            assert "User_2" not in str(messages)
            assert "conditions" in messages[0]["content"] and "negation" in messages[0]["content"]
            return Batch(input_ids=np.array([[10, 11, 12]]))

        def decode(self, generated, **kwargs):
            assert generated == state.generated and kwargs["skip_special_tokens"] is True
            return '["Do not proceed unless the test passes and is signed."]'

    class Model:
        config = SimpleNamespace(max_position_embeddings=16)
        generation_config = SimpleNamespace(eos_token_id=1)

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 5
            state.calls += 1
            return np.array([prefix + state.generated])

    def load_tokenizer(path, **kwargs):
        assert path == "local" and kwargs == {"local_files_only": True, "use_fast": causal}
        return Tokenizer()

    def model_loader(is_causal):
        def load_model(path, **kwargs):
            assert is_causal == causal and path == "local" and kwargs["local_files_only"] is True
            return Model()
        return SimpleNamespace(from_pretrained=load_model)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fp32", inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=model_loader(True), AutoModelForSeq2SeqLM=model_loader(False),
    ))
    extractor = LocalPropositionizer("local", backend, "cpu", 5)
    assert extractor.tokenizer.padding_side == ("left" if causal else "right")
    extracted = extractor.extract(source())
    assert (extracted["status"], extracted["input_tokens"], extracted["output_tokens"]) == ("ok", 3, 3)
    assert extracted["units"] == ["Do not proceed unless the test passes and is signed."]

    state.generated = [7, 8]
    truncated = extractor.extract(source())
    assert truncated["status"] == "fallback" and truncated["reason"] == "truncated"
    assert truncated["units"] == [body] and truncated["raw_output"].startswith("[")

    extractor.input_limit = 2
    long_input = extractor.extract(source())
    assert long_input["reason"] == "input_too_long" and long_input["output_tokens"] == 0
    assert long_input["input_tokens"] == 0 and long_input["tokenized_input_tokens"] == 3
    assert state.calls == 2


@pytest.mark.parametrize("causal", [False, True])
def test_batch_generation_masks_padding_and_preserves_source_alignment(monkeypatch, causal):
    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    short, too_long, medium, last = [
        {**source(f"Msg_{i}"), "content": text}
        for i, text in enumerate(("Short.", "Long source.", "Medium.", "Last."))
    ]
    tokens = {"Short.": [11, 12], "Long source.": list(range(10)),
              "Medium.": [21, 22, 23, 24], "Last.": [31, 32, 33]}
    eligible = [tokens[message["content"]] for message in (short, medium, last)]
    if causal:
        input_ids = [[0, 0, 11, 12], [21, 22, 23, 24], [0, 31, 32, 33]]
        masks = [[0, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]
    else:
        input_ids = [[11, 12, 0, 0], [21, 22, 23, 24], [31, 32, 33, 0]]
        masks = [[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]]

    def pad(value, **kwargs):
        assert value == {"input_ids": eligible}
        assert kwargs == {"padding": True, "return_attention_mask": True, "return_tensors": "pt"}
        return Batch(input_ids=np.array(input_ids), attention_mask=np.array(masks))

    decoded = []

    def decode(value, **kwargs):
        assert kwargs["skip_special_tokens"]
        decoded.append(value)
        return '["Short proposition."]' if value[0] == 40 else '["Last proposition."]'

    calls = []

    def generate(**kwargs):
        calls.append(kwargs["max_new_tokens"])
        assert kwargs["input_ids"].tolist() == input_ids
        assert kwargs["attention_mask"].tolist() == masks
        prefixes = input_ids if causal else [[0]] * 3
        continuations = [[40, 1, 0, 0], [50, 51, 52, 53], [60, 61, 2, 0]]
        return np.array([prefix + continuation for prefix, continuation in zip(prefixes, continuations)])

    extractor = LocalPropositionizer.__new__(LocalPropositionizer)
    extractor.causal, extractor.input_limit, extractor.max_new_tokens = causal, 5, 4
    extractor.device, extractor.eos = "cpu", {1, 2}
    extractor.torch = SimpleNamespace(inference_mode=nullcontext)
    extractor.tokenizer = SimpleNamespace(pad=pad, decode=decode)
    extractor.model = SimpleNamespace(generate=generate)
    monkeypatch.setattr(extractor, "_tokenize", lambda content: Batch(input_ids=np.array([tokens[content]])))
    perf = iter((3.0, 5.0))
    monkeypatch.setattr("sgmem.propositions.perf_counter", lambda: next(perf))

    rows = extractor.extract_many([short, too_long, medium, last])
    assert calls == [4] and decoded == [[40, 1], [50, 51, 52, 53], [60, 61, 2]]
    assert [row["message_id"] for row in rows] == [m["id"] for m in (short, too_long, medium, last)]
    assert [row["reason"] for row in rows] == [None, "input_too_long", "truncated", None]
    assert [row["units"] for row in rows] == [
        ["Short proposition."], [too_long["content"]], [medium["content"]], ["Last proposition."],
    ]
    assert [row["input_tokens"] for row in rows] == [2, 0, 4, 3]
    assert [row["tokenized_input_tokens"] for row in rows] == [2, 10, 4, 3]
    assert [row["output_tokens"] for row in rows] == [2, 0, 4, 3]
    assert sum(row["seconds"] for row in rows) == 2.0


@pytest.fixture
def http_extractor(monkeypatch):
    template = "local chat template"

    def tokenize(messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
                          "return_dict": False, "truncation": False}
        assert messages[0] == {"role": "system", "content": propositions.QWEN_PROMPT}
        assert messages[1]["role"] == "user"
        return [0] * (7681 if messages[1]["content"] == "over limit" else 7)

    def load(path, **kwargs):
        assert path == "local" and kwargs == {"local_files_only": True, "use_fast": True}
        return SimpleNamespace(chat_template=template, apply_chat_template=tokenize)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    extractor = propositions.HttpPropositionizer("local", propositions.QWEN_HTTP_ENDPOINT)
    assert extractor.chat_template_sha256 == hashlib.sha256(("{%- set enable_thinking = false -%}" + template).encode()).hexdigest()
    return extractor


def _http_completion(raw='["Proposition."]', finish="stop", *, model=propositions.QWEN_MODEL, prompt_tokens=7):
    return {"model": model, "choices": [{"finish_reason": finish, "message": {"content": raw}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 512 if finish == "length" else 6}}


def test_http_batches_are_bounded_concurrent_ordered_and_match_fallback(monkeypatch, http_extractor):
    rows = [{**source(str(i)), "content": text} for i, text in enumerate((
        "first", "second", "object", "empty", "over limit", "sixth", "seventh", "eighth", "ninth",
    ))]
    completion_order, caps = [], []
    second_finished = Event()

    def pool(**kwargs):
        caps.append(kwargs["max_workers"])
        return ThreadPoolExecutor(**kwargs)

    class Connection:
        def __init__(self, host, port, *, timeout):
            assert (host, port, timeout) == ("127.0.0.1", 18951, 180)

        def request(self, method, path, data=None, headers=None):
            if method == "GET":
                assert path == "/version"
                self.data = {"version": "test-version"}
                return
            assert (method, path, headers) == ("POST", "/v1/chat/completions", {"Content-Type": "application/json"})
            body = json.loads(data)
            messages = body.pop("messages")
            assert body == {"model": propositions.QWEN_MODEL, "temperature": 0, "seed": 42, "max_tokens": 512}
            assert messages[0]["content"] == propositions.QWEN_PROMPT
            assert "GOLD-DO-NOT-READ" not in str(messages)
            content = messages[1]["content"]
            assert content != "over limit"
            if content == "first":
                assert second_finished.wait(5), "Second request was never issued concurrently"
            completion_order.append(content)
            if content == "second":
                second_finished.set()
            self.data = {"second": _http_completion('["Partial."]', "length"),
                         "object": _http_completion('{"text":"P"}'), "empty": _http_completion('[]')}.get(
                             content, _http_completion())

        def getresponse(self):
            response = StringIO(json.dumps(self.data))
            response.status = 200
            return response

        def close(self):
            pass

    monkeypatch.setattr(propositions, "ThreadPoolExecutor", pool)
    monkeypatch.setattr(propositions, "HTTPConnection", Connection)
    assert http_extractor.server_version() == "test-version"
    perf = iter((3.0, 12.0))
    monkeypatch.setattr(propositions, "perf_counter", lambda: next(perf))
    records = http_extractor.extract_many(rows)
    assert caps == [9] and completion_order.index("second") < completion_order.index("first")
    assert [record["message_id"] for record in records] == [message["id"] for message in rows]
    assert [r["reason"] for r in records[:5]] == [None, "truncated", "malformed", "empty", "input_too_long"]
    assert [r["units"] for r in records[1:5]] == [[message["content"]] for message in rows[1:5]]
    assert records[1]["output_tokens"] == 512 and records[4]["input_tokens"] == 0
    assert sum(r["seconds"] for r in records) == 9.0


@pytest.mark.parametrize("response", [
    _http_completion(prompt_tokens=8), _http_completion(model="other-model"),
    _http_completion(finish="content_filter"), {"choices": []},
])
def test_http_schema_fault_aborts_instead_of_falling_back(monkeypatch, http_extractor, response):
    monkeypatch.setattr(http_extractor, "_request", lambda path, payload: response)
    with pytest.raises(ValueError, match="Invalid Qwen HTTP response for Finance/Msg_1"):
        http_extractor.extract(source())


def test_http_does_not_follow_redirects(monkeypatch, http_extractor):
    connection = SimpleNamespace(request=lambda *args: None, getresponse=lambda: SimpleNamespace(status=302),
                                 close=lambda: None)
    monkeypatch.setattr(propositions, "HTTPConnection", lambda *args, **kwargs: connection)
    with pytest.raises(RuntimeError, match="status 302"):
        http_extractor.extract(source())


class Encoder:
    def encode(self, texts, **kwargs):
        assert kwargs == {"batch_size": 64, "normalize_embeddings": True,
                          "convert_to_numpy": True, "show_progress_bar": False}
        lookup = {"passage: leading": [0, 1], "passage: best": [1, 0], "passage: equal": [1, 0],
                  "query: User_2 what happened?": [1, 0]}
        return [lookup[text] for text in texts]


def test_unique_parent_top_k_keeps_provenance_and_stable_ties():
    first, second, third = source(), source("Msg_2"), source("Msg_3")
    rows = [record(second, "best"), record(first, "leading", "best", "equal"), record(third, "leading")]
    hits = PropositionRetriever([first, second, third], rows, Encoder()).retrieve(
        "what happened?", 2, asking_user_id="User_2",
    )
    assert len(hits) == 2 and hits[0]["message"] is first and hits[1]["message"] is second
    assert hits[0]["score"] == 1 and hits[0]["winning_unit"] == {"ordinal": 1, "text": "best", "status": "ok"}
    assert hits[0]["message"]["reply_to"] == "Msg_0"


def test_reject_cross_domain_index():
    first, other = source(), source(domain="Technology")
    with pytest.raises(ValueError, match="one domain per index"):
        PropositionRetriever([first, other], [record(first, "best"), record(other, "best")], Encoder())


def test_reject_missing_wrong_domain_and_malformed_records():
    message = source()
    valid = record(message, "best")
    for rows in ([], [valid, valid], [{**valid, "domain": "Manufacturing"}], [{**valid, "units": ["best", "best"]}],
                 [{**valid, "input_tokens": -1}],
                 [{**valid, "status": "fallback", "reason": "truncated"}]):
        with pytest.raises(ValueError):
            PropositionRetriever([message], rows, Encoder())


def test_reject_stale_source_and_units_changed_after_extraction():
    message = source()
    valid = record(message, "best")
    with pytest.raises(ValueError, match="source content changed"):
        PropositionRetriever([{**message, "content": "A different statement."}], [valid], Encoder())
    with pytest.raises(ValueError, match="recorded model output"):
        PropositionRetriever([message], [{**valid, "units": ["leading"]}], Encoder())


@pytest.mark.parametrize("vectors", ([], [[np.nan, 0.0]], [[0.0, 0.0]]))
def test_reject_invalid_embedding_vectors(vectors):
    message = source()
    encoder = SimpleNamespace(encode=lambda texts, **kwargs: vectors)
    with pytest.raises(ValueError, match="invalid or empty embeddings"):
        PropositionRetriever([message], [record(message, "best")], encoder)


def test_resume_keeps_completed_rows_and_rejects_changed_model(tmp_path, monkeypatch):
    from sgmem import propositions

    build = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "build_propositions.py"))["build"]
    messages = tmp_path / "messages.jsonl"
    messages.write_text("".join(json.dumps(item) + "\n" for item in (source(), source(domain="Technology"))),
                        encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"model_type":"t5"}', encoding="utf-8")
    weight = model / "model.safetensors"
    weight.write_bytes(b"original weights")
    output = tmp_path / "output"
    args = Namespace(messages=messages, model=model, output=output, device="cpu", backend="flan-t5",
                     model_revision="pinned-revision", max_new_tokens=512, batch_size=1, resume=False)
    attempted = []

    class Interrupted:
        def __init__(self, *args):
            pass

        def extract_many(self, batch):
            message, = batch
            attempted.append(message["domain"])
            if len(attempted) == 2:
                raise RuntimeError("mock generation interrupted")
            return [record(message, "best")]

    monkeypatch.setattr(propositions, "LocalPropositionizer", Interrupted)
    with pytest.raises(RuntimeError, match="mock generation interrupted"):
        build(args)
    manifest = output / "manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "incomplete"
    rows = output / "propositions.jsonl"
    with rows.open("ab") as handle:
        handle.write(b'{"domain":')
    args.resume = True
    config.write_text('{"model_type":"qwen3"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Resume rejected"):
        build(args)
    assert attempted == ["Finance", "Technology"]
    config.write_text('{"model_type":"t5"}', encoding="utf-8")
    args.model_revision = "different-revision"
    with pytest.raises(ValueError, match="Resume rejected"):
        build(args)
    args.model_revision = "pinned-revision"
    args.batch_size = 2
    with pytest.raises(ValueError, match="Resume rejected"):
        build(args)
    args.batch_size = 1
    weight.write_bytes(b"modified weights")
    with pytest.raises(ValueError, match="Resume rejected"):
        build(args)
    weight.write_bytes(b"original weights")
    assert build(args) == rows and build(args) == rows
    parsed = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()]
    assert [item["domain"] for item in parsed] == ["Finance", "Technology"]
    assert attempted == ["Finance", "Technology", "Technology"]
    complete = json.loads(manifest.read_text(encoding="utf-8"))
    assert complete["status"] == "complete"
    assert [session["status"] for session in complete["sessions"]] == ["interrupted", "complete"]
    assert all(session["wall_seconds"] > 0 for session in complete["sessions"])
    assert complete["timing_complete"] is True
    # Simulate a hard kill after the final durable row but before timing/manifest completion.
    complete["status"] = "incomplete"
    complete["sessions"][-1] = {"status": "running", "started_at_utc": "2026-01-01T00:00:00+00:00"}
    manifest.write_text(json.dumps(complete), encoding="utf-8")
    assert build(args) == rows
    recovered = json.loads(manifest.read_text(encoding="utf-8"))
    assert recovered["status"] == "complete" and recovered["timing_complete"] is False
    assert recovered["sessions"][-1]["wall_seconds"] is None
    assert attempted == ["Finance", "Technology", "Technology"]


def test_build_resume_keeps_fixed_batch_boundaries_and_one_timing_receipt(tmp_path, monkeypatch):
    from sgmem import propositions

    build = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "build_propositions.py"))["build"]
    messages = tmp_path / "messages.jsonl"
    messages.write_text("".join(json.dumps(source(f"Msg_{i}")) + "\n" for i in range(3)), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    args = Namespace(messages=messages, model=model, output=tmp_path / "output", device="cpu", backend="qwen",
                     model_revision="pinned-revision", max_new_tokens=128, batch_size=2, resume=False)
    calls = []

    class Extractor:
        def __init__(self, *args):
            pass

        def extract_many(self, batch):
            calls.append([message["id"] for message in batch])
            return [record(message, "best") for message in batch]

    monkeypatch.setattr(propositions, "LocalPropositionizer", Extractor)
    output = build(args)
    assert calls == [["Msg_0", "Msg_1"], ["Msg_2"]]
    lines = output.read_text(encoding="utf-8").splitlines()
    first, second, third = [json.loads(line) for line in lines]
    assert first["batch_start_index"] == 0 and first["batch_source_rows"] == first["batch_generated_rows"] == 2
    assert first["batch_wall_seconds"] > 0 and "batch_wall_seconds" not in second
    assert third["batch_start_index"] == 2 and third["batch_source_rows"] == 1

    # Simulate losing the rows after the first durable write of batch zero.
    output.write_text(lines[0] + "\n", encoding="utf-8")
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"]["batch_size"] == 2
    manifest["status"] = "incomplete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args.resume = True
    assert build(args) == output
    assert calls[2:] == [["Msg_0", "Msg_1"], ["Msg_2"]]
    rerun = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["message_id"] for row in rerun] == ["Msg_0", "Msg_1", "Msg_2"]
    assert rerun[0] == first and rerun[1]["batch_start_index"] == 0 and rerun[1]["batch_wall_seconds"] > 0


def test_http_build_locks_service_profile_across_restarts(tmp_path, monkeypatch):
    build = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "build_propositions.py"))["build"]
    messages = tmp_path / "messages.jsonl"
    messages.write_text(json.dumps(source()) + "\n", encoding="utf-8")
    model = tmp_path / "snapshots" / propositions.QWEN_REVISION
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    template = "{%- set enable_thinking = false -%}local chat template"
    digest = hashlib.sha256(template.encode()).hexdigest()
    versions = {"torch": "torch-test", "transformers": "transformers-test", "vllm": "first-version"}
    state = SimpleNamespace(version=versions["vllm"], batches=0)

    def service(name, **changes):
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        chat = folder / "chat-template.jinja"
        chat.write_text(template, encoding="utf-8")
        manifest = folder / "service.json"
        command = ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
                   "--served-model-name", propositions.QWEN_MODEL, "--host", "127.0.0.1", "--port", "18951",
                   "--max-model-len", "8192", "--chat-template", str(chat)]
        manifest.write_text(json.dumps({"status": "ready", "server_command": command,
                                        "runtime_versions": versions, "chat_template_sha256": digest,
                                        "started_at_utc": name, **changes}), encoding="utf-8")
        return manifest

    class Extractor:
        chat_template_sha256 = digest

        def __init__(self, path, endpoint):
            assert path == model and endpoint == propositions.QWEN_HTTP_ENDPOINT

        def server_version(self):
            return state.version

        def extract_many(self, batch):
            state.batches += 1
            return [record(message, "best") for message in batch]

    monkeypatch.setattr(propositions, "HttpPropositionizer", Extractor)
    args = Namespace(messages=messages, model=model, output=tmp_path / "output", device="cpu", backend="qwen-http",
                     model_revision=propositions.QWEN_REVISION, max_new_tokens=512, batch_size=8, resume=False,
                     endpoint=propositions.QWEN_HTTP_ENDPOINT, server_manifest=service("launch-one"))
    output = build(args)
    identity = json.loads((args.output / "manifest.json").read_text())["identity"]["http_service"]
    assert identity["runtime_versions"] == versions and "launch-one" not in str(identity)
    assert identity["chat_template_sha256"] == digest
    args.resume, args.server_manifest = True, service("launch-two")
    assert build(args) == output and state.batches == 1

    state.version = "new-version"
    with pytest.raises(ValueError, match="Running vLLM version differs"):
        build(args)
    versions["vllm"] = state.version
    args.server_manifest = service("launch-three")
    with pytest.raises(ValueError, match="Resume rejected"):
        build(args)
    args.server_manifest = service("launch-four", status="complete")
    with pytest.raises(ValueError, match="ready pinned vLLM"):
        build(args)
    state.version = versions["vllm"]
    args.server_manifest = service("launch-five")
    args.server_manifest.with_name("chat-template.jinja").write_text("different template", encoding="utf-8")
    with pytest.raises(ValueError, match="chat template differs"):
        build(args)

    for flag, value in (("--model", str(tmp_path / "wrong-model")), ("--served-model-name", "other-model"),
                        ("--max-model-len", "4096")):
        args.server_manifest = service("wrong-profile")
        manifest = json.loads(args.server_manifest.read_bytes())
        command = manifest["server_command"]
        command[command.index(flag) + 1] = value
        args.server_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="differs from the fixed Qwen profile"):
            build(args)


def test_only_one_writer_can_hold_output_lock(tmp_path):
    lock = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "build_propositions.py"))["_output_lock"]
    path = tmp_path / ".build.lock"
    with lock(path), pytest.raises(RuntimeError, match="Cannot lock proposition output directory"):
        with lock(path):
            pytest.fail("A second writer acquired the checkpoint lock")
    with lock(path):
        assert path.is_file()
