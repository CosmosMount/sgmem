"""Build a resumable proposition sidecar for an extraction pilot."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgmem import propositions
from sgmem.runner import _checkpoint, _git_info, write_json


@contextmanager
def _output_lock(path):
    """Keep one writer per checkpoint and let the OS release its lock on exit."""
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Cannot lock proposition output directory: {path.parent}") from exc
        yield


def _model_identity(path):
    if not (path / "config.json").is_file():
        raise ValueError(f"Expected a local model directory containing config.json: {path}")
    files = [item for item in sorted(path.iterdir()) if item.is_file()]
    return {"path": str(path),
            "configuration_sha256": {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                                     for item in files if item.suffix in (".json", ".txt", ".model", ".jinja")},
            "weight_sha256": {item.name: _file_digest(item) for item in files
                              if item.suffix in (".safetensors", ".bin")}}


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_versions():
    versions = {}
    for package in ("torch", "transformers", "tokenizers", "sentencepiece"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _http_service(args, model_path):
    if (args.endpoint != propositions.QWEN_HTTP_ENDPOINT or not args.server_manifest or args.device != "cpu"
            or args.max_new_tokens != 512 or args.model_revision != propositions.QWEN_REVISION
            or (model_path.parent.name == "snapshots" and model_path.name != args.model_revision)):
        raise ValueError("Qwen HTTP requires its pinned snapshot, CPU worker, 512 tokens, and the local service manifest")
    manifest = json.loads(args.server_manifest.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("Expected a vLLM service manifest object")
    command, versions = manifest.get("server_command"), manifest.get("runtime_versions")
    if (manifest.get("status") != "ready" or not isinstance(command, list)
            or not all(isinstance(argument, str) for argument in command)
            or command[1:3] != ["-m", "vllm.entrypoints.openai.api_server"]
            or not isinstance(versions, dict)
            or not all(isinstance(versions.get(name), str) and versions[name].strip()
                       for name in ("torch", "transformers", "vllm"))):
        raise ValueError("Expected the ready pinned vLLM service manifest with runtime versions")

    def option(flag):
        if command.count(flag) != 1 or command.index(flag) + 1 == len(command):
            raise ValueError(f"Invalid vLLM manifest option {flag}")
        return command[command.index(flag) + 1]

    required = {"--served-model-name": propositions.QWEN_MODEL, "--host": "127.0.0.1", "--port": "18951",
                "--max-model-len": str(propositions.QWEN_HTTP_CONTEXT)}
    if Path(option("--model")).resolve() != model_path or any(option(flag) != value for flag, value in required.items()):
        raise ValueError("vLLM model path, name, endpoint, or context differs from the fixed Qwen profile")
    extractor = propositions.HttpPropositionizer(model_path, args.endpoint)
    template = Path(option("--chat-template")).read_text(encoding="utf-8")
    digest = hashlib.sha256(template.encode()).hexdigest()
    if digest != manifest.get("chat_template_sha256") or digest != extractor.chat_template_sha256:
        raise ValueError("vLLM chat template differs from the pinned non-thinking Qwen tokenizer")
    if extractor.server_version() != versions["vllm"]:
        raise ValueError("Running vLLM version differs from the service manifest")
    # The template path changes with each service run. Its verified content identifies the profile.
    command[command.index("--chat-template") + 1] = digest
    command[command.index("--model") + 1] = str(model_path)
    return extractor, {"endpoint": args.endpoint, "server_command": command, "chat_template_sha256": digest,
                       "runtime_versions": {name: versions[name] for name in ("torch", "transformers", "vllm")}}


def build(args):
    if min(args.max_new_tokens, args.batch_size) <= 0 or not args.model_revision.strip():
        raise ValueError("Provide positive token and batch limits and the local model revision")
    model_path = args.model.resolve()
    output = args.output.resolve()
    source = args.messages.read_bytes()
    sources = propositions.source_messages(json.loads(line) for line in source.splitlines() if line.strip())
    http = args.backend == "qwen-http"
    extractor, service = None, None
    if http:
        http_started = perf_counter()
        http_started_at = datetime.now(timezone.utc).isoformat()
        extractor, service = _http_service(args, model_path)
    elif getattr(args, "endpoint", None) is not None or getattr(args, "server_manifest", None) is not None:
        raise ValueError("--endpoint and --server-manifest apply only to qwen-http")
    identity = {"schema": 1, "input_sha256": hashlib.sha256(source).hexdigest(),
                "backend": args.backend, "device": args.device, "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "project_commit": _git_info(ROOT)["commit"], "runtime_versions": _runtime_versions(),
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "extractor_sha256": hashlib.sha256(Path(propositions.__file__).read_bytes()).hexdigest(),
                "model": {**_model_identity(model_path), "declared_revision": args.model_revision}}
    if http:
        identity["http_service"] = service
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    rows_path = output / "propositions.jsonl"
    if args.resume and not manifest_path.is_file():
        raise ValueError(f"No proposition manifest to resume: {manifest_path}")
    with _output_lock(output / ".build.lock"):
        if args.resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["identity"] != identity:
                raise ValueError("Resume rejected: source, extractor, model, or generation settings changed")
            for session in manifest.get("sessions", []):
                if session["status"] == "running":
                    session.update(status="interrupted", wall_seconds=None)
            if manifest["status"] == "complete":
                with rows_path.open(encoding="utf-8") as file:
                    propositions.index_records(sources, (json.loads(line) for line in file))
                return rows_path
            if manifest["status"] != "incomplete":
                raise ValueError("Resume rejected: invalid manifest status")
        else:
            manifest = {"identity": identity, "status": "incomplete", "source_count": len(sources)}
            write_json(manifest_path, manifest)

        records = propositions.index_records(sources, _checkpoint(rows_path), partial=True)
        source_rows = list(sources.items())
        if list(records) != [key for key, _ in source_rows[:len(records)]]:
            raise ValueError("Completed propositions must follow the ordered source prefix")
        if len(records) < len(source_rows):
            session = {"status": "running", "records_before": len(records),
                       "started_at_utc": http_started_at if http else datetime.now(timezone.utc).isoformat()}
            manifest.setdefault("sessions", []).append(session)
            write_json(manifest_path, manifest)
            started = http_started if http else perf_counter()
            try:
                if extractor is None:
                    extractor = propositions.LocalPropositionizer(model_path, args.backend, args.device, args.max_new_tokens)
                first_batch = len(records) // args.batch_size * args.batch_size
                with rows_path.open("a", encoding="utf-8", newline="\n") as handle:
                    for offset in range(first_batch, len(source_rows), args.batch_size):
                        batch = dict(source_rows[offset:offset + args.batch_size])
                        batch_started = perf_counter()
                        extracted = extractor.extract_many(batch.values())
                        batch_seconds = perf_counter() - batch_started
                        indexed = propositions.index_records(batch, extracted)
                        if list(indexed) != list(batch):
                            raise ValueError("Proposition batch results do not follow the source order")
                        # Save actual extraction wall time once, even after a partial-batch replay.
                        batch_audit = {"batch_start_index": offset, "batch_source_rows": len(batch),
                                       "batch_generated_rows": sum(r["reason"] != "input_too_long" for r in extracted),
                                       "batch_wall_seconds": batch_seconds}
                        for key, record in indexed.items():
                            if key in records:
                                continue
                            record.update(batch_audit)
                            batch_audit = {}
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                            records[key] = record
            finally:
                session.update(status="complete" if len(records) == len(sources) else "interrupted",
                               wall_seconds=perf_counter() - started, records_after=len(records),
                               ended_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(manifest_path, manifest)
        manifest.update(status="complete", record_count=len(records),
                        fallback_count=sum(row["status"] == "fallback" for row in records.values()),
                        timing_complete=bool(manifest.get("sessions")) and all(
                            session.get("wall_seconds") is not None for session in manifest["sessions"]))
        write_json(manifest_path, manifest)
    return rows_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True, help="Declared revision of the local model snapshot")
    parser.add_argument("--backend", choices=("flan-t5", "qwen", "qwen-http"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of source messages per generation batch")
    parser.add_argument("--endpoint", help="Loopback Qwen API base URL for qwen-http")
    parser.add_argument("--server-manifest", type=Path, help="Ready local vLLM service.json for qwen-http")
    parser.add_argument("--resume", action="store_true")
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
