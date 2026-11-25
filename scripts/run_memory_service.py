"""Run a local non-thinking Qwen service only for the lifetime of one worker command."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def _signal_group(group, signum):
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        return False
    return True


def _stop_server(process):
    """Stop the server session, including any workers that outlive its leader."""
    deadline = time.monotonic() + 6
    if _signal_group(process.pid, signal.SIGTERM):
        while True:
            process.poll()  # Reap an exited leader while checking for its surviving workers.
            remaining = deadline - time.monotonic()
            if not _signal_group(process.pid, 0):
                break
            if remaining <= 0:
                _signal_group(process.pid, signal.SIGKILL)
                break
            time.sleep(min(0.1, remaining))
    process.wait(timeout=max(0, deadline - time.monotonic()) + 1)


def run(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (not command or not 0 < args.memory_fraction <= 1
            or min(args.max_sequences, args.max_batched_tokens, args.context_length) < 1):
        raise ValueError("Provide a worker command, valid GPU memory fraction and positive batch limits")
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template = "{%- set enable_thinking = false -%}" + tokenizer.chat_template
    example = [{"role": "system", "content": "Read the supplied data."},
               {"role": "user", "content": "A question is not a fact."}]
    expected = tokenizer.apply_chat_template(example, enable_thinking=False, add_generation_prompt=True)
    actual = tokenizer.apply_chat_template(example, chat_template=template, add_generation_prompt=True)
    if actual != expected:
        raise ValueError("Service template differs from the frozen non-thinking tokenizer")
    template_path = args.output / "chat-template.jinja"
    template_path.write_text(template, encoding="utf-8")
    server_command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                      "--model", str(args.model), "--served-model-name", "Qwen/Qwen3-8B",
                      "--host", "127.0.0.1", "--port", str(args.port),
                      "--dtype", "bfloat16", "--max-model-len", str(args.context_length),
                      "--gpu-memory-utilization", str(args.memory_fraction),
                      "--max-num-seqs", str(args.max_sequences),
                      "--max-num-batched-tokens", str(args.max_batched_tokens),
                      "--generation-config", "vllm",
                      "--chat-template", str(template_path), "--disable-log-requests"]
    if not args.cuda_graphs:
        server_command.append("--enforce-eager")
    manifest = {"server_command": server_command, "worker_command": command,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "runtime_versions": {name: version(name) for name in ("torch", "transformers", "vllm")},
                "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "starting"}
    record = args.output / "service.json"
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    process = None
    started = time.monotonic()
    launching, closing, interrupted = False, False, None

    def stop(signum, frame):
        nonlocal interrupted
        if closing or interrupted is not None:
            return
        interrupted = signum
        if not launching:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (args.output / "server.log").open("x") as log:
            launching = True
            process = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            launching = False
            if interrupted is not None:
                raise SystemExit(128 + interrupted)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Qwen server exited {process.returncode}; see {args.output / 'server.log'}")
                try:
                    with urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Qwen did not become ready within ten minutes")
            manifest.update(status="ready", loading_seconds=time.monotonic() - started)
            record.write_text(json.dumps(manifest, indent=2) + "\n")
            # Index workers use CPU embeddings/compression; the service alone owns the GPU.
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            result = subprocess.run(command, env=environment, check=False)
            manifest.update(status="complete" if result.returncode == 0 else "failed",
                            worker_exit_code=result.returncode)
            return result.returncode
    finally:
        closing = True
        try:
            if process is not None:
                _stop_server(process)
        except Exception:
            manifest["status"] = "cleanup_failed"
            raise
        finally:
            if manifest["status"] in ("starting", "ready"):
                manifest["status"] = "interrupted_or_failed"
            manifest.update(total_seconds=time.monotonic() - started,
                            ended_at_utc=datetime.now(timezone.utc).isoformat())
            record.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18951)
    parser.add_argument("--memory-fraction", type=float, default=0.38)
    parser.add_argument("--max-sequences", type=int, default=8)
    parser.add_argument("--max-batched-tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=8192,
                        help="Server input plus output token limit; use 16384 for LightMem sleep")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Allow vLLM compilation and CUDA graphs when a full card has sufficient memory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    raise SystemExit(run(parser.parse_args()))
