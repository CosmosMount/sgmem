"""Run one memory-pipeline GPU stage with a per-stage watchdog."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

def select_gpu(rows, occupied, minimum):
    eligible = [row for row in rows if row["free_mib"] >= minimum]
    eligible.sort(key=lambda row: (row["uuid"] in occupied, -row["free_mib"], row["utilization"]))
    return eligible[0] if eligible else None


def available_gpu(minimum):
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    occupied = set(subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True).splitlines())
    rows = []
    for line in text.splitlines():
        index, uuid, free, utilization = [value.strip() for value in line.split(",")]
        rows.append({"index": index, "uuid": uuid, "free_mib": int(free), "utilization": int(utilization)})
    return select_gpu(rows, occupied, minimum)


def used_seconds(rows):
    values = [row["wall_seconds"] for row in rows]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid GPU ledger duration; reconcile it before another stage")
    if len({row["label"] for row in rows}) != len(rows):
        raise ValueError("Duplicate stage label in GPU ledger")
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError("Invalid GPU ledger duration; reconcile it before another stage")
    return total


def run(args):
    import fcntl  # Execution is on the Linux coursework server.

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise ValueError("Use a simple unique stage label")
    if (not math.isfinite(args.max_hours) or args.max_hours <= 0
            or not math.isfinite(args.max_hours * 3600) or args.min_free_mib <= 0):
        raise ValueError("Stage hours and free memory must be positive")
    if (args.total_hours is not None and
            (not math.isfinite(args.total_hours) or args.total_hours <= 0
             or not math.isfinite(args.total_hours * 3600))):
        raise ValueError("Total hours must be finite and positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Provide a command after --")
    directory = args.budget_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ledger, active = directory / "budget.jsonl", directory / "active-stage.json"
    with (directory / "budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if active.exists():
            raise RuntimeError("An unfinished GPU stage needs reconciliation; do not double-book its budget")
        rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        if any(row["label"] == args.label for row in rows):
            raise ValueError("Stage label already used; retries need separate resource records")
        used = used_seconds(rows)
        round_limit = args.total_hours * 3600 if args.total_hours is not None else None
        remaining = round_limit - used if round_limit is not None else None
        limit = min(remaining, args.max_hours * 3600) if remaining is not None else args.max_hours * 3600
        if limit < 60:
            raise RuntimeError("Insufficient remaining GPU budget")
        log_path = directory / f"{args.label}.log"
        if log_path.exists():
            raise FileExistsError(f"Refusing to overwrite stage log: {log_path}")
        while True:
            gpu = available_gpu(args.min_free_mib)
            if gpu:
                time.sleep(2)
                confirmed = available_gpu(args.min_free_mib)
                if confirmed and confirmed["uuid"] == gpu["uuid"]:
                    gpu = confirmed
                    break
            print("Waiting for a GPU with enough free memory; no model loaded", flush=True)
            time.sleep(30)
        record = {"label": args.label, "gpu": gpu, "command": command,
                  "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "round_limit_seconds": round_limit, "stage_limit_seconds": limit,
                  "remaining_before_seconds": remaining}
        active.write_text(json.dumps(record, indent=2) + "\n")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["index"],
                       "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}
        started, code, process = time.monotonic(), 1, None
        print(f"Starting {args.label} on GPU {gpu['index']}; cap {limit / 3600:.3f} hours", flush=True)
        try:
            with log_path.open("x") as log:
                # timeout survives a monitor disconnect; its process group contains only this stage.
                process = subprocess.Popen(["timeout", "--signal=TERM", "--kill-after=10s",
                                            str(int(limit - 20)), *command],
                                           stdout=log, stderr=subprocess.STDOUT, env=environment,
                                           start_new_session=True)
                record["pid"] = process.pid
                active.write_text(json.dumps(record, indent=2) + "\n")
                code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record.update(wall_seconds=time.monotonic() - started, exit_code=code,
                          ended_at_utc=datetime.now(timezone.utc).isoformat())
            with ledger.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            active.unlink()
        print(json.dumps(record), flush=True)
        return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-hours", type=float, required=True, help="Per-stage cap in hours")
    parser.add_argument("--total-hours", type=float, default=None, help="Optional cumulative cap in hours")
    parser.add_argument("--min-free-mib", type=int, default=20000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    raise SystemExit(run(parser.parse_args()))
