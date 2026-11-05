#!/usr/bin/env bash
# Verify Qwen, both retrieval baselines, judging and annotation in Docker.
set -euo pipefail
cd "$(dirname "$0")/.."

repo=$PWD
cache_dir=${SGMEM_HF_CACHE:-"$HOME/.cache/huggingface"}
model=${SGMEM_QWEN_CONTAINER_MODEL:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}
port=${SGMEM_VERIFY_PORT:-18767}
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1024 || port > 65535)); then
  printf 'SGMEM_VERIFY_PORT must be an unprivileged TCP port.\n' >&2
  exit 1
fi
if [[ ! -f "$repo/models/e5-base-v2/model.safetensors" || ! -d "$repo/external/GroupMemBench" ]]; then
  printf 'E5 model or GroupMemBench submodule is missing; see README.md.\n' >&2
  exit 1
fi
python3 - "$port" <<'PY'
import socket, sys
with socket.socket() as sock:
    try:
        sock.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError as exc:
        raise SystemExit(f'Local verification port is unavailable: {exc}')
PY

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  sudo -v
  docker_cmd=(sudo docker)
fi
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sgmem-docker-pipeline.XXXXXXXX")
container_name="sgmem-verify-$(basename "$tmp_dir")"
service_started=0
cleanup() {
  result=$?
  trap - EXIT
  if ((service_started)); then
    if ! "${docker_cmd[@]}" info >/dev/null 2>&1; then
      printf 'Cannot confirm cleanup; check test container %s.\n' "$container_name" >&2
      result=1
    elif "${docker_cmd[@]}" inspect "$container_name" >/dev/null 2>&1; then
      if ! "${docker_cmd[@]}" rm -f "$container_name" >/dev/null; then
        printf 'Could not remove test container %s.\n' "$container_name" >&2
        result=1
      fi
    fi
  fi
  rm -rf -- "$tmp_dir"
  exit "$result"
}
trap cleanup EXIT

# Create separately so a startup failure remains inspectable until cleanup.
"${docker_cmd[@]}" create --network host --runtime=nvidia --gpus all --name "$container_name" \
  -v "$cache_dir:/root/.cache/huggingface:ro" \
  -e HF_HUB_OFFLINE=1 --entrypoint transformers sgmem:local \
  serve "$model" --host 127.0.0.1 --port "$port" --device cuda --dtype float16 >/dev/null
service_started=1
if ! "${docker_cmd[@]}" start "$container_name" >/dev/null; then
  printf 'Failed to start Qwen container; recent logs follow:\n' >&2
  "${docker_cmd[@]}" logs --tail 80 "$container_name" >&2 || true
  exit 1
fi
ready=0
for ((i=0; i<120; i++)); do
  if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! state=$("${docker_cmd[@]}" inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null); then
    printf 'Could not inspect Qwen container after startup.\n' >&2
    "${docker_cmd[@]}" logs --tail 80 "$container_name" >&2 || true
    exit 1
  fi
  if [[ "$state" != true ]]; then
    printf 'Qwen container exited before it became ready; recent logs follow:\n' >&2
    "${docker_cmd[@]}" inspect --format 'Exit code: {{.State.ExitCode}}; runtime error: {{.State.Error}}' "$container_name" >&2 || true
    "${docker_cmd[@]}" logs --tail 80 "$container_name" >&2 || true
    exit 1
  fi
  sleep 1
done
if (( ! ready )); then
  printf 'Qwen service did not become ready.\n' >&2
  "${docker_cmd[@]}" logs --tail 80 "$container_name" >&2 || true
  exit 1
fi

client=("${docker_cmd[@]}" run --rm --network host --runtime=nvidia --gpus all
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=sgmem -e LOGNAME=sgmem -e HF_HOME=/tmp/huggingface
  -e HF_HUB_OFFLINE=1 -e NO_PROXY=127.0.0.1,localhost
  -v "$repo:/workspace:ro" -v "$tmp_dir:/scratch" sgmem:local)
"${client[@]}" split --domains Finance --output /scratch/split.json
for method in bm25 dense; do
  if ! "${client[@]}" run --method "$method" --embedding-model /workspace/models/e5-base-v2 \
    --device cuda --batch-size 32 --answer-model "$model" --base-url "http://127.0.0.1:$port/v1" \
    --domains Finance --categories multi_hop --split /scratch/split.json --partition dev \
    --limit 1 --top-k 10 --timeout 300 --output "/scratch/$method"; then
    python3 - "$tmp_dir/$method/predictions.jsonl" <<'PYCODE'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists():
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get('status') == 'error':
            print(f"{record.get('question_id')}: {record.get('error', 'unknown error')}", file=sys.stderr)
PYCODE
    "${docker_cmd[@]}" logs --tail 30 "$container_name" >&2 || true
    exit 1
  fi
done
"${client[@]}" annotate --records /scratch/bm25/predictions.jsonl /scratch/dense/predictions.jsonl \
  --output /scratch/failures.jsonl
"${client[@]}" analyze --records /scratch/bm25/predictions.jsonl /scratch/dense/predictions.jsonl \
  --annotations /scratch/failures.jsonl --output /scratch/analysis.json >/dev/null
python3 - "$tmp_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
a = json.loads((root / 'bm25/predictions.jsonl').read_text().splitlines()[0])
b = json.loads((root / 'dense/predictions.jsonl').read_text().splitlines()[0])
assert (a['domain'], a['category'], a['question_id']) == (b['domain'], b['category'], b['question_id'])
assert a['status'] == b['status'] == 'ok'
assert isinstance(a.get('correct'), bool) and isinstance(b.get('correct'), bool), 'Judge output was not parseable'
for method in ('bm25', 'dense'):
    metrics = json.loads((root / method / 'metrics.json').read_text())
    assert metrics['overall']['errors'] == 0 and metrics['overall']['unclear'] == 0
assert len(a['retrieval']) == len(b['retrieval']) == 10
assert json.loads((root / 'analysis.json').read_text())['total_records'] == 2
print('Docker Qwen, BM25, E5 Dense, judging and failure-analysis commands completed on the same real development question.')
print('This single-question smoke test is not an experimental accuracy result.')
PY
