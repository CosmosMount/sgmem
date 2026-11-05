"""Paired BM25/E5 pilot or full benchmark using existing local Qwen weights."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgmem import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local Qwen chat model directory")
    parser.add_argument("--embedding-model", required=True, help="Existing local E5 directory")
    parser.add_argument('--model-revision', help='Provenance revision for the already downloaded QA weights')
    parser.add_argument('--embedding-revision', help='Provenance revision for the already downloaded E5 weights')
    parser.add_argument("--output", required=True, help="New directory; never overwritten")
    parser.add_argument("--domains", nargs="+", choices=runner.DOMAINS, default=["Finance", "Technology"])
    parser.add_argument("--categories", nargs="+", choices=runner.CATEGORIES, default=list(runner.CATEGORIES))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--limit', type=int, default=1, help='Questions per dev domain/category; default 1')
    mode.add_argument('--full', action='store_true', help='Run every question in selected domains/categories; retain dev/test labels')
    parser.add_argument('--resume', action='store_true', help='Resume the same output/configuration after interruption')
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if len(set(args.domains)) != len(args.domains) or len(set(args.categories)) != len(args.categories):
        parser.error('Domains and categories must not be repeated')
    for path in (args.model, args.embedding_model):
        if not Path(path).is_dir():
            parser.error(f"Local model directory does not exist: {path}")
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        parser.error("--output already exists")
    if args.resume and not output.is_dir():
        parser.error('--resume requires an existing output directory')
    source = runner._git_info(ROOT)
    benchmark = ROOT / 'external/GroupMemBench'
    if args.full and (not source['commit'] or source['status'] or runner._git_info(benchmark)['status']):
        parser.error('Full runs require a committed, clean project and benchmark')
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu or a GPU environment")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=args.resume)
    started = time.perf_counter()
    details = {
        'purpose': 'full selected benchmark; dev/test reported separately' if args.full else 'small development pilot, not final evaluation',
        "backend": "in-process Transformers; no HTTP server",
        'arguments': {k: v for k, v in vars(args).items() if k != 'resume'},
        'project_git': source,
        'benchmark_git': runner._git_info(benchmark),
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "enable_thinking": False,
        "temperature": 0.2, "top_p": 1.0, "top_k": 0,
        "answer_max_tokens": 512, "judge_max_tokens": 256,
        "seed_per_call": args.seed,
        "packages": {p: metadata.version(p) for p in
                     ("torch", "transformers", "sentence-transformers", "numpy", "rank-bm25")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "finished": False,
    }
    details['model_files'] = {}
    for label, folder in [('qa', Path(args.model)), ('embedding', Path(args.embedding_model))]:
        details['model_files'][label] = {
            'configuration_sha256': {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(folder.rglob('*.json'))},
            'weight_sizes': {str(p.relative_to(folder)): p.stat().st_size
                             for p in sorted(folder.rglob('*.safetensors'))},
        }
    patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"])
    details["tracked_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    config_path = output / 'run-config.json'
    if args.resume:
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        for key in ('arguments', 'project_git', 'benchmark_git', 'script_sha256', 'packages', 'tracked_patch_sha256', 'model_files'):
            if previous[key] != details[key]:
                parser.error(f'Resume rejected: {key} changed')
        details['sessions'] = previous.get('sessions', [])
    else:
        details['sessions'] = []
    runner.write_json(config_path, details)
    cpu_started = time.process_time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
        ).to(args.device).eval()
        eos_ids = model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        print(f"Qwen ready in {time.perf_counter() - started:.1f}s", flush=True)

        class DirectClient:
            def __init__(self, *unused, **kwargs):
                self.last_usage = None
                self.last_finish_reason = None
                self.context = None

            def complete(self, system_prompt, user_prompt, max_tokens):
                self.last_usage = None
                self.last_finish_reason = None
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                batch = tokenizer(prompt, return_tensors="pt").to(args.device)
                n_input = batch.input_ids.shape[1]
                if n_input + max_tokens > model.config.max_position_embeddings:
                    raise ValueError("Prompt exceeds model context; no silent truncation")
                torch.manual_seed(args.seed)
                if args.device == "cuda":
                    torch.cuda.manual_seed_all(args.seed)
                call_started = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(
                        **batch, do_sample=True, temperature=0.2, top_p=1.0, top_k=0,
                        max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id,
                    )
                generated = result[0, n_input:]
                ended = int(generated[-1]) in eos_ids
                self.last_finish_reason = "stop" if ended else "length"
                self.last_usage = {"prompt_tokens": n_input, "completion_tokens": len(generated),
                                   "total_tokens": int(result.shape[1])}
                with (output / "generation-audit.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        'question': self.context,
                        'stage': 'answer' if max_tokens == 512 else 'judge',
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "max_tokens": max_tokens, "output_tokens": len(generated),
                        "seconds": time.perf_counter() - call_started,
                        "finish_reason": self.last_finish_reason, "usage": self.last_usage,
                    }) + "\n")
                return tokenizer.decode(generated, skip_special_tokens=True)

        split = runner.make_split(benchmark, args.domains, args.seed, 0.25)
        split_path = output / "split.json"
        if args.resume:
            if json.loads(split_path.read_text()) != split:
                raise ValueError('Saved split differs from the frozen split')
        else:
            runner.write_json(split_path, split)
        paired_ids = None
        summary = {}
        for method in ("bm25", "dense"):
            if args.full and (runner._git_info(ROOT) != source or runner._git_info(benchmark) != details['benchmark_git']):
                raise ValueError('Source changed during a full run; do not mix code versions')
            run_args = argparse.Namespace(
                benchmark=str(benchmark), project_root=str(ROOT), output=str(output / method),
                split=str(split_path), partition='all' if args.full else 'dev', domains=args.domains,
                categories=args.categories, method=method, base_url="in-process",
                answer_model=str(Path(args.model).resolve()), judge_model=None,
                embedding_model=str(Path(args.embedding_model).resolve()), embedding_revision=None,
                device=args.device, batch_size=64, top_k=10, timeout=300,
                limit=None if args.full else args.limit,
                resume=args.resume and (output / method).exists(),
                generation_config={'seed_per_call': args.seed, 'enable_thinking': False,
                                   'temperature': 0.2, 'top_p': 1.0, 'top_k': 0,
                                   'answer_max_tokens': 512, 'judge_max_tokens': 256},
                model_provenance={'qa_revision': args.model_revision, 'embedding_revision': args.embedding_revision,
                                  'files': details['model_files']},
            )
            run_path = runner.run(run_args, client_factory=DirectClient)
            metrics = json.loads((run_path / "metrics.json").read_text())
            if metrics['overall']['errors'] or not metrics['complete']:
                raise RuntimeError(f'{method} is incomplete; see predictions.jsonl')
            records = [json.loads(line) for line in (run_path / 'predictions.jsonl').read_text().splitlines()]
            ids = {(r['domain'], r['category'], r['question_id']) for r in records}
            if len(ids) != len(records) or (paired_ids is not None and ids != paired_ids):
                raise ValueError('Paired methods have duplicate or unequal question coverage')
            paired_ids = ids
            summary[method] = metrics
        runner.write_json(output / 'benchmark-summary.json', summary)
        details["finished"] = True
    finally:
        elapsed = time.perf_counter() - started
        details['sessions'].append({'wall_seconds': elapsed, 'cpu_seconds': time.process_time() - cpu_started,
                                    'ended_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        details['wall_seconds'] = sum(s['wall_seconds'] for s in details['sessions'])
        details['cpu_seconds'] = sum(s['cpu_seconds'] for s in details['sessions'])
        if args.device == "cuda":
            details["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
            details['peak_reserved_gpu_bytes'] = torch.cuda.max_memory_reserved()
            details["gpu_name"] = torch.cuda.get_device_name()
            details['gpu_hours'] = details['wall_seconds'] / 3600
        runner.write_json(config_path, details)
    print(f"Baselines complete: {output} ({details['wall_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
