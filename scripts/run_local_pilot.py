"""Small paired BM25/E5 pilot using an existing local Qwen model, without an API server."""

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
    parser.add_argument("--output", required=True, help="New directory; never overwritten")
    parser.add_argument("--domains", nargs="+", choices=runner.DOMAINS, default=["Finance", "Technology"])
    parser.add_argument("--categories", nargs="+", choices=runner.CATEGORIES, default=list(runner.CATEGORIES))
    parser.add_argument("--limit", type=int, default=1, help="Questions per dev domain/category")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    for path in (args.model, args.embedding_model):
        if not Path(path).is_dir():
            parser.error(f"Local model directory does not exist: {path}")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("--output already exists")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu or a GPU environment")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    started = time.perf_counter()
    details = {
        "purpose": "small development pilot, not final evaluation",
        "backend": "in-process Transformers; no HTTP server",
        "arguments": vars(args),
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
    patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"])
    details["tracked_patch_sha256"] = hashlib.sha256(patch).hexdigest()
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
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "max_tokens": max_tokens, "output_tokens": len(generated),
                        "seconds": time.perf_counter() - call_started,
                        "finish_reason": self.last_finish_reason, "usage": self.last_usage,
                    }) + "\n")
                return tokenizer.decode(generated, skip_special_tokens=True)

        benchmark = ROOT / "external/GroupMemBench"
        split = runner.make_split(benchmark, args.domains, args.seed, 0.25)
        split_path = output / "split.json"
        split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        for method in ("bm25", "dense"):
            run_args = argparse.Namespace(
                benchmark=str(benchmark), project_root=str(ROOT), output=str(output / method),
                split=str(split_path), partition="dev", domains=args.domains,
                categories=args.categories, method=method, base_url="in-process",
                answer_model=str(Path(args.model).resolve()), judge_model=None,
                embedding_model=str(Path(args.embedding_model).resolve()), embedding_revision=None,
                device=args.device, batch_size=64, top_k=10, timeout=300, limit=args.limit,
            )
            run_path = runner.run(run_args, client_factory=DirectClient)
            metrics = json.loads((run_path / "metrics.json").read_text())
            if metrics["overall"]["errors"]:
                raise RuntimeError(f"{method} contains failed model calls; see predictions.jsonl")
        details["finished"] = True
    finally:
        details["wall_seconds"] = time.perf_counter() - started
        if args.device == "cuda":
            details["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
            details["gpu_name"] = torch.cuda.get_device_name()
        (output / "pilot-config.json").write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Pilot complete: {output} ({details['wall_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
