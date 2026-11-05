# Part 1/2 evidence

Full results for 715 GroupMemBench questions per method, Qwen3-8B + BM25/E5-small-v2. This folder contains compact generated records rather than models or the full benchmark corpus.

| File | Purpose |
|---|---|
| `outcomes.jsonl` | 1,430 outcomes, answers, judge verdicts, candidate IDs/scores, token counts and timings; full candidate text and raw model traces remain in the archive |
| `split.json` | Exact frozen 179-dev / 536-test split, shared by both methods |
| `metrics.json`, `comparison.json` | Recomputed overall/category/domain/partition results and paired comparisons |
| `performance.json`, `provenance.json` | Measured costs, source commits, revisions, environment and original artifact identities |
| `failure-sample.json` | Reproducible, outcome-independent ordering within eligible dev-failure cells; one question per domain/category |
| `failure-annotations.jsonl` | Source-reviewed explanations for all 38 judged failures in 24 sampled dev questions; AI-assisted, not independently human-adjudicated |
| `failure-analysis.json` | Descriptive counts with explicit sampled and unannotated denominators |
| `source-references.jsonl` | 61 source messages needed to check annotations; domain is part of the ID namespace |
| `metadata-diagnostics.json` | Post-hoc reply-neighborhood reachability of reviewer-identified support, without rerunning QA |
| `baseline-accuracy.png`, `failure-taxonomy.png` | Report figures generated from the stored tables |

Run `python scripts/verify_part12_evidence.py` from the repository root. Add `--benchmark external/GroupMemBench` to also validate source contents and reply-graph diagnostics against the pinned dataset. The verifier does not load a model, download data, call an API, or use a GPU.

Raw archive: `SGMem-Part1-full-evidence-20261105.tar.gz` (10,275,815 bytes).
SHA256: `792a534593ea1957b24ea86b87316e3736b694dadc38f0fdd6ee01711e3cd7b1`.
Server copy: `/home/ai4s_group/XZG/NLP/sgmem/results/exports/part1-full-20261105-evidence.tar.gz`.
The teammate handoff archive includes the raw archive, so access to the original server is not required once the handoff is copied.

The primary label distribution describes this quota sample, not all 992 full-benchmark errors. F9 means evaluation/reference ambiguity requiring review; it does not overwrite the original incorrect verdict. `metadata_could_help` is a diagnostic hypothesis about improving memory selection/ranking, not a measured repair rate. Timestamp fields can already be present while the QA model fails to use them.

Source excerpts are from GroupMemBench, copyright (c) 2026 The GroupMemBench Authors, under the accompanying [MIT license](GroupMemBench-LICENSE). Project source attribution is fixed to commit `461f00c3e8a1a4a3a2ed5814b729014502c4702f`. Full experimental interpretation is in [the Part 1/2 report](../../part12-full-results.md).
