---
name: svllm-benchmark
description: Run Sparse-vLLM benchmarks according to the standard benchmark design. Use when Codex needs to plan, execute, or interpret quick/final benchmark runs, maintain feature-level benchmark ledgers, choose reliable repo-local benchmarks before external ones, or explain TTFT, TPOT, prefill/decode throughput, quality deltas, cache hit metrics, and failure statuses.
---

# Sparse-vLLM Benchmark

Use this skill when running or interpreting Sparse-vLLM benchmark work. Follow the plan in `dev_docs/plan/benchmark/standard_benchmark_design.md`.

Default to pure-text benchmarks. The normal quick/final path is sanity, microbench, NIAH, LongBench, SCBench, and MathBench. Multimodal benchmarks are optional and should only be selected when the user explicitly asks for visual KV, LLaVA-OneVision, Video-MME, StreamingBench, AI2D, or visual-cache evaluation.

## Core Rules

1. Prefer repo-local reliable benchmarks before external benchmarks.
2. Do not download datasets. If a required dataset is missing, report the exact environment variable or path the user must provide.
3. Before running GPU workloads, check device idleness with `nvidia-smi`; use an idle GPU when available.
4. Use the project virtual environment by default: `.venv/bin/python`.
5. Use `SVLLM_BENCHMARK_OUTPUT_DIR` for outputs when provided; otherwise use `benchmark/results`.
6. Maintain one feature-level ledger per feature or optimization under `benchmark/results/_ledgers/<feature>.jsonl` and `.csv`.
7. Every benchmark attempt must be explicit: success, invalid_run, invalid_input, model_failed, parse_failed, metric_failed, skipped_by_policy, oom, or timeout.
8. Never silently drop failed samples from aggregate conclusions.

## Benchmark Selection

Use this priority order.

1. **Performance/microbench**: `scripts/benchmarks/bench_sparse_vllm.py` for TTFT, TPOT, prefill tok/s, decode tok/s, and peak memory.
2. **Quick correctness**: `benchmark/niah/test_niah.py` for controllable long-context retrieval.
3. **Real quality**: `benchmark/long_bench/pred.py` and `benchmark/long_bench/eval.py`.
4. **KV cache lifecycle**: `benchmark/scbench/run_scbench_preprocessed.py`.
5. **Reasoning regression**: `benchmark/math_bench/pred.py`.
6. **Optional visual cache**: `benchmark/multimodal/visual_cache/run_visual_cache.py`, Video-MME, StreamingBench, or AI2D only for explicit multimodal changes.

External text benchmarks such as RULER, MRCR, NoLiMa, HELMET, InfiniteBench, LV-Eval, and LongBench v2 are final-report additions or capability-gap fillers. Do not introduce them before the repo-local text path is working.

## Standard Runner

Use the standard runner unless the user asks for one specific benchmark.

Quick iteration:

```bash
.venv/bin/python scripts/benchmarks/run_standard_benchmark.py \
  --mode quick \
  --feature <feature_slug> \
  --objective "<what this run validates>" \
  --model_path <MODEL_PATH> \
  --primary_method <method> \
  --methods vanilla,<method> \
  --benchmarks sanity,microbench,niah,longbench
```

Final run:

```bash
.venv/bin/python scripts/benchmarks/run_standard_benchmark.py \
  --mode final \
  --feature <feature_slug> \
  --objective "<final evaluation objective>" \
  --model_path <MODEL_PATH> \
  --primary_method <method> \
  --methods vanilla,<method> \
  --benchmarks sanity,microbench,niah,scbench,longbench,mathbench
```

Use `--dry_run` first when paths, datasets, or GPU availability are uncertain.

## Required Data Environment Variables

Set only the variables needed by the chosen benchmarks.

- LongBench: `SVLLM_LONGBENCH_DATA_DIR` or `DELTAKV_LONGBENCH_DATA_DIR`, pointing to a root with `data/*.jsonl`.
- SCBench preprocessed: `SVLLM_SCBENCH_PREPROCESSED_ROOT` or `SCBENCH_PREPROCESSED_ROOT`, pointing to a directory with `scbench_*.parquet`.
- General local data: `SVLLM_BENCHMARK_DATA_DIR`, `SVLLM_DATA_DIR`, or `DELTAKV_DATA_DIR`.
- Optional multimodal: `SVLLM_LLAVA_MODEL_PATH` plus dataset-specific variables such as `SVLLM_STREAMINGBENCH_DATA_DIR`, `SVLLM_VIDEOMME_DATA_DIR`, `SVLLM_AI2D_DATA_DIR`, `SVLLM_VISUAL_CACHE_DATA_DIR`, and `SVLLM_VQAV2_DATA_DIR`.

Use `--use_proxy_7890` on the standard runner only when the user confirms mainland network constraints and proxy availability.

## Interpreting Results

Performance:

- TTFT and prefill tok/s are the primary prefill metrics.
- TPOT, ITL, and decode tok/s are decode metrics.
- Peak memory must be interpreted with cache metadata overhead.
- Short-context slowdowns can be acceptable only if the method is intended to bypass sparse paths below a threshold.

Quality:

- Compare sparse results against full attention on the same model, prompt template, sample ids, decode config, GPU, and script version.
- For LongBench, inspect per-task deltas before overall averages.
- For NIAH/MRCR/RULER-style exact retrieval, sparse-only failures require sample-level analysis.
- For SCBench, single-request correctness does not imply multi-turn or multi-request correctness.

Ledger:

- Link every benchmark output directory from the feature ledger.
- Use `decision=promote_to_final` only for runs that can appear in final reports.
- Use `decision=investigate` or `rerun` for failures or incomparable runs.
