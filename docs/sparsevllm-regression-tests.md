# SparseVLLM Regression Tests

## Purpose

This document describes how to run the fixed SparseVLLM regression harness under
`benchmark/sparsevllm_regression/`.

The harness is intended for reproducible method/model checks across:

- `quality`: LongBench-mini generation quality.
- `logits`: HF-reference vs SparseVLLM logits alignment.
- `perf`: prefill/decode throughput and memory accounting.
- `stress`: high-concurrency SparseVLLM admission/decode stress.
- `validate`: manifest and output-artifact validation.

The test plan is controlled by
`benchmark/sparsevllm_regression/manifest.json`.

## Prerequisites

Observed working environment on 2026-06-13:

- Working directory: `/root/autodl-tmp/DeltaKV`
- Conda env: `kv`
- Output root: `/root/autodl-tmp/outputs/deltakv`
- LongBench data: `/root/autodl-fs/datasets/LongBench`
- Models:
  - `/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M`
  - `/root/autodl-fs/models/Qwen3-4B-Instruct-2507`
  - `/root/autodl-fs/models/Llama-3.1-8B-Instruct`
- Compressor checkpoints:
  - `/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
  - `/root/autodl-fs/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor`
  - `/root/autodl-fs/checkpoints/compressor/Llama-3.1-8B-Instruct-Compressor`

Set the environment before running the suite:

```bash
cd /root/autodl-tmp/DeltaKV

export DELTAKV_OUTPUT_DIR=/root/autodl-tmp/outputs/deltakv
export DELTAKV_LONGBENCH_DATA_DIR=/root/autodl-fs/datasets/LongBench

export DELTAKV_MODEL_QWEN25_7B=/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M
export DELTAKV_MODEL_QWEN3_4B=/root/autodl-fs/models/Qwen3-4B-Instruct-2507
export DELTAKV_MODEL_LLAMA31_8B=/root/autodl-fs/models/Llama-3.1-8B-Instruct

export DELTAKV_COMPRESSOR_QWEN25_7B=/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor
export DELTAKV_COMPRESSOR_QWEN3_4B=/root/autodl-fs/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor
export DELTAKV_COMPRESSOR_LLAMA31_8B=/root/autodl-fs/checkpoints/compressor/Llama-3.1-8B-Instruct-Compressor

export PYTHONPATH=/root/autodl-tmp/DeltaKV:/root/autodl-tmp/DeltaKV/src:${PYTHONPATH:-}
```

The manifest also contains `qwen25_32b`, but the current local regression
commands should omit it unless there is enough GPU memory and the corresponding
model/checkpoint environment variables are set.

## Quick Unit Tests

Run the unit tests that protect the regression harness, grading, manifest
policy, and OmniKV full-layer selector:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python -m unittest \
  tests.test_sparsevllm_regression_grading \
  tests.test_omnikv_full_layer_selector \
  -v
```

Expected result for the current harness: all tests pass.

## Manifest Validation

Use `validate` before long GPU runs. It resolves runtime paths, writes the
resolved manifest, and creates empty required artifact files.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer validate \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id validate_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

Use `--no-allow_skipped_policy` when missing model/checkpoint paths should fail
the run instead of being recorded as skipped.

## Common Run Commands

All commands write to:

```text
<output_root>/sparsevllm_regression/<run_id>/
```

### Quality

Quality is LongBench-mini with:

- tasks: `qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc`
- LongBench batch size: `100`
- SparseVLLM `max_num_seqs_in_batch`: `16`
- SparseVLLM `max_decoding_seqs`: `16`
- samples per task: `50`

Run OmniKV against vanilla baselines:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id omnikv_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

For a full non-32B quality run:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,quest,deltakv,deltakv-less-memory \
  --run_id quality_3models_all_methods_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Correctness / Logits

`logits` compares HF sparse reference outputs with SparseVLLM for methods that
declare `hf_logits_reference=true`. Methods without an HF reference are graded
`N/A` by policy.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer logits \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_logits_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Performance

Performance uses:

- prompt lengths: `16000,64000`
- batch sizes: `1,4`
- output tokens: `256`
- decode CUDA graph requested where the method supports it

For sparse methods, the benchmark also runs vanilla for the same shape so the
suite can compute decode speedup.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_perf_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Stress

Stress currently uses:

- prompt length: `16000`
- request count / batch size: `80`
- output tokens: `64`
- `max_num_seqs_in_batch=80`
- `max_decoding_seqs=80`
- max decode steps after full admission: `32`

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_stress80_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Combined Layers

`nightly` runs quality, logits, and performance. It does not run stress.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer nightly \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id nightly_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

`pre-refactor` runs quality, logits, performance, and stress.

## Recent Result Records

This section records the recent runs from the current AutoDL workspace. It is a
result index, not a new pass/fail policy. Use the linked artifacts for exact
commands, raw rows, parsed rows, per-sample status, and aggregate metrics.

### 2026-06-18 Qwen3 DeltaKV Quality Fix

- run root:
  `/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality`
- layer/model/methods:
  `quality`, `qwen3_4b`, `vanilla,deltakv`
- data:
  LongBench-mini tasks
  `qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc`,
  `50` samples per task per method.
- result:
  `grade_summary.json` reports quality grade `B`; vanilla
  `overall_category_avg=59.99`, DeltaKV `overall_category_avg=59.54`,
  `score_loss=0.45`.
- status:
  all task status counts are `success`; no skipped or failed samples were
  included in the aggregate.
- source artifacts:
  [`grade_summary.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/grade_summary.json),
  [`metrics.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/metrics.json),
  [`raw_outputs.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/raw_outputs.jsonl),
  [`parsed_outputs.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/parsed_outputs.jsonl),
  [`sample_results.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/sample_results.jsonl).

The corresponding full 16-task LongBench run is recorded in
`docs/longbench_results_summary.md` as `DeltaKV Sparse-VLLM temp-slot fix` with
overall `48.18` and 3750 `success` sample rows.

### 2026-06-18 Qwen3 DeltaKV Perf/Stress

- perf run root:
  `/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_perf`
- stress run root:
  `/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_stress`
- backend:
  Sparse-VLLM with decode CUDA graph active for all recorded rows.
- important interpretation:
  the quality bug was fixed, but the perf gate did not pass. Perf grades are
  `D` because DeltaKV decode throughput was slower than vanilla at the tested
  shapes. Stress grade is `B`; it completed with full admission and no
  preemptions, but utilization was below the A-grade threshold.

| Layer | Method | Length | BS | Status/grade | Decode tok/s | Mem GB | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| perf | vanilla | 32000 | 4 | SUCCESS | 164.117 | 83.457 | baseline |
| perf | deltakv | 32000 | 4 | D | 95.363 | 17.646 | speedup `0.581`, graph active |
| perf | vanilla | 32000 | 8 | SUCCESS | 209.585 | 83.584 | baseline |
| perf | deltakv | 32000 | 8 | D | 117.542 | 23.590 | speedup `0.561`, graph active |
| perf | vanilla | 64000 | 4 | SUCCESS | 106.581 | 83.459 | baseline |
| perf | deltakv | 64000 | 4 | D | 71.724 | 25.340 | speedup `0.673`, graph active |
| perf | vanilla | 64000 | 8 | SUCCESS | 124.318 | 83.586 | baseline |
| perf | deltakv | 64000 | 8 | D | 83.581 | 35.930 | speedup `0.672`, graph active |
| stress | deltakv | 16000 | 80 | B | 66.736 | 87.269 | full admission reached, preemptions `0`, avg BS `66.0` |

Source artifacts:
[`perf.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_perf/perf.jsonl),
[`perf grade_summary.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_perf/grade_summary.json),
[`stress.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_stress/stress.json),
[`stress grade_summary.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_stress/grade_summary.json).

### 2026-06-16 Qwen2.5 128k Concurrent Decode Sanity

This was an ad hoc Sparse-VLLM throughput sanity run for whether the
full-layer-KIVI DeltaKV less-memory path can keep all requests admitted at
`128k` context and concurrent decode.

- run root:
  `/root/autodl-tmp/outputs/deltakv/perf/qwen25_7b_1m_lessmem_cudagraph_no_compressor_2settings_20260616_151628`
- model:
  `/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M`
- method:
  `deltakv-less-memory-cudagraph`, `use_compression=false`,
  full-layer KIVI4 group32/residual32, full layers `0,2,4,11,16,22`,
  decode CUDA graph active.

| Length | BS | Output tokens | Status | Decode tok/s | Avg BS | Mem GB | Full admission |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 131072 | 8 | 256 | SUCCESS | 115.441 | 7.969 | 50.299 | yes |
| 131072 | 16 | 512 | SUCCESS | 135.191 | 15.969 | 77.228 | yes |

Source artifact:
[`summary.json`](/root/autodl-tmp/outputs/deltakv/perf/qwen25_7b_1m_lessmem_cudagraph_no_compressor_2settings_20260616_151628/summary.json).

### 2026-06-20 Qwen2.5 7B Max-BS Decode Sweep

This was an ad hoc max-batch sweep using
`scripts/benchmarks/bench_sparse_vllm.py`, not the fixed regression harness.
The selection rule in the run manifest was:
`status=SUCCESS and full_admission_reached=true`; max-success batch size and
best decode throughput are reported only from usable rows.

- run root:
  `/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry`
- model:
  `/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M`
- compressor for attempted DeltaKV rows:
  `/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
- common settings:
  output length `128`, `max_decode_steps_after_full=64`,
  `engine_prefill_chunk_size=8192`, `max_num_batched_tokens=65536`,
  `gpu_memory_utilization=0.9`, graph mode requested.

| Method | Length | Max success BS | Decode tok/s at max BS | Best decode BS | Best decode tok/s | Mem GB at best | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla | 65536 | 32 | 272.710 | 20 | 272.882 | 87.037 | success |
| vanilla | 131072 | 16 | 136.030 | 12 | 136.476 | 87.022 | success |
| vanilla | 262144 | 8 | 67.788 | 5 | 67.989 | 86.990 | success |
| vanilla | 524288 | 4 | 30.590 | 2 | 30.946 | 86.527 | success |
| vanilla | 921600 | 2 | 14.648 | 1 | 15.614 | 85.667 | success |
| deltakv | 65536 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 131072 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 262144 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 524288 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 921600 | - | - | - | - | - | bs=1 failed; larger BS skipped |

DeltaKV rows are not throughput results. They failed during engine construction
because the launcher passed `prefill_keep_tokens`, which current Sparse-VLLM
rejects as an unknown config key:
`ValueError: Unknown Sparse-vLLM config keys: ['prefill_keep_tokens']`. The
sweep used `stop_after_first_failure`, so all larger DeltaKV batch sizes were
recorded as `SKIPPED_BY_POLICY`.

Source artifacts:
[`summary.json`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/summary.json),
[`summary.md`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/summary.md),
[`status.tsv`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/status.tsv),
[`deltakv_len65536_bs1.log`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/logs/deltakv_len65536_bs1.log).

## OmniKV Full-Layer Selection

OmniKV full layers are model-specific. Use
`scripts/analysis/select_omnikv_full_layers.py` before publishing a new model's
OmniKV or OmniKV-aligned DeltaKV regression numbers.

The selector runs an offline decode-attention coverage calibration on a
LongBench task, chooses `--num-full-layers` layers, and writes the selected
layer string to `selected_full_layers.json`. This is not an online runtime mode:
the selected string must be passed back as `full_attention_layers`.

Example for Qwen2.5-7B with six full layers:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python scripts/analysis/select_omnikv_full_layers.py \
  --model-path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --longbench-root /root/autodl-fs/datasets/LongBench \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir /root/autodl-tmp/outputs/deltakv/omnikv_full_layer_calibration_$(date -u +%Y%m%d)/qwen25_7b_full6 \
  --num-full-layers 6 \
  --num-samples 32 \
  --topk 2048 \
  --random-decode-points-per-sample 8 \
  --num-sink-tokens 0 \
  --num-recent-tokens 32 \
  --prefill-chunk-size 512 \
  --torch-dtype bfloat16 \
  --device cuda
```

Key outputs:

- `selected_full_layers.json`: selected layer ids and
  `full_attention_layers` string for runtime configs.
- `per_sample_points.jsonl`: sampled decode points used for calibration.
- `pair_scores.npy` and `segment_scores.npy`: raw coverage matrices for audit.
- `run_info.json`: command, git state, model/data paths, and calibration
  settings.
- `top128_kl_metrics.json`: optional validation output when running with
  `--top128-kl-only`.

To use the selected layers in an ad hoc Sparse-VLLM run, copy the
`full_attention_layers` value into `--hyper_params`:

```bash
PYTHONPATH=$PWD:$PWD/src python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_DIR> \
  --methods omnikv \
  --lengths 131072 \
  --batch_sizes 4 \
  --output_len 128 \
  --hyper_params '{"sparse_method":"omnikv","full_attention_layers":"0,2,4,11,16,22","decode_keep_tokens":4096,"recent_keep_tokens":32,"sink_keep_tokens":0,"engine_prefill_chunk_size":512}'
```

For regression runs, update `methods.omnikv.model_configs` in
`benchmark/sparsevllm_regression/manifest.json`. If a DeltaKV regression config
is intentionally aligned to OmniKV observation/full layers, update the matching
DeltaKV model config in the same manifest and record that alignment in the run
summary. The current manifest uses:

```text
qwen25_7b:  0,2,4,11,16,22
qwen3_4b:   0,1,3,9,13,16,21,28
llama31_8b: 0,2,7,13,16,26
```

Run `validate` and rerun OmniKV quality/logits/perf/stress after changing these
layers.

## Outputs

Each run writes:

- `resolved_manifest.json`: manifest after environment-variable resolution.
- `grade_summary.json`: command records, grades, and final status.
- `metrics.json`: quality aggregate records.
- `logits_alignment.json`: logits comparison summaries.
- `perf.jsonl`: flattened performance rows.
- `memory.json`: memory grades derived from performance rows.
- `stress.json`: stress rows and stress grades.
- `raw_outputs.jsonl`, `parsed_outputs.jsonl`, `sample_results.jsonl`: quality
  generation artifacts, when quality is run.
- Layer-specific logs:
  - `quality/<model>/<method>/run.log`
  - `logits/<model>/<method>/run.log`
  - `perf/<model>/<method>.log`
  - `stress/<model>/<method>.log`

Quick summary command:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/<run_id>")
data = json.loads((root / "grade_summary.json").read_text())
print("status:", data["status"])
print("worst_required_grade:", data.get("worst_required_grade"))
for grade in data.get("grades", []):
    print(grade.get("model"), grade.get("method"), grade["name"], grade["grade"], grade["status"], grade["metrics"])
PY
```

## Grade Meanings

The gate rules live in `benchmark/sparsevllm_regression/grading.py`.

- Quality:
  - A: sparse score loss `< 0.1` vs vanilla.
  - B: loss `<= 0.5`.
  - C: loss `<= 1.0`.
  - D: loss `> 1.0`, missing score, or failed quality run.
- Logits:
  - A: all decode top-1 match, mean top-5 overlap `>= 0.8`, mean top-10 overlap
    `>= 0.9`, and p99 diff is within threshold.
  - B: top-1 match and mean top-5 overlap `>= 0.8`, but misses A.
  - C: top-1 match, but misses B/A.
  - D: top-1 mismatch, missing decode metrics, or run failure.
  - N/A: no HF logits reference exists for that method.
- Performance:
  - A: decode speedup `>= 2.0` and required decode CUDA graph active.
  - B: speedup `>= 1.5`.
  - C: speedup `> 1.0`.
  - D: speedup `<= 1.0`, run failure, or expected CUDA graph inactive.
- Memory:
  - A/B/C depend on positive observed saving and absolute error from expected
    saving within `0.05/0.10/0.20`.
  - D: missing accounting, non-positive saving, or error `> 0.20`.
- Stress:
  - A: completed, no crash, no preemptions, full admission window reached, and
    utilization OK.
  - B: completed with no preemptions, but not all A conditions met.
  - C: completed with preemptions.
  - D: crashed, stuck, failed rows, or did not finish.

## Updating The Bug Matrix

Use `benchmark/sparsevllm_regression/BUGS_20260613.md` for the current
ABCD gate matrix and open blockers. When rerunning a subset, update only the
affected model/method rows and record the new run IDs near the execution
summary.

For a new campaign date, create a new `BUGS_YYYYMMDD.md` instead of overwriting
the old report.

## Troubleshooting

- Missing model or compressor paths:
  - Run `validate`.
  - Check `resolved_manifest.json`.
  - If a run should fail on missing paths, pass `--no-allow_skipped_policy`.
- Import errors:
  - Ensure `PYTHONPATH=/root/autodl-tmp/DeltaKV:/root/autodl-tmp/DeltaKV/src:${PYTHONPATH:-}`.
  - Use the `kv` conda env on the observed AutoDL machine.
- Quality dataset errors:
  - Set `DELTAKV_LONGBENCH_DATA_DIR=/root/autodl-fs/datasets/LongBench`.
- GPU memory failures:
  - Do not add fallback behavior inside the harness.
  - Record the exact run ID, model, method, layer, log path, and error in the
    bug report.
- A command exits early:
  - Inspect `<run_id>/grade_summary.json`; failed commands are recorded with
    `returncode`, `cmd`, and `log_path`.
  - Inspect the layer-specific log path from the command record.
