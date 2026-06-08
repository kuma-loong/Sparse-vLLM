#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${SVLLM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmark/results}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model directory}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DELTAKV_CHECKPOINT_PATH="${DELTAKV_CHECKPOINT_PATH:?Set DELTAKV_CHECKPOINT_PATH to the compressor checkpoint directory}"
LONG_BENCH_DATA_DIR="${SVLLM_LONGBENCH_DATA_DIR:-${DELTAKV_LONGBENCH_DATA_DIR:-}}"
if [[ -z "${LONG_BENCH_DATA_DIR}" ]]; then
  echo "Set SVLLM_LONGBENCH_DATA_DIR or DELTAKV_LONGBENCH_DATA_DIR to the LongBench root containing data/*.jsonl" >&2
  exit 2
fi

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
SVLLM_BENCHMARK_OUTPUT_DIR="${OUTPUT_ROOT}" \
SVLLM_LONGBENCH_DATA_DIR="${LONG_BENCH_DATA_DIR}" \
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${PYTHON}" -u benchmark/long_bench/pred.py \
  --task hotpotqa,trec,2wikimqa \
  --model llama31-8b-deltakv-main-results-longbench-cr30 \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "${DELTAKV_CHECKPOINT_PATH}" \
  --temperature 0 \
  --top_p 1 \
  --top_k 0 \
  --hyper_param "${HYPER_PARAM:-{\"hf_prefill_chunk_size\":2048000,\"prefill_keep_tokens\":4096,\"chunk_prefill_accel_omnikv\":false,\"deltakv_use_omnikv_selection\":true,\"decode_keep_tokens\":0.17,\"full_attention_layers\":\"0,1,2,8,18\",\"recent_keep_tokens\":128,\"sink_keep_tokens\":8,\"use_compression\":true,\"use_cluster\":true,\"deltakv_center_ratio\":0.1,\"deltakv_latent_quant_bits\":0}}" \
  --output_root "${OUTPUT_ROOT}/long_bench/main_results/llama31_8b_deltakv_cr30_hotpotqa_trec_2wikimqa"
