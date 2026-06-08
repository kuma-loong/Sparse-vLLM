#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${SVLLM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmark/results}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Llama model directory}"
DELTAKV_CHECKPOINT_PATH="${DELTAKV_CHECKPOINT_PATH:?Set DELTAKV_CHECKPOINT_PATH to the compressor checkpoint directory}"
LONG_BENCH_DATA_DIR="${SVLLM_LONGBENCH_DATA_DIR:-${DELTAKV_LONGBENCH_DATA_DIR:-}}"
if [[ -z "${LONG_BENCH_DATA_DIR}" ]]; then
  echo "Set SVLLM_LONGBENCH_DATA_DIR or DELTAKV_LONGBENCH_DATA_DIR to the LongBench root containing data/*.jsonl" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export SVLLM_BENCHMARK_OUTPUT_DIR="${OUTPUT_ROOT}"
export SVLLM_LONGBENCH_DATA_DIR="${LONG_BENCH_DATA_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

exec "${PYTHON}" -u benchmark/long_bench/pred.py \
  --task "${TASKS:-multi_news,passage_count,passage_retrieval_en,lcc,repobench-p}" \
  --ws "${WS:-1}" \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltasnapkv \
  --model "${MODEL_NAME:-llama31-8b-hf-deltasnapkv-longbench-b0p175-w16}" \
  --model_path "${MODEL_PATH}" \
  --deltakv_checkpoint_path "${DELTAKV_CHECKPOINT_PATH}" \
  --hyper_param "${HYPER_PARAM:-{\"deltasnapkv_total_budget\":0.175,\"hf_prefill_chunk_size\":4096,\"snapkv_window_size\":16,\"full_attention_layers\":\"\"}}" \
  --temperature 0 \
  --top_p 1 \
  --top_k 0
