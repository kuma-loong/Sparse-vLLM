#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/haojitai/projects/DeltaKV"
RUN_ID="comp_kv_range_text_with_compressor_gpu5_20260601_214709"
OUT_DIR="/data2/haojitai/outputs/deltakv/analysis/${RUN_ID}"

mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${PYTHONPATH:-}"

TEXT="DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. This diagnostic sentence is repeated to create enough tokens for the cluster analysis. DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. This diagnostic sentence is repeated to create enough tokens for the cluster analysis. DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. This diagnostic sentence is repeated to create enough tokens for the cluster analysis. DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. This diagnostic sentence is repeated to create enough tokens for the cluster analysis."

conda run -n svllm python -u src/deltakv/analysis/analyze_comp_kv_range.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --deltakv_checkpoint_path /data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor \
  --num_samples 1 \
  --output_dir "${OUT_DIR}" \
  --text "${TEXT}" \
  --device cuda \
  2>&1 | tee "${OUT_DIR}/run.log"
