#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/haojitai/projects/DeltaKV"
OUT_DIR="/data2/haojitai/outputs/deltakv/analysis/compressed_diff_distribution_text_gpu5_20260601_220900"

mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${PYTHONPATH:-}"

conda run -n svllm python -u scripts/tmp/analyze_compressed_diff_distribution.py \
  2>&1 | tee "${OUT_DIR}/run.log"
