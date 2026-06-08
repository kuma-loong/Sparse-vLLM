#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the LLaVA-OneVision model directory}"
VISUAL_CACHE_DATA_DIR="${SVLLM_VISUAL_CACHE_DATA_DIR:?Set SVLLM_VISUAL_CACHE_DATA_DIR to the prepared visual-cache dataset directory}"
VQAV2_DATA_DIR="${SVLLM_VQAV2_DATA_DIR:?Set SVLLM_VQAV2_DATA_DIR to the VQAv2 dataset directory}"
OUTPUT_DIR="${SVLLM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmark/results}/multimodal/visual_cache_keep10_full"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON}" -u benchmark/multimodal/visual_cache/run_visual_cache.py \
  --model_path "${MODEL_PATH}" \
  --deltakv_checkpoint_path none \
  --dataset_dir "${VISUAL_CACHE_DATA_DIR}" \
  --source_vqa_dir "${VQAV2_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples -1 \
  --max_new_tokens 8 \
  --cuda_device "${CUDA_DEVICE:-0}" \
  --methods vanilla,visual_uniform_keep \
  --visual_keep_ratio 0.1 \
  --full_attention_layers "" \
  --attn_implementation flash_attention_2 \
  --log_every 500
