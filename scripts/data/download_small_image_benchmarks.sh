#!/usr/bin/env bash
set -euo pipefail

cd /home/haojitai/projects/DeltaKV

export HF_HOME="${HF_HOME:-/data2/haojitai/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data2/haojitai/hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data2/haojitai/hf_cache/datasets}"

exec /home/haojitai/miniconda3/envs/svllm/bin/python scripts/data/download_small_image_benchmarks.py "$@"
