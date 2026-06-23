#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/haojitai/projects/DeltaKV"
RUN_ID="hotpotqa_kv_quant_error_stride0p001_center0p1_20samples_gpu5_20260601_204127"
OUT_DIR="/data2/haojitai/outputs/deltakv/analysis/${RUN_ID}"

mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${PYTHONPATH:-}"
export DELTAKV_OUTPUT_DIR="/data2/haojitai/outputs/deltakv"
export DELTAKV_LONGBENCH_DATA_DIR="/data2/haojitai/datasets/LongBench"

HYPER_PARAM='{
  "sparse_method": "deltakv-less-memory",
  "use_cluster": true,
  "use_compression": false,
  "chunk_prefill_accel_omnikv": false,
  "full_attention_layers": "0,1,2,4,7,14",
  "sink_keep_tokens": 8,
  "recent_keep_tokens": 128,
  "decode_keep_tokens": 2048,
  "prefill_keep_tokens": 4096,
  "deltakv_center_ratio": 0.1,
  "stride_alpha": 0.001,
  "deltakv_neighbor_count": 4,
  "deltakv_latent_quant_bits": 2,
  "full_layer_cluster_ratio": 0.08,
  "full_layer_stride_alpha": 0.0,
  "full_layer_kv_quant_bits": 4,
  "cluster_metric": "l2",
  "pool_kernel_size": 1
}'

conda run -n svllm python -u src/deltakv/analysis/analyze_hotpotqa_kv_quant_error.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --data_root /data2/haojitai/datasets/LongBench \
  --hyper_param "${HYPER_PARAM}" \
  --output_path "${OUT_DIR}/hotpotqa_kv_quant_error.json" \
  --analysis_mode both \
  --kivi_bits 2 \
  --kivi_group_size 32 \
  --kivi_residual_length 32 \
  --cuda_device 0 \
  --num_samples 20 \
  --max_new_tokens 1 \
  --max_model_len 121000 \
  --sample_limit_per_stat 20000 \
  --seed 42 \
  2>&1 | tee "${OUT_DIR}/run.log"
