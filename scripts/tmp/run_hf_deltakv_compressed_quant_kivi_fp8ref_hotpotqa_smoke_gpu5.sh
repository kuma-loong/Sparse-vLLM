#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="$PWD:$PWD/src"
export DELTAKV_LONGBENCH_DATA_DIR=/data2/haojitai/datasets/LongBench
export DELTAKV_OUTPUT_DIR=/data2/haojitai/outputs/deltakv

RUN_DIR="/data2/haojitai/outputs/deltakv/hf_compressed_quant_kivi_fp8ref_hotpotqa_smoke_gpu5_20260601"
mkdir -p "$RUN_DIR"

conda run -n svllm python -u benchmark/long_bench/pred.py \
  --model qwen25-7b-hf-deltakv-compressed-quant-kivi-fp8ref-smoke \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --backend hf \
  --sparse_method delta_compressed_quant_kivi_full_fp8_ref \
  --task hotpotqa \
  --num_samples 1 \
  --batch_size 1 \
  --temperature 0 \
  --ws 1 \
  --hyper_param configs/qwen2_5_7b_hf_deltakv_compressed_quant_kivi4_fp8ref_center0p1_stride0p001.json \
  --output_root "$RUN_DIR/pred" \
  2>&1 | tee "$RUN_DIR/run.log"
