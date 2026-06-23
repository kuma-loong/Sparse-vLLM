#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export DELTAKV_LONGBENCH_DATA_DIR="${DELTAKV_LONGBENCH_DATA_DIR:-/data2/haojitai/datasets/LongBench}"
export DELTAKV_OUTPUT_DIR="${DELTAKV_OUTPUT_DIR:-/data2/haojitai/outputs/deltakv}"
export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"

CONFIG_PATH="configs/qwen2_5_7b_hf_deltakv_compressed_quant_kivi4_fp8ref_kr13_center0p1_stride0p001.json"
MODEL_PATH="/data2/haojitai/models/Qwen2.5-7B-Instruct-1M"
COMPRESSOR_PATH="/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor"
RUN_ID="${RUN_ID:-longbench_qwen25_hf_compressed_quant_kivi4_fp8ref_kr13_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${DELTAKV_OUTPUT_DIR}/${RUN_ID}"
PRED_DIR="${RUN_DIR}/pred"
LOG_PATH="${RUN_DIR}/run.log"

mkdir -p "$RUN_DIR" "$PRED_DIR"

{
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "config=${CONFIG_PATH}"
  echo "run_dir=${RUN_DIR}"
  echo "pred_dir=${PRED_DIR}"
  echo "kr_estimate=1/28 + 27/28*0.1 = 0.132142857"
  echo "start_time=$(date -Is)"

  conda run -n svllm python -u benchmark/long_bench/pred.py \
    --model qwen25-7b-hf-deltakv-cq-kivi4-fp8ref-kr13 \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --deltakv_checkpoint_path "$COMPRESSOR_PATH" \
    --backend hf \
    --sparse_method delta_compressed_quant_kivi_full_fp8_ref \
    --hyper_param "$CONFIG_PATH" \
    --ws 4 \
    --batch_size 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --top_k 20 \
    --thinking_mode off \
    --output_root "$PRED_DIR"

  echo "end_time=$(date -Is)"
} 2>&1 | tee "$LOG_PATH"
