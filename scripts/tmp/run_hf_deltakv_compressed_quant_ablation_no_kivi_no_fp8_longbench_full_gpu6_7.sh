#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export DELTAKV_LONGBENCH_DATA_DIR="${DELTAKV_LONGBENCH_DATA_DIR:-/data2/haojitai/datasets/LongBench}"
export DELTAKV_OUTPUT_DIR="${DELTAKV_OUTPUT_DIR:-/data2/haojitai/outputs/deltakv}"
export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"

MODEL_PATH="/data2/haojitai/models/Qwen2.5-7B-Instruct-1M"
COMPRESSOR_PATH="/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

run_one() {
  local name="$1"
  local model_name="$2"
  local config_path="$3"
  local kr_note="$4"
  local run_id="longbench_qwen25_hf_compressed_quant_ablation_${name}_${RUN_TAG}"
  local run_dir="${DELTAKV_OUTPUT_DIR}/${run_id}"
  local pred_dir="${run_dir}/pred"
  local log_path="${run_dir}/run.log"

  mkdir -p "$run_dir" "$pred_dir"
  {
    echo "run_id=${run_id}"
    echo "host=$(hostname)"
    echo "cwd=$(pwd)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "config=${config_path}"
    echo "run_dir=${run_dir}"
    echo "pred_dir=${pred_dir}"
    echo "kr_note=${kr_note}"
    echo "start_time=$(date -Is)"

    conda run -n svllm python -u benchmark/long_bench/pred.py \
      --model "$model_name" \
      --model_path "$MODEL_PATH" \
      --tokenizer_path "$MODEL_PATH" \
      --deltakv_checkpoint_path "$COMPRESSOR_PATH" \
      --backend hf \
      --sparse_method delta_compressed_quant_kivi_full_fp8_ref \
      --hyper_param "$config_path" \
      --ws 2 \
      --batch_size 1 \
      --temperature 0.0 \
      --top_p 1.0 \
      --top_k 20 \
      --thinking_mode off \
      --output_root "$pred_dir"

    echo "end_time=$(date -Is)"
  } 2>&1 | tee "$log_path"
}

run_one \
  "no_kivi_gpu6_7" \
  "qwen25-7b-hf-deltakv-cq-no-kivi-fp8ref" \
  "configs/qwen2_5_7b_hf_deltakv_compressed_quant_fp8ref_no_kivi_paper_full_layers_center0p1_stride0p001.json" \
  "full layers raw; approximate ratio = 6/28*1 + 22/28*(0.1*0.5 + 0.9/16) = 0.297767857"

run_one \
  "no_fp8_gpu6_7" \
  "qwen25-7b-hf-deltakv-cq-kivi4-no-fp8ref" \
  "configs/qwen2_5_7b_hf_deltakv_compressed_quant_kivi4_no_fp8ref_paper_full_layers_center0p1_stride0p001.json" \
  "sparse refs fp16; approximate ratio = 6/28*0.25 + 22/28*(0.1 + 0.9/16) = 0.176339286"
