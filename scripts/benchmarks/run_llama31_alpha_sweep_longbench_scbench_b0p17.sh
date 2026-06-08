#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${SVLLM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmark/results}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Llama model directory}"
DELTAKV_CHECKPOINT_PATH="${DELTAKV_CHECKPOINT_PATH:?Set DELTAKV_CHECKPOINT_PATH to the compressor checkpoint directory}"
LONG_BENCH_DATA_DIR="${SVLLM_LONGBENCH_DATA_DIR:-${DELTAKV_LONGBENCH_DATA_DIR:-}}"
SCBENCH_PREPROCESSED_ROOT="${SVLLM_SCBENCH_PREPROCESSED_ROOT:-${SCBENCH_PREPROCESSED_ROOT:-}}"
if [[ -z "${LONG_BENCH_DATA_DIR}" ]]; then
  echo "Set SVLLM_LONGBENCH_DATA_DIR or DELTAKV_LONGBENCH_DATA_DIR to the LongBench root containing data/*.jsonl" >&2
  exit 2
fi
if [[ -z "${SCBENCH_PREPROCESSED_ROOT}" ]]; then
  echo "Set SVLLM_SCBENCH_PREPROCESSED_ROOT or SCBENCH_PREPROCESSED_ROOT to the directory containing scbench_*.parquet" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SVLLM_BENCHMARK_OUTPUT_DIR="${OUTPUT_ROOT}"
export SVLLM_LONGBENCH_DATA_DIR="${LONG_BENCH_DATA_DIR}"
export SVLLM_SCBENCH_PREPROCESSED_ROOT="${SCBENCH_PREPROCESSED_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

IFS=',' read -r -a ALPHAS <<< "${ALPHAS:-0.001,0.02,0.05,0.1}"
SCBENCH_TASKS="${SCBENCH_TASKS:-scbench_kv,scbench_qa_eng,scbench_summary_with_needles,scbench_many_shot}"

for alpha in "${ALPHAS[@]}"; do
  alpha_label="${alpha//./p}"
  hyper_param=$(cat <<JSON
{"hf_prefill_chunk_size":32768,"prefill_keep_tokens":0.17,"chunk_prefill_accel_omnikv":false,"deltakv_use_omnikv_selection":true,"decode_keep_tokens":0.17,"full_attention_layers":"0,1,2,8,18","recent_keep_tokens":128,"sink_keep_tokens":8,"use_compression":true,"use_cluster":true,"deltakv_center_ratio":0.1,"stride_alpha":${alpha},"deltakv_latent_quant_bits":0}
JSON
)

  echo "[$(date '+%F %T')] alpha=${alpha} longbench start"
  "${PYTHON}" -u benchmark/long_bench/pred.py \
    --model "llama31-8b-hf-deltakv-longbench-b0p17-alpha${alpha_label}" \
    --model_path "${MODEL_PATH}" \
    --deltakv_checkpoint_path "${DELTAKV_CHECKPOINT_PATH}" \
    --ws "${WS:-1}" \
    --batch_size 1 \
    --backend hf \
    --sparse_method deltakv \
    --temperature 0 \
    --top_p 1 \
    --top_k 0 \
    --hyper_param "${hyper_param}" \
    --output_root "${OUTPUT_ROOT}/long_bench/alpha_sweep/llama31-8b-alpha${alpha_label}"
  echo "[$(date '+%F %T')] alpha=${alpha} longbench done"

  echo "[$(date '+%F %T')] alpha=${alpha} scbench start"
  "${PYTHON}" -u benchmark/scbench/run_scbench_preprocessed.py \
    --task "${SCBENCH_TASKS}" \
    --data_root "${SCBENCH_PREPROCESSED_ROOT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_ROOT}/scbench_alpha_llama/llama31-8b-alpha${alpha_label}" \
    --attn_type deltakv \
    --kv_type dense \
    --max_seq_length 131072 \
    --hyper_param "{\"sparse_method\":\"deltakv\",\"deltakv_checkpoint_path\":\"${DELTAKV_CHECKPOINT_PATH}\",\"hf_prefill_chunk_size\":32768,\"prefill_keep_tokens\":0.17,\"chunk_prefill_accel_omnikv\":false,\"deltakv_use_omnikv_selection\":true,\"decode_keep_tokens\":0.17,\"full_attention_layers\":\"0,1,2,8,18\",\"recent_keep_tokens\":128,\"sink_keep_tokens\":8,\"use_compression\":true,\"use_cluster\":true,\"deltakv_center_ratio\":0.1,\"stride_alpha\":${alpha},\"deltakv_latent_quant_bits\":0}"
  echo "[$(date '+%F %T')] alpha=${alpha} scbench done"
done
