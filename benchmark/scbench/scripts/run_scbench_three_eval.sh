#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-$(date +%m%d_%H%M%S)}"
TASKS="scbench_kv,scbench_qa_eng,scbench_summary_with_needles"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model directory}"
DELTAKV_CHECKPOINT_PATH="${DELTAKV_CHECKPOINT_PATH:?Set DELTAKV_CHECKPOINT_PATH to the compressor checkpoint directory}"
SCBENCH_PREPROCESSED_ROOT="${SVLLM_SCBENCH_PREPROCESSED_ROOT:-${SCBENCH_PREPROCESSED_ROOT:-}}"
if [[ -z "${SCBENCH_PREPROCESSED_ROOT}" ]]; then
  echo "Set SVLLM_SCBENCH_PREPROCESSED_ROOT or SCBENCH_PREPROCESSED_ROOT to the directory containing scbench_*.parquet" >&2
  exit 2
fi

LOG_DIR="${SVLLM_BENCHMARK_LOG_DIR:-${REPO_ROOT}/benchmark/results/logs}"
OUTPUT_DIR="${SVLLM_BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/benchmark/results}/scbench_preprocessed"
KVZIP_LOG="${LOG_DIR}/scbench_three_kvzip_${TAG}.log"
DELTAKV_LOG="${LOG_DIR}/scbench_three_deltakv_${TAG}.log"

mkdir -p "${LOG_DIR}"

if [[ "${USE_PROXY_7890:-0}" == "1" ]]; then
  export https_proxy="${https_proxy:-http://localhost:7890}"
  export http_proxy="${http_proxy:-http://localhost:7890}"
fi
export SCBENCH_PREPROCESSED_ROOT

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[START] kvzip $(date)"
"${PYTHON}" -u benchmark/scbench/run_kvzip_preprocessed.py \
  --task "${TASKS}" \
  --data_root "${SCBENCH_PREPROCESSED_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --num_eval_examples -1 \
  --max_seq_length 131072 \
  --ratio 0.3 \
  --level pair \
  --kv_type retain 2>&1 | tee "${KVZIP_LOG}"

DELTAKV_HYPER_PARAM='{
  "deltakv_checkpoint_path": "'"${DELTAKV_CHECKPOINT_PATH}"'",
  "use_cluster": true,
  "deltakv_center_ratio": 0.1,
  "decode_keep_tokens": 0.11,
  "prefill_keep_tokens": 204800000,
  "recent_keep_tokens": 128,
  "sink_keep_tokens": 8,
  "hf_prefill_chunk_size": 204800000,
  "chunk_prefill_accel_omnikv": true,
  "full_attention_layers": "0,1,2,4,7,14"
}'

echo "[START] deltakv $(date)"
"${PYTHON}" -u benchmark/scbench/run_scbench_preprocessed.py \
  --task "${TASKS}" \
  --data_root "${SCBENCH_PREPROCESSED_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --attn_type deltakv \
  --num_eval_examples -1 \
  --max_seq_length 131072 \
  --hyper_param "${DELTAKV_HYPER_PARAM}" 2>&1 | tee "${DELTAKV_LOG}"

echo "[DONE] $(date)"
