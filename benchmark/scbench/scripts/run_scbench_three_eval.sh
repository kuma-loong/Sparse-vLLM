#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-$(date +%m%d_%H%M%S)}"
TASKS="${SCBENCH_TASKS:-scbench_kv,scbench_qa_eng,scbench_summary_with_needles}"

: "${SCBENCH_MODEL_PATH:?Set SCBENCH_MODEL_PATH to the local model path.}"
: "${DELTAKV_CHECKPOINT_PATH:?Set DELTAKV_CHECKPOINT_PATH to the local compressor checkpoint path.}"
: "${SCBENCH_PREPROCESSED_ROOT:?Set SCBENCH_PREPROCESSED_ROOT to the parquet data root.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${SCBENCH_LOG_DIR:-${REPO_ROOT}/outputs/bench_logs}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${SCBENCH_CONDA_ENV:-kv}"

KVZIP_LOG="${LOG_DIR}/scbench_three_kvzip_${TAG}.log"
DELTAKV_LOG="${LOG_DIR}/scbench_three_deltakv_${TAG}.log"

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[START] kvzip $(date)"
"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
  python -u benchmark/scbench/run_kvzip_preprocessed.py \
  --task "${TASKS}" \
  --data_root "${SCBENCH_PREPROCESSED_ROOT}" \
  --model_name_or_path "${SCBENCH_MODEL_PATH}" \
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
"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
  python -u benchmark/scbench/run_scbench_preprocessed.py \
  --task "${TASKS}" \
  --data_root "${SCBENCH_PREPROCESSED_ROOT}" \
  --model_name_or_path "${SCBENCH_MODEL_PATH}" \
  --attn_type deltakv \
  --num_eval_examples -1 \
  --max_seq_length 131072 \
  --hyper_param "${DELTAKV_HYPER_PARAM}" 2>&1 | tee "${DELTAKV_LOG}"

echo "[DONE] $(date)"
