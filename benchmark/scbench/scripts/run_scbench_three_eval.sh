#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-$(date +%m%d_%H%M%S)}"
TASKS="scbench_kv,scbench_qa_eng,scbench_summary_with_needles"
MODEL_PATH="/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M"
DELTAKV_CHECKPOINT_PATH="/root/autodl-fs/checkpoints/compressor/cluster_e2e_cs256_biasFalse_l2_ratio0.1_clusMean_before_rope_lr0.0002_cdownmlp_swiglud3072_cuplinear_0125_222950"

KVZIP_LOG="/root/autodl-fs/bench_logs/scbench_three_kvzip_${TAG}.log"
DELTAKV_LOG="/root/autodl-fs/bench_logs/scbench_three_deltakv_${TAG}.log"

mkdir -p /root/autodl-fs/bench_logs

export https_proxy="http://localhost:7890"
export http_proxy="http://localhost:7890"
export SCBENCH_PREPROCESSED_ROOT="/root/autodl-fs/datasets/SCBench-preprocessed"

cd /root/autodl-tmp/Sparse-vLLM
export PYTHONPATH="/root/autodl-tmp/Sparse-vLLM/src:${PYTHONPATH:-}"

echo "[START] kvzip $(date)"
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python -u benchmark/scbench/run_kvzip_preprocessed.py \
  --task "${TASKS}" \
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
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python -u benchmark/scbench/run_scbench_preprocessed.py \
  --task "${TASKS}" \
  --model_name_or_path "${MODEL_PATH}" \
  --attn_type deltakv \
  --num_eval_examples -1 \
  --max_seq_length 131072 \
  --hyper_param "${DELTAKV_HYPER_PARAM}" 2>&1 | tee "${DELTAKV_LOG}"

echo "[DONE] $(date)"
