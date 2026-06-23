#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/haojitai/projects/DeltaKV}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to a directory under /data2/haojitai/outputs.}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/qwen2_5_7b_deltakv_less_memory_int2_dynamic_stride_paper_full_layers.json}"
MODEL_PATH="${MODEL_PATH:-/data2/haojitai/models/Qwen2.5-7B-Instruct-1M}"
DATA_DIR="${DATA_DIR:-/data2/haojitai/datasets/LongBench}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
MAX_USED_MB="${MAX_USED_MB:-10000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-30}"
MAX_RANK_RETRIES="${MAX_RANK_RETRIES:-3}"
LONG_BENCH_BATCH_SIZE="${LONG_BENCH_BATCH_SIZE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-23330}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ARRAY[@]}"
PRED_DIR="$OUT_ROOT/pred"
mkdir -p "$OUT_ROOT" "$PRED_DIR"
cp "$CONFIG_PATH" "$OUT_ROOT/hyper_params.json"

STATUS_FILE="$OUT_ROOT/status.txt"
RUN_LOG="$OUT_ROOT/run.log"

cd "$REPO_DIR"

DATASETS=(
  narrativeqa
  qasper
  multifieldqa_en
  hotpotqa
  2wikimqa
  musique
  gov_report
  qmsum
  multi_news
  trec
  triviaqa
  samsum
  passage_count
  passage_retrieval_en
  lcc
  repobench-p
)

for dataset in "${DATASETS[@]}"; do
  : > "$PRED_DIR/${dataset}.jsonl"
done

timestamp() {
  date '+%F %T'
}

probe_gpu_state() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
}

is_gpu_free() {
  local gpu="$1"
  local state
  state="$(probe_gpu_state)"
  printf '%s\n' "$state"
  awk -F, -v gpu="$gpu" -v max_mem="$MAX_USED_MB" -v max_util="$MAX_UTIL_PCT" '
    {
      gsub(/ /, "", $1)
      gsub(/ /, "", $2)
      gsub(/ /, "", $3)
      if ($1 == gpu) {
        found = 1
        if ($2 <= max_mem && $3 <= max_util) {
          exit 0
        }
        exit 1
      }
    }
    END {
      if (!found) {
        exit 2
      }
    }
  ' <<< "$state"
}

wait_for_gpu() {
  local rank="$1"
  local gpu="$2"
  echo "[$(timestamp)] rank=${rank} waiting for GPU ${gpu}: memory <= ${MAX_USED_MB}MiB and util <= ${MAX_UTIL_PCT}%"
  until is_gpu_free "$gpu"; do
    rc=$?
    if [[ "$rc" -eq 2 ]]; then
      echo "[$(timestamp)] rank=${rank} GPU ${gpu} not found in nvidia-smi output"
      return 2
    fi
    echo "[$(timestamp)] rank=${rank} GPU ${gpu} busy; sleeping ${WAIT_INTERVAL_SECONDS}s"
    sleep "$WAIT_INTERVAL_SECONDS"
  done
}

run_rank() {
  local rank="$1"
  local gpu="$2"
  local rank_log="$OUT_ROOT/rank_${rank}_gpu_${gpu}.log"
  local rank_status="$OUT_ROOT/rank_${rank}.status"

  for attempt in $(seq 1 "$MAX_RANK_RETRIES"); do
    echo "waiting" > "$rank_status"
    wait_for_gpu "$rank" "$gpu"

    echo "running_attempt_${attempt}" > "$rank_status"
    echo "[$(timestamp)] rank=${rank} starting on physical GPU ${gpu}, attempt ${attempt}/${MAX_RANK_RETRIES}" | tee -a "$rank_log"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$((MASTER_PORT_BASE + rank))" \
    DELTAKV_LONGBENCH_DATA_DIR="$DATA_DIR" \
    DELTAKV_OUTPUT_DIR=/data2/haojitai/outputs/deltakv \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$REPO_DIR/src" \
    conda run -n svllm python -u benchmark/long_bench/pred.py \
      --model qwen25-7b-svllm-deltakv-deltaquant-fullres-longbench \
      --model_path "$MODEL_PATH" \
      --tokenizer_path "$MODEL_PATH" \
      --backend sparsevllm \
      --sparse_method deltakv-less-memory \
      --batch_size "$LONG_BENCH_BATCH_SIZE" \
      --temperature 0 \
      --top_p 1 \
      --top_k 20 \
      --thinking_mode off \
      --ws "$WORLD_SIZE" \
      --worker_rank "$rank" \
      --worker_world_size "$WORLD_SIZE" \
      --hyper_param "$OUT_ROOT/hyper_params.json" \
      --output_root "$PRED_DIR" \
      >> "$rank_log" 2>&1
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
      echo "completed" > "$rank_status"
      echo "[$(timestamp)] rank=${rank} completed" | tee -a "$rank_log"
      return 0
    fi

    echo "failed_attempt_${attempt}_exit_${rc}" > "$rank_status"
    echo "[$(timestamp)] rank=${rank} failed with exit code ${rc}; see ${rank_log}" | tee -a "$rank_log"
    if [[ "$attempt" -lt "$MAX_RANK_RETRIES" ]]; then
      sleep "$WAIT_INTERVAL_SECONDS"
    fi
  done

  echo "failed" > "$rank_status"
  return 1
}

echo "running_rank_watchers" > "$STATUS_FILE"
echo "[$(timestamp)] launching rank-level watchers for GPUs ${GPU_IDS}" | tee -a "$RUN_LOG"

pids=()
for rank in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$rank]}"
  gpu="${gpu// /}"
  run_rank "$rank" "$gpu" >> "$RUN_LOG" 2>&1 &
  pids+=("$!")
  echo "$!" > "$OUT_ROOT/rank_${rank}.pid"
done

failed=0
for rank in "${!pids[@]}"; do
  if ! wait "${pids[$rank]}"; then
    failed=1
    echo "[$(timestamp)] rank=${rank} watcher failed" | tee -a "$RUN_LOG"
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "failed" > "$STATUS_FILE"
  echo "[$(timestamp)] one or more ranks failed; skipping eval" | tee -a "$RUN_LOG"
  exit 1
fi

echo "evaluating" > "$STATUS_FILE"
echo "[$(timestamp)] all ranks completed; running LongBench eval" | tee -a "$RUN_LOG"
PYTHONPATH="$REPO_DIR/src" conda run -n svllm python -u benchmark/long_bench/eval.py \
  --path "$PRED_DIR" \
  >> "$RUN_LOG" 2>&1

echo "completed" > "$STATUS_FILE"
echo "[$(timestamp)] LongBench completed" | tee -a "$RUN_LOG"
