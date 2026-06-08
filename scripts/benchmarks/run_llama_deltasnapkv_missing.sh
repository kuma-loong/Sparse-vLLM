#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TASKS="${TASKS:-trec,triviaqa,samsum,multi_news,passage_count,passage_retrieval_en,lcc,repobench-p}"
WS="${WS:-2}"
source "${REPO_ROOT}/scripts/benchmarks/run_llama_deltasnapkv_remaining.sh"
