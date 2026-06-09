# Prefix Cache Benchmark Design

This benchmark evaluates Sparse-vLLM prefix caching as a KV-cache lifecycle feature, not just a single-prompt latency optimization.

## Reference Patterns

The design borrows two complementary patterns from the local references:

- vLLM `benchmark_prefix_caching.py`: controlled shared-prefix prompts, repeated requests, and fixed token-length ranges. This isolates whether cached full prefix blocks reduce prefill work.
- SGLang `benchmark/hicache/bench_warm_cache.py` and `benchmark/hicache/bench_multiturn.py`: explicit warm-cache construction, multi-client multi-round request streams, dynamic growing histories, per-request TTFT, cached tokens, and per-round cache hit summaries.

Sparse-vLLM uses a local offline engine instead of an HTTP server, so the benchmark drives `LLM.add_request()` and `LLM.step()` directly and records runtime cache-manager stats.

## Case Matrix

Run these cases on the same model, seed, prompt trace, decode config, and GPU:

| Case | Sparse method | Prefix cache | Purpose |
| --- | --- | --- | --- |
| `baseline_full` | `vanilla` | off | Full-attention baseline with no reuse |
| `prefix_full` | `vanilla` | on | Full attention plus token-block prefix cache |
| `prefix_omnikv` | `omnikv` | on | Compatible sparse method using standard token slots |
| `prefix_quest` | `quest` | on | Compatible sparse method using page-aligned cache blocks |

Do not include methods that physically prune/drop/compress KV in this prefix-cache matrix unless their cache-manager payload semantics are implemented and tested.

## Workloads

`shared_prefix`

1. Build a deterministic shared token prefix.
2. Warm that prefix once.
3. Send multiple full prompts made from the shared prefix plus unique suffixes.
4. Measure TTFT, latency, cached tokens, eligible cached tokens, and cache-manager stats.

This is the vLLM-style controlled reuse test.

`multiturn`

1. Build a deterministic shared system/tool prefix.
2. Create multiple sessions with session-specific context.
3. For each turn, send the full current conversation prompt.
4. Append generated output tokens to the session history before the next turn.

This simulates real chat/agent requests where each new request repeats the previous prompt plus dynamic assistant output and a new user message. Because `prefix_cache_cache_decode_blocks=False`, generated decode tokens are intentionally recomputed on the next turn; only prompt-prefill blocks can be reused.

For comparable performance runs, the benchmark uses deterministic synthetic assistant-history tokens by default (`--history_update synthetic`). Real model outputs are still generated and saved, but they do not feed the next prompt. This keeps all cases on the same request trace while preserving dynamic growing histories. Use `--history_update generated` only for an exploratory traffic simulation where case-to-case prompt traces may diverge after the first turn.

## Sparse-Path Guardrails

Prefix-cache performance runs must enter the sparse method path. The driver fails fast unless `--allow_short_trace` is explicitly passed.

- OmniKV prefill/TTFT sparse path requires `prompt_len > sink_keep_tokens + decode_keep_tokens + recent_keep_tokens + engine_prefill_chunk_size`, and `chunk_prefill_accel_omnikv=True`.
- OmniKV decode sparse path requires `context_len > sink_keep_tokens + decode_keep_tokens + recent_keep_tokens`.
- QuEST sparse decode path requires `context_len > quest_token_budget`; QuEST does not sparsify prefill, so TTFT mostly measures prefix-cache prefill reuse plus fixed overhead, while latency/TPOT reflect QuEST decode sparsity.
- The report records `long_prefill_requests` and `quest_sparse_decode_eligible_requests`; a sparse performance run with zeros there is invalid.

## Required Outputs

Each run writes:

- `run_info.json`: command, git metadata, environment, model, seed, and config.
- `benchmark_plan.json`: expanded case matrix and engine kwargs.
- `per_turn_results.jsonl`: per-request status, prompt tokens, generated tokens, TTFT, latency, cached tokens, eligible tokens, and turn/session ids.
- `raw_outputs.jsonl`: full prompt token ids and generated token ids/text.
- `performance.jsonl`: one summary row per case.
- `aggregate_metrics.json`: all case summaries.
- `report.md`: compact table.
- `benchmark/results/_ledgers/prefix_cache.{jsonl,csv}`: feature-level ledger entries.

Every request is recorded with an explicit status. Failed attempts remain in the ledger as `model_failed`, `oom`, `timeout`, etc.

## Primary Metrics

- `mean_ttft_ms`, `p90_ttft_ms`, `mean_latency_ms`
- `cache_hit_rate = cached_tokens / prompt_tokens`
- `eligible_cache_hit_rate = cached_tokens / eligible_cache_tokens`
- `physical_kv_reuse_rate`
- `recomputed_prompt_tokens`
- per-turn TTFT and cache hit rates
- prefix cache runtime counters: lookups, hit requests, hit tokens, materialized blocks, duplicate blocks, evictions, live/evictable/pinned blocks
- peak GPU memory

Use `eligible_cache_hit_rate` to validate correctness of the cache matching logic, and use TTFT/latency to evaluate performance.

## Functional Smoke Command

This only validates cache lifecycle, output saving, and failure handling. It is not a sparse performance benchmark, because the prompts are intentionally short and require `--allow_short_trace`.

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/benchmarks/bench_prefix_cache.py \
  --model_path /data2/guquansheng/models/Qwen2.5-7B-Instruct-1M \
  --cases baseline_full,prefix_full,prefix_omnikv,prefix_quest \
  --workloads shared_prefix,multiturn \
  --sessions 2 \
  --turns 2 \
  --system_prompt_len 64 \
  --session_prefix_len 16 \
  --user_len 8 \
  --shared_prompts 2 \
  --shared_prefix_len 64 \
  --shared_suffix_len 16 \
  --output_len 4 \
  --gpu_memory_utilization 0.55 \
  --max_active_requests 2 \
  --max_num_batched_tokens 512 \
  --chunk_prefill_size 128 \
  --num_top_tokens 128 \
  --num_top_tokens_in_prefill 128 \
  --num_recent_tokens 64 \
  --full_attn_layers 0,1,2,4,7,14 \
  --quest_token_budget 128 \
  --master_port_base 25000 \
  --case_timeout_s 900 \
  --allow_short_trace \
  --continue_on_failure
```

## Quick Realistic Command

Run only when an idle GPU is available. This trace enters OmniKV prefill sparse path and QuEST sparse decode path while keeping runtime manageable.

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> .venv/bin/python scripts/benchmarks/bench_prefix_cache.py \
  --model_path /data2/guquansheng/models/Qwen2.5-7B-Instruct-1M \
  --cases baseline_full,prefix_full,prefix_omnikv,prefix_quest \
  --workloads shared_prefix,multiturn \
  --sessions 4 \
  --turns 3 \
  --system_prompt_len 4096 \
  --session_prefix_len 512 \
  --user_len 128 \
  --shared_prompts 4 \
  --shared_prefix_len 4096 \
  --shared_suffix_len 512 \
  --output_len 64 \
  --history_update synthetic \
  --gpu_memory_utilization 0.60 \
  --max_active_requests 4 \
  --max_num_batched_tokens 8192 \
  --chunk_prefill_size 1024 \
  --num_top_tokens 512 \
  --num_top_tokens_in_prefill 512 \
  --num_recent_tokens 128 \
  --num_sink_tokens 8 \
  --full_attn_layers 0,1,2,4,7,14 \
  --quest_token_budget 1024 \
  --master_port_base 26000 \
  --case_timeout_s 1800 \
  --continue_on_failure
```
