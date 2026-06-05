# MInference Prefill

Sparse-vLLM supports MInference as an optional prefill attention backend. It is
orthogonal to the cache-manager sparse method:

- `sparse_method="vanilla"` plus `prefill_attention_backend="minference"` runs
  MInference during prefill and standard full-cache decode.
- `sparse_method="snapkv"` plus `prefill_attention_backend="minference"` runs
  MInference during prefill and keeps SnapKV's decode-time cache pruning.

V1 intentionally supports only MInference `vertical_and_slash` patterns. Pattern
entries such as `block_sparse`, `stream_llm`, or `flex_vertical_and_slash` fail
fast so unsupported experiments cannot silently mix semantics.

Implementation ownership:

- MInference is selected through cache-manager prefill hooks, not through a
  method-specific branch in `layers/attention.py`.
- `src/sparsevllm/engine/cache_manager/minference.py` owns the MInference
  prefill dispatch and full-prompt scheduling constraint.
- `src/sparsevllm/triton_kernel/minference_prefill.py` owns the custom sparse
  prefill kernel and metadata conversion.

Required parameters:

- `prefill_attention_backend="minference"`
- `minference_config_path`: path to a MInference best-pattern JSON file.

Optional parameters:

- `minference_starting_layer`: first layer that uses MInference, default `0`.
- `minference_ratio`: multiplier for configured vertical/slash budgets, default `1.0`.

Chunk prefill is not supported yet. When MInference is enabled, each prompt must
be admitted in one prefill step. Set `max_num_batched_tokens` large enough for
the evaluated prompt length; otherwise Sparse-vLLM raises an explicit error.
Setting `engine_prefill_chunk_size` to the prompt length is useful for batch size
1 because config normalization raises `max_num_batched_tokens` to at least that
value, but MInference does not use the chunk size as its step cap.

For short prompts or dense effective patterns, the MInference prefill kernel
runs the standard dense prefill kernel instead of the sparse path. Treat this as
part of the backend semantics when comparing throughput, and record prompt
length plus `minference_ratio` with benchmark results.
