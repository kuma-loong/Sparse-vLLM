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

Required parameters:

- `prefill_attention_backend="minference"`
- `minference_config_path`: path to a MInference best-pattern JSON file.

Optional parameters:

- `minference_starting_layer`: first layer that uses MInference, default `0`.
- `minference_ratio`: multiplier for configured vertical/slash budgets, default `1.0`.

Chunk prefill is not supported yet. When MInference is enabled, each prompt must
be admitted in one prefill step. Set `engine_prefill_chunk_size` large enough for
the evaluated prompt length; otherwise Sparse-vLLM raises an explicit error.
