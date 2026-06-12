# Qwen3.5 模型适配实现细节

## Adapter 接口

`ModelAdapter` 提供以下能力：

- `normalize_config(hf_config)`：把外部 HF config 转换为 Sparse-vLLM 可消费的 config。
- `create_model(config)`：创建 repo 内模型类。
- `map_weight_name(name)`：把 safetensors 中的权重名映射到 repo 内参数名；返回 `None` 表示显式跳过。
- `attention_layer_indices(config)`：声明哪些层是真正的 KV full-attention 层。
- `validate_engine_config(config)`：在配置阶段 fail fast。

Qwen3.5 adapter 将顶层 `qwen3_5` config 归一化为 `text_config`，并设置：

- `sparsevllm_model_type = "qwen3_5"`
- `sparsevllm_source_model_type = "qwen3_5"`
- `sparsevllm_attention_layer_indices = (3, 7, 11, 15, 19, 23, 27, 31)`

## Qwen3.5 full-attention 层

`Qwen3_5SparseAttention` 对齐官方结构：

- `q_proj` 输出包含 query 和 gate 两部分。
- `q_norm/k_norm` 使用 Transformers `Qwen3_5RMSNorm`。
- RoPE 使用 Transformers `Qwen3_5TextRotaryEmbedding`，并只旋转 `partial_rotary_factor` 对应的 head 维度。
- attention compute 调用 Sparse-vLLM `Attention`，因此仍由 cache manager 写入/读取 KV。
- attention 输出乘以 `sigmoid(gate)` 后进入 `o_proj`。

## Qwen3.5 linear-attention 层

`linear_attention` 层直接复用 Transformers `Qwen3_5GatedDeltaNet`，不复制公式。为了适配 Sparse-vLLM 的 packed token 输入：

- prefill 使用 `context.cu_seqlens_q` 切分每个序列的当前 chunk。
- decode 没有 `cu_seqlens_q` 时，每个序列对应一个 token。
- 每个 `seq_id` 独立维护一个 `DynamicCache(config=text_config)`。
- 释放序列时，`ModelRunner._free_slots_one()` 同步调用 `model.free_seq_state(seq_id)`。

这样可以保持 chunked prefill/decode 的 linear recurrent state 连续性。

## 权重加载

Qwen3.5 safetensors 的主干权重命名形如：

- `model.language_model.embed_tokens.weight`
- `model.language_model.layers.3.self_attn.q_proj.weight`
- `lm_head.weight`

Sparse-vLLM 模型命名形如：

- `model.embed_tokens.weight`
- `model.layers.3.self_attn.q_proj.weight`
- `lm_head.weight`

adapter 规则：

- `model.language_model.` -> `model.`
- `model.visual.` -> skip
- `model.image_newline` -> skip
- `mtp.` -> skip

loader 的 `weight_name_mapper` 是通用入口，后续新模型可以复用该机制。

## SparseController 约束

Qwen3.5 的 `linear_attention` 层不产生 KV cache attention score。为避免 SnapKV 错误驱逐这些层：

- `SparseController` 读取 `hf_config.sparsevllm_attention_layer_indices`。
- `_needs_attn_score()` 对非 full-attention 层直接返回 `False`。
- SnapKV/StreamingLLM prefill 和 decode 驱逐循环只遍历 full-attention 层。

## KV cache 层映射

`CacheManager` 维护：

- `attention_layer_indices`
- `attention_layer_index_set`
- `layer_to_cache_idx`
- `num_cache_layers`

对于 Qwen3.5：

- 逻辑层号仍是 0-31，保证模型层、SparseController 层号和权重名一致。
- KV 存储层只包含 8 个 full-attention 层。
- `StandardCacheManager.get_layer_kv_cache()` 和 `SnapKVCacheManager.get_layer_kv_cache()` 通过 `cache_layer_idx()` 将逻辑层号映射到 KV 存储层号。
- SnapKV 的 `_prepare_prefill()`、`_prepare_decode()`、`free_seq()` 和静态 decode metadata 只遍历 full-attention 层。

这避免了为 24 个 Gated DeltaNet 层分配永远不会写入的 KV cache。

## 禁用组合

Qwen3.5 adapter 明确拒绝：

- `enable_prefix_caching=True`：prefix cache 只恢复 full-attention KV，不能恢复 Gated DeltaNet 的 conv/recurrent state。
- `decode_cuda_graph=True`：当前 decode 使用按 `seq_id` 查找的 Python `DynamicCache`，不满足 CUDA graph replay 的静态地址/静态批次语义。

## 当前未做的实现

- 未实现 Qwen3.5 vision tower 和 multimodal projector。
- 未实现 Qwen3.5 tensor parallel。
- 未声明 OmniKV/DeltaKV/QuEST 支持。
- 未安装 `flash-linear-attention` 或 `causal-conv1d`，linear attention 当前依赖 Transformers torch fallback。
