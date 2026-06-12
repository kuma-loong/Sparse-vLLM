# Qwen3.5 模型适配改动分析

## 主要改动

1. 新增模型 adapter 层
   - 文件：`src/sparsevllm/models/adapters.py`
   - 职责：加载并归一化 HF config、选择模型类、声明支持方法、映射/跳过权重名、声明 full-attention 层。
   - 影响：`Config` 和 `ModelRunner` 不再继续扩展硬编码模型分支。

2. 新增 Qwen3.5 text-only 模型
   - 文件：`src/sparsevllm/models/qwen3_5.py`
   - full-attention 层：使用 Sparse-vLLM `Attention`，可接入 vanilla KV cache 和 SnapKV。
   - linear-attention 层：复用 Transformers `Qwen3_5GatedDeltaNet`，以最小实现成本保持官方公式一致。
   - 状态：按 `seq_id` 保存 `DynamicCache`，在 `ModelRunner.free_slots()` 时释放。

3. loader 支持通用权重映射
   - 文件：`src/sparsevllm/utils/loader.py`
   - 新增 `weight_name_mapper` 参数。
   - Qwen3.5 将 `model.language_model.*` 映射为 `model.*`。
   - 显式跳过 `model.visual.*`、`model.image_newline`、`mtp.*`。

4. SparseController 收窄 sparse 操作层范围
   - 文件：`src/sparsevllm/engine/sparse_controller.py`
   - 新增 `sparsevllm_attention_layer_indices` 支持。
   - SnapKV/StreamingLLM 的分数收集和驱逐只遍历 full-attention 层。

5. CacheManager 收窄 KV 分配层范围
   - 文件：`src/sparsevllm/engine/cache_manager/base.py`
   - 文件：`src/sparsevllm/engine/cache_manager/standard.py`
   - 文件：`src/sparsevllm/engine/cache_manager/snapkv.py`
   - `CacheManager` 维护逻辑层号到 KV 存储层号的映射。
   - Qwen3.5 vanilla/SnapKV 只为 8 个 full-attention 层分配 KV cache，不为 24 个 Gated DeltaNet 层消耗 KV 容量。

6. Transformers 主分支兼容修复
   - 文件：`src/deltakv/modeling/llava_ov/llava_onevision_deltakv.py`
   - 适配新版 Transformers 中 `KwargsForCausalLM` 和 `is_torchdynamo_compiling` 的导出位置变化。

## 方法支持决策

Qwen3.5 adapter 首批只允许：

- `vanilla`
- `snapkv`

理由：

- `vanilla` 是必要 dense baseline。
- `snapkv` 已是 repo 内一等 cache-manager 方法，不依赖外部 compressor checkpoint。
- Qwen3.5 只有 8 个 full-attention 层，SnapKV 可以局部作用于这些层，不需要改写 Gated DeltaNet。
- OmniKV/DeltaKV/QuEST 的 query-aware 或跨层传播语义需要重新设计，当前不应默认声明支持。

## 风险和限制

- 当前 Qwen3.5 仅支持 `tensor_parallel_size=1`。
- 当前只支持 text-only Causal LM，不支持图片/视频输入。
- 当前 Qwen3.5 禁用 prefix cache 和 decode CUDA graph，以避免 linear-attention 状态不一致。
- linear-attention 层复用 Transformers torch fallback；未安装 `flash-linear-attention` 和 `causal-conv1d` 时性能不是最终形态。
- GPU 实测未执行：检查时 0-7 号卡均已有任务进程，不满足“只能使用空闲卡”的要求。

## 验证结果

- 环境版本：
  - `torch==2.8.0+cu128`
  - `triton==3.4.0`
  - `flash-attn==2.8.3`
  - `transformers==5.10.0.dev0`
- 编译：
  - `python -m py_compile` 覆盖新增/修改核心文件，通过。
- 单测：
  - `tests/test_qwen35_model_adapter.py`
  - `tests/test_prefill_schedule_policy.py`
  - `tests/test_prefix_cache.py`
  - `tests/test_research_fail_fast.py`
  - 结果：`71 passed, 52 subtests passed`
- 权重映射审计：
  - 源权重：775
  - 显式跳过视觉/MTP 权重：348
  - text 主干匹配参数：427
  - missing：0
