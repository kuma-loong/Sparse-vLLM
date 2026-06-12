# Qwen3.5 模型适配落地计划

## 目标

为 Sparse-vLLM 增加一套可复用的模型适配框架，减少 `ModelRunner`、`Config`、`loader` 中的模型特例分支，并尝试适配本地模型：

- 模型路径：`/data2/guquansheng/models/Qwen3.5-9B`
- HF 模型：`Qwen/Qwen3.5-9B`
- 首批运行方法：`vanilla` 和 `snapkv`

## 适配边界

Qwen3.5 不是普通 Qwen3 dense-attention 模型。其顶层 `model_type` 为 `qwen3_5`，文本主干在 `text_config` 中，`layer_types` 是 3:1 混合结构：

- 24 层 `linear_attention`，对应 Gated DeltaNet
- 8 层 `full_attention`，层号为 `3,7,11,15,19,23,27,31`

本次适配只接入 text-only Causal LM 主干，不接入视觉编码器、视频输入、MTP 分支。视觉和 MTP 权重会在 loader 中显式跳过，避免静默误加载。

## 分阶段计划

1. 环境对齐
   - 使用项目根目录 `.venv`。
   - 通过 `uv pip` 管理依赖，禁止 conda 和全局 pip。
   - 将 Torch 栈固定到 `torch==2.8.0+cu128`、`triton==3.4.0`。
   - 安装本地预编译 `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`。
   - 安装 Transformers 主分支，以支持 `qwen3_5` / `qwen3_5_text` 配置。

2. 模型适配框架
   - 新增 `sparsevllm.models.adapters`。
   - 将 HF config 加载、顶层多模态 config 到 text config 的归一化、模型类选择、权重名映射、方法支持范围校验都收敛到 adapter。
   - `Config` 不再直接调用 `AutoConfig`；`ModelRunner` 不再硬编码 `qwen2/qwen3` 分支。

3. Qwen3.5 text 主干
   - 新增 `sparsevllm.models.qwen3_5`。
   - full-attention 层使用 Sparse-vLLM `Attention`，继续走 cache manager、SparseController 和 SnapKV 逻辑。
   - linear-attention 层复用 Transformers 的 `Qwen3_5GatedDeltaNet`，按 `seq_id` 保存独立 recurrent/conv 状态，保证 chunked prefill 和 decode 不丢状态。
   - 仅支持 `tensor_parallel_size=1`，避免非 TP 模块在 TP 场景产生错误结果。

4. 稀疏方法接入
   - 首批支持 `vanilla` 和 `snapkv`。
   - SnapKV 的注意力分数收集和物理驱逐只作用于 full-attention 层。
   - Standard/SnapKV cache manager 的 KV 分配只按 full-attention 层计入容量，不为 Gated DeltaNet 层分配无用 KV。
   - Qwen3.5 暂不支持 prefix cache 和 decode CUDA graph：这两者都需要恢复/重放 linear-attention recurrent state。
   - `omnikv`、DeltaKV、QuEST、StreamingLLM 等暂不声明为 Qwen3.5 支持方法，配置阶段 fail fast。

5. 验证与交付
   - 编译变更文件。
   - 跑 adapter、prefill policy、prefix cache、research fail-fast 相关单测。
   - 做 Qwen3.5 safetensors 权重映射审计。
   - GPU 推理/吞吐验证只在检测到空闲卡后执行；当前 0-7 号 GPU 均已有进程占用，因此不抢占运行。

## 后续扩展

- 为 Qwen3.5 增加 tensor parallel，需要替换当前 text 主干里的 `nn.Linear/nn.Embedding` 为 repo 内并行层，并处理 gated q projection 的切分。
- 若要支持 OmniKV，需要重新定义 observation/full layer routing，使其只在 full-attention 段内传播，不能跨越 Gated DeltaNet 层假设 KV attention。
- 若要支持视觉输入，需要接入 vision tower、multimodal projector、mRoPE 四维 position ids 和视觉 token 拼接流程。
- 若要支持 prefix cache，需要在 prefix block 中保存/恢复 Gated DeltaNet 的 conv/recurrent state，或在 cache hit 后重放被跳过的 prefix。
