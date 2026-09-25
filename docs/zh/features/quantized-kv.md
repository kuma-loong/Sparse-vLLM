# KV cache 量化压缩

`kivi`、`turboquant`、`fp8_kv` 保留所有 token，只压缩 KV 表示，不量化模型权重。

独立 `kivi` 和 `turboquant` 方法目前均存在已知的 decode 效率问题：功能和
CUDA Graph 验证通过，不代表延迟已达到可用水平。请将两者视为实验性实现；
用于延迟敏感场景前，务必实测目标工作负载。KIVI 的寄存器溢出修复尚未解决
整体效率问题；仅启用 CUDA Graph 也未解决 TurboQuant 的性能问题。

```python
from sparseengine import LLM

llm = LLM(
    model_path,
    sparse_method="fp8_kv",  # 也可选 "kivi" 或 "turboquant"
    decode_graph=True,  # False 以 eager 执行同一量化 decode 计算路径
    enable_prefix_caching=False,
    tensor_parallel_size=1,
    max_model_len=4096,
    max_num_seqs_in_batch=4,
    kv_quant_page_size=32,
    fp8_kv_scale_path="/path/to/calibrated-kv-scales.json",
)
```

| 方法 | 参数 | 表示 |
| --- | --- | --- |
| `kivi` | `kivi_bits=2` 或 `4`（默认） | K 按通道、V 按 token 内通道组做非对称整数量化 |
| `turboquant` | `turboquant_bits=2/3/4`（默认 4）；`turboquant_seed=0` | 固定种子正交旋转及非均匀标量量化 |
| `fp8_kv` | `fp8_kv_scale_path` | E4M3，K/V 分别使用逐层固定 scale |

支持 CUDA、FP16/BF16 的 Llama/Qwen2/Qwen3/Qwen3-MoE、head dimension 64/128/256，
以及模型合法的 TP head 分片。Qwen3-MoE 也支持 EP 和现有 outer-TP/EP 布局。
量化只处理本 rank 的 KV heads，不新增通信；模型已有的 TP/EP 通信不变。
独立推理副本各自维护 KV cache。本 KV 量化路径支持的模型要求
`data_parallel_size=1`；GLM DP attention 使用单独的 MLA latent cache。
FP8 需要 GPU 原生 FP8 支持和 FlashInfer 分页 attention；当前 FP8 prefill
提供者支持 SM90。启动时必须提供与权重匹配的离线 scale 文件；引擎会校验
权重目录名、模型配置哈希、层集合和 scale 数值，不匹配则报错。页大小必须是 16～128 的 2 的幂，
且能整除 head dimension。支持 decode CUDA Graph；prefix cache/offload 和 sparse prefill
组合会明确报错。[Palu](palu.md) 使用独立的低秩缓存路径。

`fp8_kv` 在写入时直接将所有 token 存为 FP8，包括未满页。存在历史 KV 的 prefill
及 decode 由 FlashInfer 直接读取 FP8 页；首个 prefill 块可以直接用仍在计算中的
BF16 K/V 交给 SGL FA3，同时写入 FP8 cache。
后续块通过 FlashInfer FA2 读取 FP8 历史；应按提示长度和显存预算调整
`engine_prefill_chunk_size`，因为这条路径可能主导长提示延迟。KIVI 和
TurboQuant 仍保留高精度未满页，并使用有界的 prefill 历史 KV 工作区。

先用有代表性的 tokenized 提示离线校准：

```bash
python scripts/calibrate_fp8_kv.py \
  --model /path/to/checkpoint \
  --input /path/to/prepared-samples.json \
  --output /path/to/calibrated-kv-scales.json \
  --max-model-len 65536
```

输入为包含 `samples` 数组的 JSON 或 JSONL，每条数据含 `prompt_token_ids`
或 `prompt`。脚本在 dense 模型的 KV 写入边界测量 K/V，每层分别输出两个 scale。
可用 `--tensor-parallel-size`、`--expert-parallel-size` 指定校准拓扑。
验证精度时，应将校准和评测样本分开。用户指定的文件优先于内置预设。
目前内置 `Qwen3-30B-A3B-Instruct-2507-FP8` 的预设；其他权重需显式提供 scale 文件。
该预设由 45 条 LongBench v2 short 提示校准，与 12 条精度评测提示不重合；JSON
中记录了数据集和输入文件的哈希。
目录名和配置哈希只能校验布局，不能验证权重文件内容；修改过的权重应重新校准。

这些是服务引擎适配版，不宣称与官方完全一致：KIVI 只保留未满页，不额外配置独立
residual window；TurboQuant 使用 MSE 风格高斯码本，不包含 QJL 残差修正；FP8
使用离线校准的固定 scale。整数打包对齐、KIVI/TurboQuant 的 scale 元数据与
未满页、算子工作区会降低实际压缩收益。FP8 存储不保证端到端加速，需在目标
工作负载上验证精度和性能。
