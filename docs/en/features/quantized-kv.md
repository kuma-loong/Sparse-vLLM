# Quantized KV cache

`kivi`, `turboquant`, and `fp8_kv` compress KV representations without dropping
tokens. They do not quantize model weights.

The standalone `kivi` and `turboquant` methods currently have known
decode-efficiency problems:
functional and CUDA Graph validation does not imply acceptable latency.
Treat both as experimental and do not select them for latency-sensitive
workloads without measuring your workload. The KIVI register-spill repair
does not resolve its overall efficiency issue; CUDA Graph support alone does
not resolve TurboQuant's performance issue either.

```python
from sparseengine import LLM

llm = LLM(
    model_path,
    sparse_method="fp8_kv",  # alternatively "kivi" or "turboquant"
    decode_graph=True,  # False uses the same quantized decode computation eagerly
    enable_prefix_caching=False,
    tensor_parallel_size=1,
    max_model_len=4096,
    max_num_seqs_in_batch=4,
    kv_quant_page_size=32,
    fp8_kv_scale_path="/path/to/calibrated-kv-scales.json",
)
```

| Method | Controls | Representation |
| --- | --- | --- |
| `kivi` | `kivi_bits=2` or `4` (default) | Asymmetric integer quantization: per-channel K, grouped per-token V |
| `turboquant` | `turboquant_bits=2`, `3`, or `4` (default); `turboquant_seed=0` | Seeded orthogonal rotation and nonuniform scalar quantization |
| `fp8_kv` | `fp8_kv_scale_path` | E4M3 with separate, fixed per-layer K and V scales |

The implementation supports CUDA, FP16/BF16 Llama/Qwen2/Qwen3/Qwen3-MoE models,
and head dimensions 64/128/256. TP is supported with the model's legal head
sharding; Qwen3-MoE also supports EP and its existing outer-TP/EP layout.
Quantization uses rank-local KV heads and adds no communication. Model TP/EP
collectives are unchanged. Independent inference replicas keep separate KV
caches. The models supported by this KV quantization path require
`data_parallel_size=1`; GLM DP attention uses its separate MLA latent cache.
FP8 requires native FP8 support and the FlashInfer paged attention provider;
the current FP8 prefill provider supports SM90. Supply a scale file calibrated
for the loaded weights. Startup checks the checkpoint directory name, model
config hash, layer set, and scale values and fails if no matching file is found.
`kv_quant_page_size` must be a power of two from 16 to 128 and divide
the head dimension. Decode CUDA Graphs are supported. Prefix caching/offload and sparse prefill
combinations are rejected. [Palu](palu.md) uses a separate low-rank cache path.

For `fp8_kv`, every written token, including an incomplete page, is stored in
FP8. FlashInfer prefill reads FP8 history and decode reads FP8 pages. A first
prefill chunk can use its still-live BF16 K/V with SGL FA3 while writing the
FP8 cache. Later chunks use FlashInfer FA2 to read FP8 history; tune
`engine_prefill_chunk_size` for the prompt lengths and memory budget because
this path can dominate long-prompt latency. KIVI and TurboQuant retain their existing
high-precision incomplete page and bounded dense prefill workspace.

Calibrate from representative tokenized prompts before serving:

```bash
python scripts/calibrate_fp8_kv.py \
  --model /path/to/checkpoint \
  --input /path/to/prepared-samples.json \
  --output /path/to/calibrated-kv-scales.json \
  --max-model-len 65536
```

The input is JSON with a `samples` array or JSONL; each item contains
`prompt_token_ids` or `prompt`. The script measures K/V at the cache write
boundary on the dense model path and writes one K and one V scale per layer.
Use `--tensor-parallel-size` and `--expert-parallel-size` for the calibration
topology. Keep calibration and evaluation samples separate when measuring
quality. A user-provided file takes precedence over any packaged preset. A
preset is included for `Qwen3-30B-A3B-Instruct-2507-FP8`; other checkpoints
require an explicit scale file. This preset was measured on 45 LongBench v2
short prompts, separate from the 12 prompts used for quality checks; its JSON
records the dataset and input hashes. The name and config hash check layout but do
not verify weight-file contents, so recalibrate modified checkpoints.

These are serving adaptations, not exact official implementations. KIVI has
one incomplete-page residual rather than an independently configured residual
window. TurboQuant uses the MSE-style Gaussian-codebook recipe, without QJL
residual correction. FP8 uses offline calibrated scales. Integer packing
padding, KIVI/TurboQuant scale metadata and raw tails, and provider workspaces
reduce effective memory savings. FP8 storage does not guarantee an end-to-end
speedup. Validate quality and performance on your workload.
