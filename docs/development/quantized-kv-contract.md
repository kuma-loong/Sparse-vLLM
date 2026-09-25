# Quantized KV cache contract

The `kivi`, `turboquant`, and `fp8_kv` methods retain every token. They share
page allocation and expose compressed storage to a prepared decode provider.
They do not use sparse attention scores, token selection, or a compressor
checkpoint. Logical orchestration uses `PassThroughRuntime`.

| Axis | Initial contract |
| --- | --- |
| Identity | `kivi`, `turboquant`, `fp8_kv`; FP8 requires a checkpoint-bound scale file |
| Representation | KIVI asymmetric min/max int2/int4; TurboQuant MSE random rotation and Gaussian Lloyd-Max codebook, int2/int3/int4; calibrated per-layer K/V E4M3 FP8 |
| Persistent state | Cache-manager-owned pages; KIVI/TurboQuant scales and high-precision incomplete pages; FP8 data pages and two scales per layer |
| Selection and scores | All logical tokens; no score output |
| Model/layout | Homogeneous explicit KV, Llama/Qwen2/Qwen3, FP16/BF16 activations, head dimension 64/128/256 |
| Prefill | KIVI/TurboQuant materialize historical compressed KV; FP8 writes directly into physical pages and uses FlashInfer FA2 paged attention on SM90 |
| Decode | KIVI/TurboQuant read packed pages plus raw tail; FP8 FlashInfer reads FP8 pages with static K/V scales |
| Admission | Page-rounded costs, method-specific raw tails, row metadata, and provider workspaces included in allocation budget |
| Lifecycle | Append and free; no prefix attach/fork, cache offload, or rollback API advertised |
| Graph/topology | Shared eager/decode-CUDA-Graph updates; model-validated TP and Qwen3-MoE EP/outer-TP; no quantization collectives; internal DP remains model-rejected |
| Validation | Independent numerical codec/attention oracles, allocation failure atomicity and reclamation, configuration/provider rejection, actual GPU generation when an idle device is available |

KIVI quantizes K over tokens within a page and V over channel groups within a
token. Both full pages are quantized together; the incomplete page stays in
the activation dtype. This intentionally differs from the official residual
window lifecycle and does not retain an additional sink window.

TurboQuant implements the MSE variant with a seeded orthogonal rotation and
a Gaussian approximation to the rotated-coordinate distribution. It does not
implement QJL residual correction or mixed outlier bit budgets. Integer codes
are packed into 32-bit words without straddling word boundaries; padding is
included in byte accounting, particularly for int3.

FP8 uses separate fixed K/V scales per layer, measured offline at the dense
cache write boundary. Calibration records maxima across participating ranks;
the exported dequant multipliers are `safety_margin * max_abs / 448`.
Every token is quantized at its final physical slot, including incomplete
pages. FlashInfer FA2 paged prefill and paged decode consume those pages and
apply scales inside attention. The checkpoint name and canonical config hash
bind the file at startup. The current SM90 FA3 mixed BF16-query/FP8-KV prefill
JIT does not compile, so the FP8 path uses FA2. No silent dense fallback is
selected.

KIVI/TurboQuant's mixed packed-page/raw-tail payload requires a repository
provider. FP8 uses upstream FlashInfer with a repository-owned writer and
page-table adapter. There is no runtime provider reselection.

## Efficiency status

Standalone TurboQuant also has a known unresolved decode-efficiency issue.
Numerical and CUDA Graph lifecycle validation do not establish performance
readiness; enabling CUDA Graph alone does not resolve the observed latency
problem. Treat its current decode implementation as experimental, not as an
end-to-end acceleration claim or evidence about official TurboQuant kernels.

Standalone KIVI has a known unresolved decode-efficiency issue. Bounding live
dequantization tiles removes severe register spilling in validated shapes,
but does not establish acceptable serving latency. Keep functional/graph
support separate from performance readiness. Before further KIVI tuning,
assess adapting DeltaKV's existing grouped-Q-head matrix-multiply decode;
the two paths have different packed layouts and raw-tail metadata, so they
are not interchangeable launchers. This issue is not a support predicate and
must not silently route compressed requests to dense attention.

## Decode graph contract

CPU admission reserves pages and publishes row/length/write-slot metadata
outside capture. KIVI/TurboQuant append a raw token or encode a newly
completed page. FP8 writes each token directly into its physical FP8 slot.
Negative write slots suppress padded writes even when padded read rows alias
a live request. TurboQuant rotation is unchanged. Prefill still executes eagerly.

KIVI/TurboQuant decode prepares a capacity-bounded workspace and fixed split
grid for graph replay. FP8 uses the FlashInfer paged decode wrapper with
caller-owned graph buffers and replans page indices and lengths before replay.
Cache storage and provider workspaces remain alive with their owners; request
release reclaims pages without replacing captured tensors.

For KIVI/TurboQuant, append work is O(batch * KV_heads * head_dim) for raw
tails and O(completed_pages * page_size * KV_heads * head_dim) for page encoding.
FP8 writes each new token once and has no page-completion rewrite or BF16 tail.
When all visible tokens are in the current prefill chunk, the cache manager
certifies their BF16 K/V for SGL FA3 varlen attention while still persisting
FP8 pages. Later chunks consume FP8 history through FlashInfer FA2.
TurboQuant retains its per-head rotation cost. FP8 calibration uses a collective
maximum across participating model ranks; normal FP8 inference adds none.

Reuse review: vLLM `reshape_and_cache_kernel` (v0.26.0,
`568afb3a13806beb53bb2e6bd518269357b237c0`,
`csrc/libtorch_stable/cache_kernels.cu`) provides the negative-slot write-mask
pattern, but not this mixed compressed-page/raw-tail lifecycle. SGL kernel 0.4.5's
`sgl_per_token_quant_fp8` and `sgl_per_token_group_quant_fp8` interfaces consume
dense token tensors, not page completion state or KIVI/TurboQuant metadata.
Neither directly substitutes for the KIVI/TurboQuant page-completion
transition. FP8 uses a direct Triton writer and upstream FlashInfer attention;
no upstream implementation is copied.

Validation must distinguish kernel replay/codec equivalence, real-model
generation and resource reclamation, and matched eager/graph throughput.
Single-GPU validation does not establish multi-rank replay correctness.
