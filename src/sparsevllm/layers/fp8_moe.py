from __future__ import annotations

import torch


FP8_BLOCK_SIZE = 128


def require_fp8_moe_alignment(
    *,
    model_name: str,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    if hidden_size % FP8_BLOCK_SIZE or intermediate_size % FP8_BLOCK_SIZE:
        raise ValueError(
            f"{model_name} packed FP8 experts require hidden/intermediate "
            f"dimensions aligned to {FP8_BLOCK_SIZE}, got "
            f"{hidden_size}/{intermediate_size}."
        )


def copy_fp8_expert_shard(
    *,
    model_name: str,
    expert_id: int,
    projection: str,
    loaded_weight: torch.Tensor,
    loaded_scale: torch.Tensor | None,
    weight_target: torch.Tensor,
    scale_target: torch.Tensor,
    expected_scale_dtype: torch.dtype = torch.float32,
    expected_scale_dtype_name: str = "FP32",
) -> None:
    if loaded_scale is None:
        raise ValueError(
            f"Missing FP8 weight_scale_inv for {model_name} expert={expert_id}, "
            f"projection={projection}."
        )
    if loaded_weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"{model_name} expert weight must be FP8 E4M3, got "
            f"{loaded_weight.dtype}."
        )
    if loaded_scale.dtype != expected_scale_dtype:
        raise TypeError(
            f"{model_name} expert weight_scale_inv must be "
            f"{expected_scale_dtype_name}, got {loaded_scale.dtype}."
        )
    if tuple(loaded_weight.shape) != tuple(weight_target.shape):
        raise ValueError(
            f"{model_name} expert weight shape mismatch for expert={expert_id}, "
            f"projection={projection}: expected={tuple(weight_target.shape)}, "
            f"got={tuple(loaded_weight.shape)}."
        )
    if tuple(loaded_scale.shape) != tuple(scale_target.shape):
        raise ValueError(
            f"{model_name} expert scale shape mismatch for expert={expert_id}, "
            f"projection={projection}: expected={tuple(scale_target.shape)}, "
            f"got={tuple(loaded_scale.shape)}."
        )
    weight_target.copy_(loaded_weight)
    scale_target.copy_(loaded_scale.to(dtype=scale_target.dtype))


def flashinfer_fp8_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale_inv: torch.Tensor,
    w2_scale_inv: torch.Tensor,
    *,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.tllm_enums import ActivationType

    output = torch.empty_like(hidden_states)
    cutlass_fused_moe(
        hidden_states,
        topk_ids.to(dtype=torch.int32),
        topk_weights.to(dtype=torch.float32),
        w13_weight,
        w2_weight,
        hidden_states.dtype,
        quant_scales=[w13_scale_inv, w2_scale_inv],
        ep_size=int(ep_size),
        ep_rank=int(ep_rank),
        output=output,
        use_deepseek_fp8_block_scale=True,
        use_fused_finalize=False,
        enable_pdl=False,
        activation_type=ActivationType.Swiglu,
    )
    return output
