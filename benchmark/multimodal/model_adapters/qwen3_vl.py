from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


SUPPORTED_METHODS = {"vanilla", "deltakv", "divprune", "divprune_official", "fastv", "fastvid"}

CUSTOM_CONFIG_KEYS = {
    "kv_compressed_size",
    "seq_chunk_size",
    "k_neighbors",
    "compressor_token_group_size",
    "deltakv_neighbor_count",
    "layer_chunk_size",
    "recon_mode",
    "ref_mode",
    "use_nonlinear_compressor",
    "compressor_intermediate_size",
    "compressor_down_type",
    "compressor_up_type",
    "compressor_down_intermediate_size",
    "compressor_up_intermediate_size",
    "collect_kv_before_rope",
    "compressor_linear_bias",
    "split_kv",
    "cluster_metric",
    "cluster_on_kv",
    "cluster_ratio",
    "stride_alpha",
    "cluster_temp",
    "cluster_soft_assignment",
    "tail_token_size",
    "num_recent_tokens",
    "full_attn_layers",
    "num_top_tokens",
    "num_top_tokens_in_prefill",
    "num_sink_tokens",
    "omnikv_score_method",
    "deltakv_use_omnikv_selection",
    "use_compression",
    "use_cluster",
    "chunk_prefill_size",
    "snapkv_window_size",
    "pool_kernel_size",
    "chunk_prefill_accel_omnikv",
    "deltakv_cache_impl",
    "hf_sparse_cache_impl",
    "kv_quant_bits",
    "kv_quant_group_size",
    "full_layer_kv_quant_bits",
    "full_layer_cluster_ratio",
    "full_layer_stride_alpha",
    "full_layer_kivi_group_size",
    "full_layer_kivi_residual_length",
    "enable_full_layer_kivi_quant",
    "enable_sparse_ref_fp8",
}


def require_qwen3_vl_transformers():
    try:
        from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-VL evaluation requires a Transformers build with "
            "Qwen3VLForConditionalGeneration. Install a recent/source Transformers "
            "in the evaluation environment before running --model_family qwen3_vl."
        ) from exc
    return AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration


def ensure_left_padding(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def iter_requested_methods(methods: str):
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method in SUPPORTED_METHODS:
            yield raw_method, method
            continue
        raise ValueError(f"Qwen3-VL adapter supports only {sorted(SUPPORTED_METHODS)}. Unsupported method={raw_method!r}.")


@dataclass
class Qwen3VLRuntime:
    model: Any
    processor: Any
    method: str = "vanilla"

    @property
    def supports_batch_generation(self) -> bool:
        return False


def load_processor(model_path: str):
    AutoProcessor, _, _ = require_qwen3_vl_transformers()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    ensure_left_padding(processor)
    return processor


def load_vanilla_model(args: Any, dtype: torch.dtype, device: torch.device):
    _, _, model_cls = require_qwen3_vl_transformers()
    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    return model, None, "vanilla"


def resolve_compressor_path(args: Any) -> Path | None:
    checkpoint_path = str(args.deltakv_checkpoint_path)
    return Path(checkpoint_path) if checkpoint_path.lower() not in {"", "none", "null"} else None


def migrate_checkpoint_infer_config(infer_config: dict) -> dict:
    migrated = dict(infer_config)
    if "seq_chunk_size" in migrated:
        value = migrated.pop("seq_chunk_size")
        if "compressor_token_group_size" in migrated and migrated["compressor_token_group_size"] != value:
            raise ValueError("Checkpoint config has conflicting seq_chunk_size/compressor_token_group_size.")
        migrated["compressor_token_group_size"] = value
    if "k_neighbors" in migrated:
        value = migrated.pop("k_neighbors")
        if "deltakv_neighbor_count" in migrated and migrated["deltakv_neighbor_count"] != value:
            raise ValueError("Checkpoint config has conflicting k_neighbors/deltakv_neighbor_count.")
        migrated["deltakv_neighbor_count"] = value
    return migrated


def parse_full_layers_for_budget(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(part) for part in value]


def estimate_qwen3vl_deltakv_budget(infer_config: dict) -> dict | None:
    try:
        num_layers = 36
        full_layers = parse_full_layers_for_budget(
            infer_config.get("full_attn_layers", infer_config.get("full_attention_layers", ""))
        )
        full_layer_ratio = len(full_layers) / num_layers
        sparse_layer_ratio = 1.0 - full_layer_ratio
        center_ratio = float(infer_config.get("cluster_ratio", infer_config.get("deltakv_center_ratio", 0.0)) or 0.0)
        latent_dim = int(infer_config.get("kv_compressed_size", infer_config.get("deltakv_latent_dim", 0)) or 0)
        quant_bits = int(infer_config.get("kv_quant_bits", infer_config.get("deltakv_latent_quant_bits", 0)) or 0)
        if latent_dim <= 0:
            return None
        raw_kv_dim = 2048  # 2 * 8 KV heads * 128 head_dim for Qwen3-VL-8B text.
        cache_impl = str(infer_config.get("deltakv_cache_impl", "") or "")
        full_layer_quant_bits = int(infer_config.get("full_layer_kv_quant_bits", 0) or 0)
        full_layer_bit_ratio = (
            (full_layer_quant_bits / 16.0)
            if cache_impl == "delta_compressed_quant_kivi_full_fp8_ref" and full_layer_quant_bits in (2, 4)
            else 1.0
        )
        ref_bit_ratio = (
            0.5
            if cache_impl == "delta_compressed_quant_kivi_full_fp8_ref"
            and bool(infer_config.get("enable_sparse_ref_fp8", True))
            else 1.0
        )
        bit_ratio = (quant_bits / 16.0) if quant_bits in (2, 4) else 1.0
        compressed_payload_ratio = (latent_dim / raw_kv_dim) * bit_ratio
        kr_percent = 100.0 * (
            full_layer_ratio * full_layer_bit_ratio
            + sparse_layer_ratio * (center_ratio * ref_bit_ratio + compressed_payload_ratio)
        )
        return {
            "formula": "KR = full_layer_ratio*full_layer_bit_ratio + sparse_layer_ratio*(center_ratio*ref_bit_ratio + latent_dim/raw_kv_dim*quant_bits/16)",
            "num_layers": num_layers,
            "full_attention_layers": full_layers,
            "cache_impl": cache_impl,
            "center_ratio": center_ratio,
            "ref_bit_ratio": ref_bit_ratio,
            "full_layer_bit_ratio": full_layer_bit_ratio,
            "latent_dim": latent_dim,
            "raw_kv_dim": raw_kv_dim,
            "kv_quant_bits": quant_bits,
            "compressed_payload_ratio": compressed_payload_ratio,
            "estimated_kr_percent": kr_percent,
        }
    except Exception as exc:
        return {"estimate_error": str(exc)}


def build_qwen3vl_deltakv_policy(infer_config: dict) -> dict:
    uses_cluster = bool(infer_config.get("use_cluster", False))
    uses_learned_compressor = bool(infer_config.get("use_compression", False))
    kv_quant_bits = int(infer_config.get("deltakv_latent_quant_bits", infer_config.get("kv_quant_bits", 0)) or 0)
    policy = {
        "method": "qwen3vl_deltakv",
        "selection_policy": "standard_deltakv_cache",
        "uses_deltakv_wrapper": True,
        "uses_learned_compressor": uses_learned_compressor,
        "uses_cluster": uses_cluster,
        "uses_ref_tokens": uses_cluster,
        "uses_visual_uniform_pruning": False,
        "supports_batch_generation": False,
        "kv_quant_bits": kv_quant_bits,
        "note": (
            "Uses a Qwen3-VL HF DeltaKV wrapper with the same sparse cache/compressor "
            "path as text inference while preserving Qwen3-VL mRoPE and DeepStack."
        ),
    }
    estimate = estimate_qwen3vl_deltakv_budget(infer_config)
    if estimate is not None:
        policy["budget_estimate"] = estimate
    return policy


def load_deltakv_model(args: Any, dtype: torch.dtype, device: torch.device):
    _, config_cls, _ = require_qwen3_vl_transformers()
    from deltakv.modeling.qwen3vl_inference import (
        Qwen3VLDeltaKVForConditionalGeneration,
        load_deltakv_compressor_into_qwen3vl,
    )

    config = config_cls.from_pretrained(args.model_path, trust_remote_code=True)
    compressor_path = resolve_compressor_path(args)
    if compressor_path is None:
        raise ValueError("qwen3_vl deltakv requires a real --deltakv_checkpoint_path, not 'none'.")
    compressor_config = json.loads((compressor_path / "config.json").read_text())
    if isinstance(compressor_config.get("deltakv_infer_config"), dict):
        infer_config = migrate_checkpoint_infer_config(compressor_config["deltakv_infer_config"])
        infer_config_is_native = bool(compressor_config.get("deltakv_infer_config_is_native", False))
    else:
        infer_config = migrate_checkpoint_infer_config(
            {key: compressor_config[key] for key in CUSTOM_CONFIG_KEYS if key in compressor_config}
        )
        infer_config_is_native = True

    if infer_config_is_native:
        runtime_overrides = {
            "visual_token_prune_only": False,
            "full_attn_layers": args.full_attention_layers,
            "num_recent_tokens": args.recent_keep_tokens,
            "num_sink_tokens": args.sink_keep_tokens,
            "num_top_tokens": args.decode_keep_tokens,
            "num_top_tokens_in_prefill": args.prefill_keep_tokens,
            "cluster_ratio": args.deltakv_center_ratio,
            "deltakv_neighbor_count": args.deltakv_neighbor_count,
            "chunk_prefill_size": args.hf_prefill_chunk_size,
            "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
        }
        if str(getattr(args, "deltakv_cache_impl", "") or "").strip():
            runtime_overrides["deltakv_cache_impl"] = str(args.deltakv_cache_impl).strip()
        latent_quant_bits = int(getattr(args, "deltakv_latent_quant_bits", -1))
        if latent_quant_bits >= 0:
            runtime_overrides["kv_quant_bits"] = latent_quant_bits
            runtime_overrides["kv_quant_group_size"] = int(getattr(args, "deltakv_latent_quant_group_size", 0))
        full_layer_quant_bits = int(getattr(args, "full_layer_kv_quant_bits", -1))
        if full_layer_quant_bits >= 0:
            runtime_overrides["full_layer_kv_quant_bits"] = full_layer_quant_bits
            runtime_overrides["full_layer_kivi_group_size"] = int(getattr(args, "full_layer_kivi_group_size", 32))
            runtime_overrides["full_layer_kivi_residual_length"] = int(
                getattr(args, "full_layer_kivi_residual_length", 32)
            )
        if getattr(args, "enable_sparse_ref_fp8", None) is not None:
            runtime_overrides["enable_sparse_ref_fp8"] = bool(args.enable_sparse_ref_fp8)
    else:
        runtime_overrides = {
            "visual_token_prune_only": False,
            "full_attention_layers": args.full_attention_layers,
            "recent_keep_tokens": args.recent_keep_tokens,
            "sink_keep_tokens": args.sink_keep_tokens,
            "decode_keep_tokens": args.decode_keep_tokens,
            "prefill_keep_tokens": args.prefill_keep_tokens,
            "deltakv_center_ratio": args.deltakv_center_ratio,
            "deltakv_neighbor_count": args.deltakv_neighbor_count,
            "hf_prefill_chunk_size": args.hf_prefill_chunk_size,
            "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
        }
        if str(getattr(args, "deltakv_cache_impl", "") or "").strip():
            runtime_overrides["deltakv_cache_impl"] = str(args.deltakv_cache_impl).strip()
        latent_quant_bits = int(getattr(args, "deltakv_latent_quant_bits", -1))
        if latent_quant_bits >= 0:
            runtime_overrides["deltakv_latent_quant_bits"] = latent_quant_bits
            runtime_overrides["deltakv_latent_quant_group_size"] = int(getattr(args, "deltakv_latent_quant_group_size", 0))
        full_layer_quant_bits = int(getattr(args, "full_layer_kv_quant_bits", -1))
        if full_layer_quant_bits >= 0:
            runtime_overrides["full_layer_kv_quant_bits"] = full_layer_quant_bits
            runtime_overrides["full_layer_kivi_group_size"] = int(getattr(args, "full_layer_kivi_group_size", 32))
            runtime_overrides["full_layer_kivi_residual_length"] = int(
                getattr(args, "full_layer_kivi_residual_length", 32)
            )
        if getattr(args, "enable_sparse_ref_fp8", None) is not None:
            runtime_overrides["enable_sparse_ref_fp8"] = bool(args.enable_sparse_ref_fp8)
    infer_config.update(runtime_overrides)
    policy = build_qwen3vl_deltakv_policy(infer_config)
    print(
        "[qwen3vl_cache_policy] "
        f"method={policy['method']} selection={policy['selection_policy']} "
        f"cluster={policy['uses_cluster']} compressor={policy['uses_learned_compressor']} "
        f"ref_tokens={policy['uses_ref_tokens']} kv_quant_bits={policy['kv_quant_bits']}",
        flush=True,
    )
    config.deltakv_infer_config = infer_config
    config.deltakv_infer_config_is_native = infer_config_is_native
    model = Qwen3VLDeltaKVForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    incompatible = load_deltakv_compressor_into_qwen3vl(model, str(compressor_path), device="cpu")
    compressor_missing = [key for key in incompatible.missing_keys if "compress_" in key]
    if compressor_missing:
        raise RuntimeError(f"Qwen3-VL DeltaKV compressor weights were not fully loaded; missing examples: {compressor_missing[:8]}")
    return model, policy, "deltakv"


def load_model_for_method(method_kind: str, args: Any, dtype: torch.dtype, device: torch.device):
    if method_kind == "vanilla":
        return load_vanilla_model(args, dtype, device)
    if method_kind == "deltakv":
        return load_deltakv_model(args, dtype, device)
    if method_kind in {"divprune", "divprune_official", "fastvid"}:
        model, _, _ = load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.qwen3_vl_pruning import (
            Qwen3VLPruningConfig,
            apply_qwen3_vl_prefill_pruning,
        )

        policy = apply_qwen3_vl_prefill_pruning(
            model,
            Qwen3VLPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    if method_kind == "fastv":
        model, _, _ = load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.qwen3_vl_pruning import (
            Qwen3VLPruningConfig,
            apply_qwen3_vl_fastv,
        )

        policy = apply_qwen3_vl_fastv(
            model,
            Qwen3VLPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    raise AssertionError(f"Unhandled Qwen3-VL method kind: {method_kind}")


def _message_content(kind: str, media: Any, text: str) -> list[dict[str, Any]]:
    if kind == "image":
        media_items = media if isinstance(media, list) else [media]
        content = [{"type": "image", "image": item} for item in media_items]
    elif kind == "video":
        content = [{"type": "video", "video": media}]
    else:
        raise ValueError(f"Unsupported Qwen3-VL media kind: {kind}")
    content.append({"type": "text", "text": text})
    return content


def prepare_inputs(
    processor: Any,
    *,
    text: str,
    media_kind: str,
    media: Any,
    device: torch.device | None,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], int, int]:
    messages = [{"role": "user", "content": _message_content(media_kind, media, text)}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if device is None:
        input_len = int(inputs["input_ids"].shape[1])
        visual_tokens = infer_visual_token_count(inputs)
        return inputs, input_len, visual_tokens
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
    input_len = int(inputs["input_ids"].shape[1])
    visual_tokens = infer_visual_token_count(inputs)
    return inputs, input_len, visual_tokens


def infer_visual_token_count(inputs: dict[str, Any]) -> int:
    if "image_grid_thw" in inputs and torch.is_tensor(inputs["image_grid_thw"]):
        grid = inputs["image_grid_thw"]
        return int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
    if "video_grid_thw" in inputs and torch.is_tensor(inputs["video_grid_thw"]):
        grid = inputs["video_grid_thw"]
        return int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
    return 0


def decode_generated(processor: Any, output_ids: torch.Tensor, input_ids: torch.Tensor) -> list[str]:
    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
