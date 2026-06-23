from __future__ import annotations

SUPPORTED_PRUNING_METHODS = {
    "divprune",
    "divprune_official",
    "fastv",
    "fastvid_official_repo",
    "pact_official_repo",
    "visionzip",
}


def _visual_cache():
    from benchmark.multimodal.visual_cache import run_visual_cache

    return run_visual_cache


def batch_to_device(*args, **kwargs):
    return _visual_cache().batch_to_device(*args, **kwargs)


def build_llava_deltakv_policy(*args, **kwargs):
    return _visual_cache().build_llava_deltakv_policy(*args, **kwargs)


def build_visual_uniform_policy(*args, **kwargs):
    return _visual_cache().build_visual_uniform_policy(*args, **kwargs)


def ensure_left_padding(*args, **kwargs):
    return _visual_cache().ensure_left_padding(*args, **kwargs)


def load_llava_deltakv_model(*args, **kwargs):
    return _visual_cache().load_llava_deltakv_model(*args, **kwargs)


def load_vanilla_model(*args, **kwargs):
    return _visual_cache().load_vanilla_model(*args, **kwargs)


def load_visual_uniform_model(*args, **kwargs):
    return _visual_cache().load_visual_uniform_model(*args, **kwargs)


def iter_requested_methods(methods: str, *, allow_fastvid: bool = True):
    supported = {"vanilla", "deltakv", "snapkv", "omnikv", *SUPPORTED_PRUNING_METHODS}
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method == "vanilla":
            yield raw_method, "vanilla"
        elif method in {"snapkv", "llava_snapkv"}:
            yield raw_method, "snapkv"
        elif method in {"omnikv", "llava_omnikv"}:
            yield raw_method, "omnikv"
        elif method in {"deltakv", "llava_deltakv"}:
            yield raw_method, "deltakv"
        elif method in SUPPORTED_PRUNING_METHODS:
            if method == "fastvid_official_repo" and not allow_fastvid:
                raise ValueError("LLaVA-OV fastvid_official_repo is video-only; use divprune or fastv for image QA.")
            yield raw_method, method
        elif method in {"fastvid", "fastvid_official"}:
            raise ValueError(
                "The local LLaVA-OV FastVID HF ports were removed. Use method='fastvid_official_repo' "
                "to run the FastVID repository implementation."
            )
        else:
            raise ValueError(f"LLaVA-OV adapter supports {sorted(supported)}. Unsupported method={raw_method!r}.")


def load_model_for_method(method_kind: str, args, dtype, device):
    if method_kind == "vanilla":
        vc = _visual_cache()
        return vc.load_vanilla_model(args, dtype, device), None, "vanilla"
    if method_kind == "deltakv":
        vc = _visual_cache()
        model, policy = vc.load_llava_deltakv_model(args, dtype, device)
        return model, policy, policy["method"]
    if method_kind == "snapkv":
        vc = _visual_cache()
        model, policy = vc.load_llava_snapkv_model(args, dtype, device)
        return model, policy, policy["method"]
    if method_kind == "omnikv":
        vc = _visual_cache()
        model, policy = vc.load_llava_omnikv_model(args, dtype, device)
        return model, policy, policy["method"]
    if method_kind in {"divprune", "divprune_official"}:
        vc = _visual_cache()
        model = vc.load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.llava_onevision_pruning import (
            LlavaOneVisionPruningConfig,
            apply_llava_onevision_prefill_pruning,
        )

        policy = apply_llava_onevision_prefill_pruning(
            model,
            LlavaOneVisionPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    if method_kind == "visionzip":
        vc = _visual_cache()
        model = vc.load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.llava_onevision_pruning import (
            LlavaOneVisionPruningConfig,
            apply_llava_onevision_visionzip,
        )

        policy = apply_llava_onevision_visionzip(
            model,
            LlavaOneVisionPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    if method_kind == "fastvid_official_repo":
        from benchmark.multimodal.model_adapters.fastvid_official_repo import load_fastvid_official_repo_model

        model, policy = load_fastvid_official_repo_model(args, device)
        return model, policy, policy["method"]
    if method_kind == "pact_official_repo":
        from benchmark.multimodal.model_adapters.pact_official_repo import load_pact_official_repo_model

        model, policy = load_pact_official_repo_model(args, device)
        return model, policy, policy["method"]
    if method_kind == "fastv":
        vc = _visual_cache()
        model = vc.load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.llava_onevision_pruning import (
            LlavaOneVisionPruningConfig,
            apply_llava_onevision_fastv,
        )

        policy = apply_llava_onevision_fastv(
            model,
            LlavaOneVisionPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    raise AssertionError(f"Unhandled LLaVA-OV method kind: {method_kind}")


__all__ = [
    "batch_to_device",
    "build_llava_deltakv_policy",
    "build_visual_uniform_policy",
    "ensure_left_padding",
    "iter_requested_methods",
    "load_llava_deltakv_model",
    "load_model_for_method",
    "load_vanilla_model",
    "load_visual_uniform_model",
]
