from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn
from transformers import AutoConfig


ModelFactory = Callable[[object], nn.Module]


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    model_types: tuple[str, ...]
    create_model: ModelFactory
    supported_sparse_methods: frozenset[str]
    weight_prefixes_to_strip: tuple[str, ...] = ()
    ignored_weight_prefixes: tuple[str, ...] = ()
    tensor_parallel_size: int | None = None

    def normalize_config(self, hf_config):
        return hf_config

    def attention_layer_indices(self, hf_config) -> tuple[int, ...]:
        num_layers = int(getattr(hf_config, "num_hidden_layers"))
        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types is None:
            return tuple(range(num_layers))
        return tuple(
            idx
            for idx, layer_type in enumerate(layer_types)
            if str(layer_type) == "full_attention"
        )

    def map_weight_name(self, name: str) -> str | None:
        for prefix in self.ignored_weight_prefixes:
            if name.startswith(prefix):
                return None
        for prefix in self.weight_prefixes_to_strip:
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def validate_engine_config(self, config) -> None:
        if self.tensor_parallel_size is not None and int(config.tensor_parallel_size) != self.tensor_parallel_size:
            raise ValueError(
                f"{self.name} currently supports tensor_parallel_size={self.tensor_parallel_size} only; "
                f"got tensor_parallel_size={config.tensor_parallel_size}."
            )
        if self.supported_sparse_methods and config.vllm_sparse_method not in self.supported_sparse_methods:
            supported = ", ".join(
                repr(method or "vanilla") for method in sorted(self.supported_sparse_methods)
            )
            raise ValueError(
                f"Sparse-vLLM {self.name} supports sparse methods {supported}; "
                f"got vllm_sparse_method={config.vllm_sparse_method!r}."
            )


class Qwen35Adapter(ModelAdapter):
    def normalize_config(self, hf_config):
        source_model_type = getattr(hf_config, "model_type", "qwen3_5")
        if source_model_type == "qwen3_5_text":
            text_config = hf_config
        else:
            text_config = getattr(hf_config, "text_config", None)
        if text_config is None:
            raise ValueError("Qwen3.5 adapter requires a top-level config with text_config.")
        if getattr(text_config, "model_type", None) != "qwen3_5_text":
            raise ValueError(
                "Qwen3.5 adapter expected text_config.model_type='qwen3_5_text', "
                f"got {getattr(text_config, 'model_type', None)!r}."
            )
        layer_types = list(getattr(text_config, "layer_types", []) or [])
        if len(layer_types) != int(getattr(text_config, "num_hidden_layers")):
            raise ValueError(
                "Qwen3.5 text_config.layer_types must match num_hidden_layers: "
                f"len(layer_types)={len(layer_types)} num_hidden_layers={text_config.num_hidden_layers}."
            )
        unknown = sorted(set(layer_types).difference({"linear_attention", "full_attention"}))
        if unknown:
            raise ValueError(f"Unsupported Qwen3.5 layer_types: {unknown}.")

        text_config.sparsevllm_model_type = "qwen3_5"
        text_config.sparsevllm_source_model_type = source_model_type
        text_config.sparsevllm_architectures = tuple(getattr(hf_config, "architectures", ()) or ())
        text_config.sparsevllm_attention_layer_indices = self.attention_layer_indices(text_config)
        return text_config

    def validate_engine_config(self, config) -> None:
        super().validate_engine_config(config)
        if bool(getattr(config, "enable_prefix_caching", False)):
            raise ValueError(
                "Qwen3.5 prefix caching is not supported yet because linear-attention "
                "recurrent state is not stored/restored by the prefix cache."
            )
        if bool(getattr(config, "decode_cuda_graph", False)):
            raise ValueError(
                "Qwen3.5 decode_cuda_graph is not supported yet because decode uses "
                "per-sequence Python-owned linear-attention state."
            )

    def map_weight_name(self, name: str) -> str | None:
        for prefix in self.ignored_weight_prefixes:
            if name.startswith(prefix):
                return None
        language_prefix = "model.language_model."
        if name.startswith(language_prefix):
            return "model." + name[len(language_prefix) :]
        return name


def _create_qwen2(config):
    from sparsevllm.models.qwen2 import Qwen2ForCausalLM

    return Qwen2ForCausalLM(config)


def _create_qwen3(config):
    from sparsevllm.models.qwen3 import Qwen3ForCausalLM

    return Qwen3ForCausalLM(config)


def _create_qwen35(config):
    from sparsevllm.models.qwen3_5 import Qwen3_5ForCausalLM

    return Qwen3_5ForCausalLM(config)


_ADAPTERS: tuple[ModelAdapter, ...] = (
    ModelAdapter(
        name="qwen2",
        model_types=("qwen2",),
        create_model=_create_qwen2,
        supported_sparse_methods=frozenset(),
    ),
    ModelAdapter(
        name="qwen3",
        model_types=("qwen3",),
        create_model=_create_qwen3,
        supported_sparse_methods=frozenset(),
    ),
    Qwen35Adapter(
        name="qwen3_5",
        model_types=("qwen3_5", "qwen3_5_text"),
        create_model=_create_qwen35,
        supported_sparse_methods=frozenset({"", "snapkv"}),
        weight_prefixes_to_strip=("model.language_model.",),
        ignored_weight_prefixes=("model.visual.", "model.image_newline", "mtp."),
        tensor_parallel_size=1,
    ),
)


def _model_type(hf_config) -> str:
    return str(
        getattr(hf_config, "sparsevllm_model_type", None)
        or getattr(hf_config, "model_type", "")
        or ""
    )


def get_model_adapter(hf_config) -> ModelAdapter:
    model_type = _model_type(hf_config)
    for adapter in _ADAPTERS:
        if model_type in adapter.model_types:
            return adapter
    raise NotImplementedError(
        f"Unsupported Sparse-vLLM model_type={model_type!r}. "
        "Supported model types: qwen2, qwen3, qwen3_5."
    )


def load_and_normalize_hf_config(model_path: str):
    try:
        raw_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            "AutoConfig.from_pretrained failed. Refusing to silently fall back to raw "
            f"`config.json`. model={model_path} error={type(e).__name__}: {e}"
        ) from e

    raw_model_type = str(getattr(raw_config, "model_type", "") or "")
    for adapter in _ADAPTERS:
        if raw_model_type in adapter.model_types:
            return adapter.normalize_config(raw_config)
    return get_model_adapter(raw_config).normalize_config(raw_config)
