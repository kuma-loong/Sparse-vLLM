"""Validated, checkpoint-bound static scales for E4M3 KV cache storage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


SCHEME = "fp8_e4m3fn_per_layer"
CONVENTION = "dequant_multiplier"
_PRESET_DIR = Path(__file__).parent / "profiles" / "fp8_kv_scales"


@dataclass(frozen=True)
class FP8KVScales:
    key: tuple[float, ...]
    value: tuple[float, ...]
    source: str
    file_sha256: str


def model_config_sha256(model: str | Path) -> str:
    """Hash semantic checkpoint config, independent of JSON formatting."""
    path = Path(model) / "config.json"
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_scale_file(path: Path, *, model: str, layer_indices: tuple[int, ...]) -> FP8KVScales:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"Invalid FP8 KV scale schema in {path}.")
    if data.get("scheme") != SCHEME or data.get("scale_convention") != CONVENTION:
        raise ValueError(f"Unsupported FP8 KV scale scheme or convention in {path}.")
    expected_id = Path(model).name
    if data.get("checkpoint_id") != expected_id:
        raise ValueError(
            f"FP8 KV scales target checkpoint {data.get('checkpoint_id')!r}, "
            f"but the loaded checkpoint is {expected_id!r}."
        )
    expected_hash = model_config_sha256(model)
    if data.get("model_config_sha256") != expected_hash:
        raise ValueError(f"FP8 KV scales in {path} do not match the model config hash.")
    layers = data.get("layers")
    expected_layers = {str(index) for index in layer_indices}
    if not isinstance(layers, dict) or set(layers) != expected_layers:
        raise ValueError(
            f"FP8 KV scales in {path} must contain exactly the attention layers "
            f"{sorted(expected_layers, key=int)}."
        )
    key, value = [], []
    for index in layer_indices:
        entry = layers[str(index)]
        if not isinstance(entry, dict) or set(entry) != {"k", "v"}:
            raise ValueError(f"FP8 KV layer {index} requires K and V scales.")
        for name, destination in (("k", key), ("v", value)):
            scale = entry[name]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                raise ValueError(f"FP8 KV layer {index} {name} scale must be numeric.")
            scale = float(scale)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"FP8 KV layer {index} {name} scale must be finite and positive.")
            destination.append(scale)
    return FP8KVScales(tuple(key), tuple(value), str(path), hashlib.sha256(raw).hexdigest())


def resolve_fp8_kv_scales(config) -> FP8KVScales:
    """Explicit user calibration wins; presets require an exact config and name match."""
    model = str(config.model)
    layer_indices = tuple(config.runtime_layout.kv_idx_to_layer_idx)
    explicit = getattr(config, "fp8_kv_scale_path", None)
    if explicit is not None:
        return _read_scale_file(Path(explicit), model=model, layer_indices=layer_indices)
    preset = _PRESET_DIR / f"{Path(model).name}.json"
    if preset.is_file():
        return _read_scale_file(preset, model=model, layer_indices=layer_indices)
    raise ValueError(
        f"No calibrated FP8 KV scales for {Path(model).name!r}. "
        "Provide fp8_kv_scale_path with a scale file calibrated for this checkpoint."
    )


__all__ = ["FP8KVScales", "model_config_sha256", "resolve_fp8_kv_scales"]
