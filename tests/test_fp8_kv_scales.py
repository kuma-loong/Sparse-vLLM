"""Static FP8 KV scale files are checkpoint-bound and topology-independent."""

import json
from types import SimpleNamespace

import pytest

from sparseengine.configs.fp8_kv_scales import model_config_sha256, resolve_fp8_kv_scales


def _scale_file(tmp_path, *, model_name="model", layers=(0, 2)):
    model = tmp_path / model_name
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_moe", "num_hidden_layers": 3}))
    scales = {
        "schema_version": 1,
        "scheme": "fp8_e4m3fn_per_layer",
        "scale_convention": "dequant_multiplier",
        "checkpoint_id": model.name,
        "model_config_sha256": model_config_sha256(model),
        "layers": {str(index): {"k": 0.01 + index / 100, "v": 0.02 + index / 100}
                   for index in layers},
    }
    path = tmp_path / "scales.json"
    path.write_text(json.dumps(scales))
    config = SimpleNamespace(model=str(model), fp8_kv_scale_path=str(path),
                             runtime_layout=SimpleNamespace(kv_idx_to_layer_idx=layers))
    return config, path, scales


def test_explicit_scales_follow_global_attention_layer_indices(tmp_path):
    config, path, _ = _scale_file(tmp_path)
    resolved = resolve_fp8_kv_scales(config)
    assert resolved.key == (0.01, 0.03)
    assert resolved.value == (0.02, 0.04)
    assert resolved.source == str(path)
    assert len(resolved.file_sha256) == 64


@pytest.mark.parametrize("change,match", [
    (lambda data: data.update(checkpoint_id="other-model"), "checkpoint"),
    (lambda data: data.update(model_config_sha256="0" * 64), "config hash"),
    (lambda data: data["layers"]["2"].update(k=0), "finite and positive"),
    (lambda data: data["layers"].pop("2"), "exactly the attention layers"),
])
def test_rejects_mismatched_or_invalid_scales(tmp_path, change, match):
    config, path, data = _scale_file(tmp_path)
    change(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        resolve_fp8_kv_scales(config)
