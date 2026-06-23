from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.qwen2_training import Qwen2KVClusterCompress


MODEL_PATH = "/data2/haojitai/models/Qwen2.5-7B-Instruct-1M"
COMPRESSOR_PATH = "/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor"
OUT_DIR = Path("/data2/haojitai/outputs/deltakv/analysis/compressed_diff_distribution_text_gpu5_20260601_220900")

TEXT = (
    "DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. "
    "This diagnostic sentence is repeated to create enough tokens for the cluster analysis. "
    "DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. "
    "This diagnostic sentence is repeated to create enough tokens for the cluster analysis. "
    "DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. "
    "This diagnostic sentence is repeated to create enough tokens for the cluster analysis. "
    "DeltaKV stores cache tokens by referencing nearby KV states and keeping the residual. "
    "This diagnostic sentence is repeated to create enough tokens for the cluster analysis."
)


def _stats(values: list[np.ndarray]) -> dict[str, float | int]:
    if not values:
        raise ValueError("No values were collected.")
    arr = np.concatenate(values).astype(np.float64)
    out: dict[str, float | int] = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "abs_max": float(np.max(np.abs(arr))),
    }
    for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]:
        out[f"p{str(p).replace('.', '_')}"] = float(np.percentile(arr, p))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = KVQwen2Config.from_pretrained(COMPRESSOR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Qwen2KVClusterCompress.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    state_dict = load_file(str(Path(COMPRESSOR_PATH) / "model.safetensors"), device="cuda:0")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    inputs = tokenizer(TEXT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs, labels=inputs["input_ids"])

    comp_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []
    layer_stats: list[dict[str, object]] = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if attn.buffer_comp_kv is None or attn.buffer_ideal_res is None or attn.buffer_raw_kv is None:
            continue
        comp = attn.buffer_comp_kv.detach().float().cpu().numpy().reshape(-1)
        residual = attn.buffer_ideal_res.detach().float().cpu().numpy().reshape(-1)
        raw = attn.buffer_raw_kv.detach().float().cpu().numpy().reshape(-1)
        comp_values.append(comp)
        residual_values.append(residual)
        raw_values.append(raw)
        layer_stats.append(
            {
                "layer": layer_idx,
                "comp": _stats([comp]),
                "residual": _stats([residual]),
                "raw": _stats([raw]),
            }
        )

    report = {
        "model_path": MODEL_PATH,
        "compressor_path": COMPRESSOR_PATH,
        "text_token_count": int(inputs["input_ids"].shape[1]),
        "definition": {
            "compressed_diff": "compress_down(kv_rem) - compress_down(refs)",
            "residual": "kv_rem - refs",
            "raw": "concat(key_states, value_states)",
        },
        "global": {
            "compressed_diff": _stats(comp_values),
            "residual": _stats(residual_values),
            "raw": _stats(raw_values),
        },
        "layers": layer_stats,
    }
    with open(OUT_DIR / "compressed_diff_distribution_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["global"], indent=2))
    print(f"Wrote {OUT_DIR / 'compressed_diff_distribution_stats.json'}")


if __name__ == "__main__":
    main()
