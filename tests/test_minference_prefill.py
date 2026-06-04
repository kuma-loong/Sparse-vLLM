import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.triton_kernel.minference_prefill import (
    _get_vs_pattern,
    minference_context_attention_fwd,
)


def _hf_config(num_layers=2, num_heads=2, num_kv_heads=2, head_dim=32):
    return SimpleNamespace(
        model_type="qwen2",
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=num_heads * head_dim,
        head_dim=head_dim,
        max_position_embeddings=4096,
        torch_dtype=torch.float16,
    )


def _write_pattern(path: Path, num_layers=2, num_heads=2, pattern_type="vertical_and_slash"):
    layers = []
    for _ in range(num_layers):
        layers.append({str(head): [pattern_type, 30, 50, 1.0] for head in range(num_heads)})
    path.write_text(json.dumps(layers), encoding="utf-8")


class MinferencePrefillConfigTest(unittest.TestCase):
    def test_config_requires_pattern_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
                with self.assertRaisesRegex(ValueError, "minference_config_path is required"):
                    Config(model=tmp, prefill_attention_backend="minference")

    def test_config_rejects_non_snapkv_sparse_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            pattern_path = Path(tmp) / "pattern.json"
            _write_pattern(pattern_path)
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
                with self.assertRaisesRegex(NotImplementedError, "supports only vanilla/full attention and snapkv"):
                    Config(
                        model=tmp,
                        prefill_attention_backend="minference",
                        minference_config_path=str(pattern_path),
                        vllm_sparse_method="quest",
                    )

    def test_config_accepts_snapkv_combination(self):
        with tempfile.TemporaryDirectory() as tmp:
            pattern_path = Path(tmp) / "pattern.json"
            _write_pattern(pattern_path)
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
                cfg = Config(
                    model=tmp,
                    prefill_attention_backend="minference",
                    minference_config_path=str(pattern_path),
                    vllm_sparse_method="snapkv",
                )
            self.assertEqual(cfg.prefill_attention_backend, "minference")
            self.assertEqual(cfg.vllm_sparse_method, "snapkv")

    def test_pattern_rejects_unsupported_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            pattern_path = Path(tmp) / "pattern.json"
            _write_pattern(pattern_path, pattern_type="block_sparse")
            cfg = SimpleNamespace(minference_config_path=str(pattern_path), minference_ratio=1.0)
            with self.assertRaisesRegex(NotImplementedError, "vertical_and_slash"):
                _get_vs_pattern(cfg, 0, 0)

    def test_chunk_prefill_rejected_before_cuda_work(self):
        cfg = SimpleNamespace(minference_config_path="/tmp/unused.json", minference_ratio=1.0)
        with self.assertRaisesRegex(RuntimeError, "does not support chunk/prefix prefill"):
            minference_context_attention_fwd(
                torch.empty((0, 1, 32)),
                torch.empty((0, 1, 32)),
                torch.empty((0, 1, 32)),
                torch.empty((0, 1, 32)),
                torch.zeros((1,), dtype=torch.int32),
                torch.zeros((1,), dtype=torch.int32),
                torch.ones((1,), dtype=torch.int32),
                torch.ones((1,), dtype=torch.int32),
                1,
                torch.zeros((1, 1), dtype=torch.int32),
                layer_idx=0,
                config=cfg,
                rank=0,
            )


class MinferencePrefillKernelTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for MInference prefill kernel tests.")
    def test_kernel_smoke_outputs_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            pattern_path = Path(tmp) / "pattern.json"
            _write_pattern(pattern_path, num_layers=1, num_heads=2)
            cfg = SimpleNamespace(minference_config_path=str(pattern_path), minference_ratio=1.0)

            device = "cuda"
            seq_len = 128
            num_heads = 2
            num_kv_heads = 2
            head_dim = 32
            dtype = torch.float16
            q = torch.randn((seq_len, num_heads, head_dim), dtype=dtype, device=device)
            k_cache = torch.randn((seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
            v_cache = torch.randn((seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
            out = torch.empty_like(q)
            req_to_tokens = torch.arange(seq_len, dtype=torch.int32, device=device).view(1, seq_len)
            b_req_idx = torch.zeros((1,), dtype=torch.int32, device=device)
            b_start_loc = torch.zeros((1,), dtype=torch.int32, device=device)
            b_seq_len = torch.full((1,), seq_len, dtype=torch.int32, device=device)
            b_prompt_cache_len = torch.zeros((1,), dtype=torch.int32, device=device)
            attn_score = torch.zeros((1, num_heads, seq_len), dtype=torch.float32, device=device)

            minference_context_attention_fwd(
                q,
                k_cache,
                v_cache,
                out,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                b_prompt_cache_len,
                seq_len,
                req_to_tokens,
                layer_idx=0,
                config=cfg,
                rank=0,
                attn_score=attn_score,
            )
            torch.cuda.synchronize()
            self.assertTrue(torch.isfinite(out).all().item())
            self.assertGreater(float(attn_score.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
