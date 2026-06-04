import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.triton_kernel.minference_prefill import (
    MINFERENCE_BLOCK_M,
    MINFERENCE_BLOCK_N,
    _convert_vertical_slash_indexes_kernel,
    _convert_vertical_slash_row,
    _estimate_layer_pattern_density,
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

    def test_short_context_density_triggers_dense_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            pattern_path = Path(tmp) / "pattern.json"
            pattern_path.write_text(
                json.dumps([{"0": ["vertical_and_slash", 4096, 4096, 1.0]}]),
                encoding="utf-8",
            )
            cfg = SimpleNamespace(minference_config_path=str(pattern_path), minference_ratio=1.0)
            density = _estimate_layer_pattern_density(cfg, layer_idx=0, rank=0, num_heads=1, seq_len=2048)
            self.assertGreaterEqual(density, 1.0)

    def test_slash_row_conversion_matches_reference_ranges(self):
        blocks, columns = _convert_vertical_slash_row(
            [10, 70, 130],
            [190],
            end_m=192,
            block_m=64,
            block_n=64,
        )
        self.assertEqual(blocks, [0])
        self.assertEqual(columns, [])

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
    def test_gpu_converter_matches_row_reference(self):
        device = "cuda"
        block_m = MINFERENCE_BLOCK_M
        block_n = MINFERENCE_BLOCK_N
        vertical = torch.tensor(
            [
                [[10, 70, 130, 180], [5, 65, 125, 185]],
                [[0, 40, 90, 140], [20, 80, 120, 160]],
            ],
            dtype=torch.int32,
            device=device,
        )
        slash = torch.tensor(
            [
                [[190, 130, 30], [160, 90, 10]],
                [[240, 220, 200], [150, 120, 70]],
            ],
            dtype=torch.int32,
            device=device,
        )
        b_seq_len = torch.tensor([192, 96], dtype=torch.int32, device=device)
        vertical_counts = torch.full((2, 2), 4, dtype=torch.int32, device=device)
        slash_counts = torch.full((2, 2), 3, dtype=torch.int32, device=device)
        num_rows = 2
        block_count = torch.zeros((2, 2, num_rows), dtype=torch.int32, device=device)
        block_offset = torch.zeros((2, 2, num_rows, 3), dtype=torch.int32, device=device)
        column_count = torch.zeros((2, 2, num_rows), dtype=torch.int32, device=device)
        column_index = torch.zeros((2, 2, num_rows, 4), dtype=torch.int32, device=device)

        _convert_vertical_slash_indexes_kernel[(num_rows, 4)](
            b_seq_len,
            vertical,
            slash,
            vertical_counts,
            slash_counts,
            block_count,
            block_offset,
            column_count,
            column_index,
            vertical.stride(0),
            vertical.stride(1),
            vertical.stride(2),
            slash.stride(0),
            slash.stride(1),
            slash.stride(2),
            vertical_counts.stride(0),
            vertical_counts.stride(1),
            slash_counts.stride(0),
            slash_counts.stride(1),
            block_count.stride(0),
            block_count.stride(1),
            block_count.stride(2),
            block_offset.stride(0),
            block_offset.stride(1),
            block_offset.stride(2),
            block_offset.stride(3),
            column_count.stride(0),
            column_count.stride(1),
            column_count.stride(2),
            column_index.stride(0),
            column_index.stride(1),
            column_index.stride(2),
            column_index.stride(3),
            H=2,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=1,
        )
        torch.cuda.synchronize()

        block_count_cpu = block_count.cpu()
        block_offset_cpu = block_offset.cpu()
        column_count_cpu = column_count.cpu()
        column_index_cpu = column_index.cpu()
        vertical_cpu = vertical.cpu()
        slash_cpu = slash.cpu()
        for b_idx, seq_len in enumerate(b_seq_len.cpu().tolist()):
            for head_idx in range(2):
                vertical_list = vertical_cpu[b_idx, head_idx].tolist()
                slash_list = slash_cpu[b_idx, head_idx].tolist()
                for row_idx in range(num_rows):
                    end_m = (row_idx + 1) * block_m
                    if row_idx * block_m >= seq_len:
                        self.assertEqual(int(block_count_cpu[b_idx, head_idx, row_idx]), 0)
                        self.assertEqual(int(column_count_cpu[b_idx, head_idx, row_idx]), 0)
                        continue
                    expected_blocks, expected_columns = _convert_vertical_slash_row(
                        vertical_list,
                        slash_list,
                        end_m=end_m,
                        block_m=block_m,
                        block_n=block_n,
                    )
                    actual_block_count = int(block_count_cpu[b_idx, head_idx, row_idx])
                    actual_column_count = int(column_count_cpu[b_idx, head_idx, row_idx])
                    self.assertEqual(
                        block_offset_cpu[b_idx, head_idx, row_idx, :actual_block_count].tolist(),
                        expected_blocks,
                    )
                    self.assertEqual(
                        column_index_cpu[b_idx, head_idx, row_idx, :actual_column_count].tolist(),
                        expected_columns,
                    )
                    self.assertEqual(actual_block_count, len(expected_blocks))
                    self.assertEqual(actual_column_count, len(expected_columns))

        self.assertEqual(block_offset_cpu[0, 0, 0, :1].tolist(), [0])
        self.assertEqual(block_offset_cpu[0, 0, 1, :2].tolist(), [0, 128])
        self.assertEqual(int(column_count_cpu[0, 0, 0]), 0)
        self.assertEqual(int(column_count_cpu[0, 0, 1]), 0)
        self.assertEqual(int(block_count_cpu[1, 0, 0]), 0)
        self.assertEqual(int(column_count_cpu[1, 0, 0]), 3)
        self.assertEqual(column_index_cpu[1, 0, 0, :3].tolist(), [0, 40, 90])
        self.assertEqual(int(block_count_cpu[1, 0, 1]), 0)
        self.assertEqual(int(column_count_cpu[1, 0, 1]), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for MInference prefill kernel tests.")
    def test_kernel_smoke_outputs_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            pattern_path = Path(tmp) / "pattern.json"
            _write_pattern(pattern_path, num_layers=1, num_heads=2)
            cfg = SimpleNamespace(minference_config_path=str(pattern_path), minference_ratio=1.0)

            device = "cuda"
            seq_len = 256
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

            with patch("sparsevllm.triton_kernel.minference_prefill.MINFERENCE_MIN_SPARSE_SEQ_LEN", 1):
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
