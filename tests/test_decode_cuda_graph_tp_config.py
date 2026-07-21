import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphKey, DecodeCudaGraphRunner, DecodeCudaGraphState


class DecodeCudaGraphTPConfigTest(unittest.TestCase):
    def hf_config(self):
        return SimpleNamespace(
            model_type="qwen2",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

    def _config(self, method: str, *, model_name: str = "TinyModel", **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / model_name
            model_dir.mkdir()
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self.hf_config()):
                return Config(
                    model=str(model_dir),
                    vllm_sparse_method=method,
                    decode_cuda_graph=True,
                    tensor_parallel_size=2,
                    max_decoding_seqs=4,
                    **kwargs,
                )

    def test_tp_decode_cuda_graph_accepts_v1_methods(self):
        for method in ["vanilla", "streamingllm", "snapkv", "pyramidkv", "omnikv", "quest", "rkv"]:
            with self.subTest(method=method):
                cfg = self._config(method)
                self.assertTrue(cfg.decode_cuda_graph)
                self.assertEqual(cfg.tensor_parallel_size, 2)

        cfg = self._config("skipkv", model_name="DeepSeek-R1-Distill-Qwen-7B")
        self.assertEqual(cfg.vllm_sparse_method, "skipkv")

    def test_tp_decode_cuda_graph_rejects_deltakv(self):
        with self.assertRaisesRegex(ValueError, "DeltaKV is not supported"):
            self._config("deltakv", allow_missing_deltakv_path=True)

    def test_tp_decode_cuda_graph_accepts_prefix_cache_methods(self):
        for method in ["vanilla", "omnikv", "quest"]:
            with self.subTest(method=method):
                cfg = self._config(method, enable_prefix_caching=True)
                self.assertTrue(cfg.decode_cuda_graph)
                self.assertTrue(cfg.enable_prefix_caching)
                self.assertFalse(cfg.decode_cuda_graph_capture_sampling)

    def test_tp_decode_cuda_graph_rejects_capture_sampling(self):
        with self.assertRaisesRegex(ValueError, "capture_sampling is disabled"):
            self._config("snapkv", decode_cuda_graph_capture_sampling=True)


class DecodeCudaGraphDebugRefsTest(unittest.TestCase):
    def test_replay_restores_graph_captured_model_debug_refs(self):
        class DebugModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.debug_last_tensor = torch.tensor([1.0])
                self.debug_last_nested = {"items": [torch.tensor([2.0])]}
                self.not_debug = torch.tensor([3.0])

        class Engine:
            def __init__(self):
                self.model = DebugModule()

            def run_model(self, *_args):
                raise AssertionError("not called")

        engine = Engine()
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.run_model = engine.run_model
        captured_tensor = engine.model.debug_last_tensor
        captured_nested = engine.model.debug_last_nested

        refs = runner._snapshot_model_debug_refs()
        engine.model.debug_last_tensor = torch.tensor([4.0])
        engine.model.debug_last_nested = {"items": [torch.tensor([5.0])]}
        state = DecodeCudaGraphState(
            key=DecodeCudaGraphKey(
                method="",
                batch_size=1,
                context_capacity=1024,
                is_long_text=False,
                capture_sampling=False,
            ),
            model_debug_refs=refs,
        )

        runner._restore_model_debug_refs(state)

        self.assertIs(engine.model.debug_last_tensor, captured_tensor)
        self.assertIs(engine.model.debug_last_nested, captured_nested)
        tensor_refs = list(runner._debug_tensor_refs(captured_nested))
        self.assertEqual(len(tensor_refs), 1)
        self.assertIs(tensor_refs[0], captured_nested["items"][0])
        self.assertTrue(all(name != "not_debug" for _, name, _ in refs))


if __name__ == "__main__":
    unittest.main()
