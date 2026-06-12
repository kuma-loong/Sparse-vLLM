import unittest
from types import SimpleNamespace

import torch

from sparsevllm.engine.cache_manager.base import LayerBatchStates
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.models.adapters import get_model_adapter


def _qwen35_top_config(layer_types=None):
    layer_types = layer_types or [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        attn_output_gate=True,
    )
    return SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config,
    )


class Qwen35ModelAdapterTest(unittest.TestCase):
    def test_normalizes_top_level_multimodal_config_to_text_config(self):
        adapter = get_model_adapter(SimpleNamespace(model_type="qwen3_5"))
        text_config = adapter.normalize_config(_qwen35_top_config())

        self.assertEqual(text_config.model_type, "qwen3_5_text")
        self.assertEqual(text_config.sparsevllm_model_type, "qwen3_5")
        self.assertEqual(text_config.sparsevllm_source_model_type, "qwen3_5")
        self.assertEqual(text_config.sparsevllm_attention_layer_indices, (3,))

    def test_accepts_direct_text_config(self):
        adapter = get_model_adapter(SimpleNamespace(model_type="qwen3_5_text"))
        direct_text_config = _qwen35_top_config().text_config
        text_config = adapter.normalize_config(direct_text_config)

        self.assertIs(text_config, direct_text_config)
        self.assertEqual(text_config.sparsevllm_model_type, "qwen3_5")
        self.assertEqual(text_config.sparsevllm_source_model_type, "qwen3_5_text")

    def test_maps_language_weights_and_skips_vision_weights(self):
        adapter = get_model_adapter(SimpleNamespace(model_type="qwen3_5"))

        self.assertEqual(
            adapter.map_weight_name("model.language_model.layers.3.self_attn.q_proj.weight"),
            "model.layers.3.self_attn.q_proj.weight",
        )
        self.assertEqual(
            adapter.map_weight_name("model.language_model.embed_tokens.weight"),
            "model.embed_tokens.weight",
        )
        self.assertIsNone(adapter.map_weight_name("model.visual.blocks.0.norm1.weight"))
        self.assertEqual(adapter.map_weight_name("lm_head.weight"), "lm_head.weight")

    def test_rejects_unknown_layer_type(self):
        adapter = get_model_adapter(SimpleNamespace(model_type="qwen3_5"))
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3.5 layer_types"):
            adapter.normalize_config(_qwen35_top_config(["linear_attention", "unknown"]))

    def test_limits_initial_qwen35_sparse_methods(self):
        adapter = get_model_adapter(SimpleNamespace(model_type="qwen3_5"))
        adapter.validate_engine_config(SimpleNamespace(tensor_parallel_size=1, vllm_sparse_method=""))
        adapter.validate_engine_config(SimpleNamespace(tensor_parallel_size=1, vllm_sparse_method="snapkv"))

        with self.assertRaisesRegex(ValueError, "supports sparse methods"):
            adapter.validate_engine_config(SimpleNamespace(tensor_parallel_size=1, vllm_sparse_method="omnikv"))
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            adapter.validate_engine_config(SimpleNamespace(tensor_parallel_size=2, vllm_sparse_method="snapkv"))

    def test_rejects_prefix_cache_and_decode_graph(self):
        adapter = get_model_adapter(SimpleNamespace(model_type="qwen3_5"))
        with self.assertRaisesRegex(ValueError, "prefix caching is not supported"):
            adapter.validate_engine_config(
                SimpleNamespace(
                    tensor_parallel_size=1,
                    vllm_sparse_method="",
                    enable_prefix_caching=True,
                    decode_cuda_graph=False,
                )
            )
        with self.assertRaisesRegex(ValueError, "decode_cuda_graph is not supported"):
            adapter.validate_engine_config(
                SimpleNamespace(
                    tensor_parallel_size=1,
                    vllm_sparse_method="snapkv",
                    enable_prefix_caching=False,
                    decode_cuda_graph=True,
                )
            )

    def test_standard_cache_store_view_uses_cache_layer_mapping(self):
        manager = object.__new__(StandardCacheManager)
        manager.kv_cache = torch.arange(2 * 2 * 3).reshape(2, 2, 3)
        manager.layer_to_cache_idx = {3: 0, 7: 1}
        manager.attention_layer_index_set = {3, 7}
        manager.layer_batch_state = LayerBatchStates(slot_mapping=torch.tensor([2], dtype=torch.int32))

        k_cache, v_cache, slot_mapping = manager.get_layer_store_view(7)

        self.assertTrue(torch.equal(k_cache, manager.kv_cache[0, 1]))
        self.assertTrue(torch.equal(v_cache, manager.kv_cache[1, 1]))
        self.assertTrue(torch.equal(slot_mapping, manager.layer_batch_state.slot_mapping))


if __name__ == "__main__":
    unittest.main()
