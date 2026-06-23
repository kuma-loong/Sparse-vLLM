import unittest


class DeltaKVBaselineImportsTest(unittest.TestCase):
    def test_qwen2_baselines_still_import(self):
        from deltakv.modeling.qwen2.qwen2_pyramidkv import Qwen2PyramidKVForCausalLM
        from deltakv.modeling.qwen2.qwen2_snapkv import Qwen2SnapKVForCausalLM

        self.assertEqual(Qwen2SnapKVForCausalLM.__name__, "Qwen2SnapKVForCausalLM")
        self.assertEqual(Qwen2PyramidKVForCausalLM.__name__, "Qwen2PyramidKVForCausalLM")

    def test_llama_baselines_still_import(self):
        from deltakv.modeling.llama.llama_pyramidkv import LlamaPyramidKVForCausalLM
        from deltakv.modeling.llama.llama_snapkv import LlamaSnapKVForCausalLM

        self.assertEqual(LlamaSnapKVForCausalLM.__name__, "LlamaSnapKVForCausalLM")
        self.assertEqual(LlamaPyramidKVForCausalLM.__name__, "LlamaPyramidKVForCausalLM")


if __name__ == "__main__":
    unittest.main()
