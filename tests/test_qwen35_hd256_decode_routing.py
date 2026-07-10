import unittest
from unittest.mock import patch

import torch

from sparsevllm.engine.cache_manager import DecodeComputeView
from sparsevllm.layers.attention_backend import TritonAttentionBackend


class GqaDecodeRoutingTest(unittest.TestCase):
    @staticmethod
    def _make_view(*, head_dim: int, attn_score=None):
        active_slots = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        req_indices = torch.tensor([0], dtype=torch.int32)
        context_lens = torch.tensor([3], dtype=torch.int32)
        k_cache = torch.zeros(4, 4, head_dim, dtype=torch.float32)
        v_cache = torch.zeros_like(k_cache)
        return DecodeComputeView(
            k_cache=k_cache,
            v_cache=v_cache,
            active_slots=active_slots,
            req_indices=req_indices,
            context_lens=context_lens,
            attn_score=attn_score,
            max_context_len=3,
        )

    def test_gqa_decode_uses_unified_wrappers_for_d128_and_d256(self):
        for head_dim in (128, 256):
            for score_mode in ("none", "3d"):
                with self.subTest(head_dim=head_dim, score_mode=score_mode):
                    attn_score = None
                    if score_mode == "3d":
                        attn_score = torch.zeros(1, 16, 3)
                    q = torch.zeros(1, 16, head_dim)
                    view = self._make_view(head_dim=head_dim, attn_score=attn_score)
                    mid_o = torch.empty(1, 16, 1, head_dim)
                    mid_lse = torch.empty(1, 16, 1)
                    calls = {"stage1": 0, "stage1_score": 0, "stage2": 0}

                    def stage1(*args, **kwargs):
                        calls["stage1"] += 1

                    def stage1_score(*args, **kwargs):
                        calls["stage1_score"] += 1

                    def stage2(mid_o, mid_o_logsumexp, context_lens, output, block_seq):
                        calls["stage2"] += 1
                        output.fill_(float(head_dim))

                    with (
                        patch("sparsevllm.layers.attention_backend.gqa_flash_decode_stage1", side_effect=stage1),
                        patch(
                            "sparsevllm.layers.attention_backend.gqa_flash_decode_stage1_with_score",
                            side_effect=stage1_score,
                        ),
                        patch("sparsevllm.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
                    ):
                        output = TritonAttentionBackend().run_decode(
                            q,
                            view,
                            mid_o=mid_o,
                            mid_o_logexpsum=mid_lse,
                            max_len_in_batch=3,
                            block_seq=256,
                            num_heads=16,
                            num_kv_heads=4,
                        )

                    expected_stage1 = 0 if score_mode == "3d" else 1
                    expected_score = 1 if score_mode == "3d" else 0
                    self.assertEqual(
                        calls,
                        {"stage1": expected_stage1, "stage1_score": expected_score, "stage2": 1},
                    )
                    self.assertTrue(torch.equal(output, torch.full_like(output, float(head_dim))))


if __name__ == "__main__":
    unittest.main()
