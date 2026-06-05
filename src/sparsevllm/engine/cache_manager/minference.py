from __future__ import annotations

import torch

from sparsevllm.engine.sequence import Sequence
from sparsevllm.triton_kernel.minference_prefill import minference_context_attention_fwd

from .snapkv import SnapKVCacheManager
from .standard import StandardCacheManager


class MinferencePrefillMixin:
    def _minference_layer_enabled(self, layer_idx: int) -> bool:
        return int(layer_idx) >= int(self.config.minference_starting_layer)

    def run_prefill_attention(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        req_indices: torch.Tensor,
        start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prompt_cache_lens: torch.Tensor,
        max_input_len: int,
        active_slots: torch.Tensor,
        attn_score: torch.Tensor | None,
    ):
        if not self._minference_layer_enabled(layer_idx):
            return super().run_prefill_attention(
                layer_idx=layer_idx,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                out=out,
                req_indices=req_indices,
                start_loc=start_loc,
                seq_lens=seq_lens,
                prompt_cache_lens=prompt_cache_lens,
                max_input_len=max_input_len,
                active_slots=active_slots,
                attn_score=attn_score,
            )

        minference_context_attention_fwd(
            q,
            k_cache,
            v_cache,
            out,
            req_indices,
            start_loc,
            seq_lens,
            prompt_cache_lens,
            max_input_len,
            active_slots,
            layer_idx=layer_idx,
            config=self.config,
            rank=self.rank,
            attn_score=attn_score,
        )

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        if int(seq.num_prefilled_tokens) == 0:
            return remaining
        return super().remaining_prefill_tokens(seq)

    def prefill_step_tokens(
        self,
        *,
        remaining_prefill_tokens: int,
        num_batched_tokens: int,
        step_free_count: int,
        max_num_batched_tokens: int,
        target_is_long: bool,
        chunk_prefill_size: int,
        prefill_schedule_policy: str,
    ) -> int | None:
        del target_is_long, chunk_prefill_size, prefill_schedule_policy
        available_tokens = min(
            int(max_num_batched_tokens) - int(num_batched_tokens),
            int(step_free_count),
        )
        if int(remaining_prefill_tokens) <= available_tokens:
            return int(remaining_prefill_tokens)
        if int(num_batched_tokens) > 0:
            return 0
        raise RuntimeError(
            "MInference prefill requires each prompt to run in one full prefill step "
            "because chunk/prefix prefill is not supported yet. "
            f"remaining_prefill_tokens={remaining_prefill_tokens} "
            f"available_tokens={available_tokens} "
            f"max_num_batched_tokens={max_num_batched_tokens} "
            f"step_free_count={step_free_count}. "
            "Increase max_num_batched_tokens/engine_prefill_chunk_size or reduce prompt length."
        )


class MinferenceStandardCacheManager(MinferencePrefillMixin, StandardCacheManager):
    pass


class MinferenceSnapKVCacheManager(MinferencePrefillMixin, SnapKVCacheManager):
    pass
