from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    rotate_half,
)

from sparsevllm.layers.attention import Attention
from sparsevllm.utils.context import get_context


def _apply_partial_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = int(cos.shape[-1])
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


class Qwen3_5SparseAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.total_num_heads))
        self.q_size = self.total_num_heads * self.head_dim
        self.kv_size = self.total_num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.q_size * 2,
            bias=bool(config.attention_bias),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=bool(config.attention_bias),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=bool(config.attention_bias),
        )
        self.o_proj = nn.Linear(
            self.q_size,
            self.hidden_size,
            bias=bool(config.attention_bias),
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.attn = Attention(
            self.total_num_heads,
            self.head_dim,
            self.scaling,
            self.total_num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q_with_gate = self.q_proj(hidden_states).view(-1, self.total_num_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_with_gate, 2, dim=-1)
        gate = gate.reshape(-1, self.q_size)

        k = self.k_proj(hidden_states).view(-1, self.total_num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.total_num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rotary_emb(hidden_states.unsqueeze(0), positions.view(1, -1))
        q, k = _apply_partial_rotary(q, k, cos.squeeze(0), sin.squeeze(0))

        o = self.attn(q, k, v).reshape(-1, self.q_size)
        o = o * torch.sigmoid(gate)
        return self.o_proj(o)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.layer_type = str(config.layer_types[layer_idx])
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5SparseAttention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type={self.layer_type!r}.")
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        run_linear_attention,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = run_linear_attention(self.layer_idx, self.linear_attn, hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sparse_controller = None
        self._linear_state_by_seq_id: dict[int, DynamicCache] = {}

    def free_seq_state(self, seq_id: int) -> None:
        self._linear_state_by_seq_id.pop(int(seq_id), None)

    def _reset_new_sequence_states(self, seqs) -> None:
        for seq in seqs:
            if int(seq.num_prefilled_tokens) == 0 and int(seq.num_completion_tokens) == 0:
                self.free_seq_state(seq.seq_id)

    def _query_slices(self, hidden_states: torch.Tensor) -> list[tuple[int, int]]:
        context = get_context()
        seqs = getattr(context, "seqs", None)
        if seqs is None:
            raise RuntimeError("Qwen3.5 linear attention requires current seqs in sparsevllm context.")
        if context.cu_seqlens_q is None:
            if hidden_states.shape[0] != len(seqs):
                raise RuntimeError(
                    "Qwen3.5 decode expects one token per sequence when cu_seqlens_q is absent: "
                    f"tokens={hidden_states.shape[0]} seqs={len(seqs)}."
                )
            return [(idx, idx + 1) for idx in range(len(seqs))]

        cu = context.cu_seqlens_q.detach().cpu().tolist()
        if len(cu) != len(seqs) + 1:
            raise RuntimeError(
                "Qwen3.5 prefill cu_seqlens_q does not match context seqs: "
                f"len(cu)-1={len(cu) - 1} len(seqs)={len(seqs)}."
            )
        return [(int(cu[idx]), int(cu[idx + 1])) for idx in range(len(seqs))]

    def _linear_cache_for_seq(self, seq_id: int) -> DynamicCache:
        seq_id = int(seq_id)
        cache = self._linear_state_by_seq_id.get(seq_id)
        if cache is None:
            cache = DynamicCache(config=self.config)
            self._linear_state_by_seq_id[seq_id] = cache
        return cache

    def _run_linear_attention(
        self,
        layer_idx: int,
        linear_attn: Qwen3_5GatedDeltaNet,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        seqs = context.seqs
        outputs: list[torch.Tensor] = []
        for seq, (start, end) in zip(seqs, self._query_slices(hidden_states)):
            seq_hidden = hidden_states[start:end].unsqueeze(0)
            cache = self._linear_cache_for_seq(seq.seq_id)
            out = linear_attn(
                hidden_states=seq_hidden,
                cache_params=cache,
                attention_mask=None,
            )
            outputs.append(out.squeeze(0))
        if not outputs:
            return torch.empty_like(hidden_states)
        return torch.cat(outputs, dim=0)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        context = get_context()
        seqs = getattr(context, "seqs", None)
        if seqs is None:
            raise RuntimeError("Qwen3.5 requires context.seqs for linear attention state management.")
        self._reset_new_sequence_states(seqs)

        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            context.now_layer_idx = layer_idx
            hidden_states = layer(positions, hidden_states, self._run_linear_attention)
            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(layer_idx, context)
        return self.norm(hidden_states)


class Qwen3_5ForCausalLM(nn.Module):
    weight_prefix_replacements = (("model.language_model.", "model."),)
    ignored_weight_prefixes = ("model.visual.", "model.image_newline", "mtp.")

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            if context.cu_seqlens_q is None:
                raise RuntimeError("Qwen3.5 prefill logits require cu_seqlens_q.")
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        return F.linear(hidden_states, self.lm_head.weight)

    def free_seq_state(self, seq_id: int) -> None:
        self.model.free_seq_state(seq_id)
