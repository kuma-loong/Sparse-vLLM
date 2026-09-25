from __future__ import annotations

import copy
import math

import torch

from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.cache_manager.quantized import QuantizedCacheManager
from sparseengine.engine.cache_manager.quantized_pages import QuantizedPagePool
from sparseengine.engine.cache_manager.storage.quantized_kv import QuantizedKVStorage
from sparseengine.engine.cache_manager.storage.low_rank_kv import LowRankKVStorage
from sparseengine.engine.cache_manager.storage import (
    ExplicitKVStorage,
    HeterogeneousExplicitKVStorage,
    MlaLatentStorage,
)
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.sampling_params import SamplingParams
from sparseengine.utils.context import reset_context
from sparseengine.utils.log import logger

from .capacity import profiling_prefill_chunk_lengths
from .profiling import StartupMemoryProfiler


class PrefillHistoryCacheManager(StandardCacheManager):
    """Startup-only dense cache with synthetic history shared across layers."""

    def allocate_kv_cache(self) -> None:
        slots = int(self.config.num_kvcache_slots)
        storage = self.attention_cache_storage
        if isinstance(storage, LowRankKVStorage):
            storage.allocate_shared_history(num_slots=slots, device=self.device)
        elif isinstance(storage, HeterogeneousExplicitKVStorage):
            caches = {
                shape: torch.zeros(2, slots, *shape, dtype=storage.dtype, device=self.device)
                for shape in dict.fromkeys(storage.layer_shapes)
            }
            storage.kv_cache = [caches[shape] for shape in storage.layer_shapes]
        else:
            storage.allocate(num_layers=1, num_slots=slots, device=self.device)
            for tensor in storage.accounting_tensors():
                tensor.zero_()
            # Layers execute sequentially and overwrite only the current chunk.
            # The synthetic history stays zero, so one physical layer suffices.
            if isinstance(storage, ExplicitKVStorage):
                storage.kv_cache = storage.kv_cache.expand(
                    -1, self.num_kv_layers, -1, -1, -1,
                )
            elif isinstance(storage, MlaLatentStorage):
                storage.latent_cache = storage.latent_cache.expand(
                    self.num_kv_layers, -1, -1, -1,
                )
                storage.rope_cache = storage.rope_cache.expand(
                    self.num_kv_layers, -1, -1, -1,
                )
        self.kv_cache = getattr(storage, "kv_cache", None)

    def seed_history(self, seq: Sequence) -> None:
        self._allocate(seq.seq_id, int(seq.num_prefilled_tokens))


class PrefillHistoryFP8CacheManager(QuantizedCacheManager):
    """Bounded FP8 pages for startup's synthetic-history prefill measurement."""

    def allocate_kv_cache(self) -> None:
        config = self.config
        self.page_size = int(config.kv_quant_page_size)
        num_pages = math.ceil(int(config.num_kvcache_slots) / self.page_size) + int(config.max_num_seqs_in_batch)
        config.num_kvcache_slots = num_pages * self.page_size
        self.page_pool = QuantizedPagePool(num_pages, self.page_size, self.max_model_len)
        storage = QuantizedKVStorage(
            format="fp8_kv", bits=8, page_size=self.page_size,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            dtype=self.hf_config.dtype, seed=0,
            fp8_scales=config.resolved_fp8_kv_scales,
        )
        storage.allocate(num_layers=self.num_kv_layers, num_slots=config.num_kvcache_slots,
                         num_rows=self.max_buffer_rows, device=self.device)
        storage.data.zero_()
        self.attention_cache_storage = storage
        self.kv_cache = None
        self.prefill_capacity = 0
        self.prefill_kv = torch.empty((2, 0, self.num_kv_heads, self.head_dim),
                                      dtype=self.hf_config.dtype, device=self.device)
        self.prefill_rotated = None
        self.prefill_slot_map = torch.empty((0, 0), dtype=torch.int32, device=self.device)
        self._write_plan = []
        self._prefill_active = False

    def seed_history(self, seq: Sequence) -> None:
        self._allocate(seq.seq_id, int(seq.num_prefilled_tokens))


@torch.inference_mode()
def profile_prefill_history(runner):
    """Measure a full token batch with one maximum-context request in one step.

    Dense synthetic history covers history-dependent attention allocations.
    Sparse compression/scoring and concurrent long requests rely on utilization
    headroom rather than method-specific history reconstruction.
    """
    config = copy.copy(runner.config)
    context_len = int(config.max_model_len) - 1
    chunk_lengths = profiling_prefill_chunk_lengths(config)
    prompt_lengths = (context_len, *chunk_lengths[1:])
    fp8_kv = config.sparse_method == "fp8_kv"
    if not fp8_kv:
        config.sparse_method = ""
    config.prefill_sparse_method = None
    config.enable_prefix_caching = False
    config.enable_prefix_cache_offload = False
    config.resolved_prefix_cache_mode = "disabled"
    config.startup_cache_phase = "profiling"
    config.num_kvcache_slots = sum(prompt_lengths)
    manager_class = PrefillHistoryFP8CacheManager if fp8_kv else PrefillHistoryCacheManager
    manager = manager_class(config, runner.parallel_context)
    controller = SparseController(config, manager)
    runtime = RuntimeState(config, manager, runner.recurrent_state_manager)
    seqs = []
    for prompt_len, chunk_len in zip(prompt_lengths, chunk_lengths):
        seq = Sequence([0] * prompt_len, SamplingParams(max_tokens=1, temperature=0.0))
        seq.num_prefilled_tokens = prompt_len - chunk_len
        seq.current_chunk_size = chunk_len
        seqs.append(seq)
    if seqs[0].num_prefilled_tokens:
        manager.seed_history(seqs[0])

    model = runner.model.model
    previous = runner.cache_manager, runner.sparse_controller, runner.runtime_state
    previous_controller = model.sparse_controller
    profiler = StartupMemoryProfiler(runner.platform, runner.device)
    try:
        runner.cache_manager, runner.sparse_controller, runner.runtime_state = (
            manager, controller, runtime,
        )
        model.sparse_controller = controller
        controller.set_modules(model.layers)
        logger.info(
            "Startup profile phase=prefill tokens={} batch={} max_context={} "
            "history={} max_chunk={} visible_tokens={}.",
            sum(chunk_lengths), len(seqs), context_len,
            seqs[0].num_prefilled_tokens, max(chunk_lengths), sum(prompt_lengths),
        )
        # Keep the synthetic cache alive across both snapshots so its persistent
        # bytes are not counted as transient model memory.
        profiler.begin("prefill")
        runner.run(seqs, is_prefill=True)
        result = profiler.finish("prefill")
        logger.info(
            "Startup prefill transient_peak={:.2f} GiB.",
            result.measurement.transient_peak_bytes / 1024**3,
        )
        return result
    finally:
        runner.cache_manager, runner.sparse_controller, runner.runtime_state = previous
        model.sparse_controller = previous_controller
        reset_context()
        for seq in seqs:
            runtime.free_seq(seq.seq_id)
        runtime.reset_after_warmup()
        release_bindings = getattr(runner.model, "release_cache_runtime_bindings", None)
        if callable(release_bindings):
            release_bindings(manager)
