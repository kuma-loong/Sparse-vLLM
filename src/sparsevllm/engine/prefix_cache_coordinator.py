from __future__ import annotations

from dataclasses import dataclass, field

from sparsevllm.config import Config
from sparsevllm.engine.prefix_cache import (
    PrefixCacheBlock,
    PrefixBlockPayload,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
    usable_prefix_cache_tokens,
)
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.profiler import profiler


@dataclass
class MixedPrefixBlockPayload:
    kv_payload: object
    recurrent_payload: object
    token_count: int
    accounting_bytes: int
    recurrent_bytes: int


@dataclass
class _PendingMixedPrefixBlock:
    stable_block_id: bytes
    parent_block_id: bytes | None
    logical_block_idx: int
    token_ids: list[int]
    payload: MixedPrefixBlockPayload


@dataclass
class _MixedPrefixRuntimeState:
    parent_block_id: bytes | None
    next_logical_block_idx: int
    pending_tokens: list[int] = field(default_factory=list)


class PrefixCacheCoordinator:
    """Owns mixed KV+recurrent prefix radix state for mixed runtimes."""

    def __init__(self, config: Config, cache_manager, recurrent_state_manager):
        self.config = config
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.enabled = bool(config.enable_prefix_caching)
        self.block_size = int(config.prefix_cache_block_size or 0)
        self.max_recurrent_bytes = int(config.prefix_recurrent_capacity_bytes)
        self.prefix_cache = None
        if self.enabled:
            self.prefix_cache = RadixPrefixIndex(
                block_size=self.block_size,
                fingerprint=build_prefix_cache_fingerprint(config, self.block_size),
                max_blocks=config.prefix_cache_max_blocks,
            )
        self.seq_id_to_prefix_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.seq_id_to_materialized_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.runtime_states: dict[int, _MixedPrefixRuntimeState] = {}
        self.pending_blocks: dict[int, list[_PendingMixedPrefixBlock]] = {}
        self.pending_duplicate_refs: dict[int, list[bytes]] = {}
        self.pending_block_ids: set[bytes] = set()
        self.pending_recurrent_bytes = 0
        self.capacity_limited_seq_ids: set[int] = set()
        self.skipped_capacity_blocks = 0

    def _require_prefix_cache(self) -> RadixPrefixIndex:
        if self.prefix_cache is None:
            raise RuntimeError("prefix cache is not enabled for this runtime.")
        return self.prefix_cache

    def inspect(self, token_ids: list[int], *, include_subtree: bool = False) -> dict[str, object]:
        return self._require_prefix_cache().inspect_prefix(
            [int(token_id) for token_id in token_ids],
            include_subtree=include_subtree,
        )

    def match(self, token_ids: list[int]) -> dict[str, object]:
        if self.prefix_cache is None:
            return {
                "supported": True,
                "enabled": False,
                "method": str(getattr(self.config, "vllm_sparse_method", "") or ""),
                "matched_tokens": 0,
                "matched_blocks": 0,
                "match_ratio": 0.0,
                "reason": "prefix cache is not enabled for this runtime.",
            }
        token_ids = [int(token_id) for token_id in token_ids]
        usable_tokens = usable_prefix_cache_tokens(len(token_ids), self.block_size)
        hit_len, hit_last_block_id, hit_blocks = self.prefix_cache.match_longest_prefix(
            token_ids,
            max_usable_tokens=usable_tokens,
        )
        return {
            "supported": True,
            "enabled": True,
            "method": str(getattr(self.config, "vllm_sparse_method", "") or ""),
            "block_size": int(self.block_size),
            "prompt_tokens": int(len(token_ids)),
            "usable_tokens": int(usable_tokens),
            "matched_tokens": int(hit_len),
            "matched_blocks": int(hit_blocks),
            "match_ratio": 0.0 if usable_tokens <= 0 else float(hit_len) / float(usable_tokens),
            "last_block_id": None if hit_last_block_id is None else hit_last_block_id.hex(),
            "live_blocks": int(len(self.prefix_cache)),
        }

    def delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        result = self._require_prefix_cache().safe_delete_subtree(
            [int(token_id) for token_id in token_ids],
        )
        self._free_blocks(result.deleted_blocks)
        return result.to_dict()

    def set_eviction_priority(self, token_ids: list[int], *, priority: int) -> dict[str, object]:
        return self._require_prefix_cache().set_subtree_eviction_priority(
            [int(token_id) for token_id in token_ids],
            int(priority),
        )

    def stats(self) -> dict[str, int]:
        if self.prefix_cache is None:
            return {}
        accounting_bytes = 0
        recurrent_bytes = 0
        for block in self.prefix_cache.blocks.values():
            payload = block.payload
            if isinstance(payload, MixedPrefixBlockPayload):
                accounting_bytes += int(payload.accounting_bytes)
                recurrent_bytes += int(payload.recurrent_bytes)
        stats = self.prefix_cache.stats()
        stats["mixed_prefix_cache_accounting_bytes"] = int(accounting_bytes)
        stats["mixed_prefix_cache_recurrent_bytes"] = int(recurrent_bytes)
        stats["mixed_prefix_cache_pending_recurrent_bytes"] = int(
            self.pending_recurrent_bytes
        )
        stats["mixed_prefix_cache_max_recurrent_bytes"] = int(self.max_recurrent_bytes)
        stats["mixed_prefix_cache_skipped_capacity_blocks"] = int(self.skipped_capacity_blocks)
        stats["mixed_prefix_cache_evictable_slots"] = int(self.evictable_slots())
        return stats

    def evictable_slots(self) -> int:
        if self.prefix_cache is None:
            return 0
        return int(self.prefix_cache.freeable_blocks() * self.block_size)

    def prefix_hit_evictable_slots(self, seq: Sequence) -> int:
        if self.prefix_cache is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return 0
        if seq.prefix_cache_hit_last_block_id is None:
            raise RuntimeError(f"seq_id={seq.seq_id} has mixed prefix hit length but no last block id.")
        chain = self.prefix_cache.get_chain(
            seq.prefix_cache_hit_last_block_id,
            int(seq.prefix_cache_hit_block_count),
        )
        freeable_block_ids = self.prefix_cache.freeable_block_ids()
        return sum(
            self.block_size
            for block in chain
            if block.stable_block_id in freeable_block_ids
        )

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        seq.clear_prefix_cache_hit()
        if self.prefix_cache is None:
            return
        if seq.num_prefilled_tokens != 0 or seq.num_completion_tokens != 0:
            return
        usable_tokens = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.block_size)
        if usable_tokens <= 0:
            return
        with profiler.record("mixed_prefix_cache_lookup"):
            hit_len, last_block_id, hit_blocks = self.prefix_cache.lookup_longest_prefix(
                seq.prompt_token_ids,
                max_usable_tokens=usable_tokens,
            )
        if hit_len <= 0:
            return
        if last_block_id is None or hit_blocks <= 0:
            raise RuntimeError("Mixed prefix cache lookup returned an invalid hit.")
        if hit_len >= seq.num_prompt_tokens or hit_len % self.block_size != 0:
            raise RuntimeError(
                "Mixed prefix cache lookup returned an unusable hit length: "
                f"seq_id={seq.seq_id} hit_len={hit_len} prompt_len={seq.num_prompt_tokens} "
                f"block_size={self.block_size}."
            )
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(hit_len)
        seq.prefix_cache_hit_block_count = int(hit_blocks)
        seq.prefix_cache_hit_last_block_id = last_block_id
        seq.prefix_cache_block_size = self.block_size
        seq.prefix_cache_method = str(self.config.vllm_sparse_method or "")

    def attach_prefix_cache_hits(self, seqs: list[Sequence]) -> None:
        if self.prefix_cache is None:
            return
        for seq in seqs:
            self._attach_seq(seq)

    def _attach_seq(self, seq: Sequence) -> None:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        if hit_len <= 0 or seq.seq_id in self.seq_id_to_prefix_blocks:
            return
        if seq.prefix_cache_hit_last_block_id is None:
            raise RuntimeError(f"seq_id={seq.seq_id} has mixed prefix hit length but no last block id.")
        if hit_len % self.block_size != 0:
            raise RuntimeError(
                f"seq_id={seq.seq_id} mixed prefix hit length is not block aligned: "
                f"hit_len={hit_len} block_size={self.block_size}."
            )
        chain = self._require_prefix_cache().get_chain(
            seq.prefix_cache_hit_last_block_id,
            int(seq.prefix_cache_hit_block_count),
        )
        if len(chain) * self.block_size != hit_len:
            raise RuntimeError(
                "Mixed prefix cache chain length does not match scheduler metadata: "
                f"seq_id={seq.seq_id} hit_len={hit_len} blocks={len(chain)} block_size={self.block_size}."
            )

        with profiler.record("mixed_prefix_cache_attach"):
            for block in chain:
                payload = block.payload
                if not isinstance(payload, MixedPrefixBlockPayload):
                    raise RuntimeError("Mixed prefix cache block has an invalid payload.")
                self.cache_manager.attach_prefix_kv_payload(seq, payload.kv_payload)
                block.ref_count += 1
            last_payload = chain[-1].payload
            if not isinstance(last_payload, MixedPrefixBlockPayload):
                raise RuntimeError("Mixed prefix cache block has an invalid recurrent payload.")
            self.recurrent_state_manager.attach_prefix_recurrent_payload(
                seq,
                last_payload.recurrent_payload,
            )
            self.seq_id_to_prefix_blocks[int(seq.seq_id)] = chain
            self.prefix_cache.touch_chain(chain)

    def record_step_tokens(self, seqs: list[Sequence], is_prefill: bool) -> None:
        if self.prefix_cache is None:
            return
        for seq in seqs:
            if int(seq.seq_id) in self.capacity_limited_seq_ids:
                continue
            if is_prefill:
                chunk_size = int(seq.current_chunk_size or 0)
                if chunk_size <= 0:
                    continue
                start = int(seq.num_prefilled_tokens)
                end = start + chunk_size
                boundary = start + (self.block_size - (start % self.block_size))
                if start % self.block_size == 0:
                    boundary = start + self.block_size
                if end > boundary:
                    raise RuntimeError(
                        "Mixed prefix prefill chunks must not cross recurrent snapshot boundaries: "
                        f"seq_id={seq.seq_id} start={start} end={end} block_size={self.block_size}. "
                        "Schedule the prefill suffix up to the next prefix block boundary first."
                    )
                token_ids = seq.token_ids
                if len(token_ids) > chunk_size:
                    token_ids = token_ids[start:end]
                self._record_tokens(seq, [int(token_id) for token_id in token_ids])
            else:
                if seq.last_token is not None:
                    self._record_tokens(seq, [int(seq.last_token)])

    def _record_tokens(self, seq: Sequence, token_ids: list[int]) -> None:
        if not token_ids:
            return
        state = self.runtime_states.get(int(seq.seq_id))
        if state is None:
            hit_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)
            state = _MixedPrefixRuntimeState(
                parent_block_id=getattr(seq, "prefix_cache_hit_last_block_id", None),
                next_logical_block_idx=hit_blocks,
            )
            self.runtime_states[int(seq.seq_id)] = state

        pending = self.pending_blocks.setdefault(int(seq.seq_id), [])

        def add_block(block_tokens: list[int]) -> None:
            block_start = int(state.next_logical_block_idx) * self.block_size
            block_end = block_start + self.block_size
            stable_block_id = self.prefix_cache.stable_block_id(block_tokens, state.parent_block_id)
            existing = self.prefix_cache.get_block(stable_block_id)
            if existing is not None:
                self._hold_materialized_ref(seq, existing)
                state.parent_block_id = stable_block_id
                state.next_logical_block_idx += 1
                return
            if stable_block_id in self.pending_block_ids:
                duplicate_refs = self.pending_duplicate_refs.setdefault(
                    int(seq.seq_id),
                    [],
                )
                if stable_block_id not in duplicate_refs:
                    duplicate_refs.append(stable_block_id)
                state.parent_block_id = stable_block_id
                state.next_logical_block_idx += 1
                return

            recurrent_bytes = int(
                self.recurrent_state_manager.prefix_recurrent_snapshot_nbytes()
            )
            if not self._reserve_pending_block(stable_block_id, recurrent_bytes):
                self.capacity_limited_seq_ids.add(int(seq.seq_id))
                self.skipped_capacity_blocks += 1
                return
            recurrent_payload = None
            try:
                kv_payload = self.cache_manager.build_prefix_kv_payload(
                    seq,
                    block_start,
                    block_end,
                )
                recurrent_payload = self.recurrent_state_manager.build_prefix_recurrent_payload(
                    seq,
                    block_end,
                )
                actual_recurrent_bytes = int(
                    self.recurrent_state_manager.prefix_recurrent_payload_nbytes(
                        recurrent_payload
                    )
                )
                if actual_recurrent_bytes != recurrent_bytes:
                    raise RuntimeError(
                        "Mixed prefix recurrent snapshot bytes differ from the model declaration: "
                        f"declared={recurrent_bytes} actual={actual_recurrent_bytes}."
                    )
                accounting_bytes = int(
                    self.cache_manager.prefix_kv_payload_nbytes(kv_payload)
                ) + recurrent_bytes
                pending.append(
                    _PendingMixedPrefixBlock(
                        stable_block_id=stable_block_id,
                        parent_block_id=state.parent_block_id,
                        logical_block_idx=state.next_logical_block_idx,
                        token_ids=block_tokens,
                        payload=MixedPrefixBlockPayload(
                            kv_payload=kv_payload,
                            recurrent_payload=recurrent_payload,
                            token_count=self.block_size,
                            accounting_bytes=accounting_bytes,
                            recurrent_bytes=recurrent_bytes,
                        ),
                    )
                )
            except BaseException:
                if recurrent_payload is not None:
                    self.recurrent_state_manager.free_prefix_recurrent_payload(
                        recurrent_payload
                    )
                self._release_pending_reservation(stable_block_id, recurrent_bytes)
                raise
            state.parent_block_id = stable_block_id
            state.next_logical_block_idx += 1

        offset = 0
        if state.pending_tokens:
            need = self.block_size - len(state.pending_tokens)
            take = min(need, len(token_ids))
            state.pending_tokens.extend(token_ids[:take])
            offset = take
            if len(state.pending_tokens) == self.block_size:
                add_block(list(state.pending_tokens))
                state.pending_tokens = []
            else:
                return

        full_tokens = ((len(token_ids) - offset) // self.block_size) * self.block_size
        end_full = offset + full_tokens
        for block_start in range(offset, end_full, self.block_size):
            add_block(token_ids[block_start : block_start + self.block_size])
        state.pending_tokens = token_ids[end_full:] if end_full < len(token_ids) else []

    def commit_pending_blocks(self, seqs: list[Sequence]) -> None:
        if self.prefix_cache is None:
            return
        with profiler.record("mixed_prefix_cache_commit"):
            for seq in seqs:
                pending_blocks = self.pending_blocks.pop(int(seq.seq_id), [])
                materialized = self.seq_id_to_materialized_blocks.setdefault(int(seq.seq_id), [])
                for pending_idx, pending in enumerate(pending_blocks):
                    inserted = None
                    inserted_new = False
                    recurrent_released = False
                    try:
                        if self.prefix_cache.has_block(pending.stable_block_id):
                            raise RuntimeError(
                                "Mixed prefix block became duplicate after unique pending reservation."
                            )
                        block = PrefixCacheBlock(
                            stable_block_id=pending.stable_block_id,
                            parent_block_id=pending.parent_block_id,
                            block_size=self.block_size,
                            logical_block_idx=pending.logical_block_idx,
                            payload=pending.payload,
                            token_ids=tuple(pending.token_ids),
                        )
                        inserted = self.prefix_cache.insert_block(block)
                        if inserted is not block:
                            raise RuntimeError(
                                "Mixed prefix insertion returned an unexpected duplicate block."
                            )
                        inserted_new = True
                        inserted.ref_count = 1
                        materialized.append(inserted)
                        try:
                            self.cache_manager.mark_materialized_prefix_kv_payload(
                                seq,
                                pending.payload.kv_payload,
                            )
                        except BaseException:
                            self.cache_manager.rollback_materialized_prefix_kv_payload(
                                seq,
                                pending.payload.kv_payload,
                            )
                            materialized.remove(inserted)
                            inserted.ref_count = 0
                            self.prefix_cache.rollback_inserted_leaf(inserted)
                            self.recurrent_state_manager.free_prefix_recurrent_payload(
                                pending.payload.recurrent_payload
                            )
                            recurrent_released = True
                            inserted_new = False
                            raise
                    except BaseException:
                        if not inserted_new and not recurrent_released:
                            self.recurrent_state_manager.free_prefix_recurrent_payload(
                                pending.payload.recurrent_payload
                            )
                        for unprocessed in pending_blocks[pending_idx + 1 :]:
                            self.recurrent_state_manager.free_prefix_recurrent_payload(
                                unprocessed.payload.recurrent_payload
                            )
                            self._release_pending_reservation(
                                unprocessed.stable_block_id,
                                int(unprocessed.payload.recurrent_bytes),
                            )
                        self.pending_duplicate_refs.pop(int(seq.seq_id), None)
                        if not materialized:
                            self.seq_id_to_materialized_blocks.pop(
                                int(seq.seq_id),
                                None,
                            )
                        raise
                    finally:
                        self._release_pending_reservation(
                            pending.stable_block_id,
                            int(pending.payload.recurrent_bytes),
                        )
                for stable_block_id in self.pending_duplicate_refs.pop(
                    int(seq.seq_id),
                    [],
                ):
                    block = self.prefix_cache.get_block(stable_block_id)
                    if block is None:
                        raise RuntimeError(
                            "Mixed prefix duplicate reservation was not committed by its owner."
                        )
                    self._hold_materialized_ref(seq, block)

    def _hold_materialized_ref(
        self,
        seq: Sequence,
        block: PrefixCacheBlock,
    ) -> None:
        seq_id = int(seq.seq_id)
        held = [
            *self.seq_id_to_prefix_blocks.get(seq_id, []),
            *self.seq_id_to_materialized_blocks.get(seq_id, []),
        ]
        if any(existing.stable_block_id == block.stable_block_id for existing in held):
            return
        block.ref_count += 1
        self.seq_id_to_materialized_blocks.setdefault(seq_id, []).append(block)

    def _live_recurrent_bytes(self) -> int:
        total = 0
        for block in self._require_prefix_cache().blocks.values():
            payload = block.payload
            if not isinstance(payload, MixedPrefixBlockPayload):
                raise RuntimeError("Mixed prefix cache block has an invalid payload.")
            total += int(payload.recurrent_bytes)
        return int(total)

    def _reserve_pending_block(
        self,
        stable_block_id: bytes,
        recurrent_bytes: int,
    ) -> bool:
        recurrent_bytes = int(recurrent_bytes)
        if stable_block_id in self.pending_block_ids:
            return False
        if not self._evict_for_insert(
            1,
            incoming_recurrent_bytes=recurrent_bytes,
        ):
            return False
        self.pending_block_ids.add(stable_block_id)
        self.pending_recurrent_bytes += recurrent_bytes
        return True

    def _release_pending_reservation(
        self,
        stable_block_id: bytes,
        recurrent_bytes: int,
    ) -> None:
        if stable_block_id not in self.pending_block_ids:
            raise RuntimeError("Mixed prefix pending reservation is missing during release.")
        self.pending_block_ids.remove(stable_block_id)
        self.pending_recurrent_bytes -= int(recurrent_bytes)
        if self.pending_recurrent_bytes < 0:
            raise RuntimeError("Mixed prefix pending recurrent byte count became negative.")

    def _evict_for_insert(self, needed_blocks: int, *, incoming_recurrent_bytes: int) -> bool:
        incoming_recurrent_bytes = int(incoming_recurrent_bytes)
        if incoming_recurrent_bytes < 0:
            raise ValueError(
                f"incoming_recurrent_bytes must be >= 0, got {incoming_recurrent_bytes}."
            )
        if incoming_recurrent_bytes > self.max_recurrent_bytes:
            raise RuntimeError(
                "Mixed prefix recurrent payload exceeds the configured byte budget: "
                f"payload_bytes={incoming_recurrent_bytes} "
                f"max_bytes={self.max_recurrent_bytes}."
            )
        prefix_cache = self._require_prefix_cache()
        needed_blocks = int(needed_blocks)
        if prefix_cache.max_blocks is not None:
            over_capacity = (
                len(prefix_cache.blocks)
                + len(getattr(self, "pending_block_ids", ()))
                + needed_blocks
                - int(prefix_cache.max_blocks)
            )
            if over_capacity > 0:
                evicted = prefix_cache.evict_until_freeable(over_capacity)
                self._free_blocks(evicted)
                if len(evicted) != over_capacity:
                    return False
        while (
            self._live_recurrent_bytes()
            + int(getattr(self, "pending_recurrent_bytes", 0))
            + incoming_recurrent_bytes
            > self.max_recurrent_bytes
        ):
            byte_evicted = prefix_cache.evict_until_freeable(1)
            if not byte_evicted:
                return False
            self._free_blocks(byte_evicted)
        return True

    def evict_for_slots(self, needed_slots: int) -> None:
        if self.prefix_cache is None:
            return
        needed_slots = int(needed_slots)
        if needed_slots <= 0:
            return
        needed_blocks = (needed_slots + self.block_size - 1) // self.block_size
        evicted = self.prefix_cache.evict_until_freeable(needed_blocks)
        self._free_blocks(evicted)
        if len(evicted) != needed_blocks:
            raise RuntimeError(
                "Mixed prefix cache could not evict enough blocks for KV allocation: "
                f"needed_slots={needed_slots} block_size={self.block_size} "
                f"needed_blocks={needed_blocks} evicted_blocks={len(evicted)}."
            )

    def _free_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        payloads = [block.payload for block in blocks]
        if any(
            not isinstance(payload, MixedPrefixBlockPayload)
            for payload in payloads
        ):
            raise RuntimeError("Mixed prefix cache block has an invalid payload.")
        for payload in payloads:
            self.cache_manager.free_prefix_kv_payload(payload.kv_payload)
            self.recurrent_state_manager.free_prefix_recurrent_payload(payload.recurrent_payload)

    def release_seq(self, seq_id: int) -> None:
        seq_id = int(seq_id)
        for block in self.seq_id_to_prefix_blocks.pop(seq_id, []):
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError("Mixed prefix cache block ref_count became negative.")
        for block in self.seq_id_to_materialized_blocks.pop(seq_id, []):
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError("Mixed prefix cache block ref_count became negative.")
        self.runtime_states.pop(seq_id, None)
        for pending in self.pending_blocks.pop(seq_id, []):
            self.recurrent_state_manager.free_prefix_recurrent_payload(
                pending.payload.recurrent_payload
            )
            self._release_pending_reservation(
                pending.stable_block_id,
                int(pending.payload.recurrent_bytes),
            )
        self.pending_duplicate_refs.pop(seq_id, None)
        self.capacity_limited_seq_ids.discard(seq_id)

    def reset_after_warmup(self) -> None:
        if self.prefix_cache is None:
            return
        if self.seq_id_to_prefix_blocks or self.seq_id_to_materialized_blocks:
            raise RuntimeError("Cannot reset mixed prefix cache while sequences still reference blocks.")
        for pending_blocks in self.pending_blocks.values():
            for pending in pending_blocks:
                self.recurrent_state_manager.free_prefix_recurrent_payload(
                    pending.payload.recurrent_payload
                )
                self._release_pending_reservation(
                    pending.stable_block_id,
                    int(pending.payload.recurrent_bytes),
                )
        blocks = list(self.prefix_cache.blocks.values())
        referenced = [block for block in blocks if int(block.ref_count) != 0]
        if referenced:
            raise RuntimeError(
                "Cannot reset mixed prefix cache while blocks are referenced: "
                f"referenced_blocks={len(referenced)}."
            )
        self._free_blocks(blocks)
        self.prefix_cache = RadixPrefixIndex(
            block_size=self.block_size,
            fingerprint=build_prefix_cache_fingerprint(self.config, self.block_size),
            max_blocks=self.config.prefix_cache_max_blocks,
        )
        self.seq_id_to_prefix_blocks.clear()
        self.seq_id_to_materialized_blocks.clear()
        self.runtime_states.clear()
        self.pending_blocks.clear()
        self.pending_duplicate_refs.clear()
        self.pending_block_ids.clear()
        self.pending_recurrent_bytes = 0
        self.capacity_limited_seq_ids.clear()
        self.skipped_capacity_blocks = 0
