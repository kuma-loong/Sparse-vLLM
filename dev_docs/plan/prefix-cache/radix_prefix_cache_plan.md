# Block-Level Radix Prefix Cache Implementation Plan

## Summary

Replace the current hash-chain prefix cache with a block-level `RadixPrefixIndex` that keeps matching generic and payloads method-owned. The first implementation supports only vanilla, OmniKV, and QuEST, preserves current prefill/attach semantics, adds subtree control APIs, and keeps future C++ tree replacement possible.

## Core Data Model

- Replace `PrefixCacheIndex` with `RadixPrefixIndex`.
- Add `RadixTreeBackend` as the internal tree engine; no `Python` prefix in class name.
- `RadixPrefixIndex` owns namespace fingerprinting, parent-sensitive stable block id generation, the block table, commit/delete/evict stats, and token-to-block conversion.
- `RadixTreeBackend` owns only block-id tree shape: lookup, insert, leaf removal, path lookup, subtree listing, child count, leaf listing, and tree stats.
- Matching is strictly block-level: token ids are split into full blocks, edge segments store stable block ids, and node split happens only between block ids.

## Prefix Block And Payload

- Define a thin public `PrefixBlockPayload` protocol.
- Redefine `PrefixCacheBlock` as generic metadata only: stable id, parent id, block size, logical index, payload, token ids, ref count, last access, and eviction priority.
- Remove public method-specific fields: no `slots`, `page_slot`, `page_slots`, or `child_keys`.
- Keep payload dataclasses method-local:
  - `StandardPrefixBlockPayload(token_slots)`
  - `QuestPrefixBlockPayload(block_slot, token_slots)`
- QuEST remains block-level in the shared framework and aliases a block to its chunk bookkeeping internally.

## Runtime Contract

- Rename `Sequence` fields to `prefix_cache_hit_block_count` and `prefix_cache_hit_last_block_id`.
- Preserve scheduler behavior: fresh prompt lookup, hit-length admission reduction, and `num_prefilled_tokens = prefix_cache_hit_len` on admission.
- Do not implement LPM scheduling or in-batch builder in v1.
- Preserve unfinished request cache only at chunk boundary: commit complete blocks after successful prefill forward.

## Control Plane

- Add default-on, no-auth endpoints:
  - `POST /v1/prefix_cache/inspect`
  - `POST /v1/prefix_cache/delete_subtree`
  - `POST /v1/prefix_cache/set_eviction_priority`
- Endpoints are synchronous and run through the dispatcher control queue, not direct HTTP mutation.
- Prefix selector accepts exactly one of `token_ids` or exact `text`, tokenized server-side.
- Inspect returns primitive block state only: block id, logical block index, ref count, eviction priority, child count, and last access.
- Delete is best-effort safe subtree deletion, deepest blocks first, returning deleted and blocked block ids with reasons.
- Setting priority writes a numeric `eviction_priority` to each block in the matched subtree.
- `eviction_priority < 0` is hard protection, `0` is default, positive values prefer eviction.

## Stats

- Use semantically accurate stats: lookup/hit counters, committed/duplicate commits, deleted/evicted blocks, live/referenced/negative-priority/leaf blocks, tree nodes/edges, and control request counters.
- Update tests and benchmark readers instead of preserving misleading old stat names.

## Test Plan

- `RadixPrefixIndex`: parent-sensitive ids, block-level lookup, block-boundary splits, duplicate commits, leaf-only eviction, priority ordering, negative priority protection, and partial subtree delete.
- Standard/OmniKV: attach writes token slots, `free_seq()` preserves shared slots, safe delete releases eligible token slots, and referenced blocks block delete.
- QuEST: payload is method-local, block size equals `quest_chunk_size`, attach updates QuEST internal block/chunk mapping and token slots, and safe delete releases eligible QuEST block slots.
- Scheduler/model: prefix hit reduces prefill work, preempted completion replay avoids fresh lookup, and blocks commit only after successful forward.
- API server: selector validation, primitive inspect response, synchronous dispatcher control, delete/priority mutation, disabled-cache errors, and TP failure behavior with fakes where practical.

## Out Of Scope

- Token-level radix matching.
- LPM scheduling.
- In-batch prefix builder.
- DeltaKV/SnapKV/StreamingLLM prefix payloads.
- Global prefix cache reset.
- Auth for prefix-cache control API.
- Pagination for full subtree listing.
- C++ backend implementation.
- Runtime hash/radix switch.
