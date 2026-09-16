# ADR-0003 — A content-addressed prefix cache, not a radix tree

**Status:** accepted.
**Date:** 2026-09.

## Context

Multi-tenant traffic repeats itself: a long system prompt shared by every request of a
tenant, a few-shot preamble, a conversation replayed with one more turn. Every repeated
token whose KV is already in the pool is a token that does not need a forward pass.

Two data structures are used in practice:

- **Radix tree over token ids.** Shared prefixes are literal tree paths; matching walks the
  tree token by token; eviction is leaf-first.
- **Hash chain over full blocks.** Block *i*'s identity is
  `h_i = H(h_{i-1} || lora_id || tokens_i)`, defined only for *full* blocks; matching is a
  sequence of dictionary lookups; eviction is LRU over unreferenced blocks.

## Decision

The hash chain, implemented in `engine/core/prefix_cache.py` with blake2b-16 and a
domain-separated root constant, and wired to the allocator through the two-method
`BlockRecycler` protocol.

## Consequences

- **The allocator stays ignorant.** It knows three states — free, in use, retained — and
  asks the recycler two questions: "may I free this block whose last reference went away?"
  and "the free list is empty, give me back *n*". Prefix caching can be switched off by
  passing no recycler, and the allocator's own tests do not import the cache at all.
- **Matching is `O(blocks)` dictionary lookups on the host**, with no per-token descent and
  no tree rebalancing on insert or evict.
- **`lora_id` in the chain is a security property, not an optimisation.** Two tenants with
  different adapters compute different KV for the same tokens, so they must never share a
  block; mixing the adapter id into the hash makes that impossible by construction rather
  than by a check someone can forget.
- **Only full blocks are cacheable.** A partially filled tail block is never hashed, so the
  matched prefix is always a multiple of `block_size`. The cost is at most `block_size - 1`
  recomputed tokens; the benefit is that a block's hash is fixed forever once written.
- **Eviction is plain LRU and does not prefer leaves.** Evicting an interior block can
  strand still-cached descendants: they remain in the map but can never be matched, and
  they age out. Harmless, and it keeps eviction `O(1)` — but it is a real difference from a
  radix tree and it is stated in `prefix-caching.md`.
- **No cross-block deduplication within a step.** Two sequences that compute the same new
  prefix in the *same* step each keep their own block; the loser's simply does not get
  indexed.

## Alternatives considered

- **Radix tree.** Rejected on coupling, not on merit: it wants to own allocation, eviction
  order and matching together, which would put scheduling policy inside the memory
  allocator. Its leaf-first eviction is genuinely better than LRU; that is the price paid.
- **Hash the whole prompt, cache whole-prompt hits.** Rejected: it captures only exact
  repeats, and misses the common case of a shared preamble with a different tail.
- **Hash without `lora_id`, check the adapter on hit.** Rejected: a correctness property
  enforced by a conditional is a correctness property waiting for a refactor.
