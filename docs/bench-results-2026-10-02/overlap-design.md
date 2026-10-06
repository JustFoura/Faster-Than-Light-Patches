# Prefill overlap build — design (2026-10-02)

From `FINDINGS.md`: prefill is an **overlap** problem, not a read-rate problem.
~48 s of GPU work and ~39 s of non-GPU time in an ~87 s pass run serialized
against each other. The copy already sustains 1.43–1.55 GB/s, at or above the
1.37 GB/s that 300 t/s requires. Target: keep the GPU busy for the whole pass.

## The structural cause

`ggml_backend_vk_context::sync_staging` and `vk_device::sync_staging` are
**single** host-visible buffers (`ggml-vulkan.cpp:1167`, `:2448`), grown on
demand by `ggml_vk_ensure_sync_staging_buffer` (`:8453`, `:8463`).

Every large host→device copy does:

1. `ggml_backend_vk_set_tensor_2d_async` (`:17264`) records the copyBuffer from
   `sync_staging` into the destination, and parks the host→staging memcpy in
   `cpy_ctx->in_memcpys` (`:17312`).
2. `ggml_vk_submit_transfer_ctx` (`:8399`) drains `in_memcpys` into the one
   shared staging buffer, submits, and signals the transfer semaphore.
3. Compute waits on that semaphore before running (`ggml_vk_get_compute_ctx`,
   `:8360`).

Because there is only one staging buffer, copy N+1 cannot begin its host→staging
memcpy until copy N's `copyBuffer` has been consumed. That is the serialization.
The transfer queue already gives *queue-level* overlap, but the single staging
buffer caps it at one copy deep.

## The change

Turn the singleton into a small pool with a per-slot fence:

- `sync_staging_pool[]` of N slots (env-gated, default 3), each a host-visible
  buffer plus its own `vk::Fence`.
- The round-robin allocator hands out slot `k % N`. Before reusing slot k it
  waits on that slot's fence — i.e. waits for the copy that last used it, not
  for the most recent copy. With N=3, split N+1's memcpy into staging happens
  while splits N and N−1 are still being consumed by the GPU.
- Pool size is clamped against MemAvailable so it cannot reproduce the 10-01
  desktop freeze. Default 3 × ~450 MB = ~1.3 GB against ~10 GB available.

## Where the copy volume comes from

Per pass, 144 expert tensors × ~450 MB ≈ 68 GB, and every expert is routed
(P(untouched) ~1e-129), so it is irreducible. The measured 134 GB of
`read_bytes` is ~2x that and unexplained — instrument which tensors are copied
and how many times; halving redundancy is free throughput and does not depend on
the overlap landing.

## Gates

- Byte-identical greedy output vs stock, order-controlled, arms on both sides
  (the 10-02 control spread was 12.9 t/s, so a one-sided arm proves nothing).
- Guarded first run (MemAvailable floor) — this changes buffer placement.
- Report GPU busy in the pass window. The success criterion is **GPU busy% up**,
  not just t/s: if busy% does not rise, the overlap is not happening and the
  build is wrong regardless of the throughput number.