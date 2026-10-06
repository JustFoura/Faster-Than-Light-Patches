# Copy pool: built, correct, and not yet a win (2026-10-02)

Third block of the prefill copy work, after
[`copy-sync-findings.md`](copy-sync-findings.md) (where the time goes) and the
10-02 `overlap-design.md` (which plans to pool `sync_staging`). This records
what the pool actually did when built.

## Status: correct, no measured throughput win

`LLAMA_VK_COPY_POOL=<n>` gives the scheduler's host→device expert-weight copies
n staging slots with per-batch fences, and submits **without** the per-copy
`ggml_vk_synchronize()` that makes the stream one copy deep. Default 0, which is
upstream's path untouched.

**Parity: PASS byte-identical on every arm of every run**, including against the
10-02 `pd-stock-a` reference. The mechanism works.

Throughput: **no win yet.** Read the numbers below before believing any of them.

## Run 1 — per-slot pool (one submit + one fence per copy)

| arm | t/s | pass |
| --- | ---: | ---: |
| ctl-a | 166.8 | 89.7 s |
| p4-a | 173.5 | 86.2 s |
| ctl-b | 175.7 | 85.2 s |
| p8-a | 177.3 | 84.4 s |
| ctl-c | 176.4 | 84.8 s |

Controls 166.8–176.4; the treatments sit inside that. Pool log:

```
acquires=13716 blocked=13712 (0.0% ran ahead) slots=4
acquires=13716 blocked=13708 (0.1% ran ahead) slots=8
```

**The CPU blocks on essentially every acquire.** Slots alone cannot help when
every copy still costs a full submit + fence round trip and there are ~13,700 of
them per pass. GPU busy is unchanged (~40 s) across control, 4 and 8 slots.
That is what pointed at batching.

## Run 2 — batched pool: both arms hung, two real bugs

`p16` and `p32` loaded the model, took the request, and stopped dead at the
first copy drain. Controls on the same binary were fine, so it was the pool.

1. **Use after command-buffer end.** `ggml_vk_copy_slot_acquire()` flushed when
   it ran out of slots; flush → `ggml_vk_submit_transfer_ctx` → `ggml_vk_ctx_end(cpy_ctx)`.
   But `cpy_ctx` had already been taken in `set_tensor_2d_async`, and this
   copy's `copyBuffer` had not been recorded yet — so the caller recorded into an
   already-ended command buffer. Split into `ggml_vk_copy_pool_reserve()`
   (blocks; must run **before** a context is taken) and
   `ggml_vk_copy_slot_take()` (claims; safe with a live command buffer).
2. **Semaphore ordering.** Compute waits on `transfer_semaphore` at the value
   captured inside `ggml_vk_get_compute_ctx()`. A copy batch still open at that
   moment is submitted *after* the capture, so that compute never waits for it.
   The batch is now flushed at the top of `ggml_vk_get_compute_ctx`, before the
   value is read.

## Run 3 — batched pool runs, parity PASS, batching does not engage

| arm | t/s | pass |
| --- | ---: | ---: |
| ctl-a | 114.7 | 130.5 s |
| p16 | 172.2 | 86.9 s |
| ctl-b | 147.4 | 101.5 s |
| p32 | 168.9 | 88.6 s |
| ctl-c | 176.5 | 84.7 s |

All five byte-identical. **This run's throughput numbers are unusable**: controls
span 114.7–176.5 t/s, far outside the ±13 t/s band, and `ctl-a` shows 96% GPU
busy at 114 t/s. Cause: the box had ~7 GB in swap with `kswapd0` active, left by
the hung 32-slot arms of run 2. Swapped mmap pages of a 97 GB model re-read
through swap, and every arm after it inherits the state.

One number from this run is still trustworthy, because it is a counter and not a
throughput measurement:

```
p16: acquires=13716 blocked=12632 batches=12648 avg_batch=1.1 slots=16
p32: acquires=13716 blocked=11083 batches=11114 avg_batch=1.2 slots=32
```

**avg_batch ≈ 1.** The correctness flush in `ggml_vk_get_compute_ctx` fires
between nearly every copy, so a batch never gets past one member. `ggml_vk_get_compute_ctx`
is evidently called ~13,000 times per pass — the copies are interleaved with
compute contexts at fine grain, not issued as a bulk per split.

## What this says about the design

The design note was right that staging must be pooled. It was wrong that pooling
is the thing that unblocks overlap — and this run shows a second, independent
problem behind that one:

> The copy stream is not a bulk phase that can be batched and overlapped. Copies
> and compute contexts alternate roughly one-for-one. So removing the per-copy
> fence is necessary but not sufficient: with ~1 copy per compute context, there
> is no window in which to read ahead, no matter how many slots exist.

That reframes the next build. Overlap has to come from **prefetching a copy
before the compute that consumes it**, not from staging a batch ahead of a bulk
copy phase — the bulk phase does not exist.

## Next

1. **Measure the interleaving directly** rather than inferring it: count
   `ggml_vk_get_compute_ctx` calls per pass, and the copy count between
   consecutive ones. If it really is ~1:1, batching is dead and prefetch is the
   only route. This is a counter, so it survives the noise that throughput does not.
2. **Get the measurement environment under control.** Swap state swung controls
   from 176 t/s to 114 t/s. Before any further throughput comparison: settle
   swap, and have each arm start from a comparable memory state. A control that
   reads 114 t/s is the signal that the box is not comparable, not a data point.
3. Then prefetch: issue copy N+1 at the point copy N is consumed, which is the
   only place the dependency allows.

## Bookkeeping

- llama.cpp: staging-pool development change and follow-up pool work,
  uncommitted at time of writing. `LLAMA_VK_COPY_POOL=<n>`, default 0.
- Harness: `scripts/bench-copy-pool-1002.sh`. Artifacts in this folder.
- Parity has held on every arm; the change is safe to keep, it just is not fast yet.
