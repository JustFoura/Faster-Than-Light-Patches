# Copy batch size + drain concurrency: 172 → 202 t/s (2026-10-02, session 3)

Third block of the prefill copy work, after [`copy-sync-findings.md`](copy-sync-findings.md)
(where the time goes) and [`copy-pool-findings.md`](copy-pool-findings.md) (the pool built,
parity PASS, no win). This block finds the win, and it was in the pool build the previous
block had already written off.

Gate-C condition throughout: one-pass 14,960 tok, ubatch 16384, ctx 16384, `--cpu-moe
--load-mode mmap`, greedy temp-0 seed 42, `read_ahead_kb=4096`, parity byte-identical against
the 10-02 `pd-stock-a` reference on every arm.

## Result

| step | t/s | pass | gpu_busy (prompt window) |
| --- | ---: | ---: | ---: |
| pool as committed (`avg_batch` 1.1) | ~172 | 86 s | ~53% |
| + `reserve()` retires before flushing | 176.0 | 85 s | 53.6% |
| + 8-way batch drain | 188.7 / 183.0 | 79 / 82 s | 60.1% / 58.8% |
| + pread floor 4 MiB → 256 KiB | **202.2 / 201.3** | **74 s** | **66.0% / 65.7%** |

**+17% prefill, and `retire_fence` is now 0.0 s on every arm** — the copy path never blocks on
the GPU fence, which is the definition of the overlap the pool was built for.

## 1. `avg_batch ≈ 1` was the recovery path, not the caller

`copy-pool-findings.md` concluded:

> the correctness flush in `ggml_vk_get_compute_ctx` fires between nearly every copy, so
> `ggml_vk_get_compute_ctx` is evidently called ~13,000 times per pass... copies and compute
> contexts alternate about 1:1, so there is no bulk copy phase to batch or overlap.

Its own counters contradict that. `batches` (12,648) tracked `blocked` (12,632), not
`acquires / batch_max`. At n=16 with `copy_batch_max=8`, batching working looks like
`blocked ≈ 1,700` and `avg_batch ≈ 8`. Neither happened.

The cause is the order inside `ggml_vk_copy_pool_reserve()`, which was flush-then-retire:

```cpp
g_copy_pool_blocked++;
ggml_vk_copy_pool_flush(ctx);          // pushes the ONE pending slot as its own batch
ggml_vk_copy_pool_retire_oldest(ctx);
```

At the moment the pool runs dry, `copy_batch` holds exactly one un-submitted slot. So every
reclaim pushed a one-member batch before reclaiming anything — the batch was destroyed by its
own recovery path, and `avg_batch` was pinned at 1 regardless of what the caller did.

The flush blamed in the write-up is a **no-op**. `ggml_vk_get_compute_ctx` really is called
~7,420 times per pass (once per graph node, `ggml-vulkan.cpp:16195`, inside
`ggml_vk_build_graph`), but the scheduler issues every `set_tensor` copy *between* graph
computes, so `copy_batch` is empty during node iteration and every one of those flushes
returns before touching the batch counter. The `flush_by` counters now say so directly:

```
flush_by: get_compute_ctx=0 reserve=0 batch_full=1650 synchronize=3 graph_compute=126
```

**Fix:** retire the oldest in-flight batch first; only submit the open batch when nothing is
left in flight to reclaim. Predicted `avg_batch ≈ 8`, `blocked ≈ 1,700`; measured **7.7** and
**1,777**. Fence round-trips per pass fell from 12,632 to 1,777.

**This falsifies "there is no bulk copy phase".** The scheduler's per-split-input
synchronizes (~435/pass against 145 graph splits) leave runs of ~31 copies between them,
comfortably above `copy_batch_max=8`. The `get_compute_ctx` count the write-up proposed as its
next measurement would have come back at ~7,400 and confirmed the wrong hypothesis.

## 2. Drain the batch concurrently, not one copy at a time

`ggml_vk_submit_transfer_ctx` drained `in_memcpys` sequentially — one memcpy at a time, each
blocking on its own page faults, so the read stream was queue-depth-1 per copy. But a batch
already holds up to `copy_batch_max` **independent** copies, each writing its own staging
slot. `ggml_vk_drain_memcpys()` runs them across `LLAMA_VK_COPY_DRAIN_THREADS` workers.

`LLAMA_VK_COPY_DRAIN_THREADS`: 1 = old sequential drain, default 8.

Order-controlled, alternating:

| arm | threads | t/s | pass | gpu_busy | drain |
| --- | ---: | ---: | ---: | ---: | ---: |
| d1 | 1 | 170.5 | 87.7 s | 54.2% | 41.5 s @ 1.14 GB/s |
| d8 | 8 | **188.7** | 79 s | 60.1% | 27.7 s @ 1.71 GB/s |
| d1b | 1 | 141.3 | 106 s | 46.6% | 20.4 s @ 2.32 GB/s |
| d8b | 8 | **183.0** | 82 s | 58.8% | 18.5 s @ 2.56 GB/s |

Both controls lose to both treatments, and gpu_busy% separates them cleanly where t/s does
not (the control band is 29 t/s wide here). **8 is the ceiling** — 16/24/32 all regress
(164.9 / 174.9 / 176.9) on 8 physical cores.

## 3. The 4 MiB pread floor was excluding most of the stream

`ggml_vk_pread_parallel` bailed below 4 MiB, but the copy histogram averages **3.4 MB**
(3,718 calls under 1 MB, 8,952 in 1–8 MB, nothing above 64 MB). Most of the 47.3 GB was
falling back to a single-threaded memcpy fault stream. That floor was set when a copy cost a
full thread setup on its own; with a whole batch drained at once the setup is amortised, so
the floor can drop.

`LLAMA_VK_COPY_PREAD_MIN_KB` (default 4096 = old behaviour).

| arm | floor | t/s | pass | gpu_busy | drain rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| m4096 | 4 MiB | 177.5 | 84 s | 58.6% | 2.69 GB/s |
| m256 | 256 KiB | **202.2** | 74 s | 66.0% | 3.19 GB/s |
| m4096b | 4 MiB | 193.7 | 77 s | 62.9% | 2.50 GB/s |
| m256b | 256 KiB | **201.3** | 74 s | 65.7% | 3.23 GB/s |

Matched pairs both favour 256 KiB, and the drain rate moves 2.5–2.7 → 3.2 GB/s.

## Pool size is a wash once batching works

n=16 / 32 / 48, matched arms: **192.5 / 191.9 / 199.4 / 206.9 t/s** (gpu_busy 62.9 / 63.0 /
64.8 / 66.7%), drain 13.5–15.8 s. All inside the run-to-run band. What matters is
`copy_batch_max` and the drain width, not the slot total — the pool stops being the
constraint once batches form, so 16 slots (batch 8) is enough and larger buys nothing for
~2 GB more host-visible memory.

Tuning at the 256 KiB floor, four arms: drain 8 / pread depth 4 → **201.0 t/s** (gpu_busy
65.4%, drain 13.8 s @ 3.43 GB/s) is the best single reading; depth 8 and depth 4 are
equivalent within noise (195–207 cluster), drain 16 is consistently worse on 8 cores.

## Two things found outside the bench, not shipped

**`--moe-expert-cache 88` at 200k ctx does not boot on this machine.** The shipped launcher
config aborts during model load with `radv/amdgpu: Not enough memory for command submission`
→ `vk::DeviceLostError`. Isolated by bisection, all at ubatch 4096:

| ctx | expert-cache slots | result |
| --- | ---: | --- |
| 200000 | 88 | **abort — RADV out of memory** |
| 200000 | 64 | healthy |
| 200000 | 48 | healthy |
| 200000 | none | healthy |
| 16384 | 88 | healthy |

It is the *combination*: 88 slots alone fits, 200k ctx alone fits, together they do not. The
q8 KV (~2.7 GB GTT hostmem at 200k) and the slot cache together overrun what RADV will
commit. **64 slots is the ceiling that boots at 200k** — untested between 64 and 88.
Reproduces identically with `LLAMA_VK_COPY_POOL=0`, so it predates the copy work.

Not applied: the launcher is deliberately left at its shipped state until the speed targets
are met. This is recorded so the fix is one env var away (`FLASH_MOE_CACHE_SLOTS=64`).

**Real-config prefill**, launcher defaults (ctx 200000, ubatch 4096, cache 64, pool 16):
**134.6 t/s cold / 134.7 t/s warm** on the same 14,960-token prompt. The gap to the 202 t/s
bench number is the ubatch width, not the copy path: at ubatch 4096 a 15k prompt takes ~4
passes over the expert set instead of one, so it reads ~4x the bytes for the same GPU work.

## Where the pass goes now

At 202 t/s: **74 s = 48.8 s GPU busy + 14.8 s drain + ~10 s other.**

The GPU term is now the floor. 14,960 tok / 48.8 s = **306 t/s** if the drain hides entirely
and the residual goes to zero. So:

- **~300 t/s is the copy-overlap target and it is now the whole remaining gap.** The drain has
  to move off the critical path, which means staging split N+1's experts while the GPU runs
  split N. The blocker is `ggml_backend_sched_compute_splits` calling
  `ggml_backend_synchronize(split_backend)` once per split input, because with
  `sched->n_copies == 1` there are no events (`ggml-backend.cpp:1909` allocates events only
  when `n_copies > 1`) and `input_cpy` cannot be double-buffered. Forcing
  `parallel = true` allocates `GGML_SCHED_MAX_COPIES` copies of the ~77 GB of expert tensors —
  that is the VRAM blowup, and with the 10-01 desktop freeze in the history it is not a
  change to make blind.
- **Above ~306 t/s requires cutting GPU time**, not read time. The shader is the binding
  constraint and nothing in this stream touches it.

## Bookkeeping

- llama.cpp follow-up staging-pool changes, uncommitted at time of writing: `reserve()` ordering,
  `flush_by` attribution counters, `ggml_vk_drain_memcpys`, `LLAMA_VK_COPY_DRAIN_THREADS`,
  `LLAMA_VK_COPY_PREAD_MIN_KB`. Parity PASS on every arm of every run.
- Harness: `scripts/bench-copy-batch-1002.sh`
  (`<label>:<slots>:<pread_depth>:<drain_threads>:<pread_min_kb>`). Artifacts in this folder
  as `cb-*.json`, `cb-*.server.log`, `cb-*.gpu.csv`.
- Swap state affected the measured control spread; benchmark comparisons should record and
  normalize memory pressure rather than changing host swap settings as part of the harness.
