# Copy-synchronize accounting, 2026-10-02 (where the prefill pass actually goes)

Supersedes the mechanism claim in `FINDINGS.md` and `overlap-design.md`. The
conclusion those two reached — prefill is an overlap problem — survives and is
now measured. The line they blamed for it does not.

## What was measured

`LLAMA_VK_COPY_SYNC_LOG=1` (this session, unified tree `321dff9b5` + the
accounting block) times every staging-fallback `ggml_vk_synchronize()` that
`ggml_backend_vk_set_tensor_2d_async` makes, and splits it into the host
`in_memcpys` drain and the blocking fence wait. `scripts/bench-copy-sync-probe-1002.sh`.
Gate-C condition throughout: one-pass 14,960 tok, ubatch 16384, ctx 16384,
`--cpu-moe --load-mode mmap`, greedy temp-0 seed 42, read_ahead_kb 4096.

**Parity: all four arms byte-identical to the 10-02 `pd-stock-a` reference.**
The accounting is behaviour-neutral.

| depth | t/s | pass | drain | fence | drain GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (stock) | 159.0 | 94 s | 29.8 s | 44.2 s | 1.58 |
| 8 | 164.3 | 91 s | 23.2 s | 44.0 s | 2.04 |
| 16 | 173.8 | 86 s | 23.6 s | 44.1 s | 2.00 |
| 32 | 168.0 | 89 s | 25.3 s | 43.2 s | 1.87 |

Identical across all four: **13,696 staging copies, 47.2 GB, same size
histogram.** Only the read strategy differs, which is what makes the drain
column comparable run to run.

## The serialization is a fence, not the staging buffer

`overlap-design.md` blames `sync_staging` being a singleton and plans to pool
it with per-slot fences. `set_tensor_2d_async` ends its staging fallback with
`ggml_vk_synchronize(ctx)` (`ggml-vulkan.cpp:17362`), which reaches
`ggml_vk_wait_for_fence` and blocks the calling thread on the GPU fence after
**every** large copy. `git blame` puts that on `5266f24da` — upstream, not
this project. `xferq=1` in every line confirms the transfer queue is live, so
this is not the "transfer queue off" path.

Pooling the staging buffer alone cannot help: the CPU never runs far enough
ahead to reach slot 1. The pool is necessary — it is what makes it *safe* to
stop waiting — but it is not sufficient. The wait has to go too.

## Depth is a real lever that t/s could not see

Drain falls 29.8 → 23.2 s (−22%) from depth 1 to depth 8 and is flat after that.
Fence is flat at ~44 s at every depth. So queue depth buys ~6.6 s of host read
time and then saturates — the staging read path tops out near **2.0 GB/s** for
this access pattern, not the 4.09 GB/s `probe-drive-rate.py` reports for
sequential reads.

`FINDINGS.md` called the pread-depth lever neutral because 159.0 / 164.3 / 173.8
/ 168.0 t/s spans 14.8 t/s — inside the documented ±13 t/s band. The drain
column is not inside that band. The lever was real; t/s simply could not
resolve it. **Measure the mechanism, not the score, when the noise band is
wider than the effect.**

## The pass decomposes

`drain ≤ sync` holds in every one of the 215 accounting lines of both arms, so
all host-read time happens inside `set_tensor_2d_async`. For the depth-8 arm:

- **23.2 s** host read drain (CPU-bound; `read_bytes` confirms real I/O)
- **44.0 s** fence wait (GPU-bound)
- **~24 s** outside both — graph build, command recording, submission

Two independent instruments agree on the GPU term: the fence timer says 44.0 s,
`gpu_busy_percent` over the same window says 52.2% of 86 s = 44.9 s.

## Two read paths, not one

The pass reads 80–84 GB, but only **47.2 GB goes through the staging path**.
The other ~33–36 GB (**~42%**) never touches it. That is consistent with the
09-29 finding that prompt-path `mul_mat_id` runs on Vulkan and streams the mmap
through uncontrolled GPU page faults. It also explains why `read_ahead_kb=4096`
was worth 48.8 → 168.9 t/s: it accelerates *that* path, not the staged copies.

## The size model in the docs is wrong

`FINDINGS.md` and `FINDINGS-EVENING.md` both model the pass as "144 tensors ×
~450 MB ≈ 68–77 GB". The histogram says otherwise:

| bucket | calls | bytes |
| --- | ---: | ---: |
| < 1 MB | 3,718 | 3.3 GB |
| 1–8 MB | 8,952 | 32.0 GB |
| 8–64 MB | 1,026 | 12.0 GB |
| > 64 MB | **0** | **0** |

13,696 copies averaging **3.4 MB**, and nothing above 64 MB. There is no
450 MB tensor in this stream. This is why pread depth 8 helps only modestly:
it is splitting ~3.4 MB calls into ~0.4 MB reads, and `ggml_vk_pread_parallel`
only engages at ≥ 4 MB, so most of the volume never reaches it at all. The
"only 28.9 of 68 GB reaches the pread path" note from the 10-01 commit is this
same fact, mis-attributed.

## What the overlap build is now worth

Removing the per-copy wait and pooling the staging buffers hides the drain
under the GPU term:

- depth 8 today: 91 s → ~240 t/s at best if drain fully hides (`24 s other +
  44 s GPU`)
- depth 16 today: 86 s → ~240 t/s

So **overlap alone projects to roughly 220–240 t/s, not 300.** The 300 t/s
target additionally requires attacking the ~18–24 s of CPU time that is neither
a read nor a fence — graph construction and submission. That is a separate
problem and should not be folded into this one.

## Corrections to earlier documents

1. **`FINDINGS.md`'s "1.43–1.55 GB/s sustained" is a windowing artifact.**
   `bench-prefill-depth-profile-1002.sh` sums `read_bytes` to end-of-request
   but divides by the prompt-only wall time. On `pd-stock-a` (87.1 s prompt +
   14.6 s decode): prompt window **82.2 GB = 0.94 GB/s**, decode window
   52.9 GB = 3.56 GB/s. The real prefill rate is ~0.9 GB/s, i.e. *below* the
   1.37 GB/s that 300 t/s requires — not above it. The "copy stream is no
   longer the constraint" conclusion rests on this and does not hold.
2. **"Half the copy volume is redundant" does not hold.** 82 GB read against
   ~47 GB staged is not 2×; and ~42% of the read never passes through the
   staging path at all (above), which is a different question from redundancy.
3. **`gpu_busy_percent` ≈ 55% is sound.** It reproduces under correct
   windowing; the low-utilization decode phase is short next to the prompt.

## Next build

Not "pool the staging buffer". Both halves, or neither:

1. `sync_staging` → a pool of N slots (default 3, MemAvailable-clamped), each
   with its own fence; round-robin allocation waits only on the slot it reuses.
2. In `set_tensor_2d_async`, drain `in_memcpys` and submit the transfer context
   against that slot's fence **without** the blocking
   `ggml_vk_wait_for_fence`.

Gates: byte-identical greedy output vs stock with arms on both sides (the
control spread here is ~15 t/s); guarded first run; success criterion is GPU
busy% and drain-seconds falling, not t/s alone. Expected signal: `drain` stays
~23 s of *wall* time but stops costing pass time, and GPU busy% rises from
~52% toward the ~70% that a fully overlapped pass implies.