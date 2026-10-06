# Session findings, 2026-10-01 evening (same boot as the parity gate, ~4 h uptime)

> Number policy (user, night of 10-01): steady-state is the score. The numbers below
> (6.5–7.0 t/s decode at ~4 h uptime, 160–176 t/s prefill warm) are the REAL speeds —
> fresh-boot figures are diagnostics only and are not the target. Closing the
> fresh-boot-to-steady gap (~9.5 → ~6.8 decode decay) is an optimization target in
> itself, not an excuse.

Drivers and raw artifacts all in this folder. Order-controlled arms, 256-token
warm-pair decode protocol (ignore_eos), gate-C condition prefill (one-pass
14,963 tok, ubatch 16384, stock fault path), GPU-busy and CPU sampling added
this session.

## Decode: every remaining env knob is closed

Controls through the evening (base, single-text profile, fanout 24):
**6.51 / 6.79 / 7.04 t/s** — consistent with the documented evening band
(5.6–6.6 after benching; fresh boot is ~9.2–9.9).

| Arm | t/s | Verdict |
| --- | ---: | --- |
| multi-text profile (`/tmp/moe_routes_multi.log`) | 4.67 (hit 58.7%) | **LOSES 31%** — frequency-flattened ranking dilutes VRAM slot coverage; profiles must be workload-matched |
| `LLAMA_MOE_MISS_PREFETCH=1` + `LLAMA_MOE_FANOUT_PIPE=1` (the untested pair) | 6.40 (drive 187.6 MiB/tok) | NEUTRAL — speculative page-in works mechanically (storage −15%) but the real job still pays wake + pread round-trip for its wrong guesses; the wall is latency, not bytes |
| MTP (`draft-mtp`, n_max 2, cache 88, inserts 2) | 4.69 / 4.41 (hit 75.8/75.4%) | **LOSES ~30%** — verify walls scale ~2x per step for 2.06 tok/step; closed on the SN5000 exactly as on the P310 |

Decode wall anatomy re-confirmed: ~47 serialized (wake → ~7 preads → join →
matmul) chains per token; GPU busy 29–31% during decode. The structural levers
that remain are code-level (per-tensor overlap of up/gate compute with
down's preads; wake-cost trim) — both estimated +5–15%, not multipliers.

## Prefill: the "compute-bound" verdict was an artifact — the wall is the scheduler's weight copies

Three measurements this evening dismantled the 09-29 conclusion
("the one-pass wall is the MUL_MAT_ID execution itself"):

1. **GPU busy is 45% during the pass, CPU 0.89 cores-avg** (`cpu-probe.*`,
   `bench-prefill-cpu-probe-1001.sh`). Nothing is saturated.
2. **The device-staging mod never engaged.** `stage-on` kept 143,674 major
   faults — identical to stock (143–169k). Root cause chain:
   `--cpu-moe` + `--load-mode mmap` places expert weights in **CPU_Mapped**
   buffers (a plain file mmap; log: "CPU_Mapped model buffer size =
   64,969.50 MiB"); the staging gate qualified src0 via the Vulkan
   pinned-memory lookup, which cannot see mmap pointers → staging silently
   skipped 100% of ops → the staging experiment measured a no-op →
   "compute-bound" was inferred from a null result.
3. **The instrumented gate shows what the op actually sees**:
   `stage-gate src0='Vulkan0#blk.0.ffn_down_exps.weight#0' buft='Vulkan0'
   is_host=0` — the graph scheduler **copies every expert tensor from the
   mmap into a device buffer before each prompt-pass mul_mat_id**. 144
   tensors × ~450 MB = ~77 GB per one-pass pass, through a fully synchronous
   per-tensor write (read at queue-depth-1 ~0.9 GB/s → memcpy → PCIe →
   fence wait). The ubatch sweep's read volumes match copies-per-pass
   exactly: **77 / 138 / 259 GB at ubatch 16384 / 8192 / 4096**.

So the "0.88 GB/s fault stream" of the 09-29 notes was the **read side of the
scheduler's weight copy**, and the readahead fix that took gate C from 48.8 to
168.9 t/s accelerated exactly that copy. Prefill tps by ubatch (evening):
16384 → 173–176; 8192 → 135.5; 4096 → 93.1. One-pass remains right.

**Fix attempt (in-tree, default ON): rolling copy read-ahead** —
`ggml_backend_vk_buffer_set_tensor` issues `POSIX_FADV_WILLNEED` on the next
~1.5 GB of the GGUF (`LLAMA_VK_COPY_READAHEAD_MB`, 0 = off) ahead of each
≥64 MB host→device copy.

**Result: NEUTRAL** (off/on/off/on = 160.1 / 166.0 / 166.4 / 163.3 t/s,
major faults unchanged 185–205k; parity byte-identical PASS both arms).
The advise is not landing on the pages the copy reads next — candidates:
the scheduler's copy order does not follow GGUF file order (the window is
file-adjacent, the copies are graph-ordered), or the mapping resolution
misses. Next session: one-time log inside the read-ahead (resolution + fd +
window vs the set_tensor call stream), then order-aware advice (remember the
last N copy addresses; advise around the *upcoming* region) or an async
deep-queue copy pipeline in `ggml_vk_buffer_write_2d` (its per-tensor
submit→fence is fully serial today). The shader floor from the busy% data is
~35 s ⇒ ~430 t/s; the copy read side must exceed ~3 GB/s to reach the band.

Also landed: staging gate fix (host-buffer qualification + broken-flag
fallback) — dormant while weights arrive as sched device copies, but correct
for genuine host-resident inputs.

## Numbers

- Baseline (tonight, stock): 175.7 / 176.4 t/s, ~85 s, 77 GB, ~143–155k major
  faults. Read-ahead arms and parity: see `readahead-results.txt`.
- GPU busy 45%, CPU 0.9 cores, ubatch sweep and multi-profile/combo/MTP
  tables: this file + `prefillprobe-results.txt` + `nightsweep-results.txt` +
  `mtp-night-results.txt`.

## What this changes about the roadmap

- Stage 3 of the plan doc ("sorted route-order streaming, wavefront") was
  aimed at the wrong layer: the pass was never reading through the GPU fault
  path — it was reading through synchronous per-tensor sched copies. The
  wavefront executor's value (fewer expert revisits at small ubatch) matters
  only for multi-ubatch prompts; one-pass is copy-bound and now read-ahead.
- The 300–400 band is the copy floor + shader floor: ~77 GB at page-cache
  speed (memcpy + PCIe ≈ 10–20 GB/s ⇒ 4–8 s) plus the MoE shader time the
  45% GPU-busy number implies (~35 s at today's mix). If the read-ahead
  removes the copy crawl, one-pass prefill should land near the shader-bound
  number — measured in `readahead-results.txt` tonight.
- Decode to 15–20 without requant needs code-level structure change
  (per-tensor overlap in the fanout chain) plus fresh-boot discipline;
  every env knob is now measured closed (MTP, profiles, prefetch, pipe).

## Bookkeeping

- New drivers: `bench-night-sweep-1001.sh`, `bench-prefill-probe-1001.sh`,
  `bench-prefill-cpu-probe-1001.sh`, `bench-stagefix-1001.sh`,
  `bench-readahead-1001.sh`, `agent-pi-session-1001.sh`,
  `bench-mtp-night-1001.sh`.
- `scripts/bench-prefill-safe.sh`: THREADS/THREADS_BATCH passthrough (default 8/8 unchanged).
- Binary: unified tree + staging-gate fix + copy read-ahead; commit logged at write-up.
