# Session findings, 2026-09-29 (decode async pool + prefill CPU-routing validation)

Machine state: freshly rebooted (15:35), desktop up, ~11 GB MemAvailable at session start.
Model route profile restored from the committed
`docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz` into
`/tmp/moe_routes_decode.log` (12610 lines).

## Build incident (do not repeat)

The unified tree binary at `~/build/llama.cpp-unified/build-vulkan/bin/llama-server` was
built 15:14 on 09-29 — **before** the `63b8acad3` commit (15:26). Until rebuilt at 16:29
today (build 3, `63b8acad3`), the async fanout pool, `GGML_MOE_NODE_LOG` and
`GGML_VK_MOE_HOST_MAX_IDS` had **never executed anywhere**. Every "async pool" A/B and
every knob run before that rebuild measured the old code. Rule: after committing to the
unified tree, rebuild and verify `llama-server --version` reports the new commit before
benchmarking.

Second incident: the first A/B driver draft parsed env/flag arguments but never passed
`"${flags[@]}"` to the server launch — every arm silently ran cacheless. Caught by
cross-checking `/proc/$PID/cmdline` + `environ` against the arm definition. The accident
produced two honest no-cache datapoints on this fresh boot:

| Accidental arm | t/s | read |
| --- | ---: | ---: |
| nocache-accidental-fresh (no expert cache, 256 tok) | 0.48 | 806 GB ≈ 3.1 GiB/tok |
| nocache-accidental-warm (no cache, killed at n_gen 115) | 0.44 (running avg) | — |

(09-27's no-cache reference was 0.36 t/s on a long-uptime box; fresh boot looks mildly
better, consistent with cleaner page-cache/zram state.)

## Decode A/B: async fanout pool vs legacy fanout (cache-ins0 envelope)

Binary build 3 (`63b8acad3`), profile preload live (4224 slots, hit-rate 79.9% in every
arm — matches the offline 82.5% ceiling minus profile generalization), 88 slots,
inserts 0, 256 tokens, seed 42:

| Arm (order = run order) | t/s | read |
| --- | ---: | ---: |
| ins0-legacy8-fresh (fresh boot, legacy spawn/join fanout) | **1.90** | 227.5 GB |
| ins0-asyncpool (warm cache, persistent pool) | 1.78 | 240.1 GB |
| ins0-legacy8-warm (warm cache, legacy fanout) | 1.68 | — |

**The async pool is neutral for decode** (warm-vs-warm: 1.78 vs 1.68, within the ±6%
run noise; day-to-day state variance is bigger — 1.50 yesterday vs 1.90 today for the
same legacy config). The per-op pthread_create storm hypothesized as decode lever #1 in
the 09-28 FINDINGS is measured and closed: killing it does not move the number. Decode
work must go to the structural levers (98 splits/token sync structure, MTP fixed-cost
amortization). Also note fresh-boot state is worth ~+13% (1.90 vs 1.68) — benchmark
days should start from a reboot.

## Prefill: CPU-routing validation ladder (GGML_VK_MOE_HOST_MAX_IDS=1)

### Second OOM mechanism found and fixed: CPU_REPACK (not the graph reserve)

First rung attempt died during LOAD with MemAvailable diving 6.4 -> 1.0 GB (the new
guard killed the server both times; the desktop survived both). The log shows why:

```
load_tensors:   CPU_Mapped model buffer size = 64969.50 MiB   (same as stock)
load_tensors:   CPU_REPACK model buffer size = 21600.00 MiB   (stock has NO such buffer)
```

Mechanism: the load-time buffer choice (`select_weight_buft` ->
`weight_buft_supported` -> device `supports_op` probe) consults the SAME
`supports_op` the knob patches. With the knob on, the Vulkan host-buffer probe for
wide-id MUL_MAT_ID fails, selection falls through to the CPU extra buffer type —
**CPU_REPACK**, which COPIES (and re-packs) the tensor into ~21.6 GB of anonymous RAM.
The 09-29 doc's attribution to the "load-time graph reserve" was wrong; the reserve
buffers measured tiny (Vulkan0 520 MiB + Vulkan_Host 446 MiB).

Fix (unified tree `b4dde0132` + follow-up): when mmap is on, reroute any REPACK-picked
buffer to the plain CPU buffer type, which wraps the mapped pages zero-copy. Note:
CPU_REPACK's buffer type does not implement `is_host`, so the guard must key on the
name, not on `ggml_backend_buft_is_host`. After the fix the breakdown is identical to
stock and every rung below survived.

### Results (bounded ctx == ubatch == rung width, all guarded, build 63b8acad3+guard)

| Rung | prompt t/s | read GB | effective GB/s | stock reference |
| --- | ---: | ---: | ---: | --- |
| mcpu-ub512 (446-tok) | **8.11** | 77.5 | **1.41** | 6.23 t/s / 1.10 GB/s (09-27) |
| mcpu-ub2048 (446-tok) | **8.01** | 78.7 | **1.41** | 6.39 t/s / 1.14 GB/s (09-29 am) |

### Full ladder results (same boot, same 8.5k prompt where noted)

| Rung | prompt t/s | read GB | effective GB/s | verdict |
| --- | ---: | ---: | ---: | --- |
| mcpu-ub512 (446-tok) | **8.11** | 77.5 | **1.41** | +30% over stock ub512 |
| mcpu-ub2048 (446-tok) | **8.01** | 78.7 | **1.41** | +25% over stock ub2048 |
| stock-ub8704 (8.5k one-pass) | **88.19** | 77.7 | 0.88 | matches 09-28's 87.1 |
| mcpu-ub8704 (8.5k one-pass) | 55.25 | 78.9 | 0.51 | CPU kernel compute-limited |
| mcpu-ub8704-mrhs | (killed) | — | — | node log: 7.5 s compute tail per down node |

**Decision rule refined:** the CPU kernel wins narrow ubatches (+25-30%) and LOSES the
wide one-pass (55 vs 88 t/s) — at fan-in ~166 rows/expert the generic vec_dot on
non-repacked mmap weights is compute-bound (multi-RHS's f32 fallback makes it ~25x
worse: io_ms 308 vs tail_ms 7553 per node). Repack would fix the kernels but its
21.6 GB anonymous copy cannot exist on this machine.

### Prefill headline: 121 t/s at a 15k-token one-pass prompt (target band reached)

| Prompt (one pass, ubatch = ctx, stock Vulkan fault path) | prompt t/s | read GB | effective GB/s |
| --- | ---: | ---: | ---: |
| 446 tok (ub512, mcpu-routed) | 8.11 | 77.5 | 1.41 |
| 8,502 tok (ub8704) | 88.19 | 77.7 | 0.88 |
| **14,963 tok (ub16384)** | **121.05** | 77.7 | 0.87 |

Same 78 GB expert pass, amortized over more tokens — the 100-140 t/s band is
reached on the existing path with no new engineering. The pass at 15k width runs at
0.87 GB/s effective (~54% of the disk), i.e. ~60% of the wall is compute/serialization
inside the MoE node — the remaining known lever is sorting the GPU path's consumption
order (the fault stream is unsorted random 2.5 MB reads; the CPU kernel's compact
route index proves the win on CPU, where compute does not yet allow it).

Prefetcher v1 (`LLAMA_MOE_PREFETCH=1`, sequential MADV_WILLNEED ahead of the faults)
measured NEUTRAL-to-worse at ub8704: 83.95 t/s / 83.6 GB vs 88.19 / 77.7 — the
sequential lead gets evicted before the unsorted fault stream reaches it, costing
+7.6% bytes. Kept opt-in/off.

**Prefill ceiling math:** the one-pass 78 GB puts the 446-token physical ceiling at
~9.1 t/s (mcpu sits at ~88% of it) and the 8.5k ceiling at ~173 t/s (both paths sit at
51-88% of it — neither is IO-saturated).

## Decode: attribution closed — the wall is the miss-stream fault path

### Node-log attribution (attrib-nodelog arm: async pool, ins0, node log on; 1.78 t/s, 237 GB)

`GGML_MOE_NODE_LOG` lines over decode tokens (tokens=1 ops only, last 3000 = ~21
tokens' worth at ~141 CPU ops/token):

```
nodes=3000  Σ node_ms=12307 (≈578 ms/token-eq)  Σ io_ms=11075 (90% of walls)  Σ tail_ms=91
total 6.2 GiB → ~0.29 GiB/token through the CPU miss chain at ~0.5 GB/s effective
```

Per-op compute is **0.1 ms** (3 miss experts, 2.6 MB) — the 09-28 "serial per-pass
structure" hypothesis resolves to: the wall is the miss-stream faults themselves
(default readahead overshoot, shallow effective queue), NOT sched splits (~90 ms of
tails total) and NOT CPU compute. The decode graph's 98 splits and 141 small ops cost
remarkably little once the bytes arrive.

### MTP+ins0 re-test (the untested composite)

| Arm | t/s | read | acceptance / mean len |
| --- | ---: | ---: | --- |
| attrib-nodelog (no MTP) | 1.78 | 237 GB | — |
| mtp-ins0 | **1.03** | 396 GB | 52.8% / 2.06 tok/step |

Still a net loss (−42%), but the byte economics finally favor MTP: 396 GB / 248 steps ≈
1.63 GB/step for 2.06 tokens = **0.79 GiB/token vs 0.93 plain**. The loss is per-step
latency (~1.0 s/step vs 0.56 plain) — per-step cost tracks the fault path's effective
rate, not bytes. MTP flips positive only after the miss-stream fault rate improves,
which the phase-aware recipe below addresses.

### Phase-aware expert readahead (unified `90d636f7f`)

The 09-28 MADV_RANDOM+PREAD recipe fixes precisely this fault path (pure-decode io
0.9 → 0.16 GiB/tok) but lost cold benchmarks because it also killed readahead for the
prompt's one-pass stream. Now `LLAMA_MOE_MADV_RANDOM=decode` flips every registered
expert mapping MADV_NORMAL ↔ MADV_RANDOM on prompt↔decode phase transitions
(prompt-shaped = ids > 4), and `LLAMA_MOE_PREAD=1` engages only at decode/verify
widths — sequential prompts keep readahead, random decode misses get exact reads, in
one process.

### Phase-recipe A/B (standard 256-tok bench, same boot, same order protocol as the async A/B)

| Arm | t/s | read |
| --- | ---: | ---: |
| phase-legacy24-fresh (=1 recipe, legacy fanout) | 2.26 | 188.7 GB |
| phase-async24 (recipe + persistent pool) | **2.37** | 180.8 GB |
| phase-legacy24-warm (order control) | 2.30 | 189.2 GB |

**Decode standard benchmark: 1.68-1.90 → 2.26-2.37 t/s (+25%)**, read bytes −17-21%
(227-240 → 181-189 GB), prompt-pass speed unchanged (0.9 t/s — readahead kept, the
whole point of the phase gate). With exact reads the persistent pool adds ~+4%
(2.37 vs 2.26-2.30); without them it measured neutral. Hit-rate 79.9% in every arm.

New-best decode config: `LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1
LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE=... --moe-expert-cache 88
--moe-expert-cache-inserts 0` (pool default-on).

### Warm pure-decode pair on the phase recipe (cache_prompt second request)

| Measure | Value |
| --- | --- |
| pure-decode rate (r2, 256 tok, cached prompt KV) | **2.31 t/s** |
| pure-decode disk read | 127 GB / 256 tok = **506 MiB/tok** |
| effective disk rate during decode | 0.49 GiB × 2.31 ≈ **1.14 GB/s = 71% of the P310** |
| hit-rate | 79.6% (cache serving; r1 8-tok warmup 2.60 t/s) |

Two facts frame the road to ~6 t/s. (1) The disk is now CLOSE to the bottleneck but
not yet the binding constraint: 506 MiB/tok is ~2.3x the ~0.21 GiB/tok design misses
at 79.6% coverage (PLE + residual amplification unattributed — the 09-28 0.16 GiB/tok
number needed MADV+PREAD on a fully warm session), and at 6 t/s today's bytes would
need 2.9 GB/s. (2) Per-token overhead is ~430 ms (1/2.31) vs ~166 ms needed for 6 t/s.
So the ladder is: close the byte amplification (W-window miss prefetch, per-layer slot
weighting from the measured skew, 96-112 slots without the MTP sidecar), then re-test
MTP on the phase recipe (its bytes/token already improve; per-step latency should now
track bytes) — at which point the SSD becomes the binding constraint, which is the
user's signal to move the model to the second NVMe.

Accidental datapoint from a mis-flagged first pair (no expert cache, phase recipe on):
no-cache pure decode = 1.01 t/s at 1.19 GiB/tok — the exact-read recipe nearly doubles
even cacheless decode (0.48 → 1.01) by reading the full route at ~1.25 GB/s effective
instead of faulting it at ~0.5.

## Decode session 2 (2026-09-29 evening): MTP flips to parity, byte levers bounded

### The miss-stream double-read, its fix, and the ladder to 4.6 t/s

Warm pure-decode pairs (cached prompt KV, 256 tok) isolated three stacked fixes on top
of the phase recipe:

| Step | t/s | read MiB/tok | change |
| --- | ---: | ---: | --- |
| phase recipe (MADV_RANDOM=decode + PREAD + pool 24) | 2.31 | 506 | baseline (earlier today) |
| + PLE MADV_RANDOM (d6862e635) | 2.33 | 494 | PLE was NOT the amplification (24 random pages/tok is small) |
| + pread sequencing (06fecc322) | 2.58 | ~290 | the double-read found: the pool drained a node's preads while the compute threads raced the same slices, faulting them from disk at 4 KiB under MADV_RANDOM (~2x amplification: 227 MiB/tok of preads + ~267 MiB/tok of racing faults) |
| + layer-sibling merge (sequencing+) | 4.00 | 184 | one merged job pages in gate+up+down for the layer's shared miss set (same routes); per-op wait per layer instead of three serial small waits |
| + covered flag (sequencing++) | **4.57** | **191** | up/down skip their preads entirely when gate's merged job made the layer's miss set resident (cache-hit pread waits were ~2.2 ms/layer of pure overhead) |

Standard 256-tok cold benchmark (prompt included, final binary, FANOUT 12): **4.67 t/s**,
from 1.68-1.90 this morning and 1.50 on 09-28 — **3x over yesterday's best**. Cold
decode now runs nearly as fast as warm (4.2-4.3 t/s on 8-token requests) because the
merged deep preads do not depend on page-cache warmth.

Disk position: 191 MiB/tok at 4.57 t/s = 0.87 GB/s = **55% of the P310 ceiling**. The
remaining wall is the per-layer merged-job wait (~4 ms for 5-19 MB at ~0.9-1 GB/s
in-job) plus ~30 ms/tok of everything else. Next levers: deeper pread queue inside the
merged job (sub-splitting slices), and MTP warm state (bytes/token ~0.8x plain on
paper). At 6 t/s the disk would sit at ~1.1 GB/s — the SSD becomes the binding
constraint right around the target, as expected.

TO BE FILLED: MTP standard arm on tonight's stack; MTP warm pair.

Standard 256-tok bench, same boot, phase recipe (MADV_RANDOM=decode + PREAD +
FANOUT 24 + profile, inserts 0), before tonight's sequencing/merge stack:

| Arm | t/s | read | acceptance / mean len |
| --- | ---: | ---: | --- |
| plain phase recipe (pool) | 2.37 | 181 GB | — |
| MTP + phase recipe, inserts 0 | **2.38** | 184 GB | 52.8% / 2.06 |
| MTP + phase recipe, inserts 2 | 2.01 | — | 55.4% / 2.11 |

**MTP moved from a 1.03 t/s loss (−42%) to parity (2.38)** — the per-step latency now
tracks bytes as predicted. It does not yet WIN on this benchmark: the wider verify
route set is re-streamed every step because inserts 0 never lets the cache absorb the
overlap, and inserts 2 still costs more than it returns (2.01; hit-rate actually dipped
77.6-79.6%). MTP's remaining case is warm long-generation state (below) and n-max 3.

Per-layer slot reallocation: computed greedy marginal-coverage allocation from the
route dump under the same 4224-slot budget — **83.2% vs 82.5% flat**. The layer skew
differences are too small for reallocation to matter; closed without building it.
More slots don't fit either: 88 slots = 10.85 GiB leaves ~1.6 GiB VRAM (≈13 slots).

### Closed levers (evening session 2)

- **Slice sub-splitting** (778969b73, split x2-4 targeting ~2 pieces/worker): 4.59 t/s
  / 186 MiB/tok — neutral. The merged job's wait is latency-bound on ~6 random
  0.85 MB reads, not queue-depth-bound. Kept (harmless, helps wide jobs).
- **FANOUT 12 vs 24**: 4.63 / 4.57 — tied. Pool size no longer matters on this stack.
- **MTP on tonight's stack**: cold 4.09 t/s (vs 4.51 plain, -9%); warm 4.04 t/s /
  227 MiB/tok (-12%). The wider verify route set lands mostly in the UNCACHED tail
  (20.4% of 29 routes/step ≈ 5.9 misses vs 2.0 plain — ~1.4x miss bytes for 2.06
  tokens). MTP reopens only with higher cache coverage, which 16 GB VRAM cannot buy.

### Final state (2026-09-29 late evening)

Warm pure-decode: **4.57-4.63 t/s at 186-192 MiB/tok**. Standard 256-tok cold bench:
**4.67 t/s** (105 GB, prompt included). Decode day: 0.36 → 1.50 (09-28) → 2.26-2.37
(phase recipe) → **4.5-4.6** (sequencing + sibling merge + covered flag) — ~9x over
two days. Disk at ~0.85 GB/s = 55% of the P310; the binding constraint is now the
per-layer merged-job latency (~4-6 ms for 6 random 0.85 MB reads), not bandwidth.
Remaining ideas for the last stretch to ~6 t/s: shrink pool-wake overhead per job
(broadcast to 24 workers per op), and nothing else structural without more VRAM
coverage or a faster disk — at 6 t/s the disk would draw ~1.1 GB/s and become the
bottleneck, which motivates testing a higher-throughput data drive.

## Structural notes (code review, this session)

- Decode graph: **98 sched splits** (bs=1) / 145 (bs=512) per token graph — ~2 per MoE
  layer from the two-chain structure (Vulkan cache chain + CPU miss chain + ADD).
  Pipeline parallelism (which could overlap the splits) is hard-gated to multi-device
  upstream (`model.n_devices() > 1`), so the splits are fully serial today.
- Decode per-pass cost model from the pieces: pure CPU dot-product compute for the miss
  chain is ~15-25 ms/token (103 expert computes × ~12 MFLOP); page-in at 0.16 GiB/tok ≈
  100 ms if fully exposed. The remaining ~500 ms of the 667 ms/token wall is split-sync
  and per-op submit/setup overhead — the thing MTP's fixed-cost amortization and
  split-count reduction attack.
- The prompt-path MoE weights sit in a `CPU_Mapped` buffer (mmap zero-copy; the loader's
  "avoid using a host buffer when using mmap" reroute fires after the override picks
  Vulkan_Host). The routing knob's condition (host buft != cpu_buffer_type) still fires
  because `CPU_Mapped` is a distinct buffer type from `ggml_backend_cpu_buffer_type()`.
  The GPU shader path reads these pages through VK_EXT_external_memory_host imports —
  that is the "uncontrolled page fault" stream measured at ~0.8-1.1 GB/s.
- Multi-RHS kernel (`GGML_MOE_Q4_MULTI_RHS=1`) fires at >= 4 rows per expert chunk and
  dequantizes each weight row once per 4-route group — correct, opt-in, needs a prefill
  A/B on the CPU path.
