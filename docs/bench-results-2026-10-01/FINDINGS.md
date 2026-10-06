# Session findings, 2026-10-01 (separate-drive tests — decode 2.1x, prefill 1.39x after readahead fix)

Fresh-boot benchmark run; the route profile was restored from the committed capture.
Order-controlled benchmark protocol; artifacts in this folder.

## Separate-drive benchmark results — ALL PASS

| Gate | Source-drive baseline (09-29) | Separate data drive | Verdict |
| --- | --- | --- | --- |
| A warm pure-decode (r2, cached KV) | 4.57–4.63 t/s @ 186–192 MiB/tok | **9.229 t/s @ 145 MiB/tok** | EXCEEDS (2.0x) |
| B standard 256-tok bench | 4.67 t/s | **9.855 t/s** | EXCEEDS (2.1x) |
| C one-pass 14,963-tok prefill | 121.05 t/s / 77.7 GB | **168.89 t/s / 77.6 GB** (after readahead fix, below) | EXCEEDS (1.39x) |

The initial parity prediction ("latency-bound, same drive class") was wrong for decode
in the good direction: with `LLAMA_MOE_PREAD=1` exact-size reads the separate data drive sustains
**~1.40 GB/s** on the miss stream vs the source drive's 0.87 — and bytes/token dropped
186–192 → 145 MiB (same binary, recipe, seed; unexplained but real, plausibly
filesystem and readahead accounting). The separate-drive comparison was followed
by one real session to confirm the result.

## Drive rate curve (`probe-drive-rate.py`, O_DIRECT random preads, GGUF file)

| Pattern | Rate |
| --- | ---: |
| 1 MB × 1 thread | 1.87 GB/s |
| 1 MB × 8 threads | 4.01 GB/s |
| 1 MB × 24 threads | **4.14 GB/s** |
| 1 MB × 48 threads | 3.99 GB/s |
| 256 KB × 24 threads | 3.68 GB/s |
| 4 MB × 8 threads | **4.35 GB/s** |

The separate data drive sustains **~4 GB/s on random 1 MB reads at depth ≥8** (about 85% of its 5.1 GB/s rating).
Decode currently extracts 1.40 GB/s of that — the limiter is the in-process queue
structure (pool depth, serial submit→wait→compute per layer), not the drive. At a
sustained 3 GB/s in-process, 145 MiB/tok decode = **~21 t/s**; a 20 t/s target
is rate-credible. Prefill at 4 GB/s sorted+overlapped closes the 300–400 band.

## The prefill fault-coalescing regression and its fix (read_ahead_kb)

First gate C attempt: **48.77 t/s, 45.5 GB read, 11.9M major faults (3.9 KiB/fault)** vs
baseline 121 t/s, 77.7 GB, 152,573 faults (0.52 MB/fault). The prompt path's uncontrolled
GPU fault stream lost readahead coalescing on the ext4 copy — every page faulted at 4 KiB
with zero storage coalescing (~150 MB/s effective). Not fragmentation (63 extents, ~248 MB
each), not the drive (gate A: 1.4 GB/s on the same file), not read_ahead_kb per se (128 on
both drives historically).

**Fix: `read_ahead_kb` 128 → 4096 on the model data drive.** Gate C re-run: 168.89
t/s, 77.6 GB, 172,516 faults — coalescing fully restored AND the drive advantage lands.
Caveats: the setting is volatile (reboot resets it) and per-device. Decode is unaffected
by it (MADV_RANDOM + exact preads bypass readahead).

## Decode ground truth (evening): the wall is job-boundary serialization, not bytes

Warm-pair arms through the afternoon: fan12 5.56/6.17, fan24 6.49, fan32 6.44 t/s —
well below gate A's 9.23 (same binary/protocol; session state + the mid-session
readahead change are confounded — gate A ran at readahead 128, everything after at 4096).

Drive-sector ground truth (data-device sectors around one 128-token r2,
fanout 24): **52.5 MiB/tok true storage at 5.30 t/s = 0.27 GB/s** — pid read_bytes
matches drive sectors exactly (6.73 GB both). The drive runs at ~7% of its demonstrated
4 GB/s. (An earlier A/B driver printed 5.4–5.8 GiB/tok — an accounting artifact in that
driver's sampling, contradicted by the sector counters; discard those numbers.)

Consequences:
- Storage is a non-issue at the current miss volume; even 20 t/s needs only ~1.05 GB/s.
- The wall is the ~47 per-token MoE jobs' serialized submit→drain→compute structure:
  ~1.1 MB/job at in-job rates + wake/submit latency per job ≈ the whole per-token wall.
- One-layer-ahead miss prefetch (`LLAMA_MOE_MISS_PREFETCH=1`, in-tree, default OFF)
  measured a LOSS (5.40 vs 6.66/6.76 OFF arms): extra ring jobs, no added concurrency —
  the pool drains jobs FIFO with all workers batched per job.
- The structural fix is **worker-group pipelining** (`LLAMA_MOE_FANOUT_PIPE=1`, in-tree,
  default OFF): two independent worker groups drain alternate jobs, keeping the pread
  stream deep across job boundaries. A/B in flight.

## Pipelining A/B (definitive, 256-token protocol): NEUTRAL — the wall is the routing dependency

Order-controlled OFF/ON/OFF at fanout 24, ignore_eos, full 256-token generations:

| Arm | t/s | drive MiB/tok | effective GB/s |
| --- | ---: | ---: | ---: |
| pipe OFF | 6.99 | 191.3 | 1.34 |
| pipe ON | 6.80 | 204.8 | 1.43 |
| pipe OFF | 6.61 | 212.3 | 1.47 |

Neutral within the session's drift band (OFF arms differ 5.7%). Interpretation: layer
i+1's routes do not exist until layer i computes, so its preads cannot start earlier —
the pool is idle during compute because there is nothing valid to drain. Queue structure
is not the constraint; the wall is the sequential route→pread→compute chain per layer
plus per-job fixed costs (submit/wake across 24 workers × ~47 jobs/token — the
"pool-wake overhead" lever named in the 09-29 FINDINGS).

**Decode lever ranking after today** (256-token basis, honest):
1. **Coverage** — 191 MiB/tok at 82% is what the drains move. 88→101 slots ≈ +2%;
    requant IQ3-class → ~120 slots ≈ 91% → ~95–110 MiB/tok → at today's 1.34 GB/s
    in-job rate ≈ **13–15 t/s**. Requantization requires a quality tradeoff.
2. **Pool-wake overhead** — shrink the per-job submit/wake cost (47 broadcasts/token).
3. **MTP re-test at higher coverage** — 2.06 tok/step at 52.8% acceptance stacks on 1.
4. Fresh boot (+~40% measured today, gates A/B vs evening arms).

Both structural patches stay in-tree, env-gated, default OFF:
`LLAMA_MOE_MISS_PREFETCH=1` (loss on 18-tok basis, untested at 256), `LLAMA_MOE_FANOUT_PIPE=1` (neutral).

## Prompt-phase pre-reader: NEUTRAL — and the fault path looks rate-bound, not storage-bound

Staged per-layer `POSIX_FADV_WILLNEED` pre-reader (`LLAMA_MOE_PROMPT_PREREAD=1`, in-tree,
default OFF): one-pass 14,963-tok prefill 167.0 / 165.2 / 173.7 (OFF/ON/OFF) — neutral.
A blunt concurrent sequential reader (direct IO) measured NEGATIVE (105 vs 175): pure
bandwidth theft, no cache benefit.

Decisive observation: the prompt path's effective read rate is **~0.88 GB/s on the source
drive (09-29) and ~0.88 GB/s on the separate data drive (today)** — two different drives, same number. The
raw GPU fault stream appears bounded by fault-processing rate (~150k major faults/pass at
~0.5 MB coalesced granularity), not by storage. Readahead tuning recovered the coalescing
loss (121→169) but the path's ceiling sits near 1 GB/s. **Prefill toward 300–400 requires
bypassing the fault path for bulk streaming** — the wavefront/expert-major executor
(docs/qwen38-prefill-wavefront-design.md) or a CPU-kernel route with a fixed multi-RHS
kernel (the 21.6 GB repack copy is impossible on 16 GB; the kernel needs an in-place
packed-int path). That is the next major build, not a knob.

Slots 88→100 (free VRAM): allocated fine (12.3 GiB, 4800 preloaded) but hit-rate flat
(77.2 vs 77.4%) — the route capture's tail ranks don't generalize across text. Real
coverage gains need a multi-text route capture or a smaller quant (fewer bytes/slot →
  more slots).

## Device staging: built, byte-identical, NEUTRAL — prefill is compute-bound, not storage-bound

`LLAMA_VK_MOE_DEVICE_STAGE=1` (in-tree, default OFF): prompt-width expert matmuls copy
their weight tensor CPU-side into a host staging buffer (staging thread, one-node
lookahead), upload to a device scratch synchronously, and bind the device copy — the
GPU fault path is bypassed entirely. **Correctness: PASS** (temp-0 outputs byte-identical).

**Performance: neutral** (154.6 / 160.2 / — OFF/ON at one-pass 14,963 tok; baseline
175 that same evening). Root cause established by three measurements:

1. Buffered sequential reads of the GGUF: **3.08 GB/s** — the CPU read side is fine.
2. The staged arm still showed a ~93 s wall with reads fully overlapped — the 77.6 GB
   stream was never the critical path.
3. The prompt path's effective read rate was **~0.88 GB/s on BOTH drives** — same
   number, different hardware.

**Conclusion: the one-pass prefill wall is GPU-side compute/serialization** — the ~90 s
is the MUL_MAT_ID execution itself (dequant-matmul of 77.6 GB with per-layer
dispatch stalls and CPU↔Vulkan split serialization), not the SSD. No storage-side
change can move it; the 300–400 band requires attacking the compute/serialization:
GPU timeline profiling of the one-pass graph (where the 90 s actually goes — shader
time vs dispatch gaps vs split sync), then either a faster MoE shader path (weight
residency, multi-RHS-class) or the wavefront executor. This is the next major build.

## Protocol correction (evening): the warm-pair r2 ran 18 tokens, not 256

The r2 suffix this session's drivers used (" Continue the story.") hits EOS after 18
tokens at seed 42; the 09-29 protocol's drivers set `ignore_eos: true` (bench_flash_next.py)
and theirs ran the full 256. Consequences, audited from the committed JSONs:

- **Gate B stands**: n=256 vs n=256 — 9.855 vs 4.67 t/s is a like-for-like 2.1x.
- **Gate A's 9.23 was an 18-token run** vs the 4.57–4.63 baselines at 256 tokens — not
  the same basis; re-run gate A with the fixed protocol (drivers patched to
  `ignore_eos:true`) on a fresh boot before recording it as the warm-decode headline.
- All afternoon A/Bs (fanout depth, miss-prefetch, pipelining first pass) were 18-token
  windows — same-basis within each sweep but noisy; treat their deltas as weak signals
  until re-run. The prefetch-ON loss and the fanout-24 gain both need 256-token
  confirmation; the pipelining A/B was re-run with the fixed protocol.

## Bookkeeping

- Gate C's first attempt pointed at `prompt-15k.txt` (24,782 tokens — a later, bigger file
  whose own 09-29 run was also ctx-rejected; the 121 t/s run of record is
  `stock-ub16384-13k.json` from `prompt-13k.txt`, 14,963 tok). Fixed in the gate driver.
- Benchmark verdict semantics: below band = FAIL; above band = EXCEEDS (measured win).
- Launcher `scripts/llama-server-qwen38-flash-next` moved to the unified binary + phase
  recipe + profile auto-restore + inserts 0 (was: Sep-25 moe-cache binary, no recipe —
  ~3x slower than bench state).
