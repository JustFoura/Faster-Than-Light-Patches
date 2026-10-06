# Qwen3.8 Flash-Next: prefill/decode bottleneck math, live A/B results, engineering plan

**Date:** 2026-09-27 (updated after live A/B the same day)
**Scope:** where the time goes on the measured system, A/B results, what the
external research says (MTP / DFlash2 / DSpark / SSD-LLaMA / llama.cpp PRs), and the ranked
list of changes that remove bottlenecks instead of tuning around them.

Measured system: RX 9060 XT 16 GB (gfx1200, Vulkan RADV), Ryzen 7 5800X (8C/16T), 16 GB RAM
(~5-7 GB realistically available for page cache).

**Storage reality (measured this session, and it changes everything):**

| Drive | Link | Role | Measured |
| --- | --- | --- | --- |
| Crucial P310 500 GB | **PCIe Gen3 x2** ≈ 1.7 GB/s ceiling | Source model storage | **1.5-1.6 GB/s** direct sequential, **no scaling** with 4-way parallel readers |
| WD Blue SN5000 2 TB | PCIe Gen4 x4, rated 5.15 GB/s | Separate data-drive comparison | ~4-4.5 GB/s in later tests |

The initial measurements used the lower-throughput source drive; later runs used the
separate higher-throughput data drive. Storage throughput bounds the cold-read results.

## 1. The byte model (what any engine must move)

From the GGUF inspection in `docs/qwen38-engine-rd-2026-09-27.md`:

| Component | Size | Access pattern |
| --- | ---: | --- |
| Routed experts (47 MoE layers × 512 experts) | 60.94 GiB | ~2.54 MiB per expert per layer, top-10 routing |
| PLE n-gram table | 26.82 GiB | 90-byte rows, random 4 KiB pages |
| Backbone (attention/GDN/shared/router) | ~2 GiB | every token |

- One decode token's route touches ≈ 47 × 10 × 2.54 MiB ≈ **1.19 GiB** of expert weights.
- A prompt of C tokens touches `E(C) = 512·(1−(1−10/512)^C)` distinct experts per layer.
  C=64 → ~367; C≥470 → effectively all 512. **A ~500-token prompt must stream the whole
  61 GiB expert set at least once, whatever engine runs it.** That is the storage floor.
- On the source drive (1.6 GB/s demonstrated): 61-79 GiB per full-coverage prompt ≈
  **45-50 s ≈ 9-10 t/s absolute prefill ceiling** for ~450-token prompts. On the faster drive
  (~4.5-5 GB/s expected): **~25-35 t/s**, and multi-thousand-token prompts amortize the
  61 GiB into the 100s of t/s.
- Decode floor: 1.19 GiB / 1.6 GB/s ≈ 0.74 s/token ≈ **1.3 t/s** on the current disk;
  ~3.8 t/s from the faster drive; cache hits and speculation stack on top.

## 2. Live A/B, 2026-09-27 (this session)

One request each, 446-token varied prompt (`prompt-500.txt`), 1 predicted token, ubatch 512,
batch 2048, 8 threads, `LLAMA_KV_HOSTMEM=1`, no expert cache. Binary:
`~/build/llama.cpp-prefill-exp` (bccbacdb8 + expert-cache + compact-route-index +
route-histogram + IQ4 multi-RHS + `LLAMA_MOE_FANOUT` + new `GGML_MOE_MMAP_DONTNEED`;
freshly built today). Raw JSON + server logs: `docs/bench-results-2026-09-27/`.

| Run | Prompt t/s | Read bytes | Major faults |
| --- | ---: | ---: | ---: |
| base (FANOUT=8) | 6.23 | 79.1 GB | 21,511 |
| + `GGML_MOE_MMAP_DONTNEED=1` | 6.31 | 77.3 GB | 19,708 |
| + `FANOUT=16` (with DONTNEED) | 6.17 | 79.1 GB | 19,189 |
| + multi-RHS (with DONTNEED) | 6.20 | 79.0 GB | 21,853 |

**Every variant lands at ~6.2 t/s and ~79 GB.** Combined with the disk probes this says:

1. The pass already moves ~one-pass bytes (79 GB ≈ 61 GiB experts + ~18 GB PLE/backbone/
   rounding), so ubatch replay and LRU thrash are **not** what remains at ubatch 512.
2. 79.1 GB / 71.6 s = **1.10 GB/s effective, i.e. ~69 % of the disk's demonstrated 1.6 GB/s
   ceiling.** The distance to the ceiling is compute/IO serialization per MoE node, worth
   at most ~1.4× — the other ~4× to the model's potential is the disk link itself.
3. Consequently: page-cache policy (DONTNEED) and fault queue depth (FANOUT 8→16) cannot
   move the number. Multi-RHS cannot either, because (see 2.1) the CPU MoE kernel is not
   the wall — and may not even be the executor for large ubatches (see 2.2).
4. This also retro-explains the 09-24 measurements: 124 prompt + 48 decoded tokens moved
   254.6 GB — that run had replay (ubatch 256 over 2 chunks) but even then the rate tracked
   bytes ÷ ~1.1-1.3 GB/s. **Prompt t/s has been storage-link-bound the whole time.**

Found and fixed along the way:

- **Multi-RHS crash** (`GGML_ASSERT(weight_traits->to_float … ) failed`): IQ4_XS experts
  use Q8_K activations, and Q8_K has **no** `to_float` dequantizer. Fixed by calling
  `dequantize_row_q8_K()` directly (`ggml-quants.h`) for the Q8_K case; server survives,
  output token matches the non-multi-RHS run. Q8_1 (IQ4_NL `down`) is still excluded from
  the fast path; `down` falls back to the per-row kernel.
- **Route histogram only fires for 2/4-token warmup graphs in server runs** — the
  446-token request produced zero `moe-route-hist` lines even with a catch-all for
  `ids->ne[1] > 64`. A `llama-cli -st -lv 4` run with the same binary and env **did**
  print per-layer histograms for its prompt (39 MoE layers, blk.8-46; blocks 0-7 are the
  MoE-free GDN layers), so the CPU `mul_mat_id` path works and the graph is intact — the
  suppression is specific to the server request shape. Eliminated so far: batch size
  (server `--batch-size 512` is also silent). Remaining suspects to bisect, one 3-minute
  run each: `--flash-attn on`, q8_0 KV, `--fit off`; or definitive with
  `GGML_SCHED_DEBUG=2 -lv 5` to print per-node backend assignment. Resolve before
  investing in the CPU kernel or the wavefront executor — the A/B rates are unaffected
  (all variants pinned at the storage ceiling), but the histogram and multi-RHS results
  only count once the knobs are provably in the live path.

## 3. Prefill: ranked changes (highest leverage first)

### 3.0 Faster storage — the single biggest prefill win

Later measurements placed the model on a separate, higher-throughput NVMe drive. The
observed drive rate was about 4-4.5 GB/s for parallel reads. This changed the cold-read
floor; the benchmark outcomes and limitations are recorded in the dated findings. The
specific machine provisioning procedure is not part of this project.

### 3.1 With faster storage: overlap IO with compute

At 4.5 GB/s the disk pass is ~14 s while CPU MoE compute for 446 tokens is ~20-25 s
(8 cores, ~2.5 GB/s effective through quantized dot kernels) — compute becomes co-limiting,
and today IO and compute serialize inside each MoE node (fanout touches all slices, then
compute runs). Fixes in order: pipeline the fanout (keep N slices of the *next* experts
touching while the current expert computes), then the multi-RHS kernel (now fixed), then
histogram-guided tuning once the backend question (2.2) is answered.

### 3.2 The layer-wavefront executor (structural fix for long prompts)

`docs/qwen38-prefill-wavefront-design.md` remains the right design for making prompt
length amortize the 61 GiB pass with bounded activation memory (route → expert-major
compact → stream each expert once → combine). Its value is highest **after** 3.0/3.1;
correctness hazards listed there remain binding.

### 3.3 PLE overlap

PR #28136 (`--lazy-mode on-direct`, superseded by merged **#29030**) showed the mechanism:
exposed I/O **concurrency** (sorted, deduplicated rows, parallel `pread`; one worker 1.15×,
worker pool +6.6×; cold PLE stage 18 s → 2.4 s on a 9070 XT). Verify the 0.4.0 base ships
the #29030 refactor; the PLE cost then drops to a fixed 1-3 s per request and overlaps with
backbone compute.

### 3.4 Later

- Expert-packed layout (SSD-LLaMA-style): one contiguous ~7.6 MiB run per expert so one
  aligned O_DIRECT read delivers it; the HIP O_DIRECT→pinned→GPU-dequant probes in
  `experiments/qwen38-ple-prototype/` already prove the data path on this machine.
- RAM 16 → 48/64 GB would put ~30-45 GiB of experts in page cache and changes every
  ceiling above; software cannot buy that back.

## 4. Decode: how MTP / DFlash2 / DSpark work and whether they fit

Mechanisms:

- **MTP (Qwen3-Next style):** the target ships an extra next-next-token head trained with
  the backbone; the head drafts k tokens from the shared trunk, the target verifies in one
  pass. The **local MTP sidecar (2.60 GiB Q8_0) exists**; llama.cpp has a working MTP path
  for hybrid Qwen archs (the `~/build/llama.cpp-mtp` checkout is that PR branch;
  `--spec-type` hooks are in 0.4.0). Measured here: n-max 2, 74 % acceptance, 0.21 → 0.37
  t/s standalone — slower than the expert-cache build (0.65-1.01), and **MTP + expert cache
  in one build has never been tried**.
- **DFlash / DFlash2 (z-lab):** a separate lightweight block-diffusion model drafts a whole
  block of `--block-size` tokens in parallel (masked denoising), target verifies once.
  Official llama.cpp integration (PR #27342) is in 0.4.0, and the repo's
  `gfx1200-hostmem-kv-dflash.patch` adds the sliding draft window (`--spec-draft-ctx`) that
  produced 13 → ~20-36 t/s on the 35B model. Public DFlash2 checkpoints: `Qwen3.8-27B`,
  `Muse-Glimmer-30B` — **none for Flash-Next**.
- **DSpark (vLLM speculators):** DFlash + a low-rank Markov head (restores intra-block
  dependency) + a confidence head predicting per-position acceptance — i.e. it knows *how
  far a block is worth verifying*. Pretrained: GLM-5.2 only; training pipeline is public.

**The MoE byte economics (what generic writeups miss):** verifying k+1 tokens reads
`E(k+1)` distinct experts per layer — k=1 → ~20 experts ≈ 1.23 GiB, k=3 → ~37 ≈ 2.3 GiB,
k=7 → ~69 ≈ 4.3 GiB. Per *accepted* token, a cold-streamed wide verify costs the same or
more disk bytes than plain decode: a wide DFlash block is byte-hostile exactly when the
expert set is storage-resident. Speculation pays only when (a) the VRAM expert cache
absorbs the extra routes, (b) acceptance is high, and (c) verify width stays small.

Decode order of attack:

1. **Storage first (3.0):** decode is 1.19 GiB/token; on the source drive it can never beat
   ~1.3 t/s cold. On the faster drive the floor is ~3.8 t/s before caches.
2. **Combine MTP + expert cache + prefetch in one build** (never done; MTP drafts from the
   shared trunk at near-zero extra byte cost, the 88-slot cache absorbs route overlap,
   k≤2-3 keeps verify bytes ~1.2-1.5 GiB/token). Expected warm decode: from ~1.0-1.4
   toward 2-3 t/s (source drive) / 3-5 t/s (faster drive).
3. **Route-skew hot-expert preloading** into the VRAM slots from a captured route log
   (`GGML_MOE_LOG` hook exists) — raises the fraction of routed bytes served from VRAM,
   which is also what makes speculation profitable.
4. **Train a DFlash2/DSpark speculator for Flash-Next** (needs a real GPU box). DSpark's
   confidence head is the right tool for a storage-bound MoE. Interim proxy: the
   tokenizer-compatible `Qwen3.8-27B` DFlash2 draft — free to measure, acceptance likely
   mediocre. Keep the sliding-window draft-context patch; a VRAM-resident draft is what
   makes any of this cheap.

Honest ceiling: 16 GB RAM + 16 GB VRAM + one NVMe ⇒ ~3-5 t/s sustained decode is the
engineering ceiling, ~8-10 t/s only on cache-friendly, high-acceptance generations.
"Up to 10" is real; "always 10" is not, without more RAM or requantization.

## 4.6 Measured route skew (256-token generation, temp 0.6, this session)

From `moe-routes-decode-256t.log.gz` / `moe-route-analysis.txt` (39 MoE + 8 GDN-side
layers, ~2,630 routes per layer, all 48 layers route):

| Per-layer slot budget | Coverage of decode routes |
| ---: | ---: |
| top-64 (12% of experts) | 74.0% |
| **top-88 (17% — the existing VRAM cache size)** | **82.5%** |
| top-128 (25%) | 91.2% |
| top-192 (38%) | 97.5% |

Rolling-window temporal reuse (fraction of a token's experts seen in the previous W):
W=1 → 38.6%, W=2 → 46.9%, W=4 → 55.4%, W=8 → 65.2%. Early layers (0-7) are the least
skewed (60-64% at 88) and least local (7-19% at W=1/4); mid/late layers hit 44-56% W=1
reuse — a per-layer slot allocation weighted toward early layers, plus W-window
prefetch for the rest, beats a flat 88.

Decode projection with measured inputs (replacing §4.5's placeholders): coverage 82.5%
leaves 0.21 GiB/token on disk ≈ 0.13 s at the demonstrated 1.6 GB/s; exposed compute
becomes the term that matters (CPU MoE ~0.3-0.4 s → multi-RHS targets it); MTP at the
measured 74 % / n-max 2 contributes ×~1.7 passes saved. Composite: **~5-7 t/s decode on
this disk**, and ×~3 further on a 5 GB/s-class link (P310 demonstrated 1.6 vs Gen4 x4
rated 5.15 ⇒ the confirmed 2-4× window is ≈3.2× nominal). The two builds that stand
between here and the measurement: (a) profile-loaded expert slots from this exact route
log format (`GGML_MOE_LOG`), (b) MTP merged into the cache build.

### 4.6.1 Live profile-preload A/B (same session, later)

`LLAMA_MOE_PROFILE` implemented in the exp tree (`moe-profile-preload.patch`): parses the
route dump at cache init, uploads each layer's top-88 experts into the device slots
synchronously before serving (hottest = highest LRU clock), and mirrors the next ranks
into the RAM tier when enabled. Ladder, identical request (seed 42, story text, 130-256
tokens, cache 88 slots/2 inserts):

| Config | Decode t/s | Hit-rate @ step 128 |
| --- | ---: | ---: |
| no cache (route-capture run) | 0.36 | — |
| LRU from cold | 0.80 | (converging) |
| **profile preload, clocks fixed** | **0.94** | **75.1 %** |
| profile + RAM tier 2 GiB + ADMIT_ALL | 0.76 | 76.0 % (ram-hits 37) |

Findings: the first preload build assigned LRU clocks in the wrong order (hottest slot
evicted first) — 72.8 % hit-rate; reversed order → 75.1 %. The residual gap to the
82.5 % offline ceiling is profile generalization (the dump came from different text);
live coverage on in-distribution text should approach the offline number. ADMIT_ALL
floods the upload path (~1 GB/step of host→device copies) and is a net loss at 2 GiB
RAM tier scale — dropped. Also learned: this build filters libllama INFO logs at
default verbosity — use `-lv 4` for cache telemetry (hit-rate lines, preload summary).

The decisive measurement at 0.94 t/s: disk bytes ≈ 1.19 GiB × 25 % ≈ 0.30 GiB/token
≈ 0.19 s — **decode is no longer storage-bound**. The remaining ~0.85 s is CPU MoE
compute (fan-in 1 → multi-RHS never fires), PLE, and upload overhead. That is exactly
the MTP merge's territory: verify width 2-4 gives the CPU kernel its rows (multi-RHS
engages), amortizes PLE and fixed overheads over more accepted tokens, and the ×1.7
pass saving stacks on the cache. Next build: port the cache diff onto the 0.4.0 base
(`5266f24` already contains MTP `--spec-type` + DFlash + lazy improvements; the cache
surface is 3 hook sites + new files, verified portable).

### 4.6.2 The unified MTP+cache build (same session, status)

`~/build/llama.cpp-unified` — one tree on the 0.4.0 base (`5266f24`) containing:
the gfx1200 hostmem/-ot patch, the full MoE expert cache (cherry-applied from
`bccbacdb8`), the exp-tree mods (compact route index, multi-RHS, FANOUT,
DONTNEED, profile preload), and **MTP support** (`draft-mtp` type, qwen4exp
nextn layers, the gated `dsv4_hc_pre` op, shared-sidecar loading) ported from
the local MTP PR branch. The whole delta vs 0.4.0 is saved as
`patches/llama-cpp/unified-0.4.0-mtp-cache.patch` (22 files).

Port pieces that were missing from 0.4.0 and are now in: `LLM_TENSOR_NEXTN_HC_HEAD_*`
enums + nextn struct fields, `ggml_dsv4_hc_pre_gated` (op params: f32 scale, i32
gated), the null-`comb` guard in `ggml_dsv4_hc_post` (its assert killed the
reserve-graph probe), the QWEN4EXP `ctx_other` acceptance in llama-context, and
the MTP-aware `qwen4exp.cpp` (mtp_only/trunk/mtp_flags tensor gating, graph_mtp).

**Update (end of session): the crash is fixed.** Root cause: 0.4.0's
`create_memory` builds hybrid memory for every qwen4exp context, but the MTP
head's draft context needs a plain attention KV cache — the mtp branch handles
this via `mtp_on_hybrid_qwen` in `create_memory`, and its arch list lacked
QWEN4EXP. Adding `LLM_ARCH_QWEN4EXP` to that list (llama-model.cpp) lets the
draft context create the plain KV cache `graph_mtp` requires. The server now
initializes the full stack and runs generation end-to-end (§4.6.3 has the first
combined numbers).

### 4.6.3 MTP integration status (first combined runs, this session)

The unified tree runs the full stack end-to-end: target + shared MTP sidecar +
expert cache + profile preload + hit-rate telemetry. First combined numbers
(256-token story generation, seed 42, identical to the ladder above):

| Config | Decode t/s | Notes |
| --- | ---: | --- |
| MTP-only (no cache) | 0.36 | acceptance 65.6%, mean 1.66 tok/step — byte-neutral vs no-MTP cold, as §4.6 economics predict |
| MTP + cache 88 + profile | **0.25** | hit-rate 76→78.4%, acceptance unchanged — **slower than cache-only 0.94** |

The composed regression is the open question: per verify, the wide route set
(~29 experts/layer at 24% miss) should add ~0.4 s of cold streaming per step
(~1.66 tokens), yet the measured step time implies ~2.4 s/token-equivalent.
Suspects, in order: the async upload queue (2 inserts/step/layer × 47 layers
colliding with verify-frequency steps), double expert streaming between the
draft pass and the target verify pass, and the cache-chain/CPU split per
2-token ubatch. Debug handles all exist: `-lv 4` telemetry, hit-rate lines,
`/proc` io deltas, `GGML_MOE_LOG`. The non-MTP cache path (0.94 t/s) is
unaffected and remains the working baseline.

## 4.7 The committed decode path to ~6 t/s on this disk (no hardware change)

Baseline: decode with prefetch already sits **exactly** at the one-pass streaming floor
(1.36 t/s ≈ 1.19 GiB ÷ 1.6 GB/s), so every further multiple must come from bytes that
never touch the disk, plus multiple accepted tokens per pass. The stack, each piece
independently measurable:

1. **VRAM expert hot tier, profile-loaded.** The 88-slot/layer cache (~10.7 GiB VRAM at
   short context ≈ 17 % of experts) currently fills by LRU from cold. Rebuilt from a
   captured route log (`GGML_MOE_LOG` hook exists), its coverage is whatever the route
   distribution says — measured this session, see §4.6. More slots fit at short context
   (KV for this arch is tiny); at 200k the GTT KV already keeps VRAM free for them.
2. **MTP merged into the cache build.** Never attempted (separate trees). Measured
   standalone: 74 % acceptance at n-max 2 ⇒ up to ~1.7 target-passes saved per step, at
   +45 % route bytes per verify (20 → ~29 experts/layer) — net positive once the cache
   absorbs the overlap.
3. **RAM hot pages for the next tier** (madvise, no duplicates of the VRAM tier).
4. **IO/compute overlap in the MoE node** (fanout pipelining instead of touch-then-
   compute).

Decode projection arithmetic (to be replaced by measurements as each lands):
`t/s ≈ 1 / (routed_bytes × (1 − coverage) / 1.6 GB/s + compute_time)`. Hitting ~6 t/s
needs ~75 % effective coverage with MTP acceptance, or ~60 % coverage plus overlap wins.
The route-skew measurement decides whether that comes from skew alone (profile-loaded
slots) or needs the sequential-locality half (rolling-window reuse, also reported in §4.6).

Prefill on this disk, same rules: at ubatch ≥ prompt length the route coverage per pass
is complete ⇒ one expert pass per prompt ⇒ 4k-token prompts land at ~60-90 t/s and
~6-8k-token prompts cross 100 t/s once IO and compute overlap (multi-RHS fires at fan-in
~80 rows/expert there). Short prompts stay byte-bound at the single-pass floor — that is
amortization arithmetic, not a ceiling: same bytes serve 10× more tokens as the prompt
grows. The wavefront executor (`docs/qwen38-prefill-wavefront-design.md`) is the memory-
bounded form of the same argument for very long prompts.

## 5. What changed in the repo/trees today

- `~/build/llama.cpp-prefill-exp/` — fresh full build (Vulkan, `-j6`) of the moe-cache
  tree state + today's changes:
  - `GGML_MOE_MMAP_DONTNEED` post-node page release (file-backed-guarded via
    `/proc/self/maps`; skipped for expert-cache-resident ids) — validated, no regression,
    but pointless on the P310 (bytes already one-pass).
  - multi-RHS Q8_K fix (`dequantize_row_q8_K`) — validated, output preserved.
  - histogram catch-all for `ids->ne[1] > 64` — revealed the warmup-only mystery (2.2).
- `experiments/qwen38-multi-rhs-prefill/`:
  `madv-dontneed-window.patch` (the diff vs the moe-cache tree), `prompt-500.txt`,
  README status section updated (the old "never applied/built" text was stale — the
  prototype patches are in the moe-cache tree uncommitted).
- `scripts/bench-prefill-window.sh` — repeatable single-request A/B driver.
- `docs/bench-results-2026-09-27/` — JSON + server logs for every run.
- Not touched: launcher defaults, presets, model files, the moe-cache tree itself.

## 6. Next actions, in order

1. Re-run the model on higher-throughput storage (3.0) to validate the cold-read floor.
2. One short `-lv 4` run to resolve 2.2 (which backend runs big-ubatch MoE). If it is the
   CPU path after all, the fanout/DONTNEED/multi-RHS knobs become live on the fast disk;
   if it is a Vulkan/hostmem path, the same ideas (fanout touch-in, bounded window,
   multi-RHS) must be re-implemented where the work actually happens.
3. Re-run this exact A/B (`scripts/bench-prefill-window.sh`) from the new location.
4. Then: IO/compute overlap (3.1), PLE overlap check (3.3), wavefront (3.2).
5. Decode phase: combined MTP + expert-cache build (4, step 2).

## 7. References

- SSD-LLaMA (three-tier SSD/RAM/VRAM expert delivery): arXiv:2609.18110
- llama.cpp PR #28136 (`--lazy-mode on-direct`, superseded by #29030) — parallel PLE row reads
- llama.cpp expert-cache RFC (discussion #28248), two-tier cache issue #20757
- DFlash repo + llama.cpp PR #27342: github.com/z-lab/dflash (DFlash2: Qwen3.8-27B, Muse-Glimmer-30B)
- DSpark algorithm doc: github.com/vllm-project/speculators (algorithms/dspark)
- vLLM PLE auxiliary-device placement request: vllm-project/vllm#53908
- Local: `docs/qwen38-prefill-wavefront-design.md`, `docs/qwen38-engine-rd-2026-09-27.md`,
  `docs/flash-next-speed-work.md`, `docs/qwen38-flash-next-optimization.md`
