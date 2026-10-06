# Flash-Next optimization plan (2026-10-01)

Targets: **300–400 t/s prefill**, **>10 t/s decode, stretch ~15–20**, sustained at
200k context (q8 KV in GTT hostmem). Measured starting point (parity gate, fresh boot,
`docs/bench-results-2026-10-01/`): decode **9.23 warm / 9.86 standard** (2.0–2.1x the P310
final state), one-pass prefill **48.8 t/s** (regressed from 121 — fault-coalescing loss on
ext4, diagnosis in the FINDINGS doc, fix queued).

## The scaling math (why these targets are now credible, and what each requires)

Decode wall/token ≈ **108 ms ≈ 145 MiB/tok ÷ 1.40 GB/s** — reads still fully expose the
wall (zero overlap; compute hides inside it). Prefill (healthy 09-29 run) was ~72% IO /
~28% compute at 15k width. Therefore:

| Target | Needs | Levers |
| --- | --- | --- |
| decode 13–16 | rate ~1.8–2.2 GB/s, or bytes ≤120 MiB/tok, or ~30% read overlap | fanout depth, overlap, W-prefetch |
| decode 18–22 | **and** bytes ~100 MiB/tok **and** ~50% overlap | + slots/requant, MTP re-test |
| prefill 170–230 | fault stream coalesced again (≥1.5–2.5 GB/s) | readahead/THP test (Stage 1) |
| prefill 250–300 | **sorted** route-order stream at 3–4 GB/s | controlled prompt streaming (Stage 3) |
| prefill 370–440 | sorted reads **overlapped** under the ~35 s compute | pipeline/wavefront (Stage 3) |

200k context: q8 KV ≈ 2.7 GB (12 full-attn layers × 2 KV heads × 256 dims) + indexer
state — GTT hostmem (RAM), as shipped; VRAM placement would cost ~33 expert slots for no
attention win (indexer is top-2048 sparse). The 200k risk is RAM pressure on the mmap
expert stream (16 GB total), measured in Stage 4, not assumed.

## Stage 1 — close the prefill regression + measure the drive's real curves — DONE

1. **Readahead fix landed**: `read_ahead_kb` 128 → 4096 on the model data drive. Gate C:
   48.8 → **168.89 t/s** (1.39x the source-drive baseline);
   fault coalescing restored (172k faults, 77.6 GB). All three gates now EXCEED baseline —
   the separate-drive comparison is closed after one real-session confirmation.
2. THP/madvise patch: not needed (readahead alone closed it).
3. Drive curve measured (`probe-drive-rate.py`): **4.0–4.35 GB/s random 1 MB at depth
   ≥8** (85% of rating). Decode extracts 1.40 — the gap is queue structure, not the drive.
   FANOUT A/B: 12 → 5.56/6.17, **24 → 6.49**, 32 → 6.44 t/s (same session) — depth 24 is
   the new default (launcher updated); 32 flat.

## Stage 2 — decode toward 15–20 (in flight)

1. **One-layer-ahead miss prefetch** — implemented 10-01 (unified `ggml-cpu.c`,
   env-gated `LLAMA_MOE_MISS_PREFETCH=1`): each layer's merged miss-slice list is saved
   per pass; at the next pass the next layer's saved list is submitted as a
   fire-and-forget fanout-pool job, draining under compute+attention so the repeated
   slice set (W=1 temporal reuse ~39–47%) is page-cache-resident before its real job.
   A/B pending (OFF/ON/OFF interleaved; `bench-missprefetch-1001.sh`).
2. Slots 88 → ~101 (marginal), validate any requantization quality tradeoff, MTP re-test once prefetch lands.
   (Both re-tested 10-01 evening: MTP LOSES ~30% on the SN5000; slots flat with the
   single-text capture. Closed — see FINDINGS-EVENING.md.)
3. Session decay measured ~-40% through the day (9.23 fresh vs 5.6–7.0 steady).
   Per the number policy (below) the steady-state number IS the score: **closing this
   decay is the optimization**, not a benchmarking footnote. Prime suspects to measure:
   eviction of the profile-preloaded expert slots and the miss working set under
   memory pressure (madvise/mlock-class pinning or inserts-driven refresh), zram/swap
   accumulation, desktop growth. Order-control every A/B regardless.

## Stage 3 — prefill 48→170 →250 →300–400 (days, the larger campaign)

1. Gate-C recovery (Stage 1) → back to ≥121.
2. **Sorted controlled prompt stream**: route-ordered, deduplicated, deep-queue page-in
   ahead of the GPU fault stream (fanout/pread machinery exists from decode; the compact
   route index exists; `LLAMA_MOE_PREFETCH` v1 failed because it was *sequential*, not
   route-ordered — redo it route-major).
3. PLE read path: verify the lazy/direct-read mode (`LLAMA_ARG_LAZY_MODE`) engages on this
   model; the PLE table's random 4 KiB rows are the worst pattern on a DRAM-less drive.
4. **Overlap**: pipeline the expert stream under backbone/attention compute (the wavefront
   design doc's expert-major executor is the full form; its correctness hazards list is
   binding).

## Stage 4 — operationalize (parallel, cheap)

- Launcher overhauled 10-01 (unified binary + phase recipe + profile auto-restore +
  inserts 0). GUI preset `serverPath` must point at `scripts/llama-server-qwen38-flash-next`.
- Validate a 200k-ctx real session (KV in GTT, consistency target) after Stage 1 closes.
- inserts 0 vs 2 A/B in a real drifting conversation (benches say 0; live drift may want 2).

## Number policy (2026-10-01)

**The score is steady-state**: after extended use and many messages —
that is what the machine actually serves, and it is the only number that goes in the
books. Fresh-boot figures (the historical "+40% premium", "reboot before benchmark
days") are diagnostics for locating regressions, NEVER the reported speed. Practical
consequences:

- Every optimization is judged in the warm regime; a change that only helps a fresh
  boot does not count.
- The fresh-boot premium itself is now a TARGET: the ~9.2–9.9 t/s fresh decode decaying
  to 5.6–7.0 over a session is a ~40% regression to attack (candidates: page-cache
  eviction of the preloaded expert slots and the miss working set under memory
  pressure, zram/swap accumulation, desktop growth — measure which, then pin the hot
  ranges with madvise/mlock-class fixes or refresh them via cache inserts).
- Same for prefill: the steady-state one-pass number at the shipped launcher config is
  the headline, not the rebooted gate number.

## Rule of the road (from the 09-29 incidents)

Reboot before benchmark days; every knob that changes graph topology or buffer placement
gets a guarded first run (`bench-prefill-safe.sh`, oom_score_adj, MemAvailable floor);
binary must report the expected commit before benching.

## 2026-10-01 evening session (same boot; FINDINGS-EVENING.md in bench-results-2026-10-01/)

Every remaining env knob measured closed on the decode side: multi-text profile
**loses 31%** (hit 58.7% vs 77.4% — profiles must be workload-matched),
miss-prefetch+pipe combo **neutral** (storage −15%, wall flat), **MTP loses ~30%**
(4.69/4.41 vs 6.5–7.0 controls — closed on the faster drive as on the source drive). Controls 6.51/6.79/7.04;
interactive decode meanwhile ran **10.4–11.7 t/s** (conversations reuse experts
better than the bench loop).

**The prefill "compute-bound" verdict was an artifact.** Device staging never engaged
(CPU_Mapped weights invisible to the pinned-memory lookup; 143k major faults with the
flag on = a no-op arm), and the instrumented gate shows the op's src0 is the
scheduler's device copy: **the graph scheduler re-copies all ~77 GB of expert weights
from the mmap into VRAM before every prompt ubatch pass** — synchronous per-tensor
writes whose read side sustains only ~0.9 GB/s. GPU busy 45%, CPU 0.9 cores. The
ubatch sweep read volumes (77/138/259 GB at 16384/8192/4096) are copies-per-pass.

Consequences already shipped tonight:
- **Launcher ubatch 512 → 16384** (`scripts/llama-server-qwen38-flash-next`): at 512
  the launcher made every 512-token prompt chunk re-read the whole expert set — a
  live 4,397-token agent prompt prefilled at **10.4 t/s** (9 full passes, 422 s).
  One pass at 16384 is ~170 t/s today; decode passes are 1 token wide and unaffected.
- Copy read-ahead in `ggml_backend_vk_buffer_set_tensor` (rolling 1.5 GB
  `POSIX_FADV_WILLNEED` ahead of each ≥64 MB copy; `LLAMA_VK_COPY_READAHEAD_MB`,
  0=off): parity byte-identical, but measured **neutral** (160/166/166/163) — the
  window is file-adjacent while copies are graph-ordered. Next lever, in order:
  1. order-aware advise (track recent copy addresses; advise the upcoming region) or an
     async deep-queue copy pipeline in `ggml_vk_buffer_write_2d` (submit→fence is fully
     serial per tensor today). Page-cache-speed copies (≈10–15 s for 77 GB) put one-pass
     prefill at the shader floor (~35 s from the busy% data) ≈ **300–430 t/s**.
  2. decode structure: per-tensor overlap (compute up while down's slices still pread)
     + wake-cost trim — est. +12–20% stacked, on top of the live-session 10.4–11.7.
