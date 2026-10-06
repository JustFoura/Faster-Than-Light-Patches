# Session findings, 2026-09-28 (decode + prefill track)

All decode runs: 43-token story prompt, seed 42, unified tree on 0.4.0 base,
`-lv 4`, `/proc/$PID/io read_bytes` deltas. "Pure-decode" io = second request
with `cache_prompt=true` (first request warms the prompt KV), so the 43-token
prompt's one-pass expert stream (~70 GiB) is not counted.

## Fixes landed (unified tree; delta regenerated into patches/llama-cpp/)

1. **MoE cache path gated at `n_tokens == 1` (llama-graph.cpp)** — the MTP
   verify pass (1 + n_draft tokens) bypassed the expert cache entirely and
   streamed every routed expert from mmap. This was composed-regression bug #1.
   Now engages for `n_tokens <= 4` (`LLAMA_MOE_CACHE_MAX_TOKENS` env to tune).
   Slot-id lookup flattens ids to 1-D (`ggml_get_rows` has no broadcast over
   ids tokens; `selected_experts` is a strided view so `ggml_cont` first).
2. **`--moe-expert-cache-inserts 0` was a no-op** (`if (max_inserts > 0)`);
   fixed to `>= 0`.
3. **Draft decode failed every second call** (`llama_decode[1] returned -1`,
   M-RoPE X<Y violation): the unified port dropped the MTP PR's arch guard on
   `is_mem_shared` in speculative.cpp. qwen4exp sets `ctx_other` for graph
   building and was wrongly treated as position-shared (gemma4-assistant
   convention), colliding with its own draft KV. Guard restored; drafts now
   run to n-max (mean 2.02-2.56 tok/step at n-max 2, was 1.66).
4. **Read amplification fixed** (`LLAMA_MOE_MADV_RANDOM=1`, `LLAMA_MOE_PREAD=1`):
   pure-decode disk reads were ~0.9 GiB/tok vs ~0.22 GiB/tok of design misses.
   Root cause: the prompt's sequential 61 GiB pass grows the file's readahead
   window; decode's random 848 KB slice faults then overshoot into neighboring
   experts (~4-5x bytes). MADV_RANDOM on the expert mappings kills the window;
   LLAMA_MOE_PREAD=1 streams miss slices via exact-size pread (fanout workers)
   to avoid the 4 KiB fault storm MADV_RANDOM would otherwise cause.
   Result: pure-decode io 0.90 -> 0.16 GiB/tok. Prefill keeps readahead (these
   envs are opt-in).
5. Diagnostics added: `GGML_MOE_CHAIN_LOG=1` (per-tensor post-skip compute
   accounting in mul_mat_id), fanout ctx extended for pread.

## Measured (256-tok arms unless noted)

| Arm | t/s | pure-decode io GiB/tok |
| --- | ---: | ---: |
| cache-ctrl (no MTP, inserts 2) | 0.84 | (incl. prompt) |
| mtp-cache-fix (fixes 1+3, inserts 2) | 0.69 | (incl. prompt) |
| mtp-width1-draftfix (isolation) | 0.22 | (incl. prompt) |
| cache-ins0 (no MTP, inserts 0) | **1.58** | 1.04 |
| ins0 + MADV_RANDOM + FANOUT24 | **1.34-1.41** | **0.165** |
| ins0 + MADV_RANDOM + PREAD + FANOUT8 | 0.97 | 0.159 |
| MTP composite + ins0 + MADV_RANDOM + FANOUT24 | 0.60 | 0.264 |
| slots64 | 1.03 | 1.76 |
| DONTNEED / FANOUT2 variants | 1.42 / 1.46 | ~1.4 |

Key structural facts (measured):

- **GGML_MOE_CHAIN_LOG**: during decode the CPU chain computes only ~2 of 10
  routed experts per node (78-80% served by the Vulkan cache chain) — the cache
  genuinely works; telemetry hit-rate matches reality.
- **Insert path is a net loss under memory pressure**: inserts=2 halves
  throughput vs inserts=0 (upload reads + page-cache churn). The RAM tier and
  ADMIT_ALL were already known losses; plain inserts are too on this machine.
- **Utilization during decode: ~3 of 8 cores, GPU ~10% busy** — the remaining
  wall is serial per-pass structure (48 layers x small ops, per-op fanout
  pthread_create storm (24 threads per mul_mat_id, ~144 ops/token), submit and
  sync latency), not compute and not bytes.
- **MTP composite is currently a net loss**: verify pass costs ~2.2x a
  single-token pass for 2.02x tokens (acceptance 51%). With bytes fixed, MTP
  pays only after per-pass cost drops.
- slots64 loses to slots88 (coverage dominates the GTT-spill concern; VRAM at
  15.25/16 GiB with the 10.7 GiB cache is fine).

## Best-known decode config today

Standard 256-token benchmark (cold server, prompt included), final head-to-head:

- **`--moe-expert-cache 88 --moe-expert-cache-inserts 0` + profile preload:
  1.50 t/s** (was 0.94 profile-only yesterday, 0.36 no-cache — 4.2x cold).
- The MADV_RANDOM+PREAD recipe is the *warm steady-state* optimization
  (pure-decode io 0.90 -> 0.16 GiB/tok) but loses the cold benchmark (0.89)
  because exact reads give up readahead during the prompt's 70 GB one-pass
  stream. Use it for long generations on a cached prompt; leave it off for
  cold-start requests.

## Prefill ladder (8,502-token varied prompt, n_predict 1)

| ubatch | prompt t/s | read GB | passes over experts |
| ---: | ---: | ---: | ---: |
| 1024 | 16.06 | 558.8 | ~9 |
| 2048 | 27.39 | 318.9 | ~5 |
| 4096 | 41.89 | 198.2 | ~3 |
| 8192 | 55.67 | 139.0 | ~2 |
| **16384** | **87.13** | **78.9** | **1** |

- The byte model is exact: t/s tracks read bytes / ~0.9 GB/s effective (~57% of
  the disk's 1.6 GB/s — the rest is compute/IO serialization inside the MoE
  node).
- **One-pass prefill (ubatch >= prompt) lands at 87 t/s on the P310.** The
  IO/compute overlap fix (pipeline the fanout touch-in behind compute instead
  of touch-then-compute) is the remaining lever toward 100-140 t/s.
- Longer prompts amortize the same 79 GB further (16k-token prompt at ubatch
  16384: same bytes, twice the tokens).
- **Bug #2 resolved as a side effect**: `GGML_MOE_ROUTE_HISTOGRAM` fires fine
  on the server prompt path at big ubatch (141 hist lines at ubatch 8192 with
  flash-attn on, q8_0 KV, --fit off). Yesterday's silence was specific to the
  small-ubatch shapes probed, not a server-path suppression.

## Next-step analysis (decode toward ~6 t/s)

Bytes are no longer the wall (0.19 GB/s of 1.6). The ladder to ~6 t/s is:
1. Kill the per-op fanout pthread_create storm (persistent worker pool).
2. Reduce per-layer CPU<->GPU sync (bigger fused nodes; the two-chain ADD
   forces a per-layer round trip).
3. Multi-RHS on the CPU chain at verify width (needs fan-in >= 4 rows/expert:
   only fires at n-max >= 3; measured acceptance decays with position, so test
   n-max 3/4 economics after 1+2).
4. Re-test MTP composite after 1+2: with per-pass cost down, the verify pass
   should flip to a win (draft is cheap at 8-21 ms/call; acceptance 51% at
   n-max 2).
5. If still short: 96-112 slots when the MTP sidecar is absent (frees 2.6 GiB
   VRAM), wavefront/prefill-style expert batching for decode is not applicable;
   pipelining the CPU chain behind the GPU chain is the remaining lever.
