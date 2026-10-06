# Qwen3.8 Flash-Next: the prompt-path MoE never ran on the CPU kernel (2026-09-29)

Session finding that rewrites the prefill plan. Measured, not inferred — every
claim below has a probe or a bench artifact in `docs/bench-results-2026-09-29/`.

## The discovery

**On server prompt paths, the ffn expert matmuls do not execute in the CPU
`mul_mat_id` kernel. They run on VULKAN, streaming the host-mapped expert
weights through uncontrolled GPU page faults.**

Proof chain:

1. `GGML_MOE_NODE_LOG=1` (new, in `ggml-cpu.c`) prints a line per
   `mul_mat_id` node with its page-in job timing. In a full prefill run
   (8,502-token prompt, ubatch 16384) it produced **432 lines, all from
   warmup/reserve shapes (tokens=1/2/4). Zero lines at tokens=8502** — the
   prompt pass never entered `ggml_compute_forward_mul_mat_id`.
   Artifact: `prefill-ub16384-async8-nodelog.server.log`.
2. `GGML_SCHED_DEBUG=2 -lv 5` (needs BOTH; `-lv 5` is the part that was missed
   before) prints per-node backend assignment. During the request phase the
   ffn `MUL_MAT_ID` nodes split **147 CPU / 141 Vulkan** per graph — exactly
   one full set of gate/up/down nodes per MoE layer (47×3) on Vulkan, the
   non-MoE matmuls on CPU. Probe: `/tmp/schedprobe2.log` shape; summary in
   this doc.
3. The weights are mapped `buffer type overridden to Vulkan_Host`
   (`make_cpu_buft_list` puts the GPU device's host buffer first for CPU
   overrides — `llama-model.cpp` `make_cpu_buft_list`). Vulkan_Host = imported
   host memory the GPU shaders read directly over PCIe, faulting pages on
   demand. Effective rate: 78.9 GB / 97.6 s = **0.81 GB/s** at ubatch 16384.
4. Retroactive correction: **bug #2 was NOT resolved on 09-28.** The "141 hist
   lines at ubatch 8192" were warmup-shape lines, exactly like today's. The
   histogram, fanout, DONTNEED, multi-RHS and pread knobs only ever affected
   decode-shaped ubatches and llama-cli runs (which route small shapes to
   CPU). Every prefill A/B to date — including the 6.2 t/s plateau of 09-27
   and the 87 t/s one-pass ladder of 09-28 — measured the **Vulkan shader
   page-fault path**, not the CPU kernel the knobs live in.

## Today's A/B (8,502-token prompt, ubatch 16384, all CPU-knob arms — flat, as expected in hindsight)

| Arm | prompt t/s | read GB |
| --- | ---: | ---: |
| legacy8 (per-op spawn/join fanout, LLAMA_MOE_FANOUT_ASYNC=0) | 85.97 | 78.4 |
| async8 (persistent pool, dispatched) | 86.96 | 78.9 |
| async24 | 84.69 | 78.6 |
| async8 + node log | 86.86 | 78.3 |

All within noise. Correct result: none of these knobs execute on the path
that streams the 78 GB.

## What landed in the unified tree (commit `63b8acad3`)

1. **Persistent async fanout pool** (`ggml-cpu.c`): the per-op
   `pthread_create`/join fanout is replaced by a process-wide pool; thread 0
   submits the slice list and returns to compute. Overlap inside the node and
   across nodes; ring backpressure; legacy path kept under
   `LLAMA_MOE_FANOUT_ASYNC=0` (forced when DONTNEED is on). This is decode
   lever #1 (the ~24-thread create/join storm per mul_mat_id, ~144 ops/token)
   and it is measured-safe (4 full bench runs with the pool active).
2. **GGML_MOE_NODE_LOG=1** per-node phase lines (submit→drain = IO exposure,
   node wall, tail). The instrument that found all of the above; ~45 bytes of
   stderr per MoE node.
3. **GGML_VK_MOE_HOST_MAX_IDS=N** (`ggml-vulkan.cpp` supports_op): when an
   expert `mul_mat_id` with more than N id rows has its weights in a GPU host
   buffer, Vulkan rejects it so the sched routes it to the CPU backend.
   Default -1 = historical routing. This is the lever to bring the prompt path
   into the kernel we control (fanout ordering, readahead, async overlap,
   multi-RHS later).

## The OOM incidents (do not repeat)

**Four session kills** on 09-29, all from my new memory-layout knobs, none
from the stock envelope (5 stock runs completed that morning, plus the whole
09-27/09-28 sessions). Kernel `oom-kill` on `app-zcode-*` scopes at
14:46 / 15:15 / 15:19, and a full session restart at 15:35 after the fourth
run. Causes, with what each taught:

- `LLAMA_ARG_NO_HOST` on this model forces the 61 GiB of expert weights out of
  mmap (evictable file pages) into **anonymous RAM that does not exist on a
  16 GB machine**. Never use it here.
- `GGML_VK_MOE_HOST_MAX_IDS=1` at ubatch 16384 (two runs) and **also at
  ubatch 2048 with a 446-token prompt (the fourth kill)**. The memory cost is
  NOT the runtime ubatch width — it is the **load-time graph reserve**, which
  builds a full-context graph under the new GPU→CPU→GPU split pattern
  (per-split copies and buffers across ~3 splits per MoE layer). Its
  allocation profile was never bounded, and `oom_score_adj=500` on the server
  did not save the session (systemd-oomd pressure/kill path). The knob stays
  in the tree (default off, env-gated) but is **unvalidated and must be
  treated as session-hostile on this machine** until a run succeeds on a
  freshly rebooted box, watched by a human.

Benchmarking guidance: use a fresh boot for comparisons; the stock envelope
(bench-prefill-window.sh /
bench-decode-arms-0928.sh with yesterday's env set) is the only proven-safe
path; any knob that changes graph split topology or buffer placement needs a
human-supervised first run after a reboot; `scripts/bench-prefill-safe.sh`
adds MemAvailable gating, memory sampling and oom_score_adj — necessary but
NOT sufficient, as the fourth kill proved.

## Next steps (in order)

0. **Establish a clean baseline.** After repeated out-of-memory failures,
   remaining experiments need a comparable page-cache and swap state.
1. **Human-supervised validation of the routing knob** (`GGML_VK_MOE_HOST_MAX_IDS=1`):
   stock 446-token arm completed today at ubatch 2048 = **6.39 t/s, 79.4 GB**
   (Vulkan fault path, `ub2048-stock`); the CPU-routed twin arm is the one
   that killed the session at 15:35 and its log is `ub2048-moecpu.server.log`
   (partial, load-time). If it ever validates, the decision rule is: CPU arm's
   effective GB/s beats the Vulkan path's ~0.8-1.1 GB/s → prompt path moves to
   the CPU kernel and the ladder re-runs there.
   Prerequisite hardening before the next attempt: cap the sched reserve
   pattern for the CPU-MoE split (or force `--ctx-size` = prompt length so
   load-time reserve is small), so the load-time allocation is bounded by
   construction instead of by hope.
2. If CPU routing wins: widest RAM-safe ubatch for the 8.5k prompt with memory
   sampling, re-run the one-pass ladder on the CPU path, then stack IO/compute
   overlap and multi-RHS (both already in-tree, finally live on the prompt
   path).
3. Decode track (independent, proven envelope): standard 256-token benchmark
   with the async pool (LLAMA_MOE_FANOUT_ASYNC=1 is now default; A/B against
   the 1.50 t/s baseline with ASYNC=0). The pool removes the per-op thread
   storm the FINDINGS doc identified as the serial per-pass wall. No graph
   topology change — same risk class as yesterday's runs.
4. Bench artifact folder: `docs/bench-results-2026-09-29/`.
