# Decode: the 4.5 t/s figure was measured with the expert cache off (2026-10-02, session 4)

Same machine and model as the copy-path blocks earlier today, but the question
here is decode and the starting number was wrong.

## 1. The starting number was wrong

The 10-02 decode report measured **4.53 / 4.59 t/s** and concluded "decode is
byte-bound at 736 MB/token, 15-20 t/s is physically out of reach". The runs
behind that (`/tmp/decode-bench.log`, `/tmp/decode2.log`) contain **zero**
moe-cache init lines at `-lv 4`. `llama_moe_cache_init()` is called
unconditionally from the `llama_context` constructor
(`src/llama-context.cpp:102`) and always emits either
`MoE expert cache enabled: N layers x S slots` or
`... no host-resident expert layers found - disabled`. Neither is present, so
`--moe-expert-cache` was not on that command line.

`scripts/bench-decode-cache-1002.sh`, order-controlled A / B / A, 256-token
`cache_prompt` request, protocol identical to the 10-01 fanout harness:

| arm | t/s | window | MiB/token | read rate | hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `--moe-expert-cache 88` | **7.10** | 51.8 GB | 202 | 1.42 GB/s | 78.3% |
| cache off | 2.84 | 155.0 GB | 606 | 1.72 GB/s | — |
| `--moe-expert-cache 88` | **6.64** | 55.9 GB | 218 | 1.38 GB/s | 78.3% |

The cache is worth **2.4x**, and it takes 78.3% of routed experts out of the
read stream. Both cache arms sit inside the documented ±13% run-to-run band.
Cache 48 slots instead of 88 drops the hit rate to 64.1% and the rate to
~3.2-3.9 t/s, so slot count is the first-order knob on this path.

The "decode is byte-bound" conclusion therefore describes the one configuration
in which the cache was switched off.

## 2. Decode is not byte-bound; it is queue-depth-bound

`scripts/probe-drive-rate.py` at decode's exact request shape — 816 KiB
buffered random preads, one per worker, mirroring the pool:

| concurrent preads | GB/s |
| ---: | ---: |
| 1 | 1.16 |
| 2 | 2.13 |
| 4 | 3.21 |
| 8 | 3.96 |
| 16 | 4.26 |
| 24 | 4.17 |

Decode extracts **1.38-1.59 GB/s**, which is queue depth ~1.2, while holding a
24-worker pool. At 202 MiB/token, QD4 gives ~13 t/s and QD8 gives ~16 t/s — the
target band — with no change to bytes per token.

The cause is in `ggml_moe_fanout_pool_worker`: every worker takes `g == 0`, so
all 24 wait on and drain the **same** `done`-indexed job. Concurrency is bounded
by the slices in one job, not by worker count, and with the cache absorbing ~78%
of routes a decode node's miss job is ~2 slices. The pool empties between nodes.

## 3. Two fixes tried; both negative, and both recorded

**The pread barrier is not the wall.** `ggml_moe_pread_wait()` spins every
compute thread on `j->active == 0`, so each node is dispatch-all → wait-for-all
→ compute. Env-gated as `LLAMA_MOE_PREAD_WAIT` and A/B'd:

| arm | t/s | MiB/token | read rate |
| --- | ---: | ---: | ---: |
| wait on | 5.43 | 244.3 | 1.39 GB/s |
| wait off | 5.73 | 244.4 | 1.47 GB/s |
| wait on (repeat) | 5.66 | 232.2 | 1.38 GB/s |

+3% against a control mean of 5.55 — inside noise, and byte counts are
identical, so removing it does not buy the ~2x read amplification the barrier
was added to prevent. Kept on by default; the knob stays for future use.

**One-layer-ahead miss prefetch is worse, not better.**
`LLAMA_MOE_MISS_PREFETCH=1` is the only mechanism in the tree that submits a
job *ahead* of the one being drained — at each layer's gate op it hands the pool
the next layer's saved miss-slice list (`g_moe_pf_prev`). Arms:

| arm | t/s | MiB/token | hit rate |
| --- | ---: | ---: | ---: |
| prefetch off | 3.23 | 415.7 | 64.1% |
| prefetch on | (no completed request; partial eval 2.16 t/s) | — | — |
| prefetch off (repeat) | 2.53 | 404.6 | 64.1% |

The two off-arms are byte-identical on output (parity PASS) and 28% apart on
speed, so the on-arm's partial 2.16 t/s is worse than both controls rather than
inside the band. This matches 10-01's "neutral" reading of the same knob.

**A worker-side depth knob cannot work, structurally.** `LLAMA_MOE_FANOUT_DEPTH`
was built to let workers run N jobs ahead and deadlocks at N > 1: the submitter
publishes one job and the caller then blocks in `ggml_moe_pread_wait` until it
drains, so a second job is never in flight for a worker to run ahead to. Removed;
the reason is a comment at the call site so it is not re-added.

## 4. Where the remaining decode gap actually is

Three things have to change together, and none of them is the pool:

1. **Bytes.** 202 MiB/token is set by the 78.3% hit rate at 88 slots. More slots
   is the direct lever and it is capped by the 16 GB card — 88 slots already
   costs 10.8 GB of device memory, and 88 + 200k ctx is the combination that
   aborts model load with a RADV OOM.
2. **Queue depth.** The pread stream must stay deep *across* nodes, which needs
   the submitter to run ahead of the drain. Neither the barrier nor the existing
   one-layer prefetch achieves it; the submit pattern itself is the blocker, and
   changing it means touching the submit path, not the worker.
3. **Session decay.** The live agent run on 10-01 opened at 11.94 t/s and fell
   to ~1.9 t/s within the same session. Steady-state is the score per the number
   policy, and that decay is still unexplained and unmeasured.

## 5. Two configuration faults found outside the bench

**The app launches the wrong binary.** Every preset in `app/presets.json` sets
`serverPath` to `~/.local/bin/llama-server-simple`, which execs
`~/build/llama.cpp/build-vulkan/bin/llama-server` — upstream `5266f24da`
(2026-09-04). That build has **no `--moe-expert-cache` at all**
(`grep -c moe-expert-cache common/arg.cpp` → 0), and the stub exports only
`LLAMA_KV_HOSTMEM=1`, so none of the `LLAMA_MOE_*` recipe applies either.
Measured consequence: 2.84 t/s vs 7.10. There is no Flash-Next preset.

**The tuned launcher does not boot.** `scripts/llama-server-qwen38-flash-next`
defaults to `--moe-expert-cache 88` at ctx 200000, and that combination aborts
model load with a RADV out-of-memory. 64 slots boots.

## Bookkeeping

- llama.cpp development diff adds the `LLAMA_MOE_PREAD_WAIT` switch plus the
  `[moe-fanout]` start line; the worker
  job-index expression is unchanged character for character, and unset env is
  identical to stock. Parity PASS byte-identical on every completed arm.
- Harnesses: `scripts/bench-decode-cache-1002.sh`,
  `scripts/bench-decode-preadbarrier-1002.sh`,
  `scripts/bench-decode-prefetch-1002.sh`,
  `scripts/bench-decode-session-1002.py` (multi-turn, not yet run).
- Results: `decode-cache-results.txt`, `decode-preadbarrier-results.txt`,
  `decode-prefetch-results.txt`.
- **Harness label bug inherited from `bench-decode-fanout-1001.sh`:** it prints
  the window total as "MiB/tok" and divides by 1000 instead of 1.048576 to get
  "GB/s" (that is MiB/s). Every "MiB/tok" figure in the 10-01 results files is
  therefore ~256x too large. The 10-02 scripts fix both; older numbers should
  be read as window totals divided by `predicted_n`.
- Memory: server starts started losing to OOM kills as zram crept to 7.5 GB
  during this session. `--moe-expert-cache 88` is the largest single allocation
  (10.8 GB device). `swapoff /dev/zram0 && swapon /dev/zram0` reclaims it and
  needs sudo.
