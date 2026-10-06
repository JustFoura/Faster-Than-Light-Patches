# Session findings, 2026-10-02 (copy-path pread depth: landed, gated, and the profiling that redirects the roadmap)

Fresh boot ~00:10, same machine, read_ahead_kb 4096 restored (the reboot had
reset it to the 48.8 t/s state). llama.cpp at `321dff9b5`. Condition is the
gate-C prefill throughout: one-pass 14,960 tok, ubatch 16384, ctx 16384,
`--cpu-moe --load-mode mmap`, greedy temp-0 seed 42.

## Parity: PASS

`bench-prefill-depth-profile-1002.sh`, four order-controlled arms, greedy
temp-0 output compared byte for byte:

| arm | verdict |
| --- | --- |
| stock-a | reference |
| d8-a (depth 8) | **PASS byte-identical** |
| stock-b | **PASS byte-identical** |
| d16-a (depth 16) | **PASS byte-identical** |

The copy bytes are provably unchanged at both depths. The change is safe to
keep on correctness grounds.

## Speed: the +5% was noise, and the controls say so

| arm | t/s | wall | GPU busy (pass window) | read_bytes |
| --- | ---: | ---: | ---: | ---: |
| stock-a | 171.7 | 87 s | 55% | 134.5 GB |
| d8-a | 163.4 | 92 s | 57% | 136.3 GB |
| **stock-b** | **158.8** | **94 s** | **53%** | 134.3 GB |
| d16-a | 172.6 | 87 s | 55% | 131.2 GB |

**The two stock controls differ by 12.9 t/s (171.7 vs 158.8) — larger than any
treatment effect in the table.** Earlier in this session a single depth-8 arm
read 180.4 against a 171.6 control and I called it +5%; that was drift, not
signal, and the order-controlled design is what exposed it. Any prefill
conclusion from one arm against one control on this machine is unsound: the
run-to-run band is ~13 t/s.

Arithmetic on the same data: 134 GB per pass in 87–94 s is **1.43–1.55 GB/s
sustained**, against 1.39 GB/s measured for a depth-1 fault stream. The copy
stream is therefore already running at roughly depth-1-equivalent rate but
*above* what the target needs — see below. Depth 8 and depth 16 land inside
the control spread. Verdict on the pread-depth lever: **neutral on speed,
neutral on bytes, safe on correctness.**

## What the profile actually establishes: the wall is overlap, not read rate

Two numbers, and they decide the roadmap:

1. **GPU busy ~55% of an ~87 s pass = ~48 s of GPU work.** The shader floor is
   real and it is the binding constraint. If the GPU were kept busy the whole
   pass, 14,960 tok / 48 s = **313 t/s** — the bottom of your 300–400 band.

2. **The copy stream is no longer the constraint.** One-pass prefill at 300 t/s
   needs 68 GB / 50 s = 1.37 GB/s; at 400 t/s, 1.82 GB/s. Measured sustained
   is **1.43–1.55 GB/s** — already at or above the 300 t/s requirement, before
   any depth change. This is the direct, measured falsification of the
   "queue depth is the wall" hypothesis: the depth-1 memcpy path was *already*
   extracting enough bytes per second to reach 300 t/s if the copies and the
   shader were not fighting each other.

The 134 GB is also worth noting: it is ~2x the 68 GB of expert weights, so
something re-reads roughly half the copy volume. Candidates: the copy is
issued per split and some tensors are copied twice, or the PLE/other tensors
join the stream. Unresolved, and a plausible partial cause of the stalls.

**Therefore the campaign is an overlap problem, and the build is now
unambiguous:** the ~48 s of shader work and the ~39 s of non-GPU time are
serialized with each other. Making them concurrent is what produces 300–400
t/s — not more read depth, not more bandwidth. 300 t/s is achievable by
overlap alone, with the copy rate already in hand.

Concretely, next builds, in order:

1. **Overlap the copy with compute across the split boundary.** The copy for
   split N+1 is issued by `ggml_backend_vk_set_tensor_2d_async` and drained in
   `ggml_vk_submit_transfer_ctx`, gated on a single fence; compute waits on the
   transfer semaphore. Multiple staging buffers + per-buffer fences, so
   split N+1's copy is in flight while split N's shader runs, is the direct
   expression of "the two are independent". This is the 300–400 lever.
2. **Account for the 134 GB.** Instrument which tensors are copied and how
   often; halving redundant copies is free throughput.
3. Decode 15–20 t/s is untouched by all of this and still needs real code work
   (every env knob there is measured closed).

## Bookkeeping

- New: `scripts/bench-prefill-depth-profile-1002.sh` (order-controlled parity +
  GPU-busy/read_bytes profile in one run), `scripts/probe-copy-pread-1001.sh`,
  `scripts/bench-copy-pread-parity-1001.sh`.
- Artifacts: `docs/bench-results-2026-10-02/` (`pread-depth-results.txt`,
  per-arm `.json`, `.server.log`, `.gpu.csv`, `.io.csv`).
- llama.cpp: `321dff9b5` (copy pread depth; 4 in_memcpys write drains,
  env-gated, parity PASS).
- Parked: `patches/llama-cpp/order-aware-copy-readahead-WIP.patch` (the
  unmeasured 10-01 order-aware fadvise diff — neutral at best, and its 1.5 GB
  window on 16 GB is what caused the 22:47 page-allocation failure).
