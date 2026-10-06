# Flash Next speed work

## Targets and constraints

- Target: at least 10 tokens/s decode and 100 tokens/s prefill.
- Keep Qwen3.8 Flash Next IQ4_XS, the existing llama.cpp/Vulkan path, 200k context,
  and Q8 K/V cache.
- Do not requantize, change hardware, or switch models.
- Exclude the prohibited September 25 MTP/cache combination, cache-insert sweeps,
  and route-derived expert-cache preload profiles. None of their logs or results
  informed this work.

## Comparative research

Colibrì's GLM-5.3 implementation is a model-specific C engine. Its notes describe
compact expert storage, a bounded expert-major prefill workspace, context state
kept separate from expert weights, and component-level timing. These are useful
systems design references; reported rates are hardware and workload specific.
[GLM-5.3 engine notes](https://github.com/JustVugg/colibri/blob/main/docs/glm53-flash.md)

The local Flash Next path is llama.cpp's Qwen3-Next graph, with host-resident
experts, CPU `MUL_MAT_ID` work, Vulkan layers, and a memory-mapped GGUF. Its
current per-request tool already reports prefill/decode timings and process I/O
counters.

The mmap loader already marks the GGUF file sequential for `posix_fadvise`,
then marks lazy tensor ranges `POSIX_MADV_RANDOM`. The candidate therefore does
not add a second general mmap policy: it tests whether an explicit
`MADV_WILLNEED` on this graph node's selected expert extents can start those
specific page-ins early enough to overlap CPU work. Its effect is uncertain
because Linux may treat the hint as advisory; the matched run decides whether
it helps.

## Fresh diagnostic

Read-only inspection found an already-running Flash Next server on localhost
port 1235. `/props` confirmed the `-nocow` IQ4_XS model and a 200,192-token
context. The running command uses batch and ubatch 256, with Q8 K/V; no runtime
settings were changed.

One new, uncached request used a purpose-written 124-token prompt and generated
48 tokens. The server reported:

| Phase | Tokens | Time | Rate |
| --- | ---: | ---: | ---: |
| Prompt processing | 124 | 62.04 s | 2.00 t/s |
| Decode | 48 | 118.68 s | 0.396 t/s |

Across that request, the process recorded 254,596,509,696 read bytes, 86,835
major page faults, and 1,439,118 minor faults. The data makes file-backed weight
access a strong candidate bottleneck; a controlled change is still needed to
establish how much of the delay it explains.

## First implementation candidate

`patches/llama-cpp/flash-next-demand-prefetch.patch` adds an opt-in Linux
`madvise(MADV_WILLNEED)` hint in CPU `MUL_MAT_ID`. Once the current graph node's
route IDs are known, it asks the OS to read the selected experts' mapped ranges
ahead of their matrix products. Adjacent selected ranges are coalesced. It adds
no persistent cache, learned route profile, weight transformation, or KV change;
with `GGML_MOE_MMAP_WILLNEED` unset, the path is inactive.

The Flash Next launcher exposes that gate as `FLASH_MMAP_WILLNEED=1`. Its
default model path now matches the existing `-nocow` model location, while the
`FLASH_MODEL` override remains available.

This is a prototype motivated by the fresh read/fault counters. It needs a
separate build and a same-settings A/B before it can be judged. The live server
was left running and unchanged after the diagnostic request.

## Matched A/B results

The candidate was built in `/tmp/flash-next-prefetch-vulkan` from llama.cpp
commit `bccbacdb8`, using the existing Vulkan host-memory Q8 K/V patch plus the
demand-prefetch patch. Both servers used the same IQ4_XS file, 200,192 context,
batch/ubatch 256, Q8 K/V, and fixed 88-slot/2-insert cache settings. The request
used a new 141-token coding-oriented prompt, no-think wrapper, temperature 0,
seed 42, and 32 generated tokens. Prompt cache was disabled.

| Build | Run | Prompt t/s | Decode t/s | Read bytes | Major faults |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Initial | 2.12 | 0.303 | 236.1 GB | 80,677 |
| Candidate, 8 threads | Initial | 2.46 | 1.362 | 93.1 GB | 20,897 |
| Baseline | Repeat | 2.15 | 0.290 | 243.1 GB | 81,716 |
| Candidate, 8 threads | Repeat | 2.49 | 1.262 | 96.2 GB | 24,735 |

The candidate produced the same 32 tokens for the same seed. Across two runs,
the demand hint raised decode throughput by about 4.4x and lowered prompt read
volume and major faults, while prompt throughput improved only about 16%.
These are process-counter read bytes, not a claim about physical disk bandwidth.
The baseline repeat remained close to the initial baseline, which reduces (but
does not eliminate) the concern that the first candidate result was only OS
page-cache warming.

One 16-thread candidate run kept prompt processing at 2.46 t/s and reduced
decode to 0.765 t/s. Keep the 8-thread configuration for now; thread oversubscription
did not help on this 8-core/16-thread Ryzen 7 5800X.

The current best measured result is still below both requested targets
(10 t/s decode and 100 t/s prompt processing). The next work needs to attack the
large remaining CPU-MoE/read-path cost rather than adjust thread count. No MTP
sidecar, cache-insert sweep, route-derived preload profile, requantization,
hardware change, or model swap was used.
