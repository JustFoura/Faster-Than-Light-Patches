# Flash-Next expert-cache preseed tests

Date: 2026-09-25

## Setup

RX 9060 XT / RADV Vulkan, Ryzen 7 5800X, 16 GiB RAM. All server runs kept the
200,000-token context (`/props`: 200,192), Q8_0 K and V, one slot, 88 cache slots
per layer, and the same 128/64 batch/ubatch settings. The model and GTT KV path
were unchanged. Requests used the local `/completion` probe with a fixed seed,
`ignore_eos`, and no-thinking prompt wrapper. Writing runs requested 128 output
tokens; coding runs requested 64.

The seed profile was built from the saved prose routing trace at
`/tmp/flash-next-routes-prose.txt`. It ranks experts independently per layer by
frequency. The experimental llama.cpp build logged that it loaded 44 or 88
experts into each of 48 layers. Those entries occupied the existing cache slots;
the 44-slot profile left the other slots to the normal LRU cache. The KV cache
remained Q8_0 in Vulkan host memory for every run.

## Measurements

| Run | Prompt reuse | Prompt t/s | Decode t/s | Process `read_bytes` | Major faults |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard cache, cold writing request | 0 | 0.882 | 0.580 | 394.4 GB | 320,675 |
| Standard cache, repeated writing request | 66 tokens | 0.231 over 4 evaluated tokens | 0.743 | 233.9 GB | 283,222 |
| 44-slot prose-profile preseed, writing | 0 | 0.864 | 0.747 | 318.6 GB | 249,903 |
| 88-slot prose-profile preseed, writing | 0 | 0.946 | 0.816 | 241.1 GB | 214,190 |
| Standard cache after writing warm-up, coding | 0 | 0.835 | 0.526 | 267.1 GB | 258,514 |
| 88-slot prose-profile preseed, coding | 0 | 0.886 | 0.600 | 243.5 GB | 185,333 |

`read_bytes` and fault counts are process-counter deltas during each benchmark
request; they are not server-reported throughput. The 88-slot coding counters
were captured around the request from `/proc` because the initial probe was
given the wrong PID. Timing results still came from the completed server
response.

## Interpretation

Preseeding the entire cache improved the measured writing run over the cold
standard-cache run and reduced process reads and major faults. The repeated
standard-cache writing run was already close to the 44-slot preseed result.
The prose-derived profile also carried over to the coding prompt, but the coding
rate remained low. These are individual runs with a changing OS page cache, so
the differences are directional rather than a controlled estimate of a stable
gain. The route profile also overlaps the writing domain and has no independent
holdout measurement.

The experiment does not meet either intended use threshold. Startup seeding
gets the dynamic cache closer to its warmed state earlier, but expert-cache
preloading alone does not remove the large process-read and major-fault burden.
The next investigation should isolate host model-page residency and transfer
costs, while preserving the 200k Q8 GTT KV configuration. Do not increase the
cache slot allocation on this GPU without a measured VRAM budget; the test only
changed which existing slots were populated.
