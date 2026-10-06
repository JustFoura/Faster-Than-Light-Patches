# Qwen3.8 routed-row reuse prototype

## What this changes

The current CPU `MUL_MAT_ID` path first groups token rows by expert. For every
selected expert, it then calls the quantized vector-dot kernel once per routed
token and output row. That is correct, but it makes the same packed expert row
pass through the dot kernel again for each token assigned to that expert.

This prototype adds an opt-in path for the two expert formats in the local
Flash-Next GGUF:

- gate and up: IQ4_XS;
- down: IQ4_NL.

It also compacts the expert-row map. The current kernel allocates a maximum
`n_experts * top_k * n_tokens` map even though only `top_k * n_tokens` route
entries exist. At 512 tokens that is 20 MiB reserved for 40 KiB of route
entries; `compact-route-index.patch` replaces it with a prefix-summed,
expert-major array. This is an exact layout change and does not alter model
math.

When at least four token rows in an expert tile are routed to the same expert,
it reconstructs four Q8 activation rows and one expert weight row, then reuses
that weight row across all four activation rows. The ordinary quantized path
remains the fallback for other formats, small route groups, and the default
configuration.

There are three component patches and one combined patch, all against the same
engine checkout. Apply `qwen38-prefill-prototype.patch` once to get all three
components, or apply `compact-route-index.patch`, then
`route-histogram.patch`, then `iq4-multi-rhs.patch`. The compact route map is
always on after applying its patch. Set `GGML_MOE_ROUTE_HISTOGRAM=1` for a
per-layer histogram on prompt ubatches. Set `GGML_MOE_Q4_MULTI_RHS=1` to enable
the four-row weight reuse path.

The patches target checkout `bccbacdb8` with its local cache and host-streaming
changes present. They are not ready-made patches for arbitrary upstream
llama.cpp revisions. The repository's `patches/llama-cpp/build.sh` builds a
different upstream base in `~/build/llama.cpp` and force-checks out that tree;
it is not the build path for this prototype.

Status update 2026-09-27: the three component changes (compact route index,
route histogram, IQ4 multi-RHS) are applied — uncommitted — to
`~/build/llama.cpp-moe-cache/ggml/src/ggml-cpu/ggml-cpu.c`, together with that
tree's `LLAMA_MOE_FANOUT` concurrent page-in work. `madv-dontneed-window.patch`
in this directory is the additional bounded-window step built on top of them:
after each MoE node, and only for file-backed mappings verified through
`/proc/self/maps`, the routed expert pages are released with `MADV_DONTNEED`
(gate `GGML_MOE_MMAP_DONTNEED=1`), so each layer's expert range streams through
the page cache once instead of thrashing the LRU. A full build of that state
lives in `~/build/llama.cpp-prefill-exp`. A/B driver:
`scripts/bench-prefill-window.sh` with `prompt-500.txt`; results are collected
under `docs/bench-results-2026-09-27/` and summarized in
`docs/qwen38-prefill-decode-plan-2026-09-27.md`.

## Why this is model-specific

This is not a cache or a routing approximation. It keeps the exact selected
experts and GGUF weights. The model has 512 routed experts and selects 10 per
token. With uniformly distributed routes, a 64-token batch touches roughly
367 distinct experts per layer (640 route rows, or 1.74 rows per touched
expert); a 512-token batch touches nearly all 512 (about 10 rows per expert).
Real router skew can change those counts. Larger prompt batches therefore
create repeated rows per expert, while single-token decode does not. The
multi-row path targets that prefill reuse. The current launcher uses a
64-token ubatch, so the route histogram will show whether its four-row path
fires often enough there; it should be more applicable to wider prompt chunks.

The existing `MUL_MAT_ID` implementation already groups rows by expert, so
sorting routes or adding another generic grouped-GEMM layer would duplicate
work the engine already does. The prototype instead changes the inner
weight/activation loop for the exact IQ4 formats in this model.

## Expected trade and uncertainty

For each four-row group, it converts one packed weight row to F32 and uses it
against four routed activation rows, instead of decoding that row once per
route. It also converts those four Q8 activation rows to F32 once per route
group. This exchanges repeated packed-weight reads for F32 scratch traffic and
F32 dot products, so it may lose when the expert is already cache-resident,
compute is the limit, or route groups are small. A packed integer multi-RHS
kernel could avoid much of that conversion cost, but the current prototype is
the simpler probe. The four-row threshold is a first hypothesis, not a chosen
optimum.

The accumulation order changes, so logits may differ slightly from the
existing quantized dot implementation. Greedy output and logit error must be
checked before treating this as usable. The patch has **not** been built,
tested, or benchmarked; no speedup is claimed.

## Focused evaluation needed

1. Count per-layer route-row groups in bins `1`, `2–3`, `4–7`, and `8+` for a
   real prompt batch. This shows how often the new path can run.
2. Compare the old and new inner kernels on those actual expert row counts,
   including cold mapped weights and warm page-cache runs.
3. Compare prompt elapsed time, process read bytes, major faults, CPU time,
   per-layer MoE time, Vulkan transfer/kernel time, and peak activation
   workspace with the same model and batch shape.
4. Check output logits and greedy tokens against the original kernel.
5. Only then consider tile sizes 2/4/8 or a tiled integer dot implementation.

## Larger engine direction

The larger opportunity is a bounded, layer-major prompt wavefront. Today
`process_ubatch()` runs a graph through every layer, then moves to the next
ubatch. On a long prompt that revisits the same layer's routed weight set once
per ubatch. A Qwen4Exp executor could instead:

1. Keep one bounded chunk of hidden rows and the required GDN/QSA state.
2. For one layer, compute exact router IDs for all rows in that chunk.
3. Compact rows by expert, stream each selected expert through a weight-stationary
   kernel, and apply it to every row routed there before releasing the expert.
4. Complete that layer's attention/residual work and advance to the next layer.

That is a scheduling and kernel change, rather than asking the ordinary graph to
allocate a much larger ubatch. It can remove repeated expert-weight passes
across graph ubatch boundaries while keeping temporary activation storage
bounded. It must preserve causal QSA and recurrent GDN state exactly; that is a
larger refactor, so the opt-in CPU kernel is the first narrow implementation
probe. The current decode expert cache is not a prompt scheduler: its graph hook
is restricted to one-token batches.
