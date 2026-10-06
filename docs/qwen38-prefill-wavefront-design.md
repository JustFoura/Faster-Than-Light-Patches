# Qwen3.8 prompt wavefront: engine design

## Objective

Build a prompt-only execution path for the existing Qwen3.8 Flash-Next GGUF and
current machine. The goal is to stop replaying a layer's expert-weight working
set for every llama.cpp ubatch. This design does not depend on MTP or a draft
model, and it does not change the served weights.

## What the current code does

`llama_context::process_ubatch()` builds or reuses one graph for an ubatch. That
graph walks every Qwen4Exp layer before the context advances to the next
ubatch. `llama_model_qwen4exp::graph` also shows why a plain split around
`MUL_MAT_ID` is insufficient:

- each layer has a wide hyper-connection residual stream;
- a layer may run recurrent GDN or indexed sparse QSA attention;
- QSA's indexer cache must remain cell-aligned with its attention cache;
- the PLE layer hashes each token with its preceding-token history;
- the layer combines attention and MoE branches, including a shared expert.

The route tensor is therefore only one part of each layer's state. A correct
executor must carry all of those structures across the layer boundary.

## Candidate: bounded layer wavefront

Keep a prompt chunk of `C` tokens active through one layer before starting the
next layer. For each layer, keep hidden rows in a pair of bounded workspaces;
the main activation capacity is approximately
`2 * C * n_embd * hc * bytes_per_value`, plus attention/recurrent state and the
expert route/output buffers. The local GGUF has `n_embd=2560` and `hc=4`, so
two F32 ping-pong buffers would use about 40 MiB at `C=512`, 320 MiB at
`C=4096`, or 1.25 GiB at `C=16384`. These figures exclude attention/recurrent
state and temporary graph buffers. This makes chunk size an explicit memory
limit rather than relying on a giant all-layer graph allocation.

The GGUF also has 512 experts with top-10 routing. Under a uniform-route model,
the expected number of distinct experts seen in a `C`-token chunk is
`512 * (1 - (1 - 10/512)^C)`: about 367 at `C=64`, and effectively all 512 by
`C=512`. The route histogram is needed because actual router skew changes the
fan-in per expert, which determines how much the weight-stationary kernel can
reuse each row.

This gives a concrete reason to move the scheduling boundary. Eight separate
64-token ubatches issue about `8 * 367 = 2,936` expert-matrix visits per layer
under the uniform model, while one 512-token layer chunk visits about 512
distinct experts once. That is roughly 5.7 times fewer expert-matrix visits
for the same 512 prompt rows. It is a route-coverage estimate, not an SSD-byte
or speedup prediction: actual routing, GPU-cache hits, page-cache residency,
and matrix streaming determine the real traffic. The 512-token activation
ping-pong estimate above is about 40 MiB in F32, so it keeps the reuse window
wide without building one all-layer graph for the full prompt. The extra
attention, route, and backend workspace still needs to be measured against
available RAM and VRAM headroom.

For each layer in order:

1. Apply PLE at its designated layer, using the exact token history for the
   chunk.
2. Run that layer's existing GDN or QSA attention path and update its existing
   recurrent, KV, and indexer state.
3. Compute the router logits and exact top-10 routes for all `C` hidden rows.
4. Compact `(expert, token-row, route-weight)` assignments into expert-major
   lists. Keep route weights and original row indices alongside each entry.
5. For each expert with assigned rows, stream its gate/up/down matrices through
   a weight-stationary multi-row kernel. Consume all rows assigned to that
   expert before advancing, then release the matrix tile.
6. Add the shared-expert result and combine routed outputs using the original
   route weights. Complete the layer's hyper-connection residual and use it as
   the next layer's input.

This can reduce expert matrix passes across graph-ubatch boundaries: a long
prompt chunk routes against a layer's expert set once, then all rows for each
expert are computed before that weight range is left. It does not assume that
all expert weights fit in RAM or VRAM. The matrix can be streamed in tiles.

## Research boundary

Recent systems work already combines SSD expert delivery, tiered residency,
CPU/GPU assignment, and grouping prefill rows by expert. SSD-LLaMA reports
those as parts of one system, with an expert-pack layout, direct/asynchronous
delivery, and GPU-side decompression. Its end-to-end numbers are for its own
models and NVIDIA setup, so they do not predict this GGUF/Vulkan/RDNA4 machine.
[SSD-LLaMA, arXiv:2609.18110](https://arxiv.org/abs/2609.18110)

The custom work here is narrower: keep the current GGUF and engine, and change
the scheduling boundary around this model's PLE, recurrent GDN, indexed QSA,
and hyper-connection state so a wider prompt route set can share each streamed
expert. That is an implementation hypothesis, not a claim that expert-major
prefill is new. A separate CPU/GPU hybrid study also reports that CPU MoE
kernels and CPU/GPU coordination can dominate long-prefill results, but its
dual-socket CPUs, AVX-512, and RTX-class GPUs differ materially from this host.
[OSDI '26 CPU–GPU MoE study](https://www.usenix.org/conference/osdi26/presentation/wang-wenxin)

## First implementation boundary

Do not replace attention kernels or the memory system in the first cut. Refactor
the Qwen4Exp graph builder into three explicit interfaces:

- `layer_begin(layer, hidden, positions, token_history, memory_view)` returns
  attention output, attention injection, and a state-update object;
- `layer_route(layer, hidden)` returns exact selected expert IDs and normalized
  route weights;
- `layer_finish(layer, hidden, attention_result, route_result, state_update)`
  returns the next wide residual and commits the per-layer state.

The existing graph methods remain the reference implementation. The new
executor should initially call the same ggml operations for attention,
normalization, and residual math, while replacing only the scheduling and MoE
subgraph. This isolates semantic risk to routing, expert assignment, and
combination.

The CPU four-row patch in `experiments/qwen38-multi-rhs-prefill/` is a narrow
kernel probe, not this executor. It uses F32 conversion around the current
quantized formats, so its value is uncertain; a later packed integer multi-RHS
kernel is the likely high-payoff version if the probe confirms that repeated
weight reads are limiting. The route map compaction is independently useful
because it stores one entry per actual route rather than reserving space for
every expert-route combination.

## Correctness hazards to account for

- **GDN state:** recurrent and convolution states are per-layer and per-sequence;
  chunk order must remain identical.
- **QSA state:** attention KV cells and indexer cells are kept in lockstep. The
  layer scheduler cannot commit one cache without the other.
- **PLE:** row hashes depend on preceding tokens, EOS boundaries, and stored KV
  token IDs. A chunk beginning mid-sequence needs history from memory.
- **Hyper-connections:** the executor carries `hc` streams, not a single
  `n_embd` activation vector.
- **Output rows:** llama.cpp can request logits for only selected rows. The
  final layer still needs the same output-row selection and mapping.
- **Multiple sequences:** begin with one sequence per prompt chunk; mixed-sequence
  batches need per-sequence state and route-row ownership.

## Work sequence

1. Collect the per-layer route histogram using `route-histogram.patch`; confirm
   actual rows per expert for the current prompt ubatch sizes.
2. Build and compare the opt-in four-row CPU kernel on short prompts, including
   logit error and greedy-token agreement.
3. Add an experimental Qwen4Exp graph mode that carries explicit hidden rows
   between two adjacent layers while reusing the existing attention and memory
   implementations.
4. Extend that mode to all layers, first for one sequence and a bounded prompt
   chunk. Check recurrent/KV/indexer state and outputs against the reference
   graph.
5. Add prompt chunking across the full input while preserving token history and
   output selection. Only after correctness, compare process I/O, page faults,
   CPU time, GPU transfer bytes, and elapsed prompt time.

## Measurements that choose the next change

Prompt rate alone cannot tell whether the next constraint is storage, CPU
expert math, or GPU/host traffic. Add one synchronized per-layer record for the
existing path and the wavefront path with these fields:

- routed rows, unique experts, and expert fan-in bins, both before and after
  GPU-cache hits are removed;
- time in route grouping, selected-expert page-in, and expert matrix work;
- CPU wall/core time, process read-byte and major-fault deltas;
- Vulkan transfer and kernel time for the layer, plus peak activation scratch.

Interpret the first capture this way: large read/fault deltas with low CPU
utilization points at selected-weight streaming; low read deltas with saturated
CPU time points at the CPU quantized dot path; large Vulkan transfer time points
at CPU/GPU boundary traffic; and high time outside those stages points at
attention, PLE, or scheduler overhead. Keep server prompt timing as an
end-to-end check, with prompt-cache reuse disabled and the exact token count
reported. Process read bytes are accounting data, not physical SSD throughput.

The route histogram and kernel patch are not built or tested yet. No throughput
claim is attached to either artifact.
