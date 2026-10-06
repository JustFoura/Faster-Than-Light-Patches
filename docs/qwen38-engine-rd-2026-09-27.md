# Qwen3.8-Flash-Next: model-specific runtime R&D

**Date:** 2026-09-27
**Purpose:** engineer around this model's PLE table and routed experts; use token/s as instrumentation, not the objective.
**Repo impact:** this note and source-only experiment artifacts under `experiments/qwen38-ple-prototype/`. Build trees and generated binaries remain outside the repository.

## Engine used

The full-model prompt runs used **llama.cpp server**, built locally at upstream `b11200` (`81bc6b83f827df746eb129235488d325c49cae52`) with the Vulkan backend. The buffered row-read build was based on that commit; the `on-direct` build also included upstream PR #28136 commits `4ed9e239d578567801cbf0ca91363c67b8c61885` and `f5d8427d3c5b91c1bff18857be7b15d083361889`, plus the local O_DIRECT prototype patch preserved with this report. The app's installed `/usr/bin/llama-server` was a different build: llama.cpp `0.4.0-dev`, build 10809, without `--lazy-mode on-direct`. The standalone HIP programs below were probes, not an alternate model-serving engine.

## What is already here

- The requested IQ4_XS GGUF already exists at `app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf` (97,473,155,360 bytes / 90.78 GiB). It was not changed.
- A separate Q8_0 MTP sidecar exists at `app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2,786,568,256 bytes / 2.60 GiB). It was not loaded or removed.
- The GPU is an AMD Radeon RX 9060 XT, `gfx1200`, with 16 GiB VRAM. The model file is on a Btrfs volume. The active engine before this work was llama.cpp 0.4.0-dev; its `--lazy-mode` did not include `on-direct`.
- The test host has a Ryzen 7 5800X, 16 GiB system RAM and typically about 5–10 GiB available, depending on file cache and other activity. ROCm 7.2.5 is installed and `rocminfo` sees the RX 9060 XT as `gfx1200`.

## Model layout that matters for an engine

The Qwen report describes a 125B backbone, about 6B active per token, plus a separate 51B n-gram embedding capacity held off accelerator. The model uses Gated DeltaNet and sparse attention, a four-way gated residual stream, one PLE layer, and MTP for speculative decoding. [Qwen architecture report](https://arxiv.org/abs/2608.30320)

Inspection of this exact GGUF found:

- `per_layer_token_embd.weight`: IQ4_NL, 320,001,536 rows × 160 values, 28,800,138,240 bytes (26.82 GiB), starting at file offset 536,121,120. Each IQ4_NL row is 90 bytes. This is the 51.2B-value PLE table.
- The three routed-expert matrices total 65,431,142,400 bytes (60.94 GiB) across 144 tensors in 48 layers. That averages 2.54 MiB per expert per layer. Ten routed experts across all 48 layers are about 1.19 GiB of expert weights for one token's route, excluding shared experts and runtime buffers. Two or three such sets would be roughly 2.38–3.57 GiB if cached in VRAM; this is a sizing estimate, not an allocation result.
- PLE and experts have different access patterns. PLE needs a sparse set of small rows per input token. Routed experts need large matrices for selected expert IDs and are consumed by MoE matmuls.

## PLE read-path tests

Two Vulkan llama.cpp servers were built from upstream commit `81bc6b83f827df746eb129235488d325c49cae52`; the second also included the two commits from [llama.cpp PR #28136](https://github.com/ggml-org/llama.cpp/pull/28136). The same model, GPU, 4k context, batch 2048, ubatch 512, Flash Attention, eight CPU threads and `--cpu-moe` setting were used. Each run evaluated the same fresh 452-token varied-prose prompt, disabled prompt-cache reuse, and generated one token. The exact PLE file range was evicted and verified with `mincore` before each run.

| PLE path | Prompt evaluation | Prompt rate | Observation |
| --- | ---: | ---: | --- |
| Lazy mmap (`--lazy-mode on`) | 87.71 s / 452 tokens | 5.15 tok/s | Baseline |
| PR direct row reads (`--lazy-mode on-direct`) | 81.26 s / 452 tokens | 5.56 tok/s | One run; about 8% less elapsed time |
| Page-batched O_DIRECT prototype | 83.82 s / 452 tokens | 5.39 tok/s | One run; 0 PLE pages resident after the request |

These are exploratory single runs, not a speed claim. The PLE pages were cold, but expert-weight pages were not globally evicted, so this is not a full cold-model comparison. In this setup all MoE weights were assigned to CPU; it did not test a GPU expert cache. The observed rates are consistent with the CPU MoE path dominating this short prompt, while the PLE path accounts for a smaller part of total time.

The PR's original `on-direct` implementation batches row IDs, sorts and deduplicates them, then performs parallel ordinary `pread()` calls and CPU dequantization. It does **not** open the file with `O_DIRECT`; `pread()` can still use the OS page cache. The PR reports larger cold-prefill gains on other systems, including RX 9070 XT / Vulkan examples, but those results do not predict this RX 9060 XT + IQ4_XS run. [PR discussion and measurements](https://github.com/ggml-org/llama.cpp/pull/28136)

The separate O_DIRECT prototype grouped requested rows by aligned 4 KiB file pages, read each unique page into an aligned host buffer, and staged/dequantized the requested rows. A read-only probe also issued 2,048 random aligned page reads (8 MiB total) against the actual PLE range in 0.495 seconds; afterward `mincore` showed 1 of 7,031,283 PLE pages resident. On this filesystem, O_DIRECT therefore provides a concrete way to keep the large table out of the kernel page cache. It did not improve the one short prompt over buffered reads. The reader patch and source for the standalone probes are preserved in `experiments/qwen38-ple-prototype/`; the patch depends on the two upstream PR commits listed above and is an experiment, not a production-ready integration.

## SSD-to-GPU proof of concept

A separate HIP 7.2.5 probe used `hipHostMalloc` to create a 4 KiB mapped/coherent host buffer, read an actual page of the PLE tensor into it with O_DIRECT, and had the RX 9060 XT access that buffer. The GPU checksum matched the CPU. A second probe read PLE row 0 (90 IQ4_NL bytes) from the same on-disk page and dequantized all 160 values on the GPU; results matched ggml's CPU `dequantize_row_iq4_nl` exactly (0 differing values).

This proves a useful data path on this machine:

```text
NVMe --O_DIRECT--> small HIP-pinned host page buffer --GPU-visible mapping--> GPU dequant/gather
```

The read lands in pinned **system RAM**, not VRAM. The GPU accesses the mapped buffer across the host/device interconnect. A later engine can either dequantize/gather from that buffer or copy only the finished vectors into VRAM. The current full-model O_DIRECT prototype still does CPU dequantization and uses Vulkan; the HIP PLE-row kernel is a standalone proof, not yet integrated into the model graph.

AMD's current ROCm matrix lists the RX 9060 XT as supported, and HIP documents mapped pinned host allocations and async host/device transfer. I did not find an AMD equivalent to NVIDIA's cuFile GPUDirect Storage path for this Radeon/Vulkan setup; treat pinned host staging as the available starting point. NVIDIA GDS also requires aligned direct-I/O requests and may use staging for small or unaligned transfers. [ROCm Linux support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html), [HIP host memory API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html), [NVIDIA GDS overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/)

## Engine choice and next architecture

**Extend llama.cpp around its GGUF model loader and Qwen4Exp graph. Do not start with a full rewrite.** It already loads this exact GGUF and implements the model's GDN, sparse-attention, gated-residual, MoE and PLE graph. A Hugging Face engine would need compatible safetensors weights and architecture support; reverse-dequantizing IQ4_XS to safetensors cannot recover the original BF16/FP16 values and would inflate the file. Keeping GGUF avoids that round trip.

The proposed engine shape is a bounded three-tier hierarchy:

1. **NVMe:** keep the existing GGUF as the backing store for PLE and routed experts.
2. **Small pinned host ring:** read aligned PLE pages (and later expert ranges) with O_DIRECT into a fixed, bounded HIP-mapped ring. Use asynchronous reads, sort/deduplicate page IDs, and instrument bytes, waits and residency.
3. **VRAM:** keep the active transformer weights and working buffers resident; add a persistent expert-slot pool for frequently reused routed experts. Start with an explicit two/three-set budget as a capacity hypothesis, and leave graph/KV headroom.

For PLE, token IDs make the row requests knowable before the PLE layer consumes them. A first integrated version should issue page reads and GPU lookup/dequant concurrently with the earlier backbone work, double-buffer the page ring, and synchronize only when the PLE result is needed. This moves the experiment from “can the file stay disk-backed?” to a model-native asynchronous data path.

For experts, llama.cpp already supports CPU MoE placement and selective copies, while a community `--moe-expert-cache` design adds persistent VRAM slots with expert-ID remapping. That design is validated on CUDA, is mainly aimed at repeated decode routes, and has no SSD tier. It is a useful reference for a later Vulkan/HIP compatibility prototype, not a ready-made answer for this machine. The separate SSD-streaming discussions are also design/PoC material, not a production engine. [Expert-cache RFC](https://github.com/ggml-org/llama.cpp/discussions/28248), [two-tier expert-cache issue](https://github.com/ggml-org/llama.cpp/issues/20757), [SSD-streaming discussion](https://github.com/ggml-org/llama.cpp/discussions/27149)

Other engines are less direct fits for the current artifact. vLLM has model-specific PLE work and ROCm support, but the Qwen PLE auxiliary-device issue is still an open feature request, while the exact local artifact is GGUF IQ4_XS rather than a vLLM-oriented HF checkpoint. [vLLM PLE placement request](https://github.com/vllm-project/vllm/issues/53908)

## Speculative decoding status

DFlash2, DSpark and MTP target speculative **decode**: a draft/speculator proposes future tokens and the target verifies them. They do not accelerate prompt prefill or solve PLE/expert storage. The local MTP sidecar is target-specific and available for later decode work. The public DFlash2 checkpoint found for Qwen3.8 is named `Qwen3.8-27B-DFlash2`, not a verified Flash-Next speculator; DSpark's documented pre-trained table currently lists GLM-5.2, with training available for other targets. For Flash-Next, use the existing MTP sidecar only when decode work begins, or train/validate a matching DFlash2/DSpark speculator against this verifier. [Speculators getting started](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/getting_started.md), [DSpark guide](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md), [DFlash model list](https://github.com/z-lab/dflash)

## Concrete next experiment

Integrate the proven HIP page read and IQ4_NL row dequant into the Qwen4Exp PLE input path behind an experimental mode, with:

- fixed-size pinned buffers and asynchronous O_DIRECT reads;
- one/two pages per PLE row correctly handled at page boundaries;
- GPU output checked against the existing CPU row gather and model logits;
- metrics for PLE pages read, actual page-cache residency, pinned bytes, GPU staging and time spent waiting;
- the original GGUF unchanged and a buffered fallback if O_DIRECT is unavailable.

After that path is independently correct, prototype the expert-slot cache against the AMD backend. Keep the PLE and expert changes independently switchable so results explain which subsystem moved data and when.
