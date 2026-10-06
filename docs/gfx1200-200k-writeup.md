# 13 → ~30 t/s on a 16GB RDNA4 card: GTT-resident KV + block-diffusion drafting

How a Qwen3.8-27B vision model with a 200k-token context went from 13 t/s to ~30 t/s decode
(~400+ t/s prefill) on an AMD RX 9060 XT (gfx1200, 16GB) with 15GB of system RAM — using
llama.cpp, no quantization changes, no new hardware.

## The stack

| Piece | Local file | Source |
| --- | --- | --- |
| Base model (IQ4_XS, 3.84 bpw, 12.1GB) | `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf` | [byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF) |
| Vision encoder (mmproj, Q8_0, 629MB — self-quantized from the HF repo's BF16 file) | `mmproj-q8_0.gguf` | same HF repo |
| DFlash2 drafter (Q4_K_M, 1.14GB) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| Upstream base model | — | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |

Memory layout (all requirements held at once): base model + drafter on the GPU, mmproj on
the CPU, a 200k-token Q8_0 K/V cache in system RAM, peak VRAM **14.6-14.7GB of 15.9GB
including the desktop compositor**.

## The bug that set the ceiling

The old setup used `--no-kv-offload` to keep the KV cache in system RAM. What that flag
*actually* does upstream is nastier: it pins the attention computation of every
full-attention layer to the **CPU backend** (`llama-graph.cpp` calls
`ggml_backend_sched_set_tensor_backend(..., backend_cpu)`). This model has 17 such layers.

So every generated token, the GPU stopped 17 times, shipped activations to the CPU, waited
for CPU attention, and shipped them back: ~40ms of pure overhead per pass. Plain decode
capped at 13 t/s. And speculative decoding — which runs a *draft* pass, a *KV-injection*
pass, and a *verify* pass per cycle — paid the tax three times per cycle, which is why every
attempt at DFlash2 made things *worse* (4-12 t/s) instead of better.

The KV cache living in system RAM was never the problem. The CPU attention was.

## The fix

Three changes to llama.cpp (base commit `5266f24da7`; the full diff travels with this
writeup as `gfx1200-hostmem-kv-dflash.patch`):

1. **A GTT hostmem buffer type for ggml-vulkan.** A new buffer type that allocates Vulkan
   memory as `HOST_VISIBLE|HOST_COHERENT` while *excluding* `DEVICE_LOCAL` — i.e. system RAM
   the GPU can read and write directly over PCIe (amdgpu GTT). The exclusion matters: a
   naive "host-visible" request can silently land in BAR memory and eat VRAM, so this fails
   hard instead. With `LLAMA_KV_HOSTMEM=1`, the KV cache allocates in GTT and **attention
   stays on the GPU** — the flash-attention kernels read the cache over the bus directly.
   No copies, no round trips, and the 7GB cache counts against system RAM, not VRAM.

2. **`--spec-draft-ctx N` for the drafter.** The DFlash2 draft normally clones the full
   target context into its own KV cache — at 200k that's another multi-GB cache and the
   thing that previously OOM'd the card. With a sliding 8k draft window (positions remapped,
   evicted positions dropped; the draft's KV is rebuilt from injected target-layer features
   anyway), the draft cache is ~40MB on the GPU.

3. **Draft placement** — the draft no longer inherits `--no-kv-offload` from the target, so
   its KV always lands in VRAM where it belongs.

Result: one unified GPU graph per decode pass. No CPU attention, no split boundaries, no
copies — and speculative decoding flips from a liability to a ~2.5-3x amortizer.

## Launch configuration

A small wrapper script around the custom binary (sets `LD_LIBRARY_PATH` and
`LLAMA_KV_HOSTMEM=1`). Key flags:

```sh
-c 200000 -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -ctkd q4_0 -ctvd q4_0 \            # draft KV: tiny anyway, quantize for headroom
  --spec-draft-ctx 8192 \            # sliding draft window
  -b 512 -ub 128 -np 1 \
  --spec-type draft-dflash --spec-draft-n-max 3 \
  --jinja --no-mmproj-offload
```

Note there is **no `--no-kv-offload`** — that's the flag that triggered the CPU attention.

Sampling matters for speed here: DFlash2's win rate depends on how well the draft predicts
the target. Temperature 1.0 + repeat penalty 1.1 suppressed acceptance (≈2.2 tokens per
cycle). Qwen's thinking-mode recommendation — **temp 0.6, top_p 0.95, top_k 20, min_p 0,
repeat penalty off** — restored ≈2.5-2.7 and is also simply the right sampling for this
model in a coding agent.

## Measured results

| Configuration | Decode |
| --- | --- |
| `--no-kv-offload` preset (old) | 13 t/s |
| Old preset + DFlash2 | 4-12 t/s (worse, every draft pass paid the CPU tax) |
| Custom build, short-prompt benchmark | **31.6 cold / 36.5 warm t/s** |
| Custom build, ~5k-token real prompt, cached | **27-30 t/s** |
| Prompt processing (prefill) | **400-440 t/s** |
| 100k-token synthetic prompt, Q4 drafter + Q4_0 target K/V, one 128-token decode run | **15.4 t/s** |
| 100k-token synthetic prompt, Q4 drafter + Q8_0 target K/V, one 128-token decode run | **10.0 t/s** |
| 100k-token synthetic prompt, Q4 drafter + Q8_0 K / Q4_0 V, one 128-token decode run | **12.6 t/s** |
| ROCm/HIP build of the same source | 9.5 t/s plain — slower than Vulkan on RDNA4; shelved |

The 100k results are single synthetic runs, not repeated averages. Q4_0 target K/V was
fastest in these measurements, but caused a noticeable quality regression; the active preset
therefore retains Q8_0 target K/V and the Q4_K_M drafter. At 992 prompt tokens the Q4
drafter/Q8_0 target cache reached 27.2 t/s, versus 26.7 t/s with the Q8_0 drafter; that
short-context difference is within run-to-run variation. A `RADV_PERFTEST=nogttspill` run at
100k with Q4_K_M drafting and Q8_0 target K/V reached 9.0 t/s; the earlier no-flag run reached
10.0 t/s, so this flag is not enabled.

VRAM peaks at 14.6-14.7GB of 15.9GB including the desktop; the KV cache *cannot* enter VRAM
by construction (the buffer type refuses device-local memory).

## Honest caveats

- Decode tapers as a conversation genuinely fills toward 200k tokens: every pass reads the
  whole cache over PCIe, so a fully-loaded context trends toward single digits. That's
  bandwidth physics — any layout that keeps 8GB of KV in system RAM pays it. The 27-36 t/s
  numbers are the normal working range (shallow-to-mid context).
- Run-to-run draft acceptance varies ±20%, so individual requests swing mid-20s to mid-30s.
- First request after a context change pays prefill; llama.cpp prefix caching makes
  subsequent turns of the same conversation skip it almost entirely (10.8s → 0.2s observed).

## Reproducing

Build recipe (Arch with the vulkan-radeon driver; no system Vulkan headers needed):

```sh
BASE=5266f24da75dc449bd56cbed7addb9c8e4a6a73e
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch --depth 1 origin $BASE && git checkout $BASE
git apply ../gfx1200-hostmem-kv-dflash.patch

# build-time-only header deps, cloned locally (no root)
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers ../Vulkan-Headers
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers ../SPIRV-Headers
cmake -B ../spirv-build -S ../SPIRV-Headers -DCMAKE_INSTALL_PREFIX=../spirv-prefix
cmake --install ../spirv-build

cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
  -DVulkan_INCLUDE_DIR=$PWD/../Vulkan-Headers/include \
  -DVulkan_LIBRARY=/usr/lib/libvulkan.so.1 \
  -DVulkan_GLSLC_EXECUTABLE=$(command -v glslc) \
  -DSPIRV-Headers_DIR=$PWD/../spirv-prefix/share/cmake/SPIRV-Headers \
  -DLLAMA_BUILD_UI=OFF -DGGML_CCACHE=OFF
cmake --build build-vulkan --config Release -j$(nproc)
```

Then wrap the binary in a script exporting `LD_LIBRARY_PATH=<build>/bin` and
`LLAMA_KV_HOSTMEM=1`, point your frontend at the wrapper, and — again — **don't** pass
`--no-kv-offload`. Works on other amdgpu Vulkan GPUs in principle; the GTT behavior was
validated on RADV/RDNA4 only.
