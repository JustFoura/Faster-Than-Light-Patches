# Qwen3.8 PLE read-path prototypes

These are source-only records from the 2026-09-27 investigation. They exercise
the exact IQ4_XS GGUF layout used in that run; they are not a production data
path and are not integrated into the app.

## Files

- `llama.cpp-qwen4exp-odirect.patch` adds page-batched `O_DIRECT` reads to the
  upstream lazy PLE reader. Its base is llama.cpp `81bc6b83f827df746eb129235488d325c49cae52`
  plus upstream PR #28136 commits `4ed9e239d578567801cbf0ca91363c67b8c61885`
  and `f5d8427d3c5b91c1bff18857be7b15d083361889`. Apply it from the llama.cpp
  repository root after those commits.
- `odirect_probe.c` checks aligned O_DIRECT page reads and page-cache residency
  over the PLE file range.
- `hip_odirect_probe.cpp` reads one PLE page with O_DIRECT into a HIP mapped,
  pinned host allocation and compares a GPU checksum with the CPU checksum.
- `hip_ple_row_probe.cpp` reads PLE row 0, runs IQ4_NL dequantization on the GPU,
  and compares the 160 outputs with ggml's CPU dequantizer.

The probes take the model file path as their only argument. The PLE offsets,
row size and tensor layout are intentionally fixed to the exact model artifact
from this investigation; they must be re-derived before using another file.

## Build notes

Build the C page-cache probe with `cc -O2 odirect_probe.c -o odirect_probe`.
Build the mapped-page probe with `hipcc -O2 hip_odirect_probe.cpp -o
hip_odirect_probe`. For the row probe, set `LLAMA_CPP` to a llama.cpp checkout
and `GGML_LIB_DIR` to a directory containing its built `libggml-base`:

```sh
hipcc -O2 \
  -I"$LLAMA_CPP/ggml/src" -I"$LLAMA_CPP/ggml/include" \
  hip_ple_row_probe.cpp -L"$GGML_LIB_DIR" -lggml-base \
  -Wl,-rpath,"$GGML_LIB_DIR" -o hip_ple_row_probe
```

The HIP row probe was compiled against the local llama.cpp/ggml build and
ROCm 7.2.5. These commands are recorded for reuse; the source-only copies in
this directory were not rebuilt after being copied here.

The reader patch needs the PR #28136 lazy-row reader first. It opens the PLE
file with O_DIRECT where available, groups requested 90-byte rows by 4 KiB
pages, reads aligned pages into an aligned temporary buffer, then dequantizes
rows on the CPU. It is deliberately a rough prototype: that temporary buffer
scales with the number of unique pages in one gather, and read-time O_DIRECT
errors do not currently retry through buffered I/O. The HIP probes demonstrate
mapped host access and GPU dequantization separately; they do not make the
llama.cpp Vulkan server use HIP or provide direct SSD-to-VRAM transfers.

## Recorded result

On the investigation host, the C probe read 2,048 random 4 KiB pages (8 MiB)
from the PLE range in 0.495 s; `mincore` then reported 1 resident page out of
7,031,283. The mapped-page HIP checksum matched. GPU dequantization of row 0
matched ggml's CPU result with 0 differing values out of 160. The full-model
O_DIRECT reader prototype was slower than the buffered `pread` result in the
single short-prompt run recorded in `docs/qwen38-engine-rd-2026-09-27.md`.

No model weights, generated binaries, prompts, route logs or build trees are
included here.
