# Flash-Next read-path fixes: uncompressed model file + parallel expert reads

Date: 2026-09-25. Follow-up to [the followup experiments](flash-next-followup-2026-09-24.md).
Constraints honored: the 2026-09-25 voided experiments were not repeated, cited, or built
on; no insert-count sweeps, no route-derived preload profiles, no MTP/cache combination.
Context length (200k) and quantization (IQ4_XS weights) are unchanged.

## Diagnosis (new findings)

- The GGUF is stored **zstd-compressed by btrfs** (`compress=zstd:3` mount option caught
  the file on copy-in; every extent is flagged `encoded` by FIEMAP). All reads pay
  decompression, and direct IO is impossible on compressed extents.
- The file is **physically scattered**: consecutive logical extents land out of order
  (e.g. logical 0 → physical 126965788, logical 1 → 126947308). All extents are 128 KB.
- **Readahead on the encrypted block device backing the Btrfs volume is 0** — the kernel
  pipelines nothing on the fault path. (Fixing this system-wide needs root; not done.)
- Page cache is ~1.7 GB against a 97.5 GB working set, so expert reads hit NVMe every time.
- Model architecture (from GGUF metadata): 48 blocks, `full_attention_interval = 4`
  (12 full-attention layers — hence the small 3.5 GB Q8 KV in GTT at 200k), 512 experts
  per layer, 10 routed per token, expert FFN 640 on d_model 2560 → ~2.6 MB per expert.
  Decode streams ~480 expert slices (~1.25 GB minimum bytes) from NVMe per token.
- The moe-cache build's CPU `mul_mat_id` touches one expert after another, so miss reads
  run at queue depth ~1.

## Changes

1. **Uncompressed, defragmented model copy** (on the source NVMe drive):
   `btrfs property set <file> compression none`, then a
   block-resumable copy. Result: byte-identical (`cmp`), 3,511 extents of up to ~56 MB,
   none encoded. Verified with `cmp` before use. Original file retained.
   The launcher (`scripts/llama-server-qwen38-flash-next`) now accepts `FLASH_MODEL`.
2. **Parallel miss reads**: `ggml_compute_forward_mul_mat_id` in the moe-cache build
   (ggml/src/ggml-cpu/ggml-cpu.c) now issues `madvise(MADV_WILLNEED)` for every expert
   slice the op will compute (cached ids are skipped via the existing table), right after
   the row-grouping barrier. WILLNEED force-submits file readahead regardless of the
   zero device window and overlaps the NVMe reads of all routed experts with the first
   experts' GEMMs. Applies to decode misses and to prefill (which uses the stock graph
   and streams every routed expert).

## Measurements (server-reported timings, fixed seed, temperature 0, no-thinking)

Single runs, not averages; the A server had ~1 h of warm history, C was freshly started.
A = compressed file + old binary; C = uncompressed file + patched binary. Prompts are new
(`writing`: 76 prompt/128 generated tokens; `coding`: 77/64).

| Run | Prefill t/s | Decode t/s | read_bytes | major faults |
| --- | ---: | ---: | ---: | ---: |
| A writing (semi-warm server) | 0.836 | 0.955 | 293.4 GB | 298,620 |
| A coding | 0.837 | 0.380 | 346.5 GB | 337,850 |
| C writing (cold server) | 1.037 | 0.732 | 357.6 GB | 138,786 |
| C coding | 1.063 | 0.483 | 294.7 GB | 159,430 |
| C writing (warm 1) | 1.055 | 0.835 | 322.4 GB | 221,064 |
| C writing (warm 2) | 1.042 | 1.066 | 273.6 GB | 176,365 |

Interpretation: prefill improved ~25% consistently; coding decode improved ~27%;
major faults roughly halved. Writing decode climbs as the server warms (cold → warm 2)
and ends above the semi-warm A baseline; individual writing runs swing too much for the
single warm-vs-semi-warm difference to be called. Reads did not drop — the win is larger
contiguous IOs and overlap, not fewer bytes, which is expected at unchanged quantization.

The C stack (uncompressed file + patched binary) is left running on port 1235.

## Next step (started, blocked on one root action): vLLM + EXL3 evaluation

Next direction: evaluate an EXL3 quant on a mainstream engine instead of the custom
llama.cpp build. Research findings (2026-09-25):

- exllamav3 itself cannot run this model here: its CPU-offload requires experts resident
  in system RAM (16 GB total vs ~31-60 GB expert pools at 2.05-4.05 bpw); no expert disk
  streaming. RDNA4 fork (rocm_exl3) is compile-verified only and without the fused MoE
  kernel on gfx12.
- vLLM mainline supports EXL3 (`--quantization exl3`) and has expert-residency work:
  `--moe-gpu-prefetch`, the Offloader weight-prefetch PR (#51710), a disk-backed mmap
  PLE table for this model family (#54129), and cold-expert HIP-gather offload (#49321).
- AMD's ROCm vLLM support page lists the RX 9060 XT (gfx1200) with prebuilt RDNA Docker
  images (ROCm 7.14 → vLLM 0.23; ROCm 10 → vLLM 0.27). Unverified: EXL3 kernels inside
  the RDNA image, and whether non-resident experts page from disk (mmap) vs require RAM
  residency.
- Quant candidates for the Uncensored finetune: Lygodactylus EXL3 4bpw (~65 GB est.) and
  3bpw (~48 GB est.). Disk has ~81 GB free. 4.05 bpw ≈ 1/3 fewer bytes per token than the
  current file; 3.05 bpw ≈ half. Any download must be staged with `compression=none`
  (same btrfs trap).
- SGLang: no mainline EXL3. llama.cpp: no EXL3 fork exists.

Environment blockers: Docker is installed but its daemon is inactive (root needed to start);
ROCm 7.2.4 userspace is already installed via pacman. Nothing on the vLLM path has been
downloaded or run yet. The read-path lessons above apply to any engine's model files on
this btrfs.

## vLLM + EXL3 progress (evening, same day)

- Docker daemon started; CLI access enabled via
  `newgrp docker`. Image `rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`
  pull started (AMD-documented RDNA pairing; RX 9060 XT is on AMD's supported list).
- The original compressed GGUF (97.5 GB) was removed; the nocow
  copy (byte-identical, `cmp`-verified) is the serving model. Free disk: 172 GB.
- EXL3 quant availability for the Uncensored finetune (HF search): exactly two repos,
  both by Lygodactylus (3bpw, 4bpw). The 3bpw repo (`orcarouter/Qwen3.8-Flash-Next-
  Uncensored` base, abliterated) is 3.05 bpw main weights / 5 bpw lm_head+vision /
  3 bpw n-gram tables, ~72.4 GB total (52.4 GB shards + 19.8 GB `ngram_embedding.
  safetensors`). It publishes **no quality measurements** ("Not measured"). No other
  Uncensored EXL3 repo exists; turboderp's official EXL3 repo is the base (non-
  Uncensored) model. The 4bpw variant stores ~61 GB of routed experts vs ~64 GB in the
  deleted IQ4_XS GGUF — no decode-byte reduction — and was ruled out.
- General quality basis for 3.05 bpw EXL3 ≈ 4.25 bpw GGUF: third-party benchmark
  (atomic.chat EXL3-vs-GGUF), not repo-published.
- Download of the 3bpw quant was not initiated pending repository confirmation;
  staging will use `btrfs property set compression none`.

## vLLM+EXL3 probe outcome (track closed)

Both AMD ROCm vLLM images pulled and probed with GPU passthrough
(`--device /dev/kfd --device /dev/dri`; torch reports "AMD Radeon RX 9060 XT"):

| Image | vLLM | exl3 in registry | exllamav3 pkg | offload/mmap/prefetch flags |
| --- | --- | --- | --- | --- |
| rocm7.14.1_rdna (pytorch 2.11) | 0.23.1.dev1 | no | no | none |
| rocm10.0.0 unified (pytorch 2.12) | 0.27.1.dev5 | no | no | none |

EXL3 kernels are CUDA-only upstream and are not compiled into AMD's ROCm builds.
Quantization methods present in the ROCm 10 build: awq, fp8, gptq(_marlin), modelopt*,
compressed-tensors, bitsandbytes, experts_int8, moe_wna16, torchao, mxfp4, quark, etc.
Both images were removed after probing (re-pullable).

Combined with the earlier findings (SGLang: no mainline EXL3; llama.cpp: no EXL3;
exllamav3/rocm_exl3: requires experts resident in system RAM, 16 GB available vs
~46 GB expert pool at 3.05 bpw), **no EXL3 engine can serve this model on this
hardware**. The 3.05 bpw EXL3 repo (~72.4 GB) was never downloaded.

## GGUF byte-reduction candidates (Uncensored finetune, measured sizes)

| Repo | File | Size | vs current 97.5 GB |
| --- | --- | ---: | ---: |
| mradermacher i1 | IQ3_S | 88.9 GB | −8.8% |
| mradermacher i1 | IQ3_M | 89.5 GB | −8.2% |
| mradermacher i1 | Q2_K | 80.4 GB | −17.5% |
| cygnal | IQ4XS-NGQ4 | 98.4 GB | +0.9% |

The EXL3 3.05 bpw byte advantage (−26%) has no GGUF equivalent at comparable quality;
the largest quality-preserving GGUF reduction is ~9%. No quant download was made;
the decision stays with the user. Serving configuration remains the llama.cpp
moe-cache build + uncompressed model file + read-fan-out patch (port 1235).

## Same-day follow-ups: ubatch, admission policy, drive ceiling

- Raw O_DIRECT sequential read of the uncompressed file through the encrypted Btrfs volume:
  **1.5 GB/s** at queue depth 1 (P310-class drive; the decode fan-out
  reaches ~2.3 GB/s effective). The drive/stack, not the fault path, is the wall.
- `FLASH_BATCH_SIZE` env added to the launcher; tested `--batch-size 256
  --ubatch-size 256`: prefill 1.23–1.28 t/s on short prompts (single-ubatch prompts
  gain ~20% from larger-batch efficiency; the per-ubatch expert-read amortization
  only appears on multi-ubatch prompts). Warm writing decode ~1.03 t/s. Kept.
- Admission-policy change (env-gated `LLAMA_MOE_ADMIT_ALL=1`: queue every
  newly-routed miss instead of the 2 inserts/step cap): **measured worse** —
  reads 850–880 GB/request (vs 256–350), decode 0.23–0.27 t/s, prefill down to
  0.80. Cause: 88 slots < ~90-expert working set per 32-token window, so fast
  admission evicts still-needed experts and each upload re-reads from disk.
  Env stays available but off; the throttle is capacity-protective, not arbitrary.
- Final serving config: uncompressed file + fan-out patch + batch/ubatch 256,
  default throttled admission.

## Ceiling summary (measured quantities)

Decode floor = routed-expert bytes per token (~1.25 GB at IQ4_XS) ÷ effective
drive bandwidth (~2.3 GB/s demonstrated, 1.5 GB/s sequential floor) ≈ 2–3 t/s
upper bound at the current quant on this drive, before CPU-side GEMM and graph
costs. Measured warm decode 1.03–1.07 t/s (writing), 0.45–0.48 t/s (coding),
prefill 1.24–1.28 t/s (short prompts). The 10–20 t/s coding and 3–5 t/s writing
targets are above this ceiling at IQ4_XS; reaching them requires a ≥60% byte
reduction (EXL3-class 3 bpw, unavailable on RDNA4 engines), a higher cache-hit
fraction (requires VRAM beyond the 88-slot budget), or faster storage.
