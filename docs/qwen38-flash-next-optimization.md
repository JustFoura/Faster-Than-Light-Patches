# Qwen3.8 Flash-Next IQ4_XS optimization log

Date: 2026-09-24

## Hardware and artifacts

- GPU: AMD RX 9060 XT, Vulkan RADV GFX1200, 16 GiB class device.
- Host: Ryzen 7 5800X, 16 GiB RAM.
- Target GGUF: `app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf` (97,473,155,360 bytes).
- MTP sidecar: `app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2,786,568,256 bytes; SHA-256 `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`).

The cache-build rows used a 512-token context, one server slot, the same 36-token request, 64 generated tokens, temperature 0.6, top-p 0.95, top-k 20, min-p 0, and repeat penalty 1.0. The MTP 64-token row used those same request settings; the initial short MTP check used a separate 21-token prompt and returned 6 tokens. Results below are server-reported measured decode rates from individual runs, not projections or averages. “Warm” means the same prompt was repeated in that server process; llama-server reused 32 prompt tokens. Prompt processing is reported separately by the server.

## Measurements

| Backend and settings | Run | Decode | Prompt processing | VRAM used after load |
| --- | --- | ---: | ---: | ---: |
| Qwen4exp MTP build `6fcaa16f4`, MTP n-max 2, main MoE weights CPU/mmap | first short 6-token check | 0.21 t/s | 0.46 t/s | 8.15 GB |
| Qwen4exp MTP build `6fcaa16f4`, MTP n-max 2, main MoE weights CPU/mmap | 64 tokens; 37/50 draft tokens accepted (74%) | 0.37 t/s | 0.65 t/s | 8.15 GB |
| MoE GPU-cache build `bccbacdb8`, 64 slots/layer, 2 inserts/step | cold | 0.47 t/s | 0.64 t/s | 13.36 GB |
| MoE GPU-cache build `bccbacdb8`, 64 slots/layer, 2 inserts/step | warm | 0.65 t/s | 0.25 t/s | 13.36 GB |
| MoE GPU-cache build `bccbacdb8`, 64 slots/layer, 8 inserts/step | cold | 0.51 t/s | 0.67 t/s | 13.54 GB |
| MoE GPU-cache build `bccbacdb8`, 64 slots/layer, 8 inserts/step | warm | 0.59 t/s | 0.28 t/s | 13.54 GB |
| MoE GPU-cache build `bccbacdb8`, 72 slots/layer, 2 inserts/step | cold | 0.54 t/s | 0.65 t/s | 14.43 GB |
| MoE GPU-cache build `bccbacdb8`, 72 slots/layer, 2 inserts/step | warm | 0.81 t/s | 0.30 t/s | 14.43 GB |
| MoE GPU-cache build `bccbacdb8`, 80 slots/layer, 2 inserts/step | cold | 0.53 t/s | 0.67 t/s | 15.45 GB |
| MoE GPU-cache build `bccbacdb8`, 80 slots/layer, 2 inserts/step | warm | 0.89 t/s | 0.30 t/s | 15.45 GB |
| MoE GPU-cache build `bccbacdb8`, 88 slots/layer, 2 inserts/step | cold | 0.55 t/s | 0.68 t/s | 16.48 GB |
| MoE GPU-cache build `bccbacdb8`, 88 slots/layer, 2 inserts/step | warm | 1.01 t/s | 0.26 t/s | 16.48 GB |

The cache build outperformed standalone MTP in these runs. Increasing uploads from 2 to 8 did not improve the warm decode. Increasing slots from 64 to 88 improved the repeated-prompt decode in this specific setup, while reducing reported free VRAM to about 0.6 GB. The 88-slot configuration completed both requests without an allocation error; that narrow margin has not been validated with longer contexts or other GPU workloads.

## 200k Q8 host-memory validation

The selected run was restarted with `--ctx-size 200000`, `--cache-type-k q8_0`, `--cache-type-v q8_0`, and `LLAMA_KV_HOSTMEM=1`. `/props` reports an actual context of 200192 tokens. The KV allocator selects a Vulkan GTT buffer type that requires host-visible/coherent memory and excludes device-local memory; this keeps K/V in system memory and leaves attention on the GPU. `--no-kv-offload` is not used.

| Run | Decode | Prompt processing | Output/context | Memory after load/request |
| --- | ---: | ---: | --- | --- |
| First completion | 0.25 t/s | 0.28 t/s | 13 output tokens, 29 prompt tokens | VRAM 16.86 GB; system reports 7.2 GiB available |
| Repeated completion | 0.59 t/s | 0.25 t/s | 13 output tokens, 25 of 29 prompt tokens reused | VRAM 16.82 GB; system reports 6.9 GiB available |

These are single observed runs. The 88-slot expert cache uses nearly all GPU memory; at the first reading about 0.24 GB remained free. Longer prompts have not been tested at 200k. System RAM figures include reclaimable file cache and are not a direct measurement of the KV allocation alone.

## Current running configuration

The best measured setting is running locally on `127.0.0.1:1235` with the following command shape:

```sh
LLAMA_KV_HOSTMEM=1 llama-server \
  --model app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf \
  --gpu-layers all --cpu-moe --load-mode mmap --fit off \
  --ctx-size 200000 --parallel 1 --batch-size 128 --ubatch-size 64 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --moe-expert-cache 88 --moe-expert-cache-inserts 2 \
  --host 127.0.0.1 --port 1235 --no-webui
```

`--cpu-moe` leaves the full expert tensors host/file-backed; llama.cpp’s experimental LRU cache copies selected expert slices into GPU-resident slots. The model is memory-mapped and not locked into RAM. System file cache remains reclaimable by the OS. The configured context is 200k, but only short requests have been exercised so far.

The MTP head is the right model-specific draft artifact, but its standalone run had lower measured decode than the GPU expert cache. The tested MTP server and GPU-cache server are separate llama.cpp builds; combining the MTP implementation with the experimental cache patch has not been done.

The local GTT integration for the MoE-cache build is captured in `patches/llama-cpp/gfx1200-moe-cache-hostmem-kv.patch`. `scripts/llama-server-qwen38-flash-next` starts the current 200k/Q8 profile using the local cache-build binary; `LLAMA_SERVER_BIN` can override that binary path.
