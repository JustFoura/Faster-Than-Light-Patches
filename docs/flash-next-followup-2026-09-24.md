# Flash-Next follow-up experiments (2026-09-24)

Hardware and model are described in [the original optimization log](qwen38-flash-next-optimization.md). These are single-request observations, not averages. Tests used the same local Vulkan/MoE-cache build, one server slot, Q8 K/V, and the same model file. The repeat requests used the same server process. The benchmark script is `scripts/bench_flash_next.py`; it records server timings and process disk reads/page faults. `--no-thinking` wraps the prompt in the model's chat markers with an empty think block, so generation is prose.

## Measurements

| Profile and request | Prompt tokens | Generated tokens | Prompt t/s | Decode t/s | Process disk reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 200k context, 88 slots, 2 inserts, raw prompt | 27 | 8 | 0.48 | 0.39 | 98.1 GB |
| 8k context, 88 slots, 2 inserts, raw prompt, first/repeat | 27 | 8 | 0.52 / 0.53 | 0.28 / 0.51 | 103.3 / 86.9 GB |
| 8k context, 88 slots, 8 inserts, raw prompt, first/repeat | 27 | 8 | 0.50 / 0.56 | 0.28 / 2.60 | 106.0 / 68.0 GB |
| 8k context, 88 slots, 8 inserts, raw prompt, longer warm run | 27 | 32 | 0.56 | 0.53 | 145.1 GB |
| 8k context, **no expert cache**, raw prompt | 27 | 32 | 0.56 | 0.23 | 249.2 GB |
| 8k context, 88 slots, 8 inserts, **prose**, first/repeat | 39 | 32 | 0.63 / 0.65 | 0.41 / 0.67 | 178.3 / 138.0 GB |
| Same warmed prose profile, temperature 0.6 | 39 | 32 | 0.68 | 0.66 | 134.2 GB |
| 8k context, 88 slots, 32 inserts, prose, first/repeat | 39 | 32 | 0.67 / 0.68 | 0.45 / 0.66 | 166.9 / 135.6 GB |
| Same profile with prompt-prefix reuse | 4 evaluated (35 reused) | 32 | 0.34 on the 4 evaluated tokens | 0.66 | 78.9 GB |
| 8k context, 88 slots, 8 inserts, 16 CPU threads, prose, first run | 39 | 32 | 0.64 | 0.34 | 173.3 GB |

The short repeated 8-token result with 8 inserts did not persist over a 32-token continuation. Increasing the admission limit further to 32 did not improve the warm prose result. The single-token microbatch experiment was stopped after it read more than 199 GB before generating any output; prompt batches currently keep the faster setting of 64. The 16-thread repeat was interrupted to finish this round of work, so the single cold measurement is not enough to change the thread default.

## Routing and bottleneck

The `GGML_MOE_LOG` trace recorded 48 expert layers. In a 32-token continuation, the median layer selected about 90 distinct experts; the GPU cache has 88 slots per layer. A simple offline LRU replay of the prose trace reached about 84% expert-selection hits after warming. A two-hit admission rule made the cold replay worse and changed the later hit rate little. This simulation omits asynchronous uploads and is a guide for choosing experiments, not a measured server hit rate.

The model is 97.5 GB while system RAM and VRAM are each 16 GB. The measured disk reads and major page faults show that model-weight residency is the dominant constraint on this machine. Smaller context, more admission slots per step, and more CPU threads did not remove that constraint. The GPU cache does reduce storage traffic and improves decode compared with disabling it, so it remains enabled in the original launcher profile.

The server was restored to the original 200k context, 88-slot, 2-insert configuration after these experiments. The launcher now accepts `FLASH_CTX_SIZE`, `FLASH_MOE_CACHE_SLOTS`, `FLASH_MOE_CACHE_INSERTS`, `FLASH_UBATCH_SIZE`, and `FLASH_THREADS`/`FLASH_THREADS_BATCH` environment overrides for controlled comparisons; defaults are unchanged.
