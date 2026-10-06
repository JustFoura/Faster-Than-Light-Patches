# 9 → 40+ t/s decode at 200k on a 16GB RDNA4 card: GTT-pinned experts + VRAM-resident KV

How the **Hermes3.6 35B A3B** MoE preset (Qwen3.6-35B-A3B hybrid SSM/MoE, 200k context,
16GB RX 9060 XT, 15GB system RAM) went from decode that collapsed with context depth to a
flat ~40-57 t/s across the whole context window — with prompt processing above 400 t/s at
every depth. No quant, context, or mmproj changes; same GGUF file.

## The shape of the problem

The model (`qwen35moe`) is 40 blocks, only 10 of them full attention (`full_attention_interval
= 4`), 256 routed experts with 8 active. Two consequences:

- The 200k KV cache is tiny: 10 layers × 2 KV heads × 256 head dim ≈ **2.0 GiB** at q8_0.
- The weights are almost entirely expert weights (≈ 16.6 GiB of the file) and **do not fit
  in the 15.9 GiB of VRAM** alongside that cache, the compute buffers, and the desktop.

What the old preset (plain `-ngl 99`) actually produced, per the loader log:

```
Vulkan0 model buffer size = 16350 MiB   ← weights, assigned to VRAM
Vulkan0 KV buffer size    =  2077 MiB   ← assigned AFTER the weights
```

Everything is "on Vulkan0", but that's 18.4 GiB of buffers in 15.9 GiB of physical VRAM.
RADV fills VRAM with the weights (loaded first) and the KV — loaded last — spills into
host-visible GTT. So every decoded token read the whole cache over PCIe: at 192k depth
that is ~2.1 GiB/token ≈ 84 ms of bus time, and decode died with depth:

| Depth | PP (old) | Decode (old) |
| --- | --- | --- |
| shallow | 727-1246 t/s | 56 t/s |
| 64k | 982 t/s | 22.5 t/s |
| 128k | 719 t/s | 13.1 t/s |
| 192k | 451 t/s | 9.1 t/s |

Naively un-setting `LLAMA_KV_HOSTMEM` (letting the KV "go to VRAM") changes nothing: the
weights still fill VRAM first, the KV still spills last — and the compute buffer gets
squeezed into GTT, collapsing prompt processing to ~250-300 t/s. Placement has to be
controlled explicitly.

## The fix

Two coordinated changes (preset + one patch extension, already in
`patches/llama-cpp/gfx1200-hostmem-kv-dflash.patch`):

1. **The GTT hostmem buffer type gets a name: `Vulkan-GTT`.** It was already in the build
   (added for the dense model's KV), but only reachable through the `LLAMA_KV_HOSTMEM`
   env path. The patch gives the type its own `get_name`, teaches
   `ggml_backend_buffer_is_vk()` and `ggml_backend_vk_device_supports_buft()` to accept it,
   and registers it for `-ot` tensor overrides. That last part lives in `common/arg.cpp`:
   the type is not a registered device, so `-ot` could never name it before.

2. **The preset pins a slice of expert weights to GTT via `-ot`, which frees enough VRAM
   that the KV — allocated after the weights — physically lands in VRAM:**

   ```
   --kv-vram -ot blk\.(29|3[0-9])\.ffn_.*_exps=Vulkan-GTT
   ```

   `--kv-vram` is a sentinel argument consumed by the wrapper
   (`~/.local/bin/llama-server-simple`): it un-sets `LLAMA_KV_HOSTMEM` for that server only,
   so the dense 27B preset keeps its GTT-resident KV by default.

   Layers 29-39's routed experts (4470 MiB) sit in GTT; the GPU still executes every
   kernel and reads those weights over PCIe exactly like GTT-resident KV. Activated
   spilled experts cost ~4-5 ms/token of bus time. The loader log after the change:

   ```
   Vulkan-GTT model buffer size =  4470 MiB   ← experts in system RAM, GPU-computed
   Vulkan0     model buffer size = 11880 MiB   ← rest of the weights in VRAM
   Vulkan0     KV buffer size    =  2077 MiB   ← KV physically in VRAM
   Vulkan0     compute buffer    =   529 MiB   ← stays in VRAM too
   ```

   VRAM ends at ~14.9/15.9 GiB including the desktop; system RAM carries the 4.5 GiB of
   experts with ~5.5 GiB still free.

## Measured results (same machine, same GGUF)

| Depth | PP | Decode |
| --- | --- | --- |
| shallow | 907 t/s (steady), 466 t/s first-call warmup | 57 t/s |
| 64k | 896 t/s | 49.5 t/s |
| 128k | 645 t/s | 43.9 t/s |
| 192k | 480 t/s | 39.6 t/s |

Decode is nearly flat across the entire 200k window and beats the dense Qwen3.8 27B
preset at every depth (the dense preset measures 10-15 t/s at 100k). Prompt processing
stays above 400 t/s everywhere.

## Notes

- The MoE KV is ~⅓ the size of the dense model's, which is *why* it fits in VRAM here
  while the dense preset's 7 GiB cache must stay in GTT. Same patch, opposite placements,
  selected per preset.
- The `-ot` pattern deliberately matches only routed-expert tensors (`ffn_(up|down|gate)_exps`,
  including a fused `ffn_gate_up_exps` variant) — never `ffn_*_shexp` or attention/SSM
  weights, which are read every token and must stay in VRAM.
- `--no-kv-offload` remains off in both MoE and dense presets; that flag still pins
  attention to the CPU and is never wanted on this build.
- Re-running `patches/llama-cpp/build.sh` rebuilds from base + patch; it now force-checks
  out the base first so re-applying is idempotent.
- The bench scripts used for the tables live in `scripts/bench-moe.py` and
  `scripts/bench-depth-ladder.py`.
