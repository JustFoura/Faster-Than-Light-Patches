# Faster-Than-Light

A custom Vulkan/gfx1200 build of **llama.cpp** for Qwen3.8-Flash-Next (a large MoE)
on AMD RDNA4, tuned for prefill and decode throughput against NVMe.

This repo is the **research, the patches, and the write-ups**. It is not the
engine — the upstream tree is cloned fresh at build time.

- **Upstream base:** `ggml-org/llama.cpp @ 5266f24da75dc449bd56cbed7addb9c8e4a6a73e`
  (the 0.4.0 version bump; upstream tags no revision *at* that commit)
- **Hardware:** gfx1200 (RDNA4) — RX 9060 XT 16 GB, on a WD Blue SN5000 2 TB
- **Status:** the copy-path work landed; the 300 t/s prefill target did not

Measurements were taken through a local GUI backed by llama.cpp. Nothing here
requires that application.

---

## Status — read this before you trust a number

**The shipped launcher does not boot at its defaults.** The default is
`--moe-expert-cache 88` at ctx 200000. On a 16 GB card that **aborts during model
load**:

```
radv/amdgpu: Not enough memory for command submission.
ggml_vulkan: device lost on Vulkan0
terminate called: vk::DeviceLostError / vk::Queue::submit: ErrorDeviceLost
```

It is the *combination* — the q8 KV (~2.7 GB GTT hostmem at 200k) plus an 88-slot
expert cache overruns what RADV will commit. Bisected at ubatch 4096:

| ctx | `--moe-expert-cache` | result |
| --- | ---: | --- |
| 200000 | 88 | **abort — RADV OOM** |
| 200000 | 64 | healthy |
| 200000 | 48 | healthy |
| 200000 | none | healthy |
| 16384 | 88 | healthy |

The fix is one env var:

```sh
FLASH_MOE_CACHE_SLOTS=64 scripts/llama-server-qwen38-flash-next
```

**But the fix costs decode.** The ~7 t/s decode figure below was measured at 88
slots. At 48 slots the hit rate drops from 78.3% to 64.1% and the rate to
**~3.2–3.9 t/s** (`docs/bench-results-2026-10-02/decode-cache-findings.md:29-30`).
The two numbers you can actually have together at ctx 200000 on a 16 GB card are
"boots" and "~3–4 t/s", not "boots" and "~7 t/s". 88 slots is reachable at
ctx 16384, where the KV is small enough.

**The 300 t/s prefill target was not met.** The best measured prefill is
202 t/s under bench conditions and **134.6 t/s at the launcher's own defaults**.
An earlier version of this README claimed a 313 t/s ceiling; that claim was
wrong, and is retracted below.

**One measurement here is unexplained and is reported as such:** in an interactive
session on 2026-10-01, decode opened at 11.94 t/s and fell to ~1.9 t/s within
the same session (`decode-cache-findings.md:107-108`). The bench-loop numbers do
not capture real-session behaviour.

---

## Requirements

| | |
| --- | --- |
| GPU | gfx1200 (RDNA4). 16 GB is what the numbers below were measured on; 88 expert slots want ~10.8 GB of device memory, leaving ~1 GB for the desktop compositor |
| Backend | **Vulkan.** A ROCm/HIP build of the same source measured 9.5 t/s and was shelved |
| RAM | ≥10 GB. The launcher refuses to start below a 6 GB `MemAvailable` floor (`FLASH_MEM_FLOOR_GB`) — a wide ubatch at 200k ctx previously exhausted host memory |
| Model | ~93 GiB of GGUF (Flash-Next IQ4_XS + an MTP sidecar). Not in this repo |
| Disk | NVMe storage. Headline figures use the model on a separate data drive; storage layout and workload affect results |
| Build tools | git, cmake, a C++17 compiler, ninja/make, `glslc`, a Vulkan loader |

## Build

```sh
patches/llama-cpp/build.sh
```

Clones upstream to `~/build/llama.cpp`, force-checks-out the base commit,
applies `unified-0.4.0-mtp-cache.patch`, and builds
`~/build/llama.cpp/build-vulkan/bin/llama-server`.

```sh
build.sh gfx1200-hostmem-kv-dflash.patch   # foundation-only build instead
CHECK_ONLY=1 build.sh                      # verify the patch applies, build nothing
```

**Patch choice matters and is easy to get wrong.** `build.sh` defaults to the
unified patch because it is the only one that produces a binary
`scripts/llama-server-qwen38-flash-next` can actually run. The foundation-only
patch has **no** `--moe-expert-cache` and **no** `LLAMA_MOE_*` knobs, so that
launcher will reject its flags; pair it with `scripts/llama-server-simple`
instead. The six patches are **alternatives, not a stack** — see
[`patches/llama-cpp/README.md`](patches/llama-cpp/README.md).

`build.sh` resets its selected target tree to the pinned upstream commit on each
run, discarding tracked local edits in that target. Use a dedicated build path
or back up local changes first. Patch application itself is all-or-nothing.

### Build dependencies (not system-installed by the script)

| Dep | Where |
| --- | --- |
| Vulkan headers | `~/build/Vulkan-Headers` (cloned by `build.sh`, **pinned by SHA**) |
| SPIR-V headers | `~/build/SPIRV-Headers` → installed to `~/build/spirv-prefix` (pinned) |
| `glslc` | must be on `$PATH` |
| Vulkan loader | discovered at `/usr/lib`, `/usr/lib64` or the Debian multiarch dir |

CMake flags that matter: `-DGGML_VULKAN=ON`, `-DLLAMA_BUILD_UI=OFF`, `-DGGML_CCACHE=OFF`.

## Patches

Six patches plus the 18-commit series they came from. Full detail, including
which commit each applies to, is in
[`patches/llama-cpp/README.md`](patches/llama-cpp/README.md).

| Patch | Lines | What it does |
| --- | ---: | --- |
| `unified-0.4.0-mtp-cache.patch` | 4985 | **The whole stack, and what `build.sh` applies.** Adds a Vulkan GTT *hostmem* buffer type — system RAM the GPU addresses directly, never device-local, so it can never consume VRAM — selected for the KV cache via `LLAMA_KV_HOSTMEM=1` and registered for `-ot` overrides. Adds `--spec-draft-ctx N`, giving the DFlash2 draft a sliding context window instead of cloning the full target context. Plus `--moe-expert-cache`, the persistent async fanout pool, pread sequencing, the layer-sibling prefetch and covered-flag work, and the prompt-path copy staging pool. 26 files: `llama-model`, `llama-context`, `llama-arch`, `llama-kv-cache`, `ggml-cpu`, `ggml-vulkan`, `src/CMakeLists.txt`. |
| `gfx1200-hostmem-kv-dflash.patch` | 463 | The foundation alone: hostmem + `--spec-draft-ctx` only. **Builds no binary the Flash-Next launcher accepts.** |
| `gfx1200-moe-cache-hostmem-kv.patch` | 183 | **Provenance only.** Despite the name it contains no MoE code — it is an earlier, overlapping export of the same hostmem buffer type, and its `ggml-vulkan.h` hunk is byte-identical to the dflash patch's. |
| `copy-sync-accounting.patch` | 184 | Instrumentation only, behind `LLAMA_VK_COPY_SYNC_LOG=1`, default off. Measures what the staging fallback in `ggml_backend_vk_set_tensor_2d_async` actually costs — every large host→device copy ends in a fence wait that blocks the calling thread. This is what established that the prefill copy wall is a **per-copy serialization**, not a staging-pool size problem. |
| `flash-next-demand-prefetch.patch` | 91 | Optional Linux read-ahead for expert matrices selected by the current graph. A demand hint, not a learned route profile. |
| `order-aware-copy-readahead-WIP.patch` | 80 | **WIP.** Order-aware `madvise` — the scheduler copies the same tensor sequence every pass, so the previous pass's address order predicts this pass. |
| `series/0001..0018.patch` | — | The same work as `unified`, split into the 18 commits it was developed as, in order, with messages. `git am` them onto the base. |

`copy-sync-accounting.patch` is a `git format-patch` export of a development
change in this project (not an upstream ggml-org commit). The other five are
hand-made diffs.

## Launch

```sh
FLASH_MOE_CACHE_SLOTS=64 scripts/llama-server-qwen38-flash-next
```

The recipe of record, and what each knob is for:

```
LLAMA_KV_HOSTMEM=1              KV cache in system RAM (GTT), attention stays on GPU
LLAMA_MOE_MADV_RANDOM=decode    decode/verify reads experts exactly
LLAMA_MOE_PREAD=1               prompt passes keep readahead
LLAMA_MOE_FANOUT=24             persistent fanout pool
```

`FLASH_FANOUT=24` was the best of a 12/24/32 sweep (6.17 / 6.49 / 6.44 t/s), but
that margin sits **inside** the run-to-run band below and an earlier sweep found
12 and 24 tied. Treat it as "24 is fine", not "24 is optimal".

`scripts/llama-server-simple` is the older launcher — self-contained apart from
needing a `--model`, and points at the foundation build at `~/build/llama.cpp/build-vulkan`.

### Launcher environment

| Var | Default | |
| --- | --- | --- |
| `FLASH_MOE_CACHE_SLOTS` | 88 | **use 64** — see "Status". Costs decode |
| `FLASH_CTX_SIZE` | 200000 | q8 KV is ~2.7 GB GTT hostmem — RAM, not VRAM |
| `FLASH_MOE_CACHE_INSERTS` | 0 | bench-final; try 2 in a real drifting conversation |
| `FLASH_FANOUT` | 24 | see above |
| `FLASH_UBATCH_SIZE` | auto | `ctx > 32768` → 4096, else 16384 |
| `FLASH_BATCH_SIZE` | auto | 0 means "same as ubatch" |
| `FLASH_MEM_FLOOR_GB` | 6 | refuses to start below this `MemAvailable` |
| `FLASH_MODEL` | `$PROJECT_ROOT/app/models/…` | **see caveat** |
| `FLASH_PORT` | 1235 | bound to `127.0.0.1` |
| `FLASH_MOE_PROFILE` | `/tmp/moe_routes_decode.log` | restored from the committed capture on cold start |
| `FLASH_THREADS` / `FLASH_THREADS_BATCH` | unset | set to pin thread counts |
| `FLASH_SPEC_TYPE` | unset | speculative-decode type |
| `LLAMA_SERVER_BIN` | `$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server` | point at your `build.sh` output |

Two caveats:

1. `FLASH_MODEL` defaults to `$PROJECT_ROOT/app/models/…`, and models are **not**
   in this repo. Set `FLASH_MODEL` to an absolute path, or the launcher exits.
2. On cold start (when `/tmp/moe_routes_decode.log` is absent — i.e. after a
   reboot) the launcher restores the route profile by gunzipping
   `docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz`. That file is
   committed for exactly this reason. The script runs under `set -euo pipefail`,
   so if it goes missing the launcher aborts.

Note that the launcher never passes `--spec-draft-ctx`, so the DFlash2 sliding
draft window — the foundation patch's headline feature — is **inactive** in the
shipped configuration.

## Measured results

Every number below is from the separate-data-drive configuration, except where
noted. **Three different models appear in this repo's docs; they are not
comparable to each other**, and are listed separately.

### Qwen3.8-Flash-Next (512 experts, top-10) — the subject of this repo

Prefill is gate C: one-pass 14,960 tokens, ubatch 16384, ctx 16384,
`--cpu-moe --load-mode mmap`, greedy temp-0.

| Configuration | Prefill |
| --- | --- |
| Best measured (bench conditions, ctx 16384) | **202.2 / 201.3 t/s** |
| At the launcher's own defaults (ctx 200k, ubatch 4096) | **134.6 t/s cold / 134.7 warm** |

| Configuration | Decode | MiB/token | Cache hit |
| --- | ---: | ---: | ---: |
| `--moe-expert-cache 88`, best arm | **7.10 t/s** | 202 | 78.3% |
| `--moe-expert-cache 88`, other arm | 6.64 t/s | 218 | 78.3% |
| `--moe-expert-cache 48` | ~3.2–3.9 t/s | — | 64.1% |
| Expert cache **off** | ~4.5 t/s | — | — |
| Interactive session (10-01) | 11.94 → **~1.9 t/s** | — | — |

Decode is queue-depth-bound, not bandwidth-bound.

**The 202 t/s figure is real but comes with a caveat.** It was measured against a
working tree that was uncommitted at the time
(`copy-batch-findings.md:168`); that work was subsequently incorporated and **is**
in the re-exported `unified` patch. It has
not been rebuilt and re-measured since, so treat it as "reproducible in
principle, confirmed once".

### Qwen3.8-27B dense + vision — `docs/gfx1200-200k-writeup.md`

A different model. Included because it is where the 200k-context preset work
landed.

| Configuration | Result |
| --- | --- |
| Custom build, short-prompt benchmark | **31.6 cold / 36.5 warm t/s decode** |
| Custom build, ~5k-token real prompt, cached | **27–30 t/s decode** |
| Prompt processing (prefill) | **400–440 t/s** |
| 100k synthetic prompt (Q8_0/Q8_0, Q4_0/Q8_0, Q4_0/Q4_0) | 10.0 / 12.6 / 15.4 t/s |

The old `--no-kv-offload` preset managed 13 t/s. Adding DFlash2 naively made it
*worse* (4–12 t/s) because every draft pass paid the CPU tax.

### Hermes3.6-35B-A3B — `docs/qwen36-moe-35b-a3b-writeup.md`

A third model: 9 → **40+ t/s decode**, prefill above **400 t/s**.

### What the measurements actually established

**Run-to-run band is ~13 t/s.** Two stock controls in the same order-controlled
sweep differed by 12.9 t/s (171.7 vs 158.8) — larger than any treatment effect
measured. An earlier single-arm "+5%" was drift. *Any prefill conclusion from one
arm against one control on this machine is unsound.* The bench scripts run
order-controlled arms with byte-for-byte output comparison for this reason.

**A correction, stated because this repo previously got it wrong.** The
2026-10-02 profile initially reported the copy stream at "1.43–1.55 GB/s
sustained" and computed a 313 t/s "GPU-busy" ceiling from it. That number was a
**windowing artifact** — `read_bytes` was summed to end-of-request but divided by
prompt-only wall time. The real prefill read rate is **~0.9 GB/s**, which is
*below* the 1.37 GB/s that 300 t/s requires, not above it. The conclusion that
"the copy stream is no longer the constraint" **does not hold**
(`copy-sync-findings.md:113-119`). Overlap alone now projects to roughly
**220–240 t/s, not 300**; reaching 300 t/s additionally requires attacking the
~18–24 s of CPU time that is neither a read nor a fence
(`copy-sync-findings.md:106`).

**Decode is queue-depth-bound, not bandwidth-bound.** The 4.5 t/s figure that
circulated for a while was measured with the expert cache switched off.

**The raw data is not here.** Benchmarks can be *re-run* with
`scripts/bench-*.sh`, but the per-arm JSON/CSV/`.server.log` and every
`*-results.txt` the findings cite were not kept — **the published numbers cannot
be recovered from this repo, only re-measured.** Prompt corpora survive under
`experiments/qwen38-multi-rhs-prefill/`.

### Docs

Full detail, newest first:

| | |
| --- | --- |
| [`docs/bench-results-2026-10-02/FINDINGS.md`](docs/bench-results-2026-10-02/FINDINGS.md) | pread depth, and the profile that redirected the roadmap |
| [`docs/bench-results-2026-10-02/overlap-design.md`](docs/bench-results-2026-10-02/overlap-design.md) | the copy/compute overlap design |
| [`docs/bench-results-2026-10-02/copy-sync-findings.md`](docs/bench-results-2026-10-02/copy-sync-findings.md) | the copy path — **and the corrections to the above** |
| [`docs/bench-results-2026-10-02/copy-pool-findings.md`](docs/bench-results-2026-10-02/copy-pool-findings.md) | staging pool, arm by arm |
| [`docs/bench-results-2026-10-02/copy-batch-findings.md`](docs/bench-results-2026-10-02/copy-batch-findings.md) | batched submit, and the 202 t/s result |
| [`docs/bench-results-2026-10-02/decode-cache-findings.md`](docs/bench-results-2026-10-02/decode-cache-findings.md) | decode with the expert cache |
| [`docs/qwen38-flash-next-main-llm-plan-2026-10-01.md`](docs/qwen38-flash-next-main-llm-plan-2026-10-01.md) | plan of record |
| [`docs/qwen38-prefill-decode-plan-2026-09-27.md`](docs/qwen38-prefill-decode-plan-2026-09-27.md) | the prefill/decode plan |
| [`docs/qwen38-prefill-wavefront-design.md`](docs/qwen38-prefill-wavefront-design.md) | bounded layer-major prompt wavefront design |
| [`docs/qwen38-engine-rd-2026-09-27.md`](docs/qwen38-engine-rd-2026-09-27.md) | read-path research |
| [`docs/qwen38-prefill-backend-discovery-2026-09-29.md`](docs/qwen38-prefill-backend-discovery-2026-09-29.md) | backend discovery, including the ROCm/HIP dead end |
| [`docs/qwen38-flash-next-optimization.md`](docs/qwen38-flash-next-optimization.md) | the slot/insert sweep |
| [`docs/gfx1200-200k-writeup.md`](docs/gfx1200-200k-writeup.md) | the 200k preset, on the 27B dense model |
| [`docs/qwen36-moe-35b-a3b-writeup.md`](docs/qwen36-moe-35b-a3b-writeup.md) | the 35B-A3B MoE write-up |
| [`docs/flash-next-speed-work.md`](docs/flash-next-speed-work.md) | the speed-work log |
| [`docs/flash-next-readpath-2026-09-25.md`](docs/flash-next-readpath-2026-09-25.md) | read-path measurements |
| [`docs/flash-next-prefetch-test-2026-09-25.md`](docs/flash-next-prefetch-test-2026-09-25.md) | prefetch test |
| [`docs/flash-next-followup-2026-09-24.md`](docs/flash-next-followup-2026-09-24.md) | model-size, memory, and context-length tradeoffs |
| [`docs/bench-results-2026-10-01/FINDINGS.md`](docs/bench-results-2026-10-01/FINDINGS.md) | the `read_ahead_kb` fault-coalescing regression and its fix |
| [`docs/bench-results-2026-10-01/FINDINGS-EVENING.md`](docs/bench-results-2026-10-01/FINDINGS-EVENING.md) | steady-state numbers, and the "steady state is the score" number policy |
| [`docs/bench-results-2026-09-29/FINDINGS.md`](docs/bench-results-2026-09-29/FINDINGS.md) | the expert-cache final state |
| [`docs/bench-results-2026-09-28/FINDINGS.md`](docs/bench-results-2026-09-28/FINDINGS.md) | decode arms |
| [`docs/bench-results-2026-09-27/moe-route-analysis.txt`](docs/bench-results-2026-09-27/moe-route-analysis.txt) | route-skew analysis |

## Layout

| Path | |
| --- | --- |
| `patches/llama-cpp/` | the 6 patches, the 18-commit `series/`, `build.sh`, and a patch-by-patch README |
| `scripts/` | launchers and benchmark/probe scripts |
| `docs/` | Engineering write-ups, plans, and per-session findings |
| `experiments/` | two approaches that did not ship, each with a README: multi-RHS prefill, PLE/O_DIRECT probes |

Scope, and what is deliberately not here: [`README-FTL-notes.md`](README-FTL-notes.md).

## Not yet done

- **The 300 t/s prefill target is unmet** — 134.6 t/s at launcher defaults.
- **The 202 t/s configuration has not been re-verified** since the copy-path work
  was committed into the unified patch.
- **Real-session decode decay is unexplained** — 11.94 → ~1.9 t/s in one session.
- **The decode cost of 64 slots is unquantified.** It was measured at 48
  (~3.2–3.9 t/s); the recommended setting is 64 and nothing is interpolated
  between them.
- **The bench scripts need regenerating input.** `bench-night-sweep-1001.sh` and
  friends expect `/tmp/moe_routes_multi.log`, produced only from session files
  that are not in this repo.

## License

MIT — see [LICENSE](LICENSE). The patches derive from
[llama.cpp](https://github.com/ggml-org/llama.cpp), also MIT. Upstream is not
affiliated with this repository.
