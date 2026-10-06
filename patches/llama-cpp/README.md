# patches/llama-cpp

Six patches, one ordered commit series, and `build.sh`.

All of them apply to upstream `ggml-org/llama.cpp` at
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e` (the 0.4.0 version bump; upstream
tags no revision *at* that commit — the SHA is the pin).

## The patches are alternatives, not a stack

Only **one** patch is applied per build. They are **not** layered: applying two
of them will conflict, because several are alternative exports of the same code.

| Patch | Lines | Applies to | What it is |
| --- | ---: | --- | --- |
| `unified-0.4.0-mtp-cache.patch` | 4985 | base | **The whole stack.** The branch tip — 26 files. Expert cache, MTP, GTT hostmem, `--spec-draft-ctx`, persistent fanout pool, pread sequencing, and the prompt-path copy staging pool. **This is what `build.sh` defaults to.** |
| `gfx1200-hostmem-kv-dflash.patch` | 463 | base | **Foundation only.** GTT hostmem buffer type + `--spec-draft-ctx`. Builds a binary with **no** `--moe-expert-cache` and **no** `LLAMA_MOE_*` knobs, so `scripts/llama-server-qwen38-flash-next` will reject its flags. Pair it with `scripts/llama-server-simple`. |
| `gfx1200-moe-cache-hostmem-kv.patch` | 183 | base | **Provenance only — not a layer on top of the foundation.** Despite the name it contains no MoE code at all (`grep -ci moe` → 0); it is an earlier, overlapping export of the same GTT hostmem buffer type, and its `ggml-vulkan.h` hunk is byte-identical to the one in the dflash patch. Kept because the launcher history references it. |
| `copy-sync-accounting.patch` | 184 | after `321dff9b5` | Instrumentation only, behind `LLAMA_VK_COPY_SYNC_LOG=1`, default off. Measures what the staging fallback in `ggml_backend_vk_set_tensor_2d_async` costs — every large host→device copy ends in a fence wait that blocks the calling thread. This is what established that the prefill copy wall is a **per-copy serialization**, not a staging-pool size problem. |
| `flash-next-demand-prefetch.patch` | 91 | base | Optional Linux read-ahead for expert matrices selected by the current graph. A demand hint, not a learned route profile. Conflicts with `unified`. |
| `order-aware-copy-readahead-WIP.patch` | 80 | after `2fc48d245` | **WIP.** Order-aware `madvise` — the scheduler copies the same tensor sequence every pass, so the previous pass's address order predicts this pass. |

`copy-sync-accounting.patch` is a `git format-patch` export of a development
change in this project (not an upstream ggml-org commit). The other five are
hand-made diffs.

## series/ — the ordered history

`series/0001..0018.patch` is the same work as `unified-0.4.0-mtp-cache.patch`,
split into the 18 commits it was developed as, in order, with their messages:

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch --depth 1 origin 5266f24da75dc449bd56cbed7addb9c8e4a6a73e
git checkout -f FETCH_HEAD
git am /path/to/Faster-Than-Light/patches/llama-cpp/series/*.patch
```

Use the series when you want to read *why* each change was made, or bisect. Use
`unified-0.4.0-mtp-cache.patch` when you just want the tree.

The two foundation patches use `i/`/`w/` prefixes with no `diff --git` lines.
Plain `git apply` handles that; `git am`, `git apply -3` and `git apply --index`
do not. If you need those, regenerate with `git diff` from the series.

## build.sh

```sh
build.sh                                  # unified patch -> ~/build/llama.cpp
build.sh gfx1200-hostmem-kv-dflash.patch  # foundation -> ~/build/llama.cpp
build.sh unified-0.4.0-mtp-cache.patch my-tree   # -> ~/build/my-tree
CHECK_ONLY=1 build.sh                     # verify the patch applies, build nothing
```

It pins the Vulkan and SPIR-V header repos by SHA, refuses to start without
`glslc` and a Vulkan loader, and backs the target tree out to the base commit
before every apply — so re-running is safe and always starts from a clean base.
