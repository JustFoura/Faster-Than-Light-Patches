#!/usr/bin/env bash
# Rebuild the custom llama.cpp used by the Flash-Next stack (Vulkan, gfx1200).
#
# Base: upstream ggml-org/llama.cpp @ 5266f24da75dc449bd56cbed7addb9c8e4a6a73e
#       (the 0.4.0 version bump; note upstream tags no revision *at* that commit)
#
# Usage:  build.sh [patch-name] [out-dir-name]
#   patch-name    default: unified-0.4.0-mtp-cache.patch
#   out-dir-name  default: llama.cpp          (builds at ~/build/<out-dir-name>)
#
# Choosing a patch:
#   unified-0.4.0-mtp-cache.patch  the whole stack -- expert cache, MTP, hostmem,
#                                  --spec-draft-ctx, fanout pool, pread sequencing,
#                                  and the copy-path work. THIS is the default
#                                  because it is the only one that builds a binary
#                                  scripts/llama-server-qwen38-flash-next can run.
#   gfx1200-hostmem-kv-dflash.patch
#                                  the foundation only: GTT hostmem buffer type and
#                                  --spec-draft-ctx. Produces a binary with NO
#                                  --moe-expert-cache and no LLAMA_MOE_* knobs, so
#                                  the Flash-Next launcher will reject its flags.
#                                  Pair it with scripts/llama-server-simple.
#   gfx1200-moe-cache-hostmem-kv.patch
#                                  NOT a layer on top of the foundation -- it is an
#                                  earlier, overlapping export of the same GTT
#                                  hostmem buffer type. Kept for provenance only.
#   see also: patches/llama-cpp/README.md
#
# What the default patch does (details in the repo README):
#   1. ggml-vulkan: a hostmem (GTT) buffer type -- system RAM the GPU reads and
#      writes directly, never device-local, so it can never consume VRAM.
#      Selected for the KV cache when LLAMA_KV_HOSTMEM=1. Reported as
#      "Vulkan-GTT" and accepted by supports_buft / buffer_is_vk, so -ot
#      overrides can pin weight tensors to GTT while kernels stay on the GPU
#      (common/arg.cpp registers the type for -ot under that name).
#   2. llama: --spec-draft-ctx N -- a sliding context window for the DFlash2
#      draft model (position remap + eviction in speculative.cpp) instead of
#      cloning the full target context, so the draft KV stays tiny on GPU.
#   3. server: the DFlash draft no longer inherits --no-kv-offload.
#   4. llama-model / llama-context: --moe-expert-cache, the persistent async
#      fanout pool, pread sequencing, and the prompt-path copy staging pool.
#   5. ggml-vulkan / ggml-cpu: prompt-path copy staging and read-ahead, plus
#      LLAMA_KV_GPU_ATTN=1 (reference only -- the shipped setup does NOT use it;
#      with GTT KV you don't pass --no-kv-offload at all).
#
# Not installed system-wide by this script (see README "Build dependencies"):
#   git, cmake, a C++17 compiler, ninja/make, glslc, and a Vulkan loader.
#   Vulkan/SPIR-V headers are cloned into ~/build and handed to CMake directly.
set -euo pipefail

die() { printf 'build.sh: %s\n' "$1" >&2; exit 1; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=5266f24da75dc449bd56cbed7addb9c8e4a6a73e

PATCH="${1:-unified-0.4.0-mtp-cache.patch}"
OUTDIR="${2:-llama.cpp}"
[ -f "$SCRIPT_DIR/$PATCH" ] || die "no such patch: $SCRIPT_DIR/$PATCH"

BLD="${BUILD_ROOT:-$HOME/build}"
REPO="$BLD/$OUTDIR"

# Header deps are pinned: an unpinned HEAD silently changes the build.
VULKAN_HEADERS_SHA=6802bb4733b63ed5efd3adb308a6c885ef180ea1   # 2026-09-18
SPIRV_HEADERS_SHA=2f88364fbce81d98ee71113cd55e0076034c9ba4    # 2026-09-17

for tool in git cmake; do
    command -v "$tool" >/dev/null || die "$tool not found on PATH"
done
# Without this check, `command -v glslc` fails *inside* the cmake argument list,
# `set -e` does not fire (cmake is the command being tested), and the build dies
# much later with an opaque error.
GLSLC="$(command -v glslc || true)"
[ -n "$GLSLC" ] || die "glslc not found on PATH (install shaderc/glslc)"

# Vulkan loader: /usr/lib on Arch, /usr/lib64 on Fedora, multiarch on Debian.
VULKAN_LIB=""
for cand in /usr/lib/libvulkan.so.1 /usr/lib64/libvulkan.so.1 \
            /usr/lib/x86_64-linux-gnu/libvulkan.so.1; do
    if [ -e "$cand" ]; then VULKAN_LIB="$cand"; break; fi
done
[ -n "$VULKAN_LIB" ] || die "libvulkan.so.1 not found; install the Vulkan loader"

mkdir -p "$BLD"

if [ ! -d "$REPO/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$REPO"
else
    ORIGIN="$(git -C "$REPO" remote get-url origin 2>/dev/null)" \
        || die "target exists but has no origin remote: $REPO"
    case "$ORIGIN" in
        https://github.com/ggml-org/llama.cpp|https://github.com/ggml-org/llama.cpp.git|\
        git@github.com:ggml-org/llama.cpp|git@github.com:ggml-org/llama.cpp.git)
            ;;
        *) die "target origin is not ggml-org/llama.cpp: $ORIGIN" ;;
    esac
fi
cd "$REPO"
git fetch --depth 1 origin "$BASE" || die "cannot fetch base $BASE"
# -f: the patch file is the source of truth; reset to the pinned base before
# every apply. This discards tracked local edits in the selected target tree.
git checkout -f "$BASE"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    git apply --check "$SCRIPT_DIR/$PATCH" && echo "OK: $PATCH applies to $BASE"
    exit 0
fi
git apply "$SCRIPT_DIR/$PATCH" || die "$PATCH did not apply to $BASE"

# build-time-only header deps (no root needed)
clone_pinned() {  # url dest sha
    local url="$1" dest="$2" sha="$3"
    if [ ! -d "$dest/.git" ]; then
        git clone "$url" "$dest"
    fi
    git -C "$dest" fetch --depth 1 origin "$sha" || die "cannot fetch $sha for $dest"
    git -C "$dest" checkout -f "$sha"
}
clone_pinned https://github.com/KhronosGroup/Vulkan-Headers.git \
              "$BLD/Vulkan-Headers" "$VULKAN_HEADERS_SHA"
if [ ! -d "$BLD/spirv-prefix" ]; then
    clone_pinned https://github.com/KhronosGroup/SPIRV-Headers.git \
                  "$BLD/SPIRV-Headers" "$SPIRV_HEADERS_SHA"
    cmake -B "$BLD/spirv-build" -S "$BLD/SPIRV-Headers" \
          -DCMAKE_INSTALL_PREFIX="$BLD/spirv-prefix" >/dev/null
    cmake --install "$BLD/spirv-build" >/dev/null
fi

# Vulkan links are RAM-hungry; a full nproc OOMs on many-core machines.
JOBS="${BUILD_JOBS:-$(( $(nproc) / 2 ))}"
[ "$JOBS" -ge 1 ] || JOBS=1

# NOTE: build-vulkan/ caches its CMake configuration between runs. If you change
# the toolchain or the loader path, delete it first.
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
    -DVulkan_INCLUDE_DIR="$BLD/Vulkan-Headers/include" \
    -DVulkan_LIBRARY="$VULKAN_LIB" \
    -DVulkan_GLSLC_EXECUTABLE="$GLSLC" \
    -DSPIRV-Headers_DIR="$BLD/spirv-prefix/share/cmake/SPIRV-Headers" \
    -DLLAMA_BUILD_UI=OFF -DGGML_CCACHE=OFF
cmake --build build-vulkan --config Release -j"$JOBS"

echo
echo "binary: $REPO/build-vulkan/bin/llama-server"
echo "patch:  $PATCH (base $BASE)"
echo
echo "next:   cp \"\$REPO/scripts/llama-server-simple\" ~/.local/bin/"
echo "        LLAMA_SERVER_BIN=$REPO/build-vulkan/bin/llama-server \\"
echo "        FLASH_MODEL=/absolute/path/to/model.gguf \\"
echo "        FLASH_MOE_CACHE_SLOTS=64 \\"
echo "        ~/.local/bin/llama-server-simple"
