#!/usr/bin/env bash
# Where does the prefill pass actually spend its wall clock? 2026-10-02.
#
# FINDINGS.md concluded the prompt pass is an OVERLAP problem and pointed at
# sync_staging being a singleton. The committed build (overlap-design.md) pools
# that buffer with per-slot fences. That only works if the CPU can run more than
# one copy ahead, and ggml_backend_vk_set_tensor_2d_async ends its staging
# fallback with ggml_vk_synchronize() -> ggml_vk_wait_for_fence(), which blocks
# the calling thread on the GPU fence after EVERY large copy. If that blocks,
# the pool sits at slot 0 and the build is inert.
#
# So measure before building. LLAMA_VK_COPY_SYNC_LOG=1 makes the Vulkan backend
# account every staging-fallback synchronize in set_tensor_2d_async:
#
#   sync   = wall time inside those ggml_vk_synchronize() calls
#   drain  = ... of which the host memcpy/pread drain (in_memcpys)
#   fence  = ... of which the blocking fence wait
#
# drain + fence should account for nearly all of a pass; the split says whether
# the pass is paced by host reads or by waiting on the GPU. Also prints a size
# histogram, which answers the open "which tensors are copied how many times"
# question from FINDINGS.md item 2.
#
# Windowing: the 10-02 depth harness (bench-prefill-depth-profile-1002.sh) sums
# read_bytes from request start to END OF REQUEST but divides by the prompt-only
# wall time, which inflated the read rate to 1.43-1.55 GB/s against a true
# 0.88-0.94 GB/s. Every window here is bounded at t_start + prompt_ms.
#
# Results: docs/bench-results-2026-10-02/copy-sync-results.txt
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export ROOT
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
source "$SCRIPT_DIR/lib/storage-device.sh"
OUT=$ROOT/docs/bench-results-2026-10-02
RES=$OUT/copy-sync-results.txt
PORT=1251
MEM_FLOOR_GB=5
mkdir -p "$OUT"

arm() {  # arm <label> <pread_depth>
    local label=$1 depth=$2
    sleep 3
    local avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
    if (( avail_gb < MEM_FLOOR_GB )); then
        echo "$label: SKIP — MemAvailable ${avail_gb}GB < ${MEM_FLOOR_GB}GB floor" | tee -a "$RES"
        return 1
    fi
    env LLAMA_VK_COPY_PREAD_DEPTH="$depth" LLAMA_VK_COPY_SYNC_LOG=1 LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/cs-$label.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "$label: server failed" | tee -a "$RES"; kill -9 $pid 2>/dev/null; return 1
    fi
    sleep 2

    : > "$OUT/cs-$label.gpu.csv"; : > "$OUT/cs-$label.io.csv"
    ( while kill -0 "$pid" 2>/dev/null; do
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo NA)" \
            >> "$OUT/cs-$label.gpu.csv"
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(awk '/^read_bytes/{print $2}' /proc/$pid/io 2>/dev/null || echo NA)" \
            >> "$OUT/cs-$label.io.csv"
        sleep 0.2
      done ) &
    local sampler=$!

    python3 - "$PORT" "$OUT/cs-$label.json" "$label" "$OUT/cs-$label.server.log" <<'PYEOF'
import os
import json, re, sys, time, urllib.request
port, out, label, logpath = sys.argv[1:5]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "user\n" + prompt + "\nassistant\n<think>\n\n</think>\n\n"
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                     "seed": 42, "ignore_eos": True, "cache_prompt": False,
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
json.dump(r, open(out, "w"))
t = r["timings"]
pms = t["prompt_ms"]

# Window the [copy-sync] counter lines to the prompt pass. The backend prints a
# cumulative line every 256 calls; each carries a wall-clock ms stamp, so the
# lines bracketing [t0, t0+prompt_ms] give the pass's totals directly.
RE = re.compile(r"^\[copy-sync\] (\d+) n=(\d+) ([\d.]+)GB sync=([\d.]+)s drain=([\d.]+)s "
                r"fence=([\d.]+)s avg=([\d.]+)ms xferq=(\d)")
rows = []
for line in open(logpath, errors="replace"):
    m = RE.match(line)
    if m:
        # groups: wall_ms, n, GB, sync_s, drain_s, fence_s, avg_ms, xferq
        rows.append((float(m.group(1)) / 1000.0,
                     [float(m.group(i)) for i in range(2, 9)]))
if not rows:
    print("%s: NO [copy-sync] LINES — is the binary the instrumented build?" % label)
    raise SystemExit(1)
# [copy-sync] lines are stamped when the call RETURNS, i.e. after the fence, so
# the last line at or before the end of the pass brackets the pass.
base = max((r for r in rows if r[0] <= t0), default=None)
end  = max((r for r in rows if r[0] <= t0 + pms/1000.0), default=rows[-1])
sel  = end[1]
base = base[1] if base else [0.0]*7

d_n    = sel[0] - base[0]
d_gb   = sel[1] - base[1]
d_sync = sel[2] - base[2]
d_drain= sel[3] - base[3]
d_fence= sel[4] - base[4]
xferq  = sel[6]

print("%s: %.1f t/s (%.0f s, %d tok) | sync calls %d, %.1f GB | "
      "sync %.1fs (%.0f%% of pass) = drain %.1fs + fence %.1fs | xferq=%d"
      % (label, t["prompt_n"]*1000.0/pms, pms/1000.0, t["prompt_n"],
         d_n, d_gb, d_sync, 100.0*d_sync/(pms/1000.0), d_drain, d_fence, xferq))
PYEOF

    kill $sampler 2>/dev/null || true
    kill -TERM $pid 2>/dev/null || true; sleep 5; kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

: > "$RES"
echo "=== prefill copy-synchronize accounting $(date '+%F %T') ===" | tee -a "$RES"
echo "read_ahead_kb=$(storage_read_ahead_kb)  MemAvailable=$(awk '/MemAvailable/{printf "%.1fGB",$2/1024/1024}' /proc/meminfo)" | tee -a "$RES"
echo "binary mtime: $(stat -c %y "$BIN")" | tee -a "$RES"

# Controls first: the copy bytes are bit-identical across depths (same call
# count, same size histogram), so the drain/fence split compares across runs
# even when the controls are skipped.
if [[ "${SKIP_DEFAULTS:-0}" != "1" ]]; then
    arm stock 1
    arm d16   16
fi

# Extra arms for a depth curve, if asked for: bench-copy-sync-probe-1002.sh 32 8
for spec in "$@"; do
    arm "d$spec" "$spec"
done

python3 - "$@" <<'PYEOF' | tee -a "$RES"
import json, os, sys
base = os.environ["ROOT"] + "/docs/bench-results-2026-10-02"
ref = None
print("\n--- parity vs the 10-02 reference output (instrumentation must be neutral) ---")
for l in ["stock", "d16"] + ["d%s" % a for a in sys.argv[1:]]:
    p = f"{base}/cs-{l}.json"
    if not os.path.exists(p):
        print(f"{l}: MISSING"); continue
    c = json.load(open(p)).get("content")
    if ref is None:
        ref = c; print(f"{l}: reference")
    else:
        print(f"{l}: {'PASS byte-identical' if c == ref else 'FAIL — OUTPUTS DIFFER'}")
r = f"{base}/pd-stock-a.json"
if os.path.exists(r):
    c = json.load(open(r)).get("content")
    print(f"vs pd-stock-a: {'PASS byte-identical' if c == ref else 'FAIL — OUTPUTS DIFFER'}")
PYEOF

echo "COPY-SYNC DONE" | tee -a "$RES"
