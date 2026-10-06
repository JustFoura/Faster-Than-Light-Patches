#!/usr/bin/env bash
# Copy-pool batch-size fix, 2026-10-02. Tests whether the pool's avg_batch~1 is
# self-inflicted by ggml_vk_copy_pool_reserve() flushing the open batch on every
# slot reclaim, rather than a property of the caller's copy/compute interleaving.
#
# Background: the 10-02 pool run measured avg_batch 1.1 at 16 slots and blocked
# 12,632 of 13,716 acquires, and concluded "copies and compute contexts alternate
# ~1:1, there is no bulk copy phase". Its own counters contradict that reading:
# batches (12,648) tracked blocked (12,632), not acquires/batch_max. reserve()
# was flush-then-retire, so at the moment the pool ran dry the open batch held
# exactly one un-submitted slot and every reclaim pushed a 1-member batch. The
# flush attributed in the write-up (get_compute_ctx, ~7,420 calls/pass, one per
# graph node) is a no-op: the scheduler issues every copy between graph computes,
# so copy_batch is empty during node iteration.
#
# The fix retires the oldest in-flight batch first and only submits the open batch
# when nothing is left in flight to reclaim. PREDICTION at n=16 (batch_max=8):
#   avg_batch 1.1 -> ~8, blocked ~12,600 -> ~1,700, batches ~13,700 -> ~1,700.
# If blocked stays ~12,600 the fix is wrong and the old story stands.
#
# Order-controlled, control on BOTH sides: the run-to-run band on this machine is
# ~13 t/s, wider than any single-arm effect. Success criterion is gpu_busy% over
# the prompt window plus the counters, not t/s alone.
#
# Usage: bench-copy-batch-1002.sh <label>:<slots>:<pread_depth> [...]
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
RES=$OUT/copy-batch-results.txt
PORT=1254
MEM_FLOOR_GB=5
mkdir -p "$OUT"

arm() {  # arm <label> <pool_slots> <pread_depth>
    local label=$1 pool=$2 depth=$3 drain=${4:-8} minkb=${5:-4096}
    sleep 3
    local avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
    if (( avail_gb < MEM_FLOOR_GB )); then
        echo "$label: SKIP — MemAvailable ${avail_gb}GB < ${MEM_FLOOR_GB}GB floor" | tee -a "$RES"
        return 1
    fi
    # LLAMA_VK_COPY_POOL=0 keeps the upstream singleton path untouched, so the
    # control arm is the same binary and the same run conditions.
    env LLAMA_VK_COPY_POOL="$pool" LLAMA_VK_COPY_POOL_LOG=1 \
        LLAMA_VK_COPY_PREAD_DEPTH="$depth" LLAMA_VK_COPY_DRAIN_THREADS="$drain" LLAMA_VK_COPY_PREAD_MIN_KB="$minkb" LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/cb-$label.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "$label: server failed (see $OUT/cb-$label.server.log)" | tee -a "$RES"
        kill -9 $pid 2>/dev/null; return 1
    fi
    sleep 2

    : > "$OUT/cb-$label.gpu.csv"
    ( while kill -0 "$pid" 2>/dev/null; do
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo NA)" \
            >> "$OUT/cb-$label.gpu.csv"
        sleep 0.2
      done ) &
    local sampler=$!

    python3 - "$PORT" "$OUT/cb-$label.json" "$label" "$OUT/cb-$label.gpu.csv" <<'PYEOF'
import os
import json, sys, time, urllib.request
port, out, label, gpucsv = sys.argv[1:5]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "user\n" + prompt + "\nassistant\n<think>\n\n</think>\n\n"
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                     "seed": 42, "ignore_eos": True, "cache_prompt": False,
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=3600))
except Exception as e:
    print("%s: REQUEST FAILED: %s" % (label, e)); raise SystemExit(1)
json.dump(r, open(out, "w"))
t = r["timings"]; pms = t["prompt_ms"]
# gpu_busy over the prompt window only: request start .. start+prompt_ms
t_start = int(t0 * 1000); t_end = t_start + int(pms)
busy = []
for line in open(gpucsv):
    p = line.split()
    if len(p) >= 2 and p[1] != "NA" and t_start <= int(p[0]) <= t_end:
        busy.append(float(p[1]))
mb = (sum(busy) / len(busy)) if busy else -1
print("%s: %.1f t/s (%.0f s, %d tok) | gpu_busy %.1f%% (=%.1f s of the pass)"
      % (label, t["prompt_n"] * 1000.0 / pms, pms / 1000.0, t["prompt_n"], mb, mb * pms / 100.0))
PYEOF

    kill $sampler 2>/dev/null || true
    kill -TERM $pid 2>/dev/null || true; sleep 5; kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

: > "$RES"
echo "=== copy-pool batch-size fix $(date '+%F %T') ===" | tee -a "$RES"
echo "read_ahead_kb=$(storage_read_ahead_kb)  MemAvailable=$(awk '/MemAvailable/{printf "%.1fGB",$2/1024/1024}' /proc/meminfo)" | tee -a "$RES"
echo "binary: $(stat -c %y "$BIN")  commit: $(cd "$HOME/build/llama.cpp-unified" && git rev-parse --short HEAD)" | tee -a "$RES"

for spec in "$@"; do
    arm "${spec%%:*}" "$(echo "$spec" | cut -d: -f2)" "$(echo "$spec" | cut -d: -f3)" "$(echo "$spec" | cut -d: -f4)" "$(echo "$spec" | cut -d: -f5)"
done

echo "" | tee -a "$RES"
echo "--- pool counters (the prediction) ---" | tee -a "$RES"
grep -h "copy-pool" "$OUT"/cb-*.server.log 2>/dev/null | tee -a "$RES"

LABELS=$(for spec in "$@"; do echo "${spec%%:*}"; done | tr '\n' ' ')
echo "" | tee -a "$RES"
echo "--- arms ---" | tee -a "$RES"
python3 - $LABELS <<'PYEOF' | tee -a "$RES"
import json, os, sys
base = os.environ["ROOT"] + "/docs/bench-results-2026-10-02"
for l in sys.argv[1].split():
    p = f"{base}/cb-{l}.json"
    if not os.path.exists(p):
        print("%-7s MISSING" % l); continue
    t = json.load(open(p))["timings"]
    print("%-7s %6.1f t/s  %5.1f s" % (l, t["prompt_n"] * 1000.0 / t["prompt_ms"], t["prompt_ms"] / 1000.0))
PYEOF

echo "" | tee -a "$RES"
echo "--- parity: greedy temp-0 output must be byte-identical across every arm ---" | tee -a "$RES"
python3 - $LABELS <<'PYEOF' | tee -a "$RES"
import json, os, sys
base = os.environ["ROOT"] + "/docs/bench-results-2026-10-02"
ref = None
for l in sys.argv[1].split():
    p = f"{base}/cb-{l}.json"
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
    print(f"vs pd-stock-a (10-02 reference): {'PASS byte-identical' if c == ref else 'FAIL — OUTPUTS DIFFER'}")
PYEOF

echo "COPY-BATCH DONE" | tee -a "$RES"
