#!/usr/bin/env bash
# Copy/compute overlap by pooling the prompt-path staging buffers. 2026-10-02.
#
# The 10-02 accounting (docs/bench-results-2026-10-02/copy-sync-findings.md) split
# a one-pass 14,960-token prefill into ~23 s of host read drain and ~44 s of GPU
# fence wait in a 91 s pass, strictly alternating: set_tensor_2d_async ends every
# staging copy with ggml_vk_synchronize(), which spin-waits on the GPU fence, so
# copy N+1 cannot start reading until copy N's copyBuffer is consumed.
#
# LLAMA_VK_COPY_POOL=<n> gives those copies n staging slots, each with its own
# fence, and submits without the wait — so the host read for copy N+1 proceeds
# while the GPU is still draining N-k..N.
#
# Order-controlled with a control on BOTH sides of every treatment: the prefill
# run-to-run band on this machine is ~13 t/s, wider than any single-arm effect.
# Success criterion is pass wall time and gpu_busy% — if the overlap is not
# happening, t/s alone cannot prove it either way.
#
# Results: docs/bench-results-2026-10-02/copy-pool-results.txt
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
RES=$OUT/copy-pool-results.txt
PORT=1252
MEM_FLOOR_GB=5
mkdir -p "$OUT"

arm() {  # arm <label> <pool_slots> <pread_depth>
    local label=$1 pool=$2 depth=$3
    sleep 3
    local avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
    if (( avail_gb < MEM_FLOOR_GB )); then
        echo "$label: SKIP — MemAvailable ${avail_gb}GB < ${MEM_FLOOR_GB}GB floor" | tee -a "$RES"
        return 1
    fi
    # LLAMA_VK_COPY_POOL=0 keeps the upstream singleton path untouched.
    env LLAMA_VK_COPY_POOL="$pool" LLAMA_VK_COPY_POOL_LOG=1 \
        LLAMA_VK_COPY_PREAD_DEPTH="$depth" LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/cp-$label.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "$label: server failed (see $OUT/cp-$label.server.log)" | tee -a "$RES"
        kill -9 $pid 2>/dev/null; return 1
    fi
    sleep 2

    : > "$OUT/cp-$label.gpu.csv"
    ( while kill -0 "$pid" 2>/dev/null; do
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo NA)" \
            >> "$OUT/cp-$label.gpu.csv"
        sleep 0.2
      done ) &
    local sampler=$!

    python3 - "$PORT" "$OUT/cp-$label.json" "$label" "$OUT/cp-$label.gpu.csv" <<'PYEOF'
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
echo "=== copy-pool overlap $(date '+%F %T') ===" | tee -a "$RES"
echo "read_ahead_kb=$(storage_read_ahead_kb)  MemAvailable=$(awk '/MemAvailable/{printf "%.1fGB",$2/1024/1024}' /proc/meminfo)" | tee -a "$RES"
echo "binary: $(stat -c %y "$BIN")" | tee -a "$RES"

# control, treatment, control, treatment, control — depth 8 throughout, since
# the accounting showed depth saturates there and only muddies a pool comparison.
# Slot count sets batch size (slots/2), so p16 = 8 copies per submit, p32 = 16.
arm ctl-a 0 8
arm p16   16 8
arm ctl-b 0 8
arm p32   32 8
arm ctl-c 0 8

echo "" | tee -a "$RES"
echo "--- pool log lines ---" | tee -a "$RES"
grep -h "copy-pool" "$OUT"/cp-*.server.log 2>/dev/null | tee -a "$RES"

echo "" | tee -a "$RES"
echo "--- arms ---" | tee -a "$RES"
python3 - <<'PYEOF' | tee -a "$RES"
import json, os
base = os.environ["ROOT"] + "/docs/bench-results-2026-10-02"
for l in ["ctl-a", "p16", "ctl-b", "p32", "ctl-c"]:
    p = f"{base}/cp-{l}.json"
    if not os.path.exists(p):
        print("%-7s MISSING" % l); continue
    t = json.load(open(p))["timings"]
    print("%-7s %6.1f t/s  %5.1f s" % (l, t["prompt_n"] * 1000.0 / t["prompt_ms"], t["prompt_ms"] / 1000.0))
PYEOF

echo "" | tee -a "$RES"
echo "--- parity: greedy temp-0 output must be byte-identical across every arm ---" | tee -a "$RES"
python3 - <<'PYEOF' | tee -a "$RES"
import json, os
base = os.environ["ROOT"] + "/docs/bench-results-2026-10-02"
ref = None
for l in ["ctl-a", "p16", "ctl-b", "p32", "ctl-c"]:
    p = f"{base}/cp-{l}.json"
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

echo "COPY-POOL DONE" | tee -a "$RES"
