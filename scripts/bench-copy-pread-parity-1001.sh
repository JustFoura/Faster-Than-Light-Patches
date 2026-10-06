#!/usr/bin/env bash
# Order-controlled parity + speed gate for the prefill page-in depth change
# (LLAMA_VK_COPY_PREAD_DEPTH), 2026-10-01 night.
#
#   depth 1  = stock memcpy out of the CPU_Mapped mmap (queue depth 1)
#   depth N  = N concurrent preads into the same staging buffer
#
# Protocol: the gate-C condition prefill from 10-01 (one-pass 14,963 tok,
# ubatch 16384) so the numbers are comparable with the 168.89 t/s gate and the
# 173-176 t/s evening band. Every arm is a fresh server, greedy temp-0,
# ignore_eos, and the generated text must be byte-identical across all arms —
# the copy bytes must not change, only who moves them.
#
# Results: docs/bench-results-2026-10-01/pread-parity-results.txt
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export ROOT
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
source "$SCRIPT_DIR/lib/storage-device.sh"
PROMPT=$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt
OUT=$ROOT/docs/bench-results-2026-10-01
RES=$OUT/pread-parity-results.txt
PORT=1246

arm() {  # arm <label> <depth>
    local label=$1 depth=$2
    sleep 3
    # MemAvailable floor: a wide ubatch at this ctx stacks the q8 KV on the
    # graph buffers; the 10-01 desktop freeze came from running under it.
    local avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
    if (( avail_gb < 5 )); then
        echo "$label: SKIP — MemAvailable ${avail_gb}GB < 5GB floor" | tee -a "$RES"
        return 1
    fi
    env LLAMA_VK_COPY_PREAD_DEPTH="$depth" LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/pread-$label.server.log" 2>&1 &
    local pid=$!
    local i
    for i in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 2
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "$label: server failed to come up" | tee -a "$RES"
        kill -9 $pid 2>/dev/null
        return 1
    fi
    sleep 2
    # major-fault delta around the pass = did we actually change the read path
    local f0 f1
    f0=$(awk '/^majflt/{print $2}' /proc/$pid/stat 2>/dev/null || echo 0)
    python3 - "$PORT" "$OUT/pread-$label.json" "$label" <<'PYEOF'
import os
import json, sys, urllib.request
port, out, label = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "user\n" + prompt + "\nassistant\n<think>\n\n</think>\n\n"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                     "seed": 42, "ignore_eos": True, "cache_prompt": False,
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
json.dump(r, open(out, "w"))
t = r["timings"]
print("%s: %.1f t/s  (%.0f s, %d prompt tok)" % (label, t["prompt_n"]*1000.0/t["prompt_ms"], t["prompt_ms"]/1000.0, t["prompt_n"]))
PYEOF
    f1=$(awk '/^majflt/{print $2}' /proc/$pid/stat 2>/dev/null || echo 0)
    echo "$label: major faults delta = $((f1 - f0))" | tee -a "$RES"
    kill -TERM $pid 2>/dev/null || true
    sleep 6
    kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    return 0
}

: > "$RES"
echo "=== copy pread-depth parity gate $(date '+%F %T') ===" | tee -a "$RES"
echo "read_ahead_kb=$(storage_read_ahead_kb)  MemAvailable=$(awk '/MemAvailable/{printf "%.1fGB",$2/1024/1024}' /proc/meminfo)" | tee -a "$RES"

# order-controlled: baseline, depth 8, baseline, depth 16
arm d1-a 1
arm d8-a 8
arm d1-b 1
arm d16-a 16

python3 - <<'PYEOF' | tee -a "$RES"
import json, os
base = os.environ["ROOT"] + "/docs/bench-results-2026-10-01"
labels = ["d1-a", "d8-a", "d1-b", "d16-a"]
ref = None
print("\n--- parity vs first depth-1 arm ---")
for l in labels:
    p = f"{base}/pread-{l}.json"
    if not os.path.exists(p):
        print(f"{l}: MISSING"); continue
    c = json.load(open(p)).get("content")
    if ref is None:
        ref = c; print(f"{l}: reference")
    else:
        print(f"{l}: {'PASS byte-identical' if c == ref else 'FAIL — OUTPUTS DIFFER'}")
PYEOF

echo "PREAD-PARITY DONE" | tee -a "$RES"
