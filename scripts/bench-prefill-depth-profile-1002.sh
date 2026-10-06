#!/usr/bin/env bash
# Prefill copy-depth parity gate + pass profile, 2026-10-02.
#
# Two jobs in one run, because both need the same expensive arms:
#
#   1. PARITY. Order-controlled arms (stock, depth 8, stock, depth 16) with
#      greedy temp-0 outputs. The copy bytes must not change — only who moves
#      them — so every arm must produce byte-identical text.
#
#   2. PROFILE. The 10-01 evening left an unexplained gap: depth-8 preads
#      bought only +5% (171.6 -> 180.4 t/s) even though the drive does
#      4.09 GB/s at depth 8 against the ~0.80 GB/s the memcpy path was
#      getting. Either the copy is no longer the wall (shader is), or the
#      copies are not overlapping with the GPU as the transfer queue implies.
#      Sample gpu_busy_percent and /proc/<pid>/io read_bytes through each
#      pass so we can separate the two:
#
#        GPU busy high + reads paced by the GPU  -> shader-bound, the read
#          depth work is done and the bigger lever is overlap or the shader
#        GPU busy low  + reads running ahead    -> copy serialization, more
#          depth / more overlap still to win
#
# Results: docs/bench-results-2026-10-02/pread-depth-results.txt
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
RES=$OUT/pread-depth-results.txt
PORT=1248
mkdir -p "$OUT"

arm() {  # arm <label> <depth>
    local label=$1 depth=$2
    sleep 3
    local avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
    if (( avail_gb < 5 )); then
        echo "$label: SKIP — MemAvailable ${avail_gb}GB < 5GB floor" | tee -a "$RES"
        return 1
    fi
    env LLAMA_VK_COPY_PREAD_DEPTH="$depth" LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/pd-$label.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "$label: server failed" | tee -a "$RES"; kill -9 $pid 2>/dev/null; return 1
    fi
    sleep 2

    # sample gpu_busy_percent + the server's own read_bytes for the whole pass
    : > "$OUT/pd-$label.gpu.csv"; : > "$OUT/pd-$label.io.csv"
    ( while kill -0 "$pid" 2>/dev/null; do
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo NA)" \
            >> "$OUT/pd-$label.gpu.csv"
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(awk '/^read_bytes/{print $2}' /proc/$pid/io 2>/dev/null || echo NA)" \
            >> "$OUT/pd-$label.io.csv"
        sleep 0.2
      done ) &
    local sampler=$!

    python3 - "$PORT" "$OUT/pd-$label.json" "$label" "$OUT/pd-$label.gpu.csv" "$OUT/pd-$label.io.csv" <<'PYEOF'
import os
import json, sys, urllib.request
port, out, label, gpucsv, iocsv = sys.argv[1:6]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "user\n" + prompt + "\nassistant\n<think>\n\n</think>\n\n"
t0 = __import__("time").time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                     "seed": 42, "ignore_eos": True, "cache_prompt": False,
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
json.dump(r, open(out, "w"))
t = r["timings"]
# Window the samples to the measured prompt span so startup, decode and
# teardown are excluded. This needs BOTH bounds: an earlier version took only
# a lower bound, so the byte total spanned the whole request while the divisor
# was the prompt-only wall time — that reported 134 GB / 87 s = 1.43-1.55 GB/s
# when the true prompt-window rate is 82.2 GB / 87.1 s = 0.94 GB/s.
t_start = int(t0 * 1000)
t_end = t_start + int(t["prompt_ms"])
def window(path, col):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) >= col + 1:
            ts, v = int(p[0]), p[col]
            if v not in ("NA",):
                rows.append((ts, float(v)))
    return [v for ts, v in rows if t_start - 500 <= ts <= t_end]
busy = window(gpucsv, 1)
io   = window(iocsv, 1)
print("%s: %.1f t/s (%.0f s, %d tok) | gpu_busy %.0f%% (n=%d) | read_bytes %.1f GB" % (
    label, t["prompt_n"]*1000.0/t["prompt_ms"], t["prompt_ms"]/1000.0, t["prompt_n"],
    (sum(busy)/len(busy)) if busy else -1, len(busy),
    (io[-1]-io[0])/1e9 if len(io) > 1 else -1))
PYEOF
    kill $sampler 2>/dev/null || true
    kill -TERM $pid 2>/dev/null || true; sleep 5; kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

: > "$RES"
echo "=== prefill copy-depth parity + pass profile $(date '+%F %T') ===" | tee -a "$RES"
echo "read_ahead_kb=$(storage_read_ahead_kb)  MemAvailable=$(awk '/MemAvailable/{printf "%.1fGB",$2/1024/1024}' /proc/meminfo)" | tee -a "$RES"

arm stock-a 1
arm d8-a    8
arm stock-b 1
arm d16-a  16

python3 - <<'PYEOF' | tee -a "$RES"
import json, os
base = os.environ["ROOT"] + "/docs/bench-results-2026-10-02"
labels = ["stock-a", "d8-a", "stock-b", "d16-a"]
ref = None
print("\n--- parity: greedy temp-0 output must be byte-identical ---")
for l in labels:
    p = f"{base}/pd-{l}.json"
    if not os.path.exists(p):
        print(f"{l}: MISSING"); continue
    c = json.load(open(p)).get("content")
    if ref is None:
        ref = c; print(f"{l}: reference")
    else:
        print(f"{l}: {'PASS byte-identical' if c == ref else 'FAIL — OUTPUTS DIFFER'}")
PYEOF

echo "PREAD-DEPTH DONE" | tee -a "$RES"
