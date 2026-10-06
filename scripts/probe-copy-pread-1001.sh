#!/usr/bin/env bash
# Single-arm diagnostic: does the copy pread path actually engage on the prompt
# path, and with what memcpy sizes? Runs one depth-8 prefill with
# GGML_VK_COPY_PREAD_DEBUG=1 and reports HIT/MISS counts, total bytes paged,
# and the observed memcpy size distribution.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export ROOT
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
OUT=$ROOT/docs/bench-results-2026-10-01
PORT=1247
DEPTH="${1:-8}"

sleep 2
avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
if (( avail_gb < 5 )); then echo "SKIP: MemAvailable ${avail_gb}GB < 5GB"; exit 1; fi

env LLAMA_VK_COPY_PREAD_DEPTH="$DEPTH" GGML_VK_COPY_PREAD_DEBUG=1 LLAMA_KV_HOSTMEM=1 \
    "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
    --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
    --host 127.0.0.1 --port $PORT --no-webui > "$OUT/pread-diag-d$DEPTH.server.log" 2>&1 &
pid=$!
for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server failed"; kill -9 $pid; exit 1; }
sleep 2

python3 - "$PORT" <<'PYEOF'
import os
import json, sys, urllib.request
port = sys.argv[1]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "user\n" + prompt + "\nassistant\n<think>\n\n</think>\n\n"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                     "seed": 42, "ignore_eos": True, "cache_prompt": False,
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
t = r["timings"]
print("PREFILL: %.1f t/s (%.0f s, %d tok)" % (t["prompt_n"]*1000.0/t["prompt_ms"], t["prompt_ms"]/1000.0, t["prompt_n"]))
PYEOF

echo "=== pread diagnostics (depth=$DEPTH) ==="
grep -c "\[pread\] HIT" "$OUT/pread-diag-d$DEPTH.server.log" | sed 's/^/HITs: /'
grep -c "\[pread\] MISS" "$OUT/pread-diag-d$DEPTH.server.log" | sed 's/^/MISSes: /'
echo "--- MISS reasons ---"
grep "\[pread\] MISS" "$OUT/pread-diag-d$DEPTH.server.log" | sed 's/size=.*//' | sort | uniq -c | head
echo "--- total paged bytes ---"
grep "\[pread\] HIT" "$OUT/pread-diag-d$DEPTH.server.log" | grep -oE "size=[0-9]+" | cut -d= -f2 | awk '{s+=$1} END{printf "%.1f GB over %d hits\n", s/1e9, NR}'
echo "--- sample HIT sizes ---"
grep "\[pread\] HIT" "$OUT/pread-diag-d$DEPTH.server.log" | head -3

kill -TERM $pid 2>/dev/null || true
sleep 5
kill -9 $pid 2>/dev/null || true
wait $pid 2>/dev/null || true
echo "DIAG DONE"
