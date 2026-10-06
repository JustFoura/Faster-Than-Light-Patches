#!/usr/bin/env bash
# Decode FANOUT depth A/B on the SN5000, 2026-10-01.
#
# On the P310 the pool size stopped mattering (per-layer merged-job wait was
# latency-bound). At the SN5000's 1.4 GB/s the question reopens: does t/s
# scale with fanout depth (queue depth -> drive parallelism)?
# Protocol: gate A warm pair (r1 8-tok warmup populates prompt KV, r2 256 tok
# measured + /proc io delta). Arms: 12 / 24 / 32 / 12 (last = same-session
# order control against the first).
# Usage: scripts/bench-decode-fanout-1001.sh
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-10-01
mkdir -p "$OUT"
[ -f "$PROFILE" ] || gunzip -c "$ROOT/docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz" > "$PROFILE"

PROMPT='<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

run_arm() {
  local fan=$1
  local label="fan${fan}-$(date +%H%M%S)"
  sleep 2
  env LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
      LLAMA_MOE_FANOUT=$fan LLAMA_MOE_PROFILE=$PROFILE \
      "$BIN" -m "$MODEL" -lv 4 \
      --moe-expert-cache 88 --moe-expert-cache-inserts 0 \
      --gpu-layers all --cpu-moe --load-mode mmap --fit off \
      --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --threads 8 --threads-batch 8 \
      --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$label.server.log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break
    sleep 2
  done
  curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "$label: server failed"; kill -9 $pid 2>/dev/null; return 1; }
  sleep 2
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"'","n_predict":8,"temperature":0.6,"top_p":0.95,"top_k":20,
    "seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' -o /dev/null --max-time 600
  local io0=$(grep read_bytes /proc/$pid/io | awk '{print $2}')
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"' Continue the story.","n_predict":256,"temperature":0.6,
    "top_p":0.95,"top_k":20,"seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' \
    -o "$OUT/$label.resp.json" --max-time 1200
  local io1=$(grep read_bytes /proc/$pid/io | awk '{print $2}')
  python3 - "$io0" "$io1" "$OUT/$label.resp.json" <<'EOF' | tee -a "$OUT/fanout-ab-results.txt"
import json, sys
io0, io1, fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t = json.load(open(fn))["timings"]
mib = (io1 - io0) // 2**20
gen = t["predicted_n"] * 1000.0 / t["predicted_ms"]
print("%s: %.2f t/s @ %d MiB/tok (%.2f GB/s effective)" % (
    sys.argv[3].rsplit("/", 1)[-1].replace(".resp.json", ""), gen, mib,
    mib * 1.048576 * gen / 1000))
EOF
  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

: > "$OUT/fanout-ab-results.txt"
for f in 12 24 32 12; do run_arm "$f"; done
echo "FANOUT AB DONE"
