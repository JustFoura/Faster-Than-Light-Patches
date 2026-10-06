#!/usr/bin/env bash
# MTP A/B for the unified tree: MTP-only vs MTP+cache+profile decode.
# Usage: scripts/bench-mtp-arms.sh [NPREDICT]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
SIDECAR=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-09-27
NPREDICT=${1:-256}

run_arm() {
  local label=$1; shift
  sleep 2
  env "$@" LLAMA_KV_HOSTMEM=1 "$BIN" -m "$MODEL" -lv 4 \
    --spec-type draft-mtp --spec-draft-model "$SIDECAR" --spec-draft-n-max 2 \
    --gpu-layers all --cpu-moe --load-mode mmap --fit off \
    --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 8 --threads-batch 8 \
    --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$label.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break
    sleep 2
  done
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "n_predict":'"$NPREDICT"',"temperature":0.6,"top_p":0.95,"top_k":20,"seed":42,
    "cache_prompt":false,"stream":false}' -o "$OUT/$label.resp.json" --max-time 2400
  python3 -c "
import json
r=json.load(open('$OUT/$label.resp.json'))
t=r['timings']
print('$label: %d tok %.1f s = %.2f t/s'%(t['predicted_n'], t['predicted_ms']/1000, t['predicted_n']*1000/t['predicted_ms']))
"
  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  grep -E "hit-rate|draft|accept" "$OUT/$label.server.log" | tail -4 || true
}

run_arm mtp-only
run_arm mtp-cache-profile LLAMA_MOE_PROFILE=$PROFILE --moe-expert-cache 88 --moe-expert-cache-inserts 2
