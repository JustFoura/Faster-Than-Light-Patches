#!/usr/bin/env bash
# Decode attribution arms, 2026-09-29 evening:
#   attrib-nodelog: GGML_MOE_NODE_LOG=1 on the default (async pool) ins0 config —
#                   per-op io/node phase split for the 667 ms/token wall
#   mtp-ins0:       MTP + fixes + inserts 0 (the untested composite: 09-28 ran
#                   MTP arms with inserts 2 only)
# Same envelope as bench-decode-async-ab-0929.sh (256 tok, seed 42, -b 512 -ub 512).
# Usage: scripts/bench-decode-attrib-0929.sh [NPREDICT]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
SIDECAR=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-09-29
mkdir -p "$OUT"
NPREDICT=${1:-256}

[ -f "$PROFILE" ] || { echo "missing $PROFILE" >&2; exit 1; }

PROMPT='<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

run_arm() {
  local label=$1; shift
  local envs=() flags=()
  while [ $# -gt 0 ]; do
    if [[ "$1" == *=* && "$1" != --* ]]; then envs+=("$1"); else flags+=("$1"); fi
    shift
  done
  sleep 2
  env "${envs[@]}" LLAMA_KV_HOSTMEM=1 "$BIN" -m "$MODEL" -lv 4 \
    "${flags[@]}" \
    --gpu-layers all --cpu-moe --load-mode mmap --fit off \
    --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 8 --threads-batch 8 \
    --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$label.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break
    sleep 2
  done
  curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "$label: server failed to start"; kill -9 $pid 2>/dev/null || true; return 1; }
  sleep 2
  local io0=$(grep read_bytes /proc/$pid/io | awk '{print $2}')
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"'",
    "n_predict":'"$NPREDICT"',"temperature":0.6,"top_p":0.95,"top_k":20,"seed":42,
    "cache_prompt":false,"stream":false}' -o "$OUT/$label.resp.json" --max-time 3600
  local io1=$(grep read_bytes /proc/$pid/io | awk '{print $2}')
  echo "$label io: read_bytes=$(( (io1 - io0)/1024/1024 )) MiB" | tee "$OUT/$label.io.txt"
  python3 -c "
import json
r=json.load(open('$OUT/$label.resp.json'))
t=r['timings']
print('$label: %d tok %.1f s = %.2f t/s (prompt %.1f t/s)'%(t['predicted_n'], t['predicted_ms']/1000, t['predicted_n']*1000/t['predicted_ms'], t['prompt_n']*1000/max(t['prompt_ms'],1)))
" | tee -a "$OUT/$label.io.txt"
  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  grep -E "hit-rate|acceptance|mean len" "$OUT/$label.server.log" | tail -3 || true
}

run_arm attrib-nodelog GGML_MOE_NODE_LOG=1 LLAMA_MOE_PROFILE=$PROFILE --moe-expert-cache 88 --moe-expert-cache-inserts 0
run_arm mtp-ins0 LLAMA_MOE_PROFILE=$PROFILE --spec-type draft-mtp --spec-draft-model "$SIDECAR" --spec-draft-n-max 2 --moe-expert-cache 88 --moe-expert-cache-inserts 0
echo "ATTRIB ARMS DONE"
