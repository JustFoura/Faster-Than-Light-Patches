#!/usr/bin/env bash
# Counter-measure arms for the read-amplification: 128-token runs with chain
# accounting + io delta. Usage: bench-amp-arms.sh [ARMS...]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-09-28
mkdir -p "$OUT"
ARMS=${@:-dontneed fanout2 slots64}

run_arm() {
  local label=$1; shift
  local envs=() flags=()
  while [ $# -gt 0 ]; do
    if [[ "$1" == *=* && "$1" != --* ]]; then envs+=("$1"); else flags+=("$1"); fi
    shift
  done
  sleep 2
  env "${envs[@]}" LLAMA_KV_HOSTMEM=1 LLAMA_MOE_PROFILE=$PROFILE GGML_MOE_CHAIN_LOG=1 "$BIN" -m "$MODEL" -lv 4 \
    "${flags[@]}" \
    --gpu-layers all --cpu-moe --load-mode mmap --fit off \
    --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 8 --threads-batch 8 \
    --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$label.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break; sleep 3; done
  curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "$label FAILED to start"; kill -9 $pid 2>/dev/null; return 1; }
  sleep 2
  local io0=$(awk '/read_bytes/{print $2}' /proc/$pid/io)
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "n_predict":128,"temperature":0.6,"top_p":0.95,"top_k":20,"seed":42,
    "cache_prompt":false,"stream":false}' -o "$OUT/$label.resp.json" --max-time 1200
  local io1=$(awk '/read_bytes/{print $2}' /proc/$pid/io)
  python3 -c "
import json
r=json.load(open('$OUT/$label.resp.json')); t=r['timings']
print('$label: %d tok %.1f s = %.2f t/s | io=%.2f GiB/tok' % (t['predicted_n'], t['predicted_ms']/1000, t['predicted_n']*1000/t['predicted_ms'], ($io1-$io0)/2**30/t['predicted_n']))"
  grep "moe-chain" "$OUT/$label.server.log" | tail -1
  kill -TERM $pid 2>/dev/null || true; sleep 5; kill -9 $pid 2>/dev/null || true
}

for arm in $ARMS; do
  case $arm in
    dontneed) run_arm dontneed GGML_MOE_MMAP_DONTNEED=1 --moe-expert-cache 88 --moe-expert-cache-inserts 0 ;;
    fanout2)  run_arm fanout2 LLAMA_MOE_FANOUT=2 --moe-expert-cache 88 --moe-expert-cache-inserts 0 ;;
    slots64)  run_arm slots64 --moe-expert-cache 64 --moe-expert-cache-inserts 0 ;;
    base128)  run_arm base128 --moe-expert-cache 88 --moe-expert-cache-inserts 0 ;;
    *) echo "unknown arm $arm"; exit 1;;
  esac
done
echo AMP ARMS DONE
