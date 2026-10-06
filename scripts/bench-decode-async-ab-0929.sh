#!/usr/bin/env bash
# Decode A/B, 2026-09-29: persistent async fanout pool (63b8acad3 default) vs
# the legacy per-op spawn/join fanout, on the proven-safe cache-ins0 envelope.
# No graph topology change (same risk class as the 09-28 runs). Measures decode
# lever #1 from docs/bench-results-2026-09-28/FINDINGS.md (per-op pthread storm).
# Arm order bounds page-cache order effects: legacy (fresh boot) / async / legacy.
# Usage: scripts/bench-decode-async-ab-0929.sh [NPREDICT]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-09-29
mkdir -p "$OUT"
NPREDICT=${1:-256}

[ -f "$PROFILE" ] || { echo "missing $PROFILE — restore from docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz" >&2; exit 1; }

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
    ${MTP_FLAGS:-} \
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
  curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "$label: server failed to start"; kill -9 $pid 2>/dev/null || true; exit 1; }
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

# fresh boot, page cache cold for the model: legacy path re-baseline (1.50 t/s reference is the 09-28 final-ins0)
run_arm ins0-legacy8-fresh LLAMA_MOE_FANOUT_ASYNC=0 LLAMA_MOE_PROFILE=$PROFILE --moe-expert-cache 88 --moe-expert-cache-inserts 0
# async pool (new default) on the now-warm cache
run_arm ins0-asyncpool LLAMA_MOE_PROFILE=$PROFILE --moe-expert-cache 88 --moe-expert-cache-inserts 0
# legacy again: bounds the warm-vs-cold order effect between the first two
run_arm ins0-legacy8-warm LLAMA_MOE_FANOUT_ASYNC=0 LLAMA_MOE_PROFILE=$PROFILE --moe-expert-cache 88 --moe-expert-cache-inserts 0
echo "ALL ARMS DONE"
