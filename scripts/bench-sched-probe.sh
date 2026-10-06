#!/usr/bin/env bash
# Probe: which backend executes the MoE cache chain / CPU chain nodes, and
# where step time goes. Short generation, GGML_SCHED_DEBUG=2 -lv 5.
# Usage: scripts/bench-sched-probe.sh [NPREDICT]
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
NPREDICT=${1:-16}

sleep 2
env LLAMA_KV_HOSTMEM=1 LLAMA_MOE_PROFILE=$PROFILE GGML_SCHED_DEBUG=2 "$BIN" -m "$MODEL" -lv 5 \
  --moe-expert-cache 88 --moe-expert-cache-inserts 2 \
  --gpu-layers all --cpu-moe --load-mode mmap --fit off \
  --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 8 --threads-batch 8 \
  --host 127.0.0.1 --port 1243 --no-webui > "$OUT/sched-probe.server.log" 2>&1 &
pid=$!
for i in $(seq 1 200); do curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break; sleep 3; done
curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "server failed"; exit 1; }
curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
  "prompt":"<|im_start|>user\nWrite a short fantasy story about a mapmaker.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "n_predict":'"$NPREDICT"',"temperature":0.6,"seed":42,"cache_prompt":false,"stream":false}' --max-time 900 -o "$OUT/sched-probe.resp.json"
kill -TERM $pid 2>/dev/null || true; sleep 5; kill -9 $pid 2>/dev/null || true
echo "--- cache-chain node backends (decode-time) ---"
grep -E "ffn_moe_cache_(up|gate|down)|ffn_moe_up|ffn_moe_gate|ffn_moe_down" "$OUT/sched-probe.server.log" | tail -12
echo "--- backend assignment summary ---"
grep -oE "GGML_OP_MUL_MAT_ID.*" "$OUT/sched-probe.server.log" | sort | uniq -c | sort -rn | head -10
python3 -c "
import json
r=json.load(open('$OUT/sched-probe.resp.json'))
t=r['timings']
print('%d tok %.2f s = %.2f t/s'%(t['predicted_n'],t['predicted_ms']/1000,t['predicted_n']*1000/t['predicted_ms']))"
