#!/usr/bin/env bash
# Fanout pipelining A/B, 2026-10-01 (LLAMA_MOE_FANOUT_PIPE).
# Two worker groups drain alternate jobs so the pread stream stays deep
# across the ~47 per-token MoE jobs (ground truth: only ~52 MiB/tok of true
# misses; the wall is job-boundary serialization, not bandwidth).
# Arms: OFF / ON / OFF (order control). Warm-pair protocol; records pid
# read_bytes AND model-drive sector deltas per request.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
source "$SCRIPT_DIR/lib/storage-device.sh"
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-10-01
mkdir -p "$OUT"
[ -f "$PROFILE" ] || gunzip -c "$ROOT/docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz" > "$PROFILE"
PROMPT='<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

run_arm() {
  local mode=$1
  local label="pipe-$mode-$(date +%H%M%S)"
  sleep 2
  if [ "$mode" = on ]; then
    env -u LLAMA_MOE_MISS_PREFETCH LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode \
        LLAMA_MOE_PREAD=1 LLAMA_MOE_FANOUT_PIPE=1 \
        LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE=$PROFILE \
        "$BIN" -m "$MODEL" -lv 4 --moe-expert-cache 88 --moe-expert-cache-inserts 0 \
        --gpu-layers all --cpu-moe --load-mode mmap --fit off --ctx-size 8192 --parallel 1 \
        -b 512 -ub 512 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
        --threads 8 --threads-batch 8 --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$label.server.log" 2>&1 &
  else
    env -u LLAMA_MOE_MISS_PREFETCH -u LLAMA_MOE_FANOUT_PIPE \
        LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
        LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE=$PROFILE \
        "$BIN" -m "$MODEL" -lv 4 --moe-expert-cache 88 --moe-expert-cache-inserts 0 \
        --gpu-layers all --cpu-moe --load-mode mmap --fit off --ctx-size 8192 --parallel 1 \
        -b 512 -ub 512 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
        --threads 8 --threads-batch 8 --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$label.server.log" 2>&1 &
  fi
  local pid=$! i
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break; sleep 2; done
  curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "$label: server failed"; kill -9 $pid 2>/dev/null; return 1; }
  sleep 2
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"'","n_predict":8,"temperature":0.6,"top_p":0.95,"top_k":20,"seed":42,
    "cache_prompt":true,"ignore_eos":true,"stream":false}' -o /dev/null --max-time 600
  local io0=$(diskstats_sectors_read)
  local p0=$(grep read_bytes /proc/$pid/io | awk '{print $2}')
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"' Continue the story.","n_predict":256,"temperature":0.6,"top_p":0.95,
    "top_k":20,"seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' -o "$OUT/$label.resp.json" --max-time 1200
  local io1=$(diskstats_sectors_read)
  local p1=$(grep read_bytes /proc/$pid/io | awk '{print $2}')
  python3 - "$io0" "$io1" "$p0" "$p1" "$OUT/$label.resp.json" <<'PYEOF' | tee -a "$OUT/fanoutpipe-ab-results.txt"
import json, sys
s0, s1, p0, p1, fn = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
t = json.load(open(fn))["timings"]
n = t["predicted_n"]
gen = n * 1000.0 / t["predicted_ms"]
dsec = (s1 - s0) * 512 // 2**20
dpid = (p1 - p0) // 2**20
gbs = (s1 - s0) * 512.0 / 2**30 * gen / 1000
print("%s: %.2f t/s | drive %d MiB (%.1f MiB/tok, %.2f GB/s) | pid %d MiB" % (
    fn.rsplit("/", 1)[-1].replace(".resp.json", ""), gen, dsec, dsec / n, gbs, dpid))
PYEOF
  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

: > "$OUT/fanoutpipe-ab-results.txt"
run_arm off
run_arm on
run_arm off
echo "PIPE AB DONE"
