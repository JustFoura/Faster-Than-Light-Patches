#!/usr/bin/env bash
# Decode A/B: does the ggml_moe_pread_wait() barrier cap the read rate?
#
# The barrier (ggml-cpu.c, LLAMA_MOE_PREAD_WAIT) makes every compute thread spin
# on j->active == 0 before touching slice data, so each MoE node runs
# dispatch-all-preads -> wait-for-all -> compute. Decode extracts 1.42 GB/s while
# the same drive does 3.2-3.4 GB/s through the prefill copy path, so the question
# is whether the barrier is the cap or whether removing it just pays the ~2x read
# amplification it was added to prevent.
#
# Arms are order-controlled A / B / A: if the two controls disagree by more than
# the run-to-run band, B means nothing.
# Protocol is identical to bench-decode-cache-1002.sh (and the 10-01 fanout
# harness) so results stay in the same series as the 7.10 / 6.64 t/s controls.
# Usage: scripts/bench-decode-preadbarrier-1002.sh
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-10-02
mkdir -p "$OUT"
[ -f "$PROFILE" ] || gunzip -c "$ROOT/docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz" > "$PROFILE"

PROMPT='user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n\nassistant\n<think>\n\n</think>\n\n'

# run_arm <label> <LLAMA_MOE_PREAD_WAIT>
run_arm() {
  local label=$1 wait=$2
  local tag="$label-$(date +%H%M%S)"

  sleep 3

  env LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
      LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE="$PROFILE" \
      LLAMA_MOE_PREAD_WAIT="$wait" \
      "$BIN" -m "$MODEL" -lv 4 \
      --moe-expert-cache 88 --moe-expert-cache-inserts 0 \
      --gpu-layers all --cpu-moe --load-mode mmap --fit off \
      --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --threads 8 --threads-batch 8 \
      --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$tag.server.log" 2>&1 &
  local pid=$!

  local i
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break
    sleep 2
  done
  if ! curl -sf http://127.0.0.1:1243/health >/dev/null; then
      echo "$tag: server failed to come up" | tee -a "$OUT/decode-preadbarrier-results.txt"
      kill -9 $pid 2>/dev/null; return 1
  fi

  sleep 2
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"'","n_predict":8,"temperature":0.6,"top_p":0.95,"top_k":20,
    "seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' -o /dev/null --max-time 900

  local io0; io0=$(awk '/read_bytes/{print $2}' /proc/$pid/io)
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"' Continue the story.","n_predict":256,"temperature":0.6,
    "top_p":0.95,"top_k":20,"seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' \
    -o "$OUT/$tag.resp.json" --max-time 1800
  local io1; io1=$(awk '/read_bytes/{print $2}' /proc/$pid/io)

  local hit; hit=$(grep -o "hit-rate=[0-9.]*%" "$OUT/$tag.server.log" | tail -1)
  python3 - "$io0" "$io1" "$OUT/$tag.resp.json" "$tag" "${hit:-n/a}" "$wait" <<'EOF' | tee -a "$OUT/decode-preadbarrier-results.txt"
import json, sys
io0, io1, fn, tag, hit, wait = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
t = json.load(open(fn))["timings"]
tot_mib = (io1 - io0) / 2**20
gen = t["predicted_n"] * 1000.0 / t["predicted_ms"]
per_tok = tot_mib / t["predicted_n"]
print("%-18s wait=%s  %6.2f t/s | %7.1f MiB/tok | %5.2f GB/s | %6.1f GB window | %s"
      % (tag, wait, gen, per_tok, per_tok * 1.048576 * gen / 1000,
         tot_mib * 1.073741824 / 1000, hit))
EOF

  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

: > "$OUT/decode-preadbarrier-results.txt"
run_arm pwait-on  1
run_arm pwait-off 0
run_arm pwait-on2 1
echo "PREAD BARRIER AB DONE"
