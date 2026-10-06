#!/usr/bin/env bash
# Decode: is the expert cache actually on, and what does it buy?
#
# The 10-02 report measured 4.5 t/s and concluded "decode is byte-bound at
# 736 MB/token, 15-20 t/s is unreachable". That log (/tmp/decode-bench.log)
# has no moe-cache init line at any verbosity, so --moe-expert-cache was not
# passed. The 10-01 artifacts with the cache on read 193-199 MiB/token at a
# 77.4% hit rate for 7.0-7.2 t/s. This harness puts both arms back to back,
# order-controlled, so the 2x is measured rather than inferred.
#
# Protocol is byte-for-byte the 10-01 fanout harness (scripts/bench-decode-fanout-1001.sh)
# so the numbers are comparable with the 7.0-7.2 t/s baseline.
# Usage: scripts/bench-decode-cache-1002.sh
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

# run_arm <label> <cache_slots|-1> <inserts>
run_arm() {
  local label=$1 slots=$2 ins=$3
  local stamp; stamp=$(date +%H%M%S)
  local tag="$label-$stamp"
  local cache_args=()
  if [ "$slots" -gt 0 ]; then
      cache_args=(--moe-expert-cache "$slots" --moe-expert-cache-inserts "$ins")
  fi

  sleep 3

  env LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
      LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE="$PROFILE" \
      "$BIN" -m "$MODEL" -lv 4 \
      "${cache_args[@]}" \
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
      echo "$tag: server failed to come up" | tee -a "$OUT/decode-cache-results.txt"
      kill -9 $pid 2>/dev/null; return 1
  fi

  # record what the server believes it loaded, before any request perturbs it
  grep -i "expert cache enabled\|no host-resident expert layers\|RAM tier\|preloaded" \
      "$OUT/$tag.server.log" | sed "s/^/  [$tag init] /"

  sleep 2
  # warmup: populates the prompt KV so the measured request is decode-only
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
  python3 - "$io0" "$io1" "$OUT/$tag.resp.json" "$tag" "${hit:-n/a}" <<'EOF' | tee -a "$OUT/decode-cache-results.txt"
import json, sys
io0, io1, fn, tag, hit = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
t = json.load(open(fn))["timings"]
tot_mib = (io1 - io0) / 2**20
gen = t["predicted_n"] * 1000.0 / t["predicted_ms"]
# The 10-01 harness printed the window total and labelled it MiB/tok, and
# divided by 1000 instead of 1.048576 to get "GB/s" (that is MiB/s). Both
# mislabels are fixed here; the raw window total is kept for cross-checking.
per_tok = tot_mib / t["predicted_n"]
print("%-14s %6.2f t/s | %7.1f MiB/tok | %5.2f GB/s | %6.1f GB window | %d tok in %.1f s | %s"
      % (tag, gen, per_tok, per_tok * 1.048576 * gen / 1000,
         tot_mib * 1.073741824 / 1000, t["predicted_n"],
         t["predicted_ms"] / 1000.0, hit))
EOF

  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

: > "$OUT/decode-cache-results.txt"
# order-controlled: A, B, A again. If the A arms disagree by more than the
# run-to-run band the box is drifting and the B number means nothing.
run_arm c88   88 0
run_arm nocache -1 0
run_arm c88b  88 0
echo "DECODE CACHE AB DONE"
