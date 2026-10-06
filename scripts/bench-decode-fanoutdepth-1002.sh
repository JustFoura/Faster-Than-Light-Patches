#!/usr/bin/env bash
# Decode A/B: fanout job depth (LLAMA_MOE_FANOUT_DEPTH).
#
# Measured cause (scripts/probe-drive-rate.py, 816 KiB buffered random preads —
# the exact shape the decode path issues):
#
#   QD 1 -> 1.16 GB/s    QD 2 -> 2.13    QD 4 -> 3.21
#   QD 8 -> 3.96 GB/s    QD 16 -> 4.26   QD 24 -> 4.17
#
# Decode extracts 1.38-1.47 GB/s, i.e. queue depth ~1.2, while holding a
# 24-worker pool. The cause is in ggml_moe_fanout_pool_worker: every worker has
# g=0, so all of them wait on and drain the SAME `done`-indexed job. Concurrency
# is therefore bounded by the slices in one job, and with the expert cache
# absorbing ~78% of routes a decode node's miss job is ~2 slices. LLAMA_MOE_
# FANOUT_DEPTH lets workers run `depth` jobs ahead so the stream stays deep
# across job boundaries.
#
# depth=1 is bit-for-bit the pre-change behaviour (the wait reduces to
# `want >= submitted`). At 244 MiB/token, QD4 predicts ~13 t/s and QD8 ~16 t/s.
#
# Order-controlled A / B / A, plus a byte-for-byte parity check against arm A:
# pread placement cannot change the arithmetic, and this repo gates on parity.
# Usage: scripts/bench-decode-fanoutdepth-1002.sh
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

# run_arm <label> <depth>
run_arm() {
  local label=$1 depth=$2
  local tag="$label-$(date +%H%M%S)"

  sleep 3

  env LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
      LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE="$PROFILE" \
      LLAMA_MOE_FANOUT_DEPTH="$depth" \
      "$BIN" -m "$MODEL" -lv 4 \
      --moe-expert-cache 48 --moe-expert-cache-inserts 0 \
      --gpu-layers all --cpu-moe --load-mode mmap --fit off \
      --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --threads 8 --threads-batch 8 \
      --host 127.0.0.1 --port 1243 --no-webui --no-warmup > "$OUT/$tag.server.log" 2>&1 &
  local pid=$!

  # The box loses server starts to OOM kills as zram creeps up (7.5 GB at the
  # time of writing). Retry once per arm rather than recording a bogus failure.
  local i ok=0
  for attempt in 1 2; do
    for i in $(seq 1 240); do
      curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break
      sleep 2
    done
    if curl -sf http://127.0.0.1:1243/health >/dev/null; then ok=1; break; fi
    echo "  [$tag] attempt $attempt: no health; avail $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB" \
        | tee -a "$OUT/decode-depth-results.txt"
    sleep 5
    env LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
        LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE="$PROFILE" \
        LLAMA_MOE_FANOUT_DEPTH="$depth" \
        "$BIN" -m "$MODEL" -lv 4 \
        --moe-expert-cache 48 --moe-expert-cache-inserts 0 \
        --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 8192 --parallel 1 -b 512 -ub 512 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port 1243 --no-webui --no-warmup > "$OUT/$tag.server.log" 2>&1 &
    pid=$!
  done
  if [ "$ok" -ne 1 ]; then
      echo "$tag: server failed to come up" | tee -a "$OUT/decode-depth-results.txt"
      kill -9 $pid 2>/dev/null; return 1
  fi
  grep -h "\[moe-fanout\]" "$OUT/$tag.server.log" | head -1 | sed "s/^/  [$tag] /"

  sleep 2
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"'","n_predict":8,"temperature":0,"top_p":0.95,"top_k":20,
    "seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' -o /dev/null --max-time 900

  local io0; io0=$(awk '/read_bytes/{print $2}' /proc/$pid/io)
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$PROMPT"' Continue the story.","n_predict":256,"temperature":0,
    "top_p":0.95,"top_k":20,"seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' \
    -o "$OUT/$tag.resp.json" --max-time 1800
  local io1; io1=$(awk '/read_bytes/{print $2}' /proc/$pid/io)

  local hit; hit=$(grep -o "hit-rate=[0-9.]*%" "$OUT/$tag.server.log" | tail -1)
  python3 - "$io0" "$io1" "$OUT/$tag.resp.json" "$tag" "${hit:-n/a}" "$depth" <<'EOF' | tee -a "$OUT/decode-depth-results.txt"
import json, sys
io0, io1, fn, tag, hit, depth = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
t = json.load(open(fn))["timings"]
tot_mib = (io1 - io0) / 2**20
gen = t["predicted_n"] * 1000.0 / t["predicted_ms"]
per_tok = tot_mib / t["predicted_n"]
print("%-16s depth=%s  %6.2f t/s | %6.1f MiB/tok | %5.2f GB/s | %6.1f GB window | %s"
      % (tag, depth, gen, per_tok, per_tok * 1.048576 * gen / 1000,
         tot_mib * 1.073741824 / 1000, hit))
EOF

  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

: > "$OUT/decode-depth-results.txt"
run_arm depth1 1
run_arm depth4 4
run_arm depth1b 1

# parity: greedy temp-0 seed 42, so every arm must return the same text
python3 - "$OUT" <<'EOF' | tee -a "$OUT/decode-depth-results.txt"
import glob, json, os, sys
out = sys.argv[1]
ref = None
for fn in sorted(glob.glob(os.path.join(out, "depth*.resp.json"))):
    c = json.load(open(fn)).get("content", "")
    if ref is None:
        ref, refn = c, os.path.basename(fn)
        continue
    print("parity %-22s vs %-22s : %s" % (os.path.basename(fn), refn,
          "PASS byte-identical" if c == ref else "FAIL (%d vs %d chars)"
          % (len(c), len(ref))))
EOF
echo "FANOUT DEPTH AB DONE"
