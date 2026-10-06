#!/usr/bin/env bash
# Evening sweep, 2026-10-01 (same boot as the parity gate, ~4 h uptime):
#   arm 1  base   single-text profile (control 1)
#   arm 2  multi  multi-text profile (/tmp/moe_routes_multi.log, 3 domains)
#   arm 3  base   control 2
#   arm 4  combo  single profile + MISS_PREFETCH=1 + FANOUT_PIPE=1
#                (the untested pair: pipe's two worker groups drain the
#                 prefetch job for layer i+1 concurrently with real job i)
#   arm 5  base   control 3
#   arm 6  mcombo multi profile + combo env (only if combo moves arm 4)
# 256-token warm-pair protocol (ignore_eos), drive sectors + GPU busy%.
# Results: docs/bench-results-2026-10-01/nightsweep-results.txt
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
source "$SCRIPT_DIR/lib/storage-device.sh"
PROFILE_SINGLE=/tmp/moe_routes_decode.log
PROFILE_MULTI=/tmp/moe_routes_multi.log
OUT=$ROOT/docs/bench-results-2026-10-01
mkdir -p "$OUT"
[ -f "$PROFILE_SINGLE" ] || gunzip -c "$ROOT/docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz" > "$PROFILE_SINGLE"
[ -f "$PROFILE_MULTI" ] || { echo "FATAL: $PROFILE_MULTI missing (run scripts/capture-routes-multi-1001.sh)" >&2; exit 1; }
PROMPT='<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
RES="$OUT/nightsweep-results.txt"

gpu_sampler() {  # gpu_sampler <out.csv> — runs until killed
    local out=$1
    : > "$out"
    while :; do
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo NA)" >> "$out"
        sleep 0.2
    done
}

run_arm() {  # run_arm <label> <profile> [extra ENV=val ...]
    local label=$1 profile=$2; shift 2
    local tag="$label-$(date +%H%M%S)"
    sleep 2
    env -u LLAMA_MOE_MISS_PREFETCH -u LLAMA_MOE_FANOUT_PIPE "$@" \
        LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
        LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE="$profile" \
        "$BIN" -m "$MODEL" -lv 4 --moe-expert-cache 88 --moe-expert-cache-inserts 0 \
        --gpu-layers all --cpu-moe --load-mode mmap --fit off --ctx-size 8192 --parallel 1 \
        -b 512 -ub 512 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
        --threads 8 --threads-batch 8 --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$tag.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 180); do curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break; sleep 2; done
    curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "$tag: server failed"; kill -9 $pid 2>/dev/null; return 1; }
    sleep 2
    curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
      "prompt":"'"$PROMPT"'","n_predict":8,"temperature":0.6,"top_p":0.95,"top_k":20,"seed":42,
      "cache_prompt":true,"ignore_eos":true,"stream":false}' -o /dev/null --max-time 900
    local io0=$(diskstats_sectors_read)
    gpu_sampler "$OUT/$tag.gpu.csv" & local gs=$!
    curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
      "prompt":"'"$PROMPT"' Continue the story.","n_predict":256,"temperature":0.6,"top_p":0.95,
      "top_k":20,"seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' \
      -o "$OUT/$tag.resp.json" --max-time 1800
    kill $gs 2>/dev/null
    local io1=$(diskstats_sectors_read)
    python3 - "$io0" "$io1" "$OUT/$tag.resp.json" "$OUT/$tag.gpu.csv" <<'PYEOF' | tee -a "$RES"
import json, sys
s0, s1, fn, gcsv = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
t = json.load(open(fn))["timings"]
n = t["predicted_n"]
gen = n * 1000.0 / t["predicted_ms"]
mib = (s1 - s0) * 512 // 2**20
gbs = (s1 - s0) * 512.0 / 2**30 * gen / 1000
busy = "NA"
try:
    rows = [int(l.split()[1]) for l in open(gcsv) if l.split()[1] != "NA"]
    busy = "%.0f" % (sum(rows) / len(rows)) if rows else "NA"
except Exception:
    pass
print("%s: %.2f t/s | drive %.1f MiB/tok (%.2f GB/s) | gpu_busy %s%%" % (
    fn.rsplit("/", 1)[-1].replace(".resp.json", ""), gen, mib / n, gbs, busy))
PYEOF
    grep -oE 'hit-rate=[0-9.]+%' "$OUT/$tag.server.log" | tail -1 | sed "s/^/$tag: /" | tee -a "$RES"
    kill -TERM $pid 2>/dev/null || true
    sleep 8
    kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

: > "$RES"
run_arm base   "$PROFILE_SINGLE"
run_arm multi  "$PROFILE_MULTI"
run_arm base   "$PROFILE_SINGLE"
run_arm combo  "$PROFILE_SINGLE" LLAMA_MOE_MISS_PREFETCH=1 LLAMA_MOE_FANOUT_PIPE=1
run_arm base   "$PROFILE_SINGLE"
run_arm mcombo "$PROFILE_MULTI"  LLAMA_MOE_MISS_PREFETCH=1 LLAMA_MOE_FANOUT_PIPE=1
echo "NIGHT SWEEP DONE"
