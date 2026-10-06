#!/usr/bin/env bash
# MTP re-test on the SN5000, 2026-10-01 evening. The 09-27 P310 verdict was
# closed (4.04-4.09 vs 4.57 plain): verify widths landed in the uncached tail
# (~1.4x miss bytes for 2.06 tok/step). Two things changed since: the drive is
# 1.6x faster on the miss stream and the route profile + recipe are in. Same
# warm-pair protocol as the night sweep (256 tok, ignore_eos, cache_prompt),
# MTP on vs the sweep's base controls; inserts 2 (the 09-27 MTP arm config).
# Results: docs/bench-results-2026-10-01/mtp-night-results.txt
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
source "$SCRIPT_DIR/lib/storage-device.sh"
SIDECAR=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
PROFILE=/tmp/moe_routes_decode.log
OUT=$ROOT/docs/bench-results-2026-10-01
RES="$OUT/mtp-night-results.txt"
RUNTXT="$OUT/prefillprobe-run.txt"
[ -f "$PROFILE" ] || gunzip -c "$ROOT/docs/bench-results-2026-09-27/moe-routes-decode-256t.log.gz" > "$PROFILE"
PROMPT='<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

for _ in $(seq 1 240); do
    grep -q "PREFILL PROBE DONE" "$RUNTXT" 2>/dev/null && break
    sleep 30
done

run_mtp() {
    local label=$1
    local tag="$label-$(date +%H%M%S)"
    sleep 2
    env -u LLAMA_MOE_MISS_PREFETCH -u LLAMA_MOE_FANOUT_PIPE \
        LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode LLAMA_MOE_PREAD=1 \
        LLAMA_MOE_FANOUT=24 LLAMA_MOE_PROFILE="$PROFILE" \
        "$BIN" -m "$MODEL" -lv 4 \
        --spec-type draft-mtp --spec-draft-model "$SIDECAR" --spec-draft-n-max 2 \
        --moe-expert-cache 88 --moe-expert-cache-inserts 2 \
        --gpu-layers all --cpu-moe --load-mode mmap --fit off --ctx-size 8192 --parallel 1 \
        -b 512 -ub 512 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
        --threads 8 --threads-batch 8 --host 127.0.0.1 --port 1243 --no-webui > "$OUT/$tag.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 240); do curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break; sleep 2; done
    curl -sf http://127.0.0.1:1243/health >/dev/null || { echo "$tag: server failed"; kill -9 $pid 2>/dev/null; return 1; }
    sleep 2
    curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
      "prompt":"'"$PROMPT"'","n_predict":8,"temperature":0.6,"top_p":0.95,"top_k":20,"seed":42,
      "cache_prompt":true,"ignore_eos":true,"stream":false}' -o /dev/null --max-time 900
    local io0=$(diskstats_sectors_read)
    curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
      "prompt":"'"$PROMPT"' Continue the story.","n_predict":256,"temperature":0.6,"top_p":0.95,
      "top_k":20,"seed":42,"cache_prompt":true,"ignore_eos":true,"stream":false}' \
      -o "$OUT/$tag.resp.json" --max-time 1800
    local io1=$(diskstats_sectors_read)
    python3 - "$io0" "$io1" "$OUT/$tag.resp.json" <<'PYEOF' | tee -a "$RES"
import json, sys
s0, s1, fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t = json.load(open(fn))["timings"]
n = t["predicted_n"]
gen = n * 1000.0 / t["predicted_ms"]
mib = (s1 - s0) * 512 // 2**20
gbs = (s1 - s0) * 512.0 / 2**30 * gen / 1000
print("%s: %.2f t/s | drive %.1f MiB/tok (%.2f GB/s) | %d tok in %.1f s" % (
    fn.rsplit("/", 1)[-1].replace(".resp.json", ""), gen, mib / n, gbs, n, t["predicted_ms"] / 1000))
PYEOF
    grep -oE 'hit-rate=[0-9.]+%' "$OUT/$tag.server.log" | tail -1 | sed "s/^/$tag: /" | tee -a "$RES"
    grep -oE 'accepted=[0-9]+|draft=[0-9]+|accept[^ ]* [0-9.]+%' "$OUT/$tag.server.log" | tail -3 | sed "s/^/$tag: /" | tee -a "$RES" || true
    kill -TERM $pid 2>/dev/null || true
    sleep 8
    kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

: > "$RES"
run_mtp mtp
run_mtp mtp
echo "MTP NIGHT DONE"
