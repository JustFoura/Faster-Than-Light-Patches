#!/usr/bin/env bash
# Device-staging A/B (LLAMA_VK_MOE_DEVICE_STAGE), 2026-10-01.
# Phase 1 — correctness: temp-0 generation with staging OFF vs ON must be
# byte-identical (staging only changes where weights are read from).
# Phase 2 — prefill: one-pass 14,963-tok gate C, OFF / ON / OFF.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
OUT=$ROOT/docs/bench-results-2026-10-01
P500=$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-500.txt
mkdir -p "$OUT"

run_server() {  # run_server <on|off>
  sleep 2
  if [ "$1" = on ]; then
    env LLAMA_KV_HOSTMEM=1 LLAMA_VK_MOE_DEVICE_STAGE=1 \
        "$BIN" -m "$MODEL" -lv 4 \
        --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 2048 --parallel 1 -b 2048 -ub 2048 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port 1244 --no-webui > "$OUT/stagetest-$1.server.log" 2>&1 &
  else
    env -u LLAMA_VK_MOE_DEVICE_STAGE LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 \
        --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 2048 --parallel 1 -b 2048 -ub 2048 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port 1244 --no-webui > "$OUT/stagetest-$1.server.log" 2>&1 &
  fi
  local pid=$! i
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:1244/health >/dev/null 2>&1 && break; sleep 2; done
  curl -sf http://127.0.0.1:1244/health >/dev/null || { echo "server $1 failed"; kill -9 $pid 2>/dev/null; return 1; }
}

echo "=== phase 1: correctness (temp 0, 446-tok prompt, 48 tok) ==="
run_server off
curl -s http://127.0.0.1:1244/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":$(python3 -c 'import json;print(json.dumps(open("'"$P500"'").read()))'),\"n_predict\":48,\"temperature\":0,\"seed\":42,\"cache_prompt\":false,\"ignore_eos\":true,\"stream\":false}" \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["content"])' > "$OUT/stagetest-off.txt"
sleep 2
run_server on
curl -s http://127.0.0.1:1244/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":$(python3 -c 'import json;print(json.dumps(open("'"$P500"'").read()))'),\"n_predict\":48,\"temperature\":0,\"seed\":42,\"cache_prompt\":false,\"ignore_eos\":true,\"stream\":false}" \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["content"])' > "$OUT/stagetest-on.txt"
if diff -q "$OUT/stagetest-off.txt" "$OUT/stagetest-on.txt" >/dev/null 2>&1; then
  echo "CORRECTNESS: PASS — outputs byte-identical"
else
  echo "CORRECTNESS: FAIL — outputs differ!"
  diff "$OUT/stagetest-off.txt" "$OUT/stagetest-on.txt" | head -6
fi

echo "=== phase 2: prefill A/B ==="
run_gc() {
  local extra=""
  [ "$1" = on ] && extra="LLAMA_VK_MOE_DEVICE_STAGE=1"
  LABEL=stage-$1 PROMPT_FILE=$P500 PROMPT_FILE2="" \
    PROMPT_FILE_15K="" OUT_DIR=$OUT CTX=16384 UBATCH=16384 BATCH=16384 \
    MEM_FLOOR_GB=2 PORT=1239 EXTRA_ENV="$extra" \
    PROMPT_FILE="$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt" \
    bash "$ROOT/scripts/bench-prefill-safe.sh"
  python3 - "$OUT/stage-$1.json" <<'PYEOF' | tee -a "$OUT/stage-ab-results.txt"
import json, sys
r = json.load(open(sys.argv[1])); t = r["timings"]; d = r.get("process_delta") or {}
tps = t["prompt_n"] * 1000.0 / t["prompt_ms"]
gb = d.get("read_bytes", 0) / 2**30
print("stage-%s: %.2f t/s (%d tok, %.1f GB read)" % (sys.argv[1].rsplit("/",1)[-1].split(".")[0].replace("stage-",""), tps, t["prompt_n"], gb))
PYEOF
}
: > "$OUT/stage-ab-results.txt"
run_gc off
run_gc on
run_gc off
echo "STAGE AB DONE"
