#!/usr/bin/env bash
# Single-request prefill A/B for the Flash-Next expert-streaming experiments.
# Starts one llama-server from the prefill-exp tree with the requested env gates,
# runs scripts/bench_flash_next.py once (1 predicted token = pure prefill cost),
# saves JSON + server stderr (route histogram lines), then stops the server.
#
# Knobs (env): LABEL, PORT, CTX, UBATCH, BATCH, FANOUT, DONTNEED, HISTO, MULTI_RHS,
#              FLASH_BIN, FLASH_CTX  (see defaults below)
# Example:
#   LABEL=dn1-fan16 UBATCH=512 DONTNEED=1 FANOUT=16 scripts/bench-prefill-window.sh
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${FLASH_BIN:-$HOME/build/llama.cpp-prefill-exp/build-vulkan/bin/llama-server}"
MODEL="${FLASH_MODEL:-$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf}"
PROMPT="${PROMPT_FILE:-$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-500.txt}"
PORT="${PORT:-1237}"
CTX="${CTX:-8192}"
UBATCH="${UBATCH:-512}"
BATCH="${BATCH:-2048}"
LABEL="${LABEL:-run}"
OUT="${OUT_DIR:-$ROOT/docs/bench-results-2026-09-27}"
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "binary not executable: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

export LLAMA_KV_HOSTMEM=1
export LLAMA_MOE_FANOUT="${FANOUT:-8}"
if [[ "${DONTNEED:-0}" == "1" ]]; then export GGML_MOE_MMAP_DONTNEED=1; fi
if [[ "${HISTO:-0}" == "1" ]]; then export GGML_MOE_ROUTE_HISTOGRAM=1; fi
if [[ "${MULTI_RHS:-0}" == "1" ]]; then export GGML_MOE_Q4_MULTI_RHS=1; fi

"$BIN" --model "$MODEL" \
    --gpu-layers all --cpu-moe --load-mode mmap --fit off \
    --ctx-size "$CTX" --parallel 1 --batch-size "$BATCH" --ubatch-size "$UBATCH" \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 8 --threads-batch 8 \
    --host 127.0.0.1 --port "$PORT" --no-webui \
    > "$OUT/$LABEL.server.log" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
done
if [[ "$ready" != "1" ]]; then echo "server failed to become ready; see $OUT/$LABEL.server.log" >&2; exit 1; fi

python3 "$ROOT/scripts/bench_flash_next.py" \
    --url "http://127.0.0.1:$PORT" --pid "$SRV" \
    --label "$LABEL" \
    --prompt-file "$PROMPT" \
    --n-predict 1 --no-thinking \
    --output "$OUT/$LABEL.json"

kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
trap - EXIT

python3 - "$OUT/$LABEL.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
t = r.get("timings") or {}
d = r.get("process_delta") or {}
print(f"{r['label']}: prompt_tokens={t.get('prompt_n')} prompt_tps={t.get('prompt_ms') and t['prompt_n']*1000.0/t['prompt_ms']:.2f} "
      f"read_GB={d.get('read_bytes',0)/2**30:.1f} major_faults={d.get('major_faults')}")
EOF
