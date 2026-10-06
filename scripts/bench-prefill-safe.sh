#!/usr/bin/env bash
# Memory-safe single-request prefill A/B driver for the Flash-Next work.
#
# Differences from bench-prefill-window.sh (which is fine on a freshly
# rebooted box but gives the OOM killer no target):
#   - the server is launched with oom_score_adj=500, so if the run ever
#     overcommits, the KERNEL KILLS THIS SERVER, not the desktop session;
#   - MemAvailable is gated before launch (abort <8 GB) and sampled every 5 s
#     into the run log, so the memory cost of every config is on record;
#   - cleanup is kill-by-PID only, never pkill.
#
# Knobs (env): LABEL, PROMPT_FILE, UBATCH, BATCH, CTX, PORT, EXTRA_ENV
#              (semicolon-separated K=V list applied to the server env),
#              FLASH_BIN, OUT_DIR
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${FLASH_BIN:-$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server}"
MODEL="${FLASH_MODEL:-$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf}"
PROMPT="${PROMPT_FILE:-$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-500.txt}"
PORT="${PORT:-1239}"
CTX="${CTX:-8192}"
UBATCH="${UBATCH:-2048}"
BATCH="${BATCH:-2048}"
LABEL="${LABEL:?LABEL required}"
OUT="${OUT_DIR:-$ROOT/docs/bench-results-2026-09-29}"
mkdir -p "$OUT"

avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
# 7 GB gate: the server below runs with oom_score_adj=500, so any
# overcommit kills THIS server first, not the desktop. With that protection
# in place, 7 GB of headroom is the safe floor; without it, use 8.
if (( avail_kb < 7 * 1024 * 1024 )); then
    echo "ABORT: MemAvailable ${avail_kb} kB < 8 GB — benchmarking would risk an OOM kill" >&2
    exit 1
fi

export LLAMA_KV_HOSTMEM=1
export LLAMA_MOE_FANOUT="${FANOUT:-8}"
if [[ -n "${EXTRA_ENV:-}" ]]; then
    IFS=';' read -ra KV <<< "$EXTRA_ENV"
    for kv in "${KV[@]}"; do export "$kv"; done
fi

"$BIN" --model "$MODEL" \
    --gpu-layers all --cpu-moe --load-mode mmap --fit off \
    --ctx-size "$CTX" --parallel 1 --batch-size "$BATCH" --ubatch-size "$UBATCH" \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads "${THREADS:-8}" --threads-batch "${THREADS_BATCH:-${THREADS:-8}}" \
    --host 127.0.0.1 --port "$PORT" --no-webui -lv 4 \
    > "$OUT/$LABEL.server.log" 2>&1 &
SRV=$!
echo 500 > "/proc/$SRV/oom_score_adj" 2>/dev/null || true

# In-run guard: MemAvailable below the floor means the run is on the path that
# killed the session four times on 09-29. Kill THE SERVER now — the kernel
# reclaims its mappings and the desktop survives; waiting for the OOM killer
# or systemd-oomd does not (oom_score_adj does not rule the pressure path).
FLOOR_GB="${MEM_FLOOR_GB:-2}"
( while kill -0 "$SRV" 2>/dev/null; do
    mem_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    awk -v ts="$(date +%H:%M:%S)" -v mb="$mem_kb" 'BEGIN {printf "%s MemAvailable %.2f GB\n", ts, mb/1048576}'
    if (( mem_kb < FLOOR_GB * 1024 * 1024 )); then
        echo "$(date +%H:%M:%S) MemAvailable ${mem_kb} kB < ${FLOOR_GB} GB floor — killing server $SRV (guard)" >> "$OUT/$LABEL.server.log"
        kill -9 "$SRV" 2>/dev/null || true
        exit 1
    fi
    sleep 5
  done ) >> "$OUT/$LABEL.server.log" &
MON=$!

cleanup() {
    kill "$SRV" 2>/dev/null || true
    kill "$MON" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SRV" 2>/dev/null || { echo "$LABEL: server died during load — see $OUT/$LABEL.server.log" >&2; exit 1; }
    sleep 2
done
if [[ "$ready" != 1 ]]; then
    echo "$LABEL: server never became healthy" >&2
    exit 1
fi

python3 "$ROOT/scripts/bench_flash_next.py" \
    --url "http://127.0.0.1:$PORT" --pid "$SRV" \
    --label "$LABEL" \
    --prompt-file "$PROMPT" \
    --n-predict 1 --no-thinking \
    --output "$OUT/$LABEL.json"

kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
trap - EXIT
kill "$MON" 2>/dev/null || true

python3 - "$OUT/$LABEL.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
t = r.get("timings") or {}
d = r.get("process_delta") or {}
print(f"{r['label']}: prompt_tokens={t.get('prompt_n')} prompt_tps={t.get('prompt_ms') and t['prompt_n']*1000.0/t['prompt_ms']:.2f} "
      f"read_GB={d.get('read_bytes',0)/2**30:.1f} major_faults={d.get('major_faults')}")
EOF
