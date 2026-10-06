#!/usr/bin/env bash
# Prompt-phase expert pre-reader A/B (LLAMA_MOE_PROMPT_PREREAD), 2026-10-01.
# One-pass 14,963-tok prefill via bench-prefill-safe.sh. Arms: OFF / ON / OFF.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUT=$ROOT/docs/bench-results-2026-10-01
mkdir -p "$OUT"

run_gc() {
  local mode=$1
  local extra=""
  [ "$mode" = on ] && extra="LLAMA_MOE_PROMPT_PREREAD=1"
  echo "=== preread $mode ==="
  LABEL=preread-$mode PROMPT_FILE=$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt \
    OUT_DIR=$OUT CTX=16384 UBATCH=16384 BATCH=16384 MEM_FLOOR_GB=2 PORT=1239 \
    EXTRA_ENV="$extra" bash $ROOT/scripts/bench-prefill-safe.sh
  python3 - "$OUT/preread-$mode.json" <<'PYEOF' | tee -a "$OUT/preread-ab-results.txt"
import json, sys
r = json.load(open(sys.argv[1])); t = r["timings"]; d = r.get("process_delta") or {}
tps = t["prompt_n"] * 1000.0 / t["prompt_ms"]
gb = d.get("read_bytes", 0) / 2**30
print("preread-%s: %.2f t/s (%d tok, %.1f GB read)" % (sys.argv[1].rsplit("/",1)[-1].split(".")[0].replace("preread-",""), tps, t["prompt_n"], gb))
PYEOF
}

: > "$OUT/preread-ab-results.txt"
run_gc off
run_gc on
run_gc off
echo "PREREAD AB DONE"
