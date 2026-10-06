#!/usr/bin/env bash
# Prefill ubatch ladder on the unified tree: long prompt, cache disabled,
# one-predict-token (pure prefill cost). Route histogram on — doubles as the
# bug #2 probe at large ubatch.
# Usage: scripts/bench-prefill-ladder-0928.sh [TOKENS] [UBS...]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TOKENS=${1:-4800}
shift || true
UBS=${@:-1024 2048 4096}

PROMPT=$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-long.txt
[ -f "$PROMPT" ] || python3 "$ROOT/scripts/gen-long-prompt.py" "$PROMPT" "$TOKENS"

for UB in $UBS; do
  echo "=== ubatch $UB ==="
  LABEL=prefill-ub$UB \
  FLASH_BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server \
  PROMPT_FILE=$PROMPT \
  OUT_DIR=$ROOT/docs/bench-results-2026-09-28 \
  CTX=16384 UBATCH=$UB BATCH=$((UB > 2048 ? UB : 2048)) \
  HISTO=1 \
  bash "$ROOT/scripts/bench-prefill-window.sh"
done
echo "PREFILL LADDER DONE"
