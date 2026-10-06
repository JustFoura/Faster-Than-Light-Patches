#!/usr/bin/env bash
# Prefill CPU-routing validation ladder, 2026-09-29.
#
# Validates GGML_VK_MOE_HOST_MAX_IDS=1 (prompt-path MoE routed from the Vulkan
# shader page-fault path into the CPU mul_mat_id kernel) under the hardening
# the 09-29 discovery doc calls for:
#   - every rung bounds the load-time graph reserve BY CONSTRUCTION:
#     ctx == ubatch == the rung's width, so min(n_ctx, n_ubatch) is the rung
#     width, never the full 8k+ context;
#   - every run goes through scripts/bench-prefill-safe.sh: oom_score_adj=500,
#     launch gate, 5 s MemAvailable sampling, and an in-run kill floor
#     (MEM_FLOOR_GB) so an overcommitting run kills the SERVER, not the session;
#   - the ladder climbs width only after the previous rung produced a result.
#
# Decision rule (discovery doc): the CPU arm's effective GB/s
# (process read bytes / prompt wall) beats the Vulkan path's ~0.8-1.1 GB/s.
# Usage: scripts/bench-prefill-moecpu-ladder-0929.sh [RUNGS...]
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUT=$ROOT/docs/bench-results-2026-09-29
P500=$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-500.txt
PLONG=$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-long.txt
[ -f "$PLONG" ] || python3 "$ROOT/scripts/gen-long-prompt.py" "$PLONG" 8502
mkdir -p "$OUT"

run_rung() {  # label prompt ubatch extra_env [floor]
  local label=$1 prompt=$2 ub=$3 extra=$4 floor=${5:-2}
  local ctx=$ub batch=$ub
  echo "=== $label (ubatch $ub ctx $ctx) ==="
  LABEL=$label PROMPT_FILE=$prompt OUT_DIR=$OUT \
  CTX=$ctx UBATCH=$ub BATCH=$batch MEM_FLOOR_GB=$floor \
  EXTRA_ENV="$extra" \
  bash "$ROOT/scripts/bench-prefill-safe.sh"
  local rc=$?
  if [ ! -f "$OUT/$label.json" ]; then
    echo "$label: NO RESULT — server did not survive; stopping the ladder here" >&2
    return 1
  fi
  return 0
}

RUNGS=${@:-mcpu-ub512 mcpu-ub2048 stock-ub8704 mcpu-ub8704 mcpu-ub8704-mrhs}
for r in $RUNGS; do
  case $r in
    mcpu-ub512)
      run_rung mcpu-ub512 "$P500" 512 "GGML_VK_MOE_HOST_MAX_IDS=1;GGML_MOE_NODE_LOG=1" 2 || break ;;
    mcpu-ub2048)
      run_rung mcpu-ub2048 "$P500" 2048 "GGML_VK_MOE_HOST_MAX_IDS=1;GGML_MOE_NODE_LOG=1" 2 || break ;;
    stock-ub8704)
      run_rung stock-ub8704 "$PLONG" 8704 "GGML_MOE_NODE_LOG=1" 2 || break ;;
    mcpu-ub8704)
      run_rung mcpu-ub8704 "$PLONG" 8704 "GGML_VK_MOE_HOST_MAX_IDS=1;GGML_MOE_NODE_LOG=1" 3 || break ;;
    mcpu-ub8704-mrhs)
      # stacked: multi-RHS IQ4 kernel + wider fanout once the CPU path is live
      run_rung mcpu-ub8704-mrhs "$PLONG" 8704 "GGML_VK_MOE_HOST_MAX_IDS=1;GGML_MOE_NODE_LOG=1;GGML_MOE_Q4_MULTI_RHS=1;LLAMA_MOE_FANOUT=16" 3 || break ;;
    *)
      echo "unknown rung $r" >&2; exit 1 ;;
  esac
done
echo "LADDER DONE"
