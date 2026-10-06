#!/usr/bin/env bash
# Drive-rate profile during a staged prefill: is the wall compute (bursty IO
# with gaps) or CPU-read-side (flat low rate)?
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODEL="${FLASH_MODEL:-$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf}"
source "$SCRIPT_DIR/lib/storage-device.sh"
OUT=$ROOT/docs/bench-results-2026-10-01

( while pgrep -x llama-server >/dev/null 2>&1; do
    awk -v device="$FTL_DISKSTATS_DEVICE" '$3 == device {print systime(), $6*512}' /proc/diskstats
    sleep 0.5
  done ) > "$OUT/stage-io-trace.txt" &
TRACE=$!

LABEL=stage-prof PROMPT_FILE="$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt" \
  OUT_DIR="$OUT" CTX=16384 UBATCH=16384 BATCH=16384 MEM_FLOOR_GB=2 PORT=1239 \
  EXTRA_ENV="LLAMA_VK_MOE_DEVICE_STAGE=1" \
  bash "$ROOT/scripts/bench-prefill-safe.sh"

kill $TRACE 2>/dev/null
python3 - "$OUT/stage-io-trace.txt" <<'PYEOF'
import sys
rows = []
for ln in open(sys.argv[1]):
    p = ln.split()
    if len(p) == 2:
        rows.append((int(p[0]), int(p[1])))
rates = []
for a, b in zip(rows, rows[1:]):
    dt = b[0] - a[0]
    if dt > 0:
        rates.append((b[1] - a[1]) / dt / 2**30)
if rates:
    rates.sort()
    n = len(rates)
    active = [r for r in rates if r > 0.05]
    print(f"samples={n} idle={n-len(active)}")
    if active:
        print(f"active rate: median {active[len(active)//2]:.2f} GB/s, p90 {active[int(n*0.9) if n>10 else -1]:.2f}, max {max(active):.2f}")
    print(f"mean overall: {sum(rates)/n:.2f} GB/s")
PYEOF
