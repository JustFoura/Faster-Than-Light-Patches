#!/usr/bin/env bash
# Prefill probe, 2026-10-01 evening: where does the one-pass wall go?
#
# Waits for the night decode sweep to finish, then runs the 14,963-token
# one-pass prefill at three ubatch widths on the stock fault path (gate C
# condition), sampling GPU busy% at 200 ms and reporting the busy fraction
# over each run's prompt_ms window. Interpretation key:
#   busy >= ~85%  the wall is inside the MoE shaders (kernel work needed)
#   busy <  ~60%  the wall is dispatch/sync stalls (submission work needed)
# Results: docs/bench-results-2026-10-01/prefillprobe-results.txt
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUT=$ROOT/docs/bench-results-2026-10-01
RES="$OUT/prefillprobe-results.txt"
RUNTXT="$OUT/nightsweep-run.txt"

# wait for the decode sweep (up to 75 min)
for _ in $(seq 1 150); do
    grep -q "NIGHT SWEEP DONE" "$RUNTXT" 2>/dev/null && break
    sleep 30
done

gpu_probe() {  # gpu_probe <csv> — until killed
    local out=$1
    : > "$out"
    while :; do
        printf '%s %s\n' "$(date +%s%3N)" \
            "$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo NA)" >> "$out"
        sleep 0.2
    done
}

arm() {  # arm <label> <ubatch>
    local label=$1 ub=$2
    gpu_probe "$OUT/$label.gpu.csv" & local gs=$!
    LABEL="$label" PROMPT_FILE="$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt" \
        OUT_DIR="$OUT" CTX=16384 UBATCH="$ub" BATCH=16384 MEM_FLOOR_GB=2 PORT=1239 \
        bash "$ROOT/scripts/bench-prefill-safe.sh"
    kill $gs 2>/dev/null
    python3 - "$OUT/$label.json" "$OUT/$label.gpu.csv" <<'PYEOF' | tee -a "$RES"
import json, sys, os
r = json.load(open(sys.argv[1]))
t = r.get("timings") or {}
pms = t.get("prompt_ms")
d = r.get("process_delta") or {}
busy = "NA"
if pms:
    end_ms = r["wall_seconds"] * 1000.0
    lo = end_ms - pms - 1500
    rows = []
    for l in open(sys.argv[2]):
        p = l.split()
        if len(p) == 2 and p[1] != "NA" and float(p[0]) >= lo:
            rows.append(int(p[1]))
    busy = "%.0f (%d samples)" % (sum(rows) / len(rows), len(rows)) if rows else "NA"
print("%s: prompt_tps=%.2f (%d tok, %.1f s) read_GB=%.1f faults=%s gpu_busy=%s%%" % (
    r["label"], t["prompt_n"] * 1000.0 / pms if pms else 0, t.get("prompt_n"),
    (pms or 0) / 1000.0, d.get("read_bytes", 0) / 2**30, d.get("major_faults"), busy))
PYEOF
}

: > "$RES"
arm p-ub16384 16384
arm p-ub8192  8192
arm p-ub4096  4096
echo "PREFILL PROBE DONE" | tee -a "$RES"
