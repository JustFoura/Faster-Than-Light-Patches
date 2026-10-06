#!/usr/bin/env bash
# CPU-side attribution probe, 2026-10-01 late evening: one-pass 14,963-tok
# prefill (ubatch 16384, gate-C condition) while sampling GPU busy%, process
# CPU jiffies, and per-thread CPU every 200 ms / 1 s. Decides where the
# GPU-idle 55% of the wall goes:
#   CPU pegged ~800%  -> CPU-side graph ops (PLE gather / router / norms)
#   CPU low, GPU low  -> latency stalls (faults, syncs, submission)
# Output: docs/bench-results-2026-10-01/prefillcpu-results.txt + samples.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUT=$ROOT/docs/bench-results-2026-10-01
RES="$OUT/prefillcpu-results.txt"

LABEL=cpu-probe PROMPT_FILE="$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt" \
    OUT_DIR="$OUT" CTX=16384 UBATCH=16384 BATCH=16384 MEM_FLOOR_GB=2 PORT=1239 \
    bash "$ROOT/scripts/bench-prefill-safe.sh" &
DRIVER=$!

# wait for the server to appear, then sample until it exits
SRV=""
for _ in $(seq 1 300); do
    SRV=$(pgrep -x llama-server | head -1)
    [[ -n "$SRV" ]] && break
    sleep 1
done
[[ -z "$SRV" ]] && { echo "no server appeared" ; exit 1; }

: > "$OUT/cpu-probe.gpu.csv"
: > "$OUT/cpu-probe.proc.csv"
( while kill -0 "$SRV" 2>/dev/null; do
    printf '%s %s\n' "$(date +%s%3N)" \
        "$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo NA)" >> "$OUT/cpu-probe.gpu.csv"
    read -r _ _ u s _ < <(awk '{print $1, $2, $14, $15, $22}' "/proc/$SRV/stat" 2>/dev/null)
    io=$(awk '/^read_bytes/ {print $2}' "/proc/$SRV/io" 2>/dev/null)
    printf '%s %s %s %s\n' "$(date +%s%3N)" "$u" "$s" "${io:-NA}" >> "$OUT/cpu-probe.proc.csv"
    sleep 0.2
done ) &
SAMPLER=$!

# per-thread snapshot every second
( while kill -0 "$SRV" 2>/dev/null; do
    total=0
    busy=""
    for t in "/proc/$SRV/task/"*; do
        tid=${t##*/}
        read -r _ _ _ _ _ _ _ _ _ _ _ _ _ u s _ < "$t/stat" 2>/dev/null || continue
        d=$((u + s))
        total=$((total + d))
        if (( d > 40 )); then busy="$busy $tid:$d"; fi
    done
    printf '%s threads_busy_jiffies:%s\n' "$(date +%H:%M:%S)" "$busy" >> "$OUT/cpu-probe.threads.txt"
    sleep 1
done ) &
TSAMPLER=$!

wait $DRIVER
kill $SAMPLER $TSAMPLER 2>/dev/null
wait 2>/dev/null

python3 - "$OUT/cpu-probe.gpu.csv" "$OUT/cpu-probe.proc.csv" >> "$RES" <<'PYEOF'
import sys
g = [l.split() for l in open(sys.argv[1])]
p = [l.split() for l in open(sys.argv[2])]
gb = [int(x[1]) for x in g if x[1] != "NA"]
# find the prefill window: process CPU jumps (the request) = jiffies delta
def col(rows, i):
    return [int(r[i]) if len(r) > i and r[i] != "NA" else 0 for r in rows]
u, s, io = col(p, 1), col(p, 2), col(p, 3)
hz = 100
# the request is the last continuous window where CPU or IO advances;
# approximate: take the window between the first and last index where
# (utime+stime) increases by >2 jiffies per sample or io jumps
act = [i for i in range(1, len(u)) if (u[i]+s[i]) - (u[i-1]+s[i-1]) > 1 or (io[i] and io[i-1] and io[i]-io[i-1] > 2**20)]
if act:
    lo, hi = min(act), max(act)
    ts = [float(r[0]) for r in p]
    # map to gpu samples by timestamp
    gts = [float(r[0]) for r in g]
    w = [b for t, b in zip(gts, gb) if ts[lo] - 500 <= t <= ts[hi] + 500]
    dur = (ts[hi] - ts[lo]) / 1000.0
    cpu_j = (u[hi]+s[hi]) - (u[lo]+s[lo])
    print("prefill window ~%.1f s | cpu %.2f cores-avg | gpu_busy %.0f%% (%d samples)" % (
        dur, cpu_j / hz / dur if dur else 0,
        sum(w) / len(w) if w else -1, len(w)))
    print("(8.0 cores = fully pegged 8-thread server; 2.0 = one hot thread)")
else:
    print("no active window found")
PYEOF
echo "CPU PROBE DONE" | tee -a "$RES"
