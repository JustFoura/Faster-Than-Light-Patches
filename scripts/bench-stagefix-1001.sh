#!/usr/bin/env bash
# Device-staging fix A/B + parity, 2026-10-01 late evening.
#
# The 10-01 fix: staging qualified src0 via the pinned_memory lookup, but
# --cpu-moe + mmap weights live in CPU_Mapped buffers (plain file mmap), so
# host_get returned null and staging silently skipped every op — today's
# stage-on run kept 143,674 major faults, the pass never left the fault path,
# and the "compute-bound" verdict measured a no-op. The fix qualifies on
# host-backed src0 and stages from the raw pointer.
#
#   1. parity: greedy (temp 0) completion, stock vs stage-on — outputs must
#      be byte-identical (same weight bytes, same shader, device-bound copy).
#   2. A/B at gate-C condition (one-pass 14,963 tok, ubatch 16384):
#      off / on / off. Engagement check: stage-on major faults must COLLAPSE
#      (weights served from VRAM copies, not the disk fault stream).
# Results: docs/bench-results-2026-10-01/stagefix-results.txt
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export ROOT
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
PROMPT_FILE=$ROOT/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt
OUT=$ROOT/docs/bench-results-2026-10-01
RES="$OUT/stagefix-results.txt"
PORT=1244

parity_arm() {  # parity_arm <label> <stage:0|1>
    local label=$1 stage=$2
    sleep 2
    env -u LLAMA_VK_MOE_DEVICE_STAGE LLAMA_KV_HOSTMEM=1 \
        LLAMA_VK_MOE_DEVICE_STAGE=$stage \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/stagefix-$label.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done
    curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "$label: server failed"; kill -9 $pid; return 1; }
    python3 - "$PORT" "$OUT/stagefix-$label.parity.json" <<'PYEOF'
import os
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                     "seed": 42, "ignore_eos": True, "cache_prompt": False,
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
json.dump(r, open(out, "w"))
import hashlib
print("%s | content sha256=%s | prompt %d tok, %.2f ms | faults见log" % (
    sys.argv[2].rsplit('/',1)[-1], hashlib.sha256((r.get("content") or "").encode()).hexdigest()[:16],
    r["timings"]["prompt_n"], r["timings"]["prompt_ms"]))
PYEOF
    kill -TERM $pid 2>/dev/null || true
    sleep 8
    kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

prefill_arm() {  # prefill_arm <label> <stage:0|1>
    local label=$1 stage=$2
    local envv="LLAMA_VK_MOE_DEVICE_STAGE=$stage"
    if [ "$stage" = 0 ]; then envv="LLAMA_VK_MOE_DEVICE_STAGE=0"; fi
    LABEL="stagefix-$label" PROMPT_FILE="$PROMPT_FILE" OUT_DIR="$OUT" \
        CTX=16384 UBATCH=16384 BATCH=16384 MEM_FLOOR_GB=2 PORT=1239 EXTRA_ENV="$envv" \
        bash "$ROOT/scripts/bench-prefill-safe.sh"
    python3 - "$OUT/stagefix-$label.json" <<'PYEOF' | tee -a "$RES"
import json, sys
r = json.load(open(sys.argv[1]))
t = r.get("timings") or {}
d = r.get("process_delta") or {}
print("%s: prompt_tps=%.2f (%d tok, %.1f s) read_GB=%.1f major_faults=%s" % (
    r["label"], t["prompt_n"]*1000.0/t["prompt_ms"], t.get("prompt_n"),
    t["prompt_ms"]/1000.0, d.get("read_bytes",0)/2**30, d.get("major_faults")))
PYEOF
}

: > "$RES"
echo "=== parity (greedy, temp 0, 12 tok) ===" | tee -a "$RES"
parity_arm off 0
parity_arm on 1
python3 - <<'PYEOF' | tee -a "$RES"
import os
import json, hashlib
a = json.load(open(os.environ["ROOT"] + "/docs/bench-results-2026-10-01/stagefix-off.parity.json"))
b = json.load(open(os.environ["ROOT"] + "/docs/bench-results-2026-10-01/stagefix-on.parity.json"))
same = a.get("content") == b.get("content")
print("PARITY:", "PASS byte-identical" if same else "FAIL — outputs differ")
if not same:
    print("  off:", repr((a.get('content') or '')[:200]))
    print("  on :", repr((b.get('content') or '')[:200]))
PYEOF
echo "=== A/B (one-pass 14,963 tok) ===" | tee -a "$RES"
prefill_arm off1 0
prefill_arm on1 1
prefill_arm off2 0
echo "STAGEFIX DONE" | tee -a "$RES"
