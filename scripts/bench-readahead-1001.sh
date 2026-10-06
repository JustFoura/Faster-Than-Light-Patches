#!/usr/bin/env bash
# Copy-readahead A/B, 2026-10-01 night.
#
# Root cause established this evening: each prompt pass, the graph scheduler
# copies every expert tensor from its CPU_Mapped home (mmap) into a device
# buffer BEFORE the mul_mat_id runs — ~77 GB per one-pass prefill through a
# fully synchronous per-tensor write (read at queue-depth-1 ~0.9 GB/s →
# memcpy → PCIe → fence). The ubatch sweep's read volumes (77/138/259 GB at
# ub 16384/8192/4096) match copies-per-pass exactly, and the instrumented
# gate shows the op's src0 is the sched's Vulkan0#...#0 device copy.
# Fix in ggml_backend_vk_buffer_set_tensor: rolling POSIX_FADV_WILLNEED
# window ahead of the copy cursor (LLAMA_VK_COPY_READAHEAD_MB, default 1536,
# 0 = off). Byte-preserving; parity checked anyway.
# Results: docs/bench-results-2026-10-01/readahead-results.txt
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
RES="$OUT/readahead-results.txt"
PORT=1244

parity_arm() {  # parity_arm <label> <ra_mb>
    local label=$1 ra=$2
    sleep 2
    env LLAMA_VK_COPY_READAHEAD_MB=$ra LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/ra-$label.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done
    curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "$label: server failed"; kill -9 $pid; return 1; }
    python3 - "$PORT" "$OUT/ra-$label.parity.json" "$label" <<'PYEOF'
import os
import json, sys, urllib.request, hashlib
port, out, label = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                     "seed": 42, "ignore_eos": True, "cache_prompt": False,
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
json.dump(r, open(out, "w"))
print("%s | sha256=%s | %.1f s prompt" % (label,
    hashlib.sha256((r.get("content") or "").encode()).hexdigest()[:16],
    r["timings"]["prompt_ms"] / 1000))
PYEOF
    kill -TERM $pid 2>/dev/null || true
    sleep 8
    kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

prefill_arm() {  # prefill_arm <label> <ra_mb>
    local label=$1 ra=$2
    LABEL="ra-$label" PROMPT_FILE="$PROMPT_FILE" OUT_DIR="$OUT" \
        CTX=16384 UBATCH=16384 BATCH=16384 MEM_FLOOR_GB=2 PORT=1239 \
        EXTRA_ENV="LLAMA_VK_COPY_READAHEAD_MB=$ra" \
        bash "$ROOT/scripts/bench-prefill-safe.sh"
    python3 - "$OUT/ra-$label.json" <<'PYEOF' | tee -a "$RES"
import json, sys
r = json.load(open(sys.argv[1]))
t = r.get("timings") or {}
d = r.get("process_delta") or {}
print("%s: prompt_tps=%.2f (%d tok, %.1f s) read_GB=%.1f major_faults=%s" % (
    r["label"], t["prompt_n"]*1000.0/t["prompt_ms"], t.get("prompt_n"),
    t["prompt_ms"]/1000.0, d.get("read_bytes",0)/2**30, d.get("major_faults")))
PYEOF
}

for _ in $(seq 1 120); do
    grep -q "MTP NIGHT DONE" "$OUT/mtp-night-run.txt" 2>/dev/null && break
    sleep 20
done

: > "$RES"
echo "=== parity (greedy, temp 0, 12 tok; window 0 vs 1536) ===" | tee -a "$RES"
parity_arm off 0
parity_arm on 1536
python3 - <<'PYEOF' | tee -a "$RES"
import os
import json
a = json.load(open(os.environ["ROOT"] + "/docs/bench-results-2026-10-01/ra-off.parity.json"))
b = json.load(open(os.environ["ROOT"] + "/docs/bench-results-2026-10-01/ra-on.parity.json"))
same = a.get("content") == b.get("content")
print("PARITY:", "PASS byte-identical" if same else "FAIL — outputs differ")
if not same:
    print("  off:", repr((a.get('content') or '')[:200]))
    print("  on :", repr((b.get('content') or '')[:200]))
PYEOF
echo "=== A/B (one-pass 14,963 tok) ===" | tee -a "$RES"
prefill_arm off1 0
prefill_arm on1 1536
prefill_arm off2 0
prefill_arm on2 1536
echo "READAHEAD DONE" | tee -a "$RES"
