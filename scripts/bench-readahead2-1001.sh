#!/usr/bin/env bash
# Order-aware copy read-ahead A/B, 2026-10-01 night (post-restart, fresh boot
# but every arm is measured STEADY-STATE: two full prefill passes in the same
# server, pass 1 learns the copy order, pass 2 is the score per the number
# policy). OFF = LLAMA_VK_COPY_READAHEAD_MB=0, ON = 1536 with the learned-
# sequence advice. Parity: greedy temp-0 outputs must be byte-identical.
# Results: docs/bench-results-2026-10-01/readahead2-results.txt
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
RES="$OUT/readahead2-results.txt"
PORT=1244

prefill_two_pass() {  # prefill_two_pass <label> <ra_mb>
    local label=$1 ra=$2
    sleep 2
    env LLAMA_VK_COPY_READAHEAD_MB=$ra LLAMA_KV_HOSTMEM=1 \
        "$BIN" -m "$MODEL" -lv 4 --gpu-layers all --cpu-moe --load-mode mmap --fit off \
        --ctx-size 16384 --parallel 1 -b 16384 -ub 16384 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
        --host 127.0.0.1 --port $PORT --no-webui > "$OUT/ra2-$label.server.log" 2>&1 &
    local pid=$! i
    for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done
    curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "$label: server failed" | tee -a "$RES"; kill -9 $pid; return 1; }
    sleep 2
    python3 - "$PORT" "$OUT/ra2-$label.pass1.json" "$OUT/ra2-$label.pass2.json" "$label" <<'PYEOF'
import os
import json, sys, urllib.request, hashlib
port, out1, out2, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
prompt = open(os.environ["ROOT"] + "/experiments/qwen38-multi-rhs-prefill/prompt-13k.txt").read()
prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def go(out):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
        data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0.0,
                         "seed": 42, "ignore_eos": True, "cache_prompt": False,
                         "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    json.dump(r, open(out, "w"))
    return r
r1 = go(out1); r2 = go(out2)
def tps(r): t = r["timings"]; return t["prompt_n"]*1000.0/t["prompt_ms"], t["prompt_ms"]/1000.0
p1, s1 = tps(r1); p2, s2 = tps(r2)
same = r1.get("content") == r2.get("content")
print("%s: pass1 %.1f t/s (%.0f s) | pass2 %.1f t/s (%.0f s) | self-parity %s" % (
    label, p1, s1, p2, s2, "same" if same else "DIFF"))
PYEOF
    kill -TERM $pid 2>/dev/null || true
    sleep 8
    kill -9 $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}

: > "$RES"
prefill_two_pass off1 0
prefill_two_pass on1 1536
prefill_two_pass off2 0
prefill_two_pass on2 1536
python3 - <<'PYEOF' | tee -a "$RES"
import os
import json
a = json.load(open(os.environ["ROOT"] + "/docs/bench-results-2026-10-01/ra2-on1.pass2.json"))
b = json.load(open(os.environ["ROOT"] + "/docs/bench-results-2026-10-01/ra2-off1.pass2.json"))
print("CROSS-PARITY on1-vs-off1:", "PASS byte-identical" if a.get("content") == b.get("content") else "FAIL — outputs differ")
PYEOF
echo "READAHEAD2 DONE" | tee -a "$RES"
