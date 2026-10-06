#!/usr/bin/env bash
# Multi-text route capture for the VRAM expert profile, 2026-10-01.
# The single-text (fiction) capture's tail ranks don't generalize; capture
# decode routes across three domains and concatenate — the profile loader
# ranks by raw per-layer frequency, so concatenation is the merge.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/process-safety.sh"
refuse_if_running_llama_server || exit 1
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN=$HOME/build/llama.cpp-unified/build-vulkan/bin/llama-server
MODEL=$ROOT/app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf
OUT=$ROOT/docs/bench-results-2026-10-01
mkdir -p "$OUT"

P_FICTION='<|im_start|>user\nWrite a long, varied fantasy story about a cartographer whose maps change overnight. Include dialogue, travel, markets, and several distinct characters and places.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
P_CODE='<|im_start|>user\nWrite a C program implementing a thread pool with a work queue, graceful shutdown, and error handling. Explain the synchronization primitives you chose.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
P_TECH='<|im_start|>user\nExplain how modern DRAM-less NVMe SSDs use Host Memory Buffer technology, how their SLC cache works, and what sustained random read performance depends on.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

capture() {
  local name=$1 prompt=$2
  sleep 2
  env -u LLAMA_MOE_PROFILE LLAMA_KV_HOSTMEM=1 LLAMA_MOE_MADV_RANDOM=decode \
      LLAMA_MOE_PREAD=1 LLAMA_MOE_FANOUT=24 GGML_MOE_LOG=$OUT/routes-$name.log \
      "$BIN" -m "$MODEL" -lv 4 --moe-expert-cache 88 --moe-expert-cache-inserts 0 \
      --gpu-layers all --cpu-moe --load-mode mmap --fit off --ctx-size 8192 --parallel 1 \
      -b 512 -ub 512 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
      --threads 8 --threads-batch 8 --host 127.0.0.1 --port 1243 --no-webui > "$OUT/capture-$name.server.log" 2>&1 &
  local pid=$! i
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:1243/health >/dev/null 2>&1 && break; sleep 2; done
  curl -s http://127.0.0.1:1243/completion -H 'Content-Type: application/json' -d '{
    "prompt":"'"$prompt"'","n_predict":192,"temperature":0.8,"top_p":0.95,"top_k":20,"seed":7,
    "cache_prompt":false,"ignore_eos":true,"stream":false}' -o /dev/null --max-time 900
  wc -l "$OUT/routes-$name.log" 2>/dev/null
  kill -TERM $pid 2>/dev/null || true
  sleep 8
  kill -9 $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

capture fiction "$P_FICTION"
capture code "$P_CODE"
capture tech "$P_TECH"
cat "$OUT"/routes-fiction.log "$OUT"/routes-code.log "$OUT"/routes-tech.log > /tmp/moe_routes_multi.log 2>/dev/null
echo "multi-text profile: $(wc -l < /tmp/moe_routes_multi.log) route lines"
echo "CAPTURE DONE"
