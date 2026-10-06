# Scope

What this repo is, and what it deliberately leaves out.

**This repo is the research, the patches, and the write-ups.** It is not the
engine and it does not contain llama.cpp — the upstream tree is cloned fresh
at build time.

The measurements were taken through a local GUI backed by llama.cpp. That
application is a separate concern: nothing in this repo requires it, and every
script here runs standalone.

## What is here

- `patches/llama-cpp/` — the llama.cpp patches, the ordered commit series, and
  `build.sh`
- `scripts/` — launchers and bench/probe scripts
- `docs/` — engineering write-ups, plans, and per-session findings
- `experiments/` — two approaches that did not ship, each with a README

## What is not here

**Model weights.** The GGUF files are ~93 GiB and are not in this repo. Point
`FLASH_MODEL` at an absolute path.

**Raw benchmark output.** The per-arm `.server.log`, JSON and CSV files were not
kept; only the distilled findings were. Two files under
`docs/bench-results-2026-09-27/` are load-bearing rather than illustrative:

- `moe-route-analysis.txt` — the only documentation the 2026-09-27 route capture
  produced.
- `moe-routes-decode-256t.log.gz` — looks like a raw capture, but
  `scripts/llama-server-qwen38-flash-next` gunzips it into
  `/tmp/moe_routes_decode.log` on cold start, under `set -euo pipefail`. If it
  goes missing the launcher aborts.

The benchmarks can be **re-run** with `scripts/bench-*.sh`; the specific numbers
in the findings cannot be **recovered** from this repo, because the per-arm
outputs they were computed from were not kept.

## Known limitations of the bench scripts

The benchmark harnesses that manage `llama-server` refuse to start if one is
already running. They only stop the server process they launched.

`scripts/bench-night-sweep-1001.sh` and friends expect
`/tmp/moe_routes_multi.log`, which is produced by
`scripts/capture-routes-multi-1001.sh` from session files that are not in this
repo. Those scripts need that capture regenerated first.
