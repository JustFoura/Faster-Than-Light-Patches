#!/usr/bin/env python3
"""Real-session decode bench: multi-turn, growing context, agent-shaped.

Why this exists: the single-shot 256-token bench reads 202 MiB/token at
1.42 GB/s and reports 6.6-7.1 t/s, but the live agent run on 10-01 opened at
11.9 t/s and decayed to 1.9 t/s within the same session. The per-token number
alone cannot see that, because the thing that changes is whether the working set
is still page-cache-resident — which only a session long enough to fill 15 GB of
RAM exposes.

Protocol:
  - launcher-real config: ctx 200000, -b/-ub 16384, --cpu-moe, mmap, q8 KV in
    hostmem, 64 expert-cache slots (88 + 200k ctx aborts on load with a RADV
    OOM; see flashnext-cache88-unbootable-at-200k)
  - a fixed 10-turn code-task conversation, each turn answering into the growing
    context so the routed working set drifts the way a real session's does
  - per turn: decode t/s, window bytes, MiB/token, and cumulative context size
  - phase 1 (--capture) additionally writes the routed-expert log via
    GGML_MOE_LOG so an agent-shaped preload profile can be committed; phase 2
    preloads it via LLAMA_MOE_PROFILE

Usage: scripts/bench-decode-session-1002.py [--capture] [--turns N]
"""
import argparse
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.expanduser("~/build/llama.cpp-unified/build-vulkan/bin/llama-server")
MODEL = os.path.join(
    ROOT,
    "app/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-nocow"
    "/Qwen3.8-Flash-Next-Uncensored.i1-IQ4_XS.gguf",
)
OUT = os.path.join(ROOT, "docs/bench-results-2026-10-02")
PORT = 1244
COMMITTED_PROFILE = os.path.join(
    ROOT, "docs/bench-results-2026-10-02/moe-routes-agent-10turn.log.gz"
)

# One agent-shaped task, then follow-ups that lean on what came before. Written
# out rather than generated so every run routes the same experts and turn N's
# numbers are comparable with turn N's from the previous run.
TURNS = [
    "Write a Python CLI tool todo.py with add/list/done/remove commands that "
    "stores tasks in a local JSON file. Standard library only. Then write tests "
    "for it in test_todo.py using unittest, run them with python3, and fix any "
    "failures until the tests pass. Keep the code clean.",
    "Now add a --priority flag accepting low/medium/high, a `list --sort` option "
    "that accepts priority, due or created, and make sure the tests cover both.",
    "The remove command should confirm before deleting unless --force is passed. "
    "Add that, test it, and show me the final test output.",
    "Add due dates. A task can have an optional --due YYYY-MM-DD. `list` should "
    "mark overdue tasks and support --overdue. Update the tests.",
    "Persist to ~/.config/todo.json by default but honour $TODO_FILE. Print the "
    "resolved path on startup so I can tell where it is writing.",
    "Add an import command that reads a JSON array of tasks and merges them, and "
    "an export that writes the same shape. Round-trip must be lossless.",
    "Rewrite the storage layer so the format version is tracked in the file and "
    "migrations run on load. Add a test that loads a v1 fixture.",
    "Add --json output to every subcommand so I can pipe into jq. Keep the human "
    "output unchanged and update the tests.",
    "Run the whole suite, then add a CI-friendly `make test` target and a short "
    "README section describing the commands.",
    "Finally, review your own code for anything a reviewer would flag — naming, "
    "dead code, error handling — fix what you find, and summarise the changes.",
]


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (PORT, path),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def read_bytes(pid):
    with open("/proc/%d/io" % pid) as f:
        for line in f:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return 0


def meminfo_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true",
                    help="write GGML_MOE_LOG and commit an agent-shaped profile")
    ap.add_argument("--turns", type=int, default=len(TURNS))
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    turns = TURNS[: args.turns]

    os.makedirs(OUT, exist_ok=True)
    label = args.label or ("agentcap" if args.capture else "agent")
    tag = "%s-%s" % (label, time.strftime("%H%M%S"))
    log = os.path.join(OUT, tag + ".server.log")
    results = os.path.join(OUT, "decode-session-results.txt")

    running = subprocess.run(
        ["pgrep", "-x", "llama-server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if running.returncode == 0:
        raise SystemExit(
            "An existing llama-server process is running. Stop it manually before benchmarking; this script will not terminate it."
        )
    time.sleep(3)

    env = dict(os.environ)
    env.update(
        LLAMA_KV_HOSTMEM="1",
        LLAMA_MOE_MADV_RANDOM="decode",
        LLAMA_MOE_PREAD="1",
        LLAMA_MOE_FANOUT="24",
        LD_LIBRARY_PATH=os.path.dirname(BIN),
    )
    # The barrier A/B harness owns LLAMA_MOE_PREAD_WAIT while it runs; do not
    # silently inherit whatever it left behind.
    env.pop("LLAMA_MOE_PREAD_WAIT", None)
    env.pop("LLAMA_MOE_PREAD_OFF", None)

    route_log = "/tmp/moe_routes_agent_%s.log" % tag
    if args.capture:
        env["GGML_MOE_LOG"] = route_log
    elif os.path.exists(COMMITTED_PROFILE):
        with gzip.open(COMMITTED_PROFILE, "rt") as f:
            data = f.read()
        with open("/tmp/moe_routes_agent_preload.log", "w") as f:
            f.write(data)
        env["LLAMA_MOE_PROFILE"] = "/tmp/moe_routes_agent_preload.log"

    cmd = [
        BIN, "-m", MODEL, "-lv", "4",
        # 64, not 88: 88 slots + 200k ctx aborts model load with a RADV OOM.
        "--moe-expert-cache", "64", "--moe-expert-cache-inserts", "0",
        "--gpu-layers", "all", "--cpu-moe", "--load-mode", "mmap", "--fit", "off",
        "--ctx-size", "200000", "--parallel", "1",
        "-b", "16384", "-ub", "16384", "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--threads", "8", "--threads-batch", "8",
        "--host", "127.0.0.1", "--port", str(PORT), "--no-webui",
    ]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    try:
        up = False
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2)
                up = True
                break
            except Exception:
                if proc.poll() is not None:
                    break
                time.sleep(2)
        if not up:
            print("%s: server failed to come up (see %s)" % (tag, log))
            return 1

        for line in open(log):
            if "expert cache enabled" in line or "RAM tier" in line:
                print("  [%s init] %s" % (tag, line.strip()))
                break
        print("%s load: MemAvailable %d MiB" % (tag, meminfo_mb()))

        # conversation history, so each turn really does grow the context
        history = ""
        with open(results, "a") as res:
            for i, turn in enumerate(turns, 1):
                prompt = history + "user\n" + turn + "\n\nassistant\n<think>\n\n</think>\n\n"
                io0 = read_bytes(proc.pid)
                t0 = time.time()
                r = post("/completion", {
                    "prompt": prompt, "n_predict": 256, "temperature": 0.6,
                    "top_p": 0.95, "top_k": 20, "seed": 42,
                    "cache_prompt": True, "ignore_eos": True, "stream": False,
                })
                wall = time.time() - t0
                io1 = read_bytes(proc.pid)
                t = r["timings"]
                gen = t["predicted_n"] * 1000.0 / t["predicted_ms"]
                mib = (io1 - io0) / 2 ** 20
                per_tok = mib / max(t["predicted_n"], 1)
                line = ("%s turn %2d | %6.2f t/s | %6.1f MiB/tok | %5.2f GB/s | "
                        "ctx %5d tok | prompt %6.2f t/s | MemAvail %5d MiB"
                        % (tag, i, gen, per_tok,
                           per_tok * 1.048576 * gen / 1000,
                           t["prompt_n"], t["prompt_n"] * 1000.0 / max(t["prompt_ms"], 1),
                           meminfo_mb()))
                print(line)
                res.write(line + "\n")
                res.flush()
                history += "user\n" + turn + "\n\nassistant\n" + r["content"] + "\n\n"

            hit = ""
            for line in open(log):
                if "hit-rate=" in line:
                    hit = line.strip()
            if hit:
                print("%s: %s" % (tag, hit))
                res.write("%s: %s\n" % (tag, hit))
    finally:
        proc.send_signal(signal.SIGTERM)
        time.sleep(8)
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    if args.capture and os.path.exists(route_log):
        with open(route_log) as f, gzip.open(COMMITTED_PROFILE, "wt") as g:
            shutil.copyfileobj(f, g)
        n = sum(1 for _ in open(route_log))
        print("captured %d route lines -> %s" % (n, COMMITTED_PROFILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
