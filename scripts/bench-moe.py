#!/usr/bin/env python3
"""PP/TG benchmark against a llama-server /completion endpoint.

Measures:
  - prompt processing (PP) at several prompt sizes (cold, cache_prompt=false)
  - decode (TG) at shallow depth and at the depth of a long cached prefix

Usage: bench-moe.py [port] [pp_sizes...] [--label LABEL]
"""
import json, sys, time, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
args = [a for a in sys.argv[2:]]
label = "run"
if "--label" in args:
    i = args.index("--label")
    label = args[i + 1]
    args = args[:i] + args[i + 2:]
pp_sizes = [int(x) for x in args] if args else [512, 4096, 32768]

BASE = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0

# words for a synthetic but varied prompt (~1.35 tokens per word for this tokenizer family)
words = ("the quick brown fox jumps over a lazy dog while printing tokens and streaming "
         "results through the little inference server that could keep up with the demand "
         "of a very long conversation about memory bandwidth graphics cards system ram "
         "and the physics of reading gigabytes over a pci express bus every single pass "
).split()

def make_prompt(n_tokens):
    # ~1.35 tok/word observed for this tokenizer; pad generously, real count from timings
    n_words = int(n_tokens / 1.2)
    out = []
    i = 0
    while len(out) < n_words:
        out.extend(words)
        i += 1
    return " ".join(out)

def wait_health():
    for _ in range(3600):
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(1)
    sys.exit("server never became healthy")

wait_health()

print(f"=== {label} ===")
for size in pp_sizes:
    p = make_prompt(size)
    body, wall = post("/completion", {
        "prompt": p, "n_predict": 4, "cache_prompt": False, "temperature": 0.0,
    })
    t = body.get("timings", {})
    print(f"PP  target={size:>6}  prompt_n={t.get('prompt_n', 0):>6}  "
          f"pp={t.get('prompt_per_second', 0):8.1f} t/s  "
          f"tg_during_pp={t.get('predicted_per_second', 0):6.2f} t/s  wall={wall:.1f}s")

# decode at depth: push a large prompt into the cache, then generate with the prefix cached
depth = pp_sizes[-1] if pp_sizes[-1] >= 4096 else 32768
if max(pp_sizes) < 4096:
    depth = 32768
p = make_prompt(depth)
body, wall = post("/completion", {"prompt": p, "n_predict": 1, "cache_prompt": True,
                                  "temperature": 0.0})
t = body.get("timings", {})
print(f"fill depth ~{depth}  prompt_n={t.get('prompt_n', 0)}  pp={t.get('prompt_per_second', 0):.1f} t/s")

for run in (1, 2):
    body, wall = post("/completion", {"prompt": p, "n_predict": 128, "cache_prompt": True,
                                      "temperature": 0.6, "top_p": 0.95, "top_k": 20})
    t = body.get("timings", {})
    print(f"TG@depth run{run}  prompt_n={t.get('prompt_n', 0)}  "
          f"predicted_n={t.get('predicted_n', 0)}  "
          f"tg={t.get('predicted_per_second', 0):6.2f} t/s  "
          f"(cache hit pp={t.get('prompt_per_second', 0):.0f} t/s)")

# decode at shallow depth (fresh tiny prompt)
body, wall = post("/completion", {"prompt": "Write a story about a fox.", "n_predict": 128,
                                  "cache_prompt": True, "temperature": 0.6, "top_p": 0.95,
                                  "top_k": 20})
t = body.get("timings", {})
print(f"TG@shallow        predicted_n={t.get('predicted_n', 0)}  "
      f"tg={t.get('predicted_per_second', 0):6.2f} t/s")
