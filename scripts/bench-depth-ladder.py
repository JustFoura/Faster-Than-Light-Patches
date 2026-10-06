#!/usr/bin/env python3
"""Depth ladder: at each depth D, prefill D tokens (PP at depth) and decode 128
tokens with the prefix cached (TG at depth). Prompts are built so each depth's
prompt strictly extends the previous one (prefix cache reuse)."""
import json, sys, time, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 1239
depths = [int(x) for x in sys.argv[2:]] or [65536, 131072, 196608]
BASE = f"http://127.0.0.1:{port}"

words = ("the quick brown fox jumps over a lazy dog while printing tokens and streaming "
         "results through the little inference server that could keep up with the demand "
         "of a very long conversation about memory bandwidth graphics cards system ram "
         "and the physics of reading gigabytes over a pci express bus every single pass "
).split()

def make_prompt(n_tokens):
    n_words = int(n_tokens / 1.2)
    out = []
    while len(out) < n_words:
        out.extend(words)
    return " ".join(out)

def post(payload, timeout=3000):
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0

def health():
    for _ in range(3600):
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(1)
    sys.exit("server not healthy")

health()
print(f"=== depth ladder on :{port} ===")
for d in depths:
    p = make_prompt(d)
    body, wall = post({"prompt": p, "n_predict": 1, "cache_prompt": True, "temperature": 0.0})
    t = body.get("timings", {})
    pp = t.get("prompt_per_second", 0)
    print(f"depth {d:>7}: prefill prompt_n={t.get('prompt_n',0):>6} pp={pp:7.1f} t/s wall={wall:6.1f}s", flush=True)
    body, wall = post({"prompt": p, "n_predict": 128, "cache_prompt": True,
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20})
    t = body.get("timings", {})
    print(f"           decode: tg={t.get('predicted_per_second',0):6.2f} t/s "
          f"(n={t.get('predicted_n',0)}, wall={wall:.1f}s)", flush=True)
