#!/usr/bin/env python3
"""Repeatable, single-request llama-server probe for the Flash-Next experiments."""

import argparse
import json
import pathlib
import time
import urllib.request


DEFAULT_PROMPT = (
    "Write the opening of a fantasy story. A cartographer discovers that the "
    "coastline on her map has moved overnight. Use concrete sensory details."
)


def process_counters(pid):
    if not pid:
        return None
    base = pathlib.Path(f"/proc/{pid}")
    try:
        io = dict(line.split(": ", 1) for line in (base / "io").read_text().splitlines())
        stat = (base / "stat").read_text().rsplit(") ", 1)[1].split()
        return {
            "read_bytes": int(io["read_bytes"]),
            "write_bytes": int(io["write_bytes"]),
            "minor_faults": int(stat[7]),
            "major_faults": int(stat[9]),
        }
    except (FileNotFoundError, PermissionError, KeyError, ValueError):
        return None


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1235")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-file", type=pathlib.Path)
    parser.add_argument("--cache-prompt", action="store_true")
    parser.add_argument("--no-thinking", action="store_true", help="Use the model's chat wrapper with an empty think block")
    args = parser.parse_args()

    prompt = args.prompt_file.read_text() if args.prompt_file else DEFAULT_PROMPT
    if args.no_thinking:
        prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    props = request_json(args.url.rstrip("/") + "/props")
    before = process_counters(args.pid)
    t0 = time.monotonic()
    response = request_json(
        args.url.rstrip("/") + "/completion",
        {
            "prompt": prompt,
            "n_predict": args.n_predict,
            "temperature": args.temperature,
            "seed": args.seed,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "ignore_eos": True,
            "cache_prompt": args.cache_prompt,
            "stream": False,
        },
    )
    wall_seconds = time.monotonic() - t0
    after = process_counters(args.pid)
    delta = (
        {key: after[key] - before[key] for key in before}
        if before is not None and after is not None
        else None
    )
    result = {
        "label": args.label,
        "server": args.url,
        "model_path": props.get("model_path"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "prompt": prompt,
        "requested_tokens": args.n_predict,
        "temperature": args.temperature,
        "seed": args.seed,
        "cache_prompt": args.cache_prompt,
        "no_thinking": args.no_thinking,
        "generated_text": response.get("content"),
        "generated_tokens": response.get("tokens_predicted"),
        "prompt_tokens": response.get("tokens_evaluated"),
        "timings": response.get("timings"),
        "wall_seconds": wall_seconds,
        "process_delta": delta,
        "response_error": response.get("error"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("label", "generated_tokens", "prompt_tokens", "timings", "wall_seconds", "process_delta", "response_error")}, indent=2))


if __name__ == "__main__":
    main()
