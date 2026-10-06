#!/usr/bin/env python3
"""Analyze a GGML_MOE_LOG route dump: expert skew and cache-coverage ceilings.

The log has one line per ffn_gate_exps mul_mat_id node:
    <tensor name> <id0> <id1> ... <id{n-1}>
where in decode each line is one token's selected experts for one layer.

Reports, per aggregate and per layer:
- routes seen, distinct experts
- top-K coverage for K in {64, 88, 128, 192, 256, 384} (fraction of routes the
  K most popular experts would have served)
- rolling-window reuse: fraction of (layer, token) routes whose experts all
  appeared in the same layer's previous W tokens, W in {1, 2, 4, 8}
"""

import collections
import re
import sys


def main(path):
    per_layer_routes = collections.defaultdict(list)
    line_re = re.compile(r"^(blk\.\d+)\.ffn_gate_exps.*?((?: \d+)+)\s*$")
    order = []
    with open(path) as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue
            layer = m.group(1)
            ids = [int(x) for x in m.group(2).split()]
            if layer not in per_layer_routes:
                order.append(layer)
            per_layer_routes[layer].append(ids)

    print(f"layers logged: {len(per_layer_routes)}")
    total_routes = sum(len(v) for v in per_layer_routes.values())
    total_tokens = min(len(per_layer_routes[l]) for l in per_layer_routes) if per_layer_routes else 0
    print(f"decode steps captured per layer (min): {total_tokens}")
    print()

    ks = [64, 88, 128, 192, 256, 384]
    ws = [1, 2, 4, 8]

    agg_counts = collections.Counter()
    agg_top = {k: 0 for k in ks}
    agg_routes = 0
    roll_hit = {w: 0 for w in ws}
    roll_tot = {w: 0 for w in ws}

    print(f"{'layer':>8} {'routes':>8} {'distinct':>8} " + " ".join(f">{k:>3}" for k in ks) + "  win1 win4")
    for layer in order:
        seqs = per_layer_routes[layer]
        counts = collections.Counter()
        for ids in seqs:
            counts.update(ids)
        n_routes = sum(counts.values())
        agg_counts.update(counts)
        agg_routes += n_routes

        cov = {}
        top = counts.most_common()
        for k in sorted(ks):
            cov[k] = sum(c for _, c in top[:k]) / n_routes if n_routes else 0.0
        for k in ks:
            agg_top[k] += sum(c for _, c in top[:k])

        # rolling-window reuse: per token, fraction of its experts seen in prev W steps
        win_frac = {}
        recent = collections.deque(maxlen=max(ws))
        for w in ws:
            pass
        hits = {w: 0 for w in ws}
        tot = {w: 0 for w in ws}
        seen_hist = []
        for ids in seqs:
            s = set(ids)
            for w in ws:
                if len(seen_hist) >= w:
                    pool = set()
                    for prev in seen_hist[-w:]:
                        pool |= prev
                    tot[w] += 1
                    hits[w] += len(s & pool) / len(s)
            seen_hist.append(s)
        for w in ws:
            if tot[w]:
                roll_hit[w] += hits[w]
                roll_tot[w] += tot[w]
        win_frac[1] = hits[1] / tot[1] if tot[1] else 0.0
        win_frac[4] = hits[4] / tot[4] if tot[4] else 0.0

        print(f"{layer.replace('blk.',''):>8} {n_routes:>8} {len(counts):>8} "
              + " ".join(f"{cov[k]*100:>4.0f}%" for k in ks)
              + f"  {win_frac[1]*100:>3.0f}% {win_frac[4]*100:>3.0f}%")

    print()
    print("AGGREGATE across layers:")
    print(f"  routes: {agg_routes}")
    top = agg_counts.most_common()
    for k in ks:
        served = agg_top[k]
        # recompute properly per-layer sum of top-k counts already aggregated above
        print(f"  top-{k:>3} experts cover {served/agg_routes*100:>5.1f}% of routes "
              f"(= {k/512*100:.0f}% of expert capacity)")
    print()
    print("rolling-window expert reuse (fraction of a token's experts seen in prev W tokens):")
    for w in ws:
        if roll_tot[w]:
            print(f"  W={w}: {roll_hit[w]/roll_tot[w]*100:>5.1f}%")
    print()
    print("NOTE: coverage numbers are per-layer top-K; a per-layer slot cache of K experts")
    print("serving 'coverage' of routes removes exactly that fraction of routed bytes from disk.")


if __name__ == "__main__":
    main(sys.argv[1])
