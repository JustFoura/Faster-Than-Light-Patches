#!/usr/bin/env python3
"""Generate a varied long prompt for prefill amortization tests.

Writes <out> with ~<target_tokens> tokens of varied prose (topics rotate so
the router sees a broad token mix; at C>=470 the expert coverage is complete
regardless, but varied text keeps the PLE table access pattern realistic).
"""
import random
import sys

random.seed(20260928)

TOPICS = [
    ("cartography", "map trade winds coastlines atlas compass meridian surveyor parchment ink expedition latitude longitude terrain elevation legend border province river delta estuary tidal sediment cartouche ornament engraving plate copperpress folio"),
    ("market economics", "merchant caravan silk spice ledger profit bargain warehouse harbor tariff cargo auction bid coinage mint debasement inflation granary surplus famine export import caravan route tollway customs seizure contraband smuggler bribe"),
    ("sea voyage", "brigantine schooner mainsail jib rigging anchor harbormaster pilot reef monsoon doldrums typhoon zenith polestar sextant chronometer bilge keel hull fouling barnacle provisions hardtack rum watered-wine scurvy quarantine"),
    ("mountain travel", "switchback scree moraine glacier crevasse serac avalanche cornice bivouac rope-team crampons ice-axe pass col saddle ridge summit plateau yak tundra lichen altitude thin-air cairn waypoint basecamp stormfront"),
    ("court intrigue", "chancellor regent vizier emissary envoy treaty alliance dowry succession heir regency coup palace guard cipher dispatch sealed letter wax signet forged alias disguise masquerade rumor faction purge banish exile pardon"),
    ("night sky", "constellation eclipse occultation transit perihelion comet meteor nebula observatory telescope astrolabe almanac ephemeris zodiac zenith horizon twilight aurora magnetism pendulum longitude prize navigation tables"),
    ("desert crossing", "dune oasis caravan camel waterskin mirage sirocco sandstorm wadi escarpment saltpan guide toll sultanate cistern ration shade turban dust simoom twilight fire camelthorn acacia"),
    ("scholars", "library scriptorium manuscript codex palimpsest vellum gloss marginalia translation scribe copying error variant recension critical edition archive catalog index citation dispute heresy orthodoxy inquisition ban"),
]

def sent(words, n):
    w = [random.choice(words) for _ in range(n)]
    w[0] = w[0].capitalize()
    return " ".join(w) + random.choice([".", ".", ".", "?", ";"])

def para(topic_words, n_sent):
    return " ".join(sent(topic_words, random.randint(8, 16)) for _ in range(n_sent))

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "prompt-4k.txt"
    target_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 4800
    # ~0.95 tokens per word for this tokenizer family on technical prose.
    target_words = int(target_tokens * 0.95)
    parts = ["<|im_start|>user\n",
             "Below is a travel almanac compiled by the cartographer's guild. "
             "Read it carefully and remember the details.\n\n"]
    words = 0
    i = 0
    while words < target_words:
        name, topic = TOPICS[i % len(TOPICS)]
        wl = topic.split()
        p = para(wl, random.randint(3, 6))
        words += len(p.split())
        parts.append(f"[Section {i+1}: {name}]\n{p}\n\n")
        i += 1
    parts.append("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    text = "".join(parts)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}: {len(text.split())} words (~{int(len(text.split())/0.95)} tokens est.)")

if __name__ == "__main__":
    main()
