"""Same-item table: DuplexJev v1.3 / v1.2 / Qwen3-Omni / closed models. Only items every listed model answered.

Headline ("public benchmarks combined"): LISTEN x4 (audio-only), StressTest and MSPB, item-weighted.
By default the 93 LISTEN neutral-text items whose source corpus is in our training data (ESD = "Emotion-Speech",
CREMA-D) are left out -> 1,035 items. --include-train-corpora restores them (1,128 items).

usage: python score.py [--include-train-corpora] [--by-corpus] [model ...]"""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
INCLUDE_TRAIN = "--include-train-corpora" in sys.argv
BY_CORPUS = "--by-corpus" in sys.argv
TRAIN_CORPORA = {"Emotion-Speech", "CREMA-D"}  # ESD and CREMA-D are in DuplexJev's emotion training data
HEADLINE = ["LISTEN-NT", "LISTEN-PARA", "LISTEN-EMM", "LISTEN-EM", "StressTest", "MSPB"]
P = json.load(open(f"{D}/answers/duplexjev_qwen3omni.json"))  # set -> id -> [Qwen3-Omni pred, DuplexJev v1.2 pred]
V13 = json.load(open(f"{D}/answers/duplexjev_v1.3.json"))     # set -> id -> DuplexJev-32B-Para v1.3 pred
SRC = json.load(open(f"{D}/items/listen_sources.json"))       # LISTEN item id -> source corpus
items = {}
for l in open(f"{D}/items/items.jsonl"):
    x = json.loads(l); items[(x["set"], x["id"])] = x
models = args or ["gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.8-live", "gpt-audio-1.5", "gpt-realtime-2.1"]
G = {}
for m in models:
    G[m] = {}
    for l in open(f"{D}/answers/{m}.jsonl"):
        r = json.loads(l)
        if r.get("final"): G[m][(r["set"], r["id"])] = r["pred"]
cols = ["DuplexJev v1.3", "DuplexJev v1.2", "Qwen3-Omni"] + models
preds = [lambda k: V13.get(k[0], {}).get(k[1]), lambda k: P[k[0]][k[1]][1], lambda k: P[k[0]][k[1]][0]] + \
        [lambda k, m=m: G[m][k] for m in models]
def in_train(k): return k[0].startswith("LISTEN") and SRC.get(k[1]) in TRAIN_CORPORA
def keys_of(s):
    ks = [(s, i) for i, (q, d) in P[s].items() if q is not None and d is not None and all((s, i) in G[m] for m in models)]
    return [k for k in ks if INCLUDE_TRAIN or not in_train(k)]
def accs(ks): return [round(100 * sum(f(k) == items[k]["gold"] for k in ks) / len(ks), 1) for f in preds]
def row(name, ks): print(f"{name} | {len(ks)} | " + " | ".join(map(str, accs(ks))))
print(("including" if INCLUDE_TRAIN else "excluding") + " LISTEN items from corpora in DuplexJev's training data (ESD, CREMA-D)\n")
print("set | n | " + " | ".join(cols))
allk = []
for s in P:
    ks = keys_of(s)
    if not ks: continue
    row(s + ("" if s in HEADLINE else " †"), ks)
    if s in HEADLINE: allk += ks
row("**public benchmarks combined** (LISTEN x4, StressTest, MSPB)", allk)
lk = [k for k in allk if k[0].startswith("LISTEN")]
print("\nLISTEN four-condition mean | " + " | ".join(str(round(sum(accs(keys_of(s))[i] for s in HEADLINE[:4]) / 4, 1)) for i in range(len(cols))))
if BY_CORPUS:
    print("\nLISTEN by source corpus\ncorpus | n | " + " | ".join(cols))
    for c in sorted({SRC.get(k[1]) for k in lk}):
        row(c, [k for k in lk if SRC.get(k[1]) == c])
print("\n† held-out speakers / sentences of corpora that are in our training data; not part of the combined score.")
