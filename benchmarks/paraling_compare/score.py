"""Same-item table: DuplexJev v1.2 / Qwen3-Omni / Gemini models. Only items every listed model answered."""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(f"{D}/answers/duplexjev_qwen3omni.json"))  # set -> id -> [Qwen3-Omni pred, DuplexJev pred]
items = {}
for l in open(f"{D}/items/items.jsonl"):
    x = json.loads(l); items[(x["set"], x["id"])] = x
models = [m for m in sys.argv[1:]] or ["gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.8-live", "gpt-audio-1.5", "gpt-realtime-2.1"]
G = {}
for m in models:
    G[m] = {}
    p = f"{D}/answers/{m}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("final"): G[m][(r["set"], r["id"])] = r["pred"]
rows = []
print("set | n | DuplexJev | Qwen3-Omni | " + " | ".join(models))
for s in P:
    keys = [(s, i) for i, (q, d) in P[s].items() if q is not None and d is not None and all((s, i) in G[m] for m in models)]
    if not keys: continue
    acc = lambda f: round(100 * sum(f(k) == items[k]["gold"] for k in keys) / len(keys), 1)
    vals = [acc(lambda k: P[s][k[1]][1]), acc(lambda k: P[s][k[1]][0])] + [acc(lambda k, m=m: G[m][k]) for m in models]
    print(f"{s} | {len(keys)} | " + " | ".join(map(str, vals)))
    rows.append(dict(set=s, n=len(keys), dj=vals[0], qwen=vals[1], **{m: v for m, v in zip(models, vals[2:])}))

