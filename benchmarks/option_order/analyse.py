# Analyse run.py output: flip rate vs majority, confidence spread, single-order vs averaged accuracy, letter-position bias.
import json, sys, gzip, collections, statistics as st
f = sys.argv[1] if len(sys.argv) > 1 else 'results_v1.3.jsonl.gz'
R = [json.loads(l) for l in (gzip.open(f, 'rt') if f.endswith('.gz') else open(f))]; R = [r for r in R if "p" in r]
by = collections.defaultdict(list)
for r in R: by[(r["set"], r["id"])].append(r)
rows = collections.defaultdict(lambda: dict(items=0, orders=0, flips=0, items_flip=0, acc1=0, accavg=0, rng=[], rng_hi=[], flips_hi=0, orders_hi=0))
pos = collections.defaultdict(lambda: [0] * 4); posgold = collections.defaultdict(lambda: [[0, 0] for _ in range(4)])
for (s, i), rs in by.items():
    n = rs[0]["n"]; g = rs[0]["gold"]
    arg = [max(range(n), key=lambda k: r["p"][k]) for r in rs]
    maj = collections.Counter(arg).most_common(1)[0][0]
    avg = [sum(r["p"][k] for r in rs) / len(rs) for k in range(n)]; a_avg = max(range(n), key=lambda k: avg[k])
    pm = [r["p"][maj] for r in rs]; hi = st.mean(pm) >= 0.8
    for key in (s, "ALL"):
        d = rows[key]; d["items"] += 1; d["orders"] += len(rs); f = sum(a != maj for a in arg); d["flips"] += f; d["items_flip"] += f > 0
        d["acc1"] += sum(a == g for a in arg) / len(rs); d["accavg"] += a_avg == g; d["rng"].append(max(pm) - min(pm))
        if hi: d["rng_hi"].append(max(pm) - min(pm)); d["flips_hi"] += f; d["orders_hi"] += len(rs)
    if n == 4:
        for r, a in zip(rs, arg):
            L = r["perm"].index(a); pos[s][L] += 1; pos["ALL"][L] += 1
            gl = r["perm"].index(g); posgold["ALL"][gl][0] += a == g; posgold["ALL"][gl][1] += 1
print("set | items | orders | flip rate % | items with any flip % | acc single-order | acc order-avg | median conf range | conf>=0.8: flip % / max range")
for s, d in sorted(rows.items(), key=lambda x: (x[0] == "ALL", x[0])):
    hi = f"{100*d['flips_hi']/max(1,d['orders_hi']):.2f} / {max(d['rng_hi']) if d['rng_hi'] else 0:.2f}"
    print(f"{s} | {d['items']} | {d['orders']} | {100*d['flips']/d['orders']:.2f} | {100*d['items_flip']/d['items']:.1f} | {100*d['acc1']/d['items']:.1f} | {100*d['accavg']/d['items']:.1f} | {st.median(d['rng']):.3f} | {hi}")
t = sum(pos["ALL"]); print("4-option items, share of answers by letter A-D (%):", [round(100 * x / t, 1) for x in pos["ALL"]])
print("4-option items, accuracy when gold is shown at A/B/C/D (%):", [round(100 * a / max(1, b), 1) for a, b in posgold["ALL"]])
