# Option-order stability check. DuplexJev-32B-Para v1.3 on vLLM, gateway prompt format,
# single-order readout per request. n<=4: all n! orders; n=5..10: 12 random orders (seeded, incl. identity). n>10 skipped.
import json, sys, os, base64, itertools, random, time
from concurrent.futures import ThreadPoolExecutor
from duplexjev.remote import VLLMClient, _post, _softmax, said_prefix, HEAD
from duplexjev.question import Question, LETTERS
ITEMS, AUD, OUT = sys.argv[1:4]; URL = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8000/v1"
items = [json.loads(l) for l in open(ITEMS)]
cl = VLLMClient(URL)
done = set()
if os.path.exists(OUT):
    for l in open(OUT): r = json.loads(l); done.add((r["set"], r["id"], tuple(r["perm"])))
def orders(n, key):
    if n <= 4: return [list(p) for p in itertools.permutations(range(n))]
    rng = random.Random(key); s = {tuple(range(n))}
    while len(s) < 12: p = list(range(n)); rng.shuffle(p); s.add(tuple(p))
    return [list(p) for p in sorted(s)]
B64 = {}
def job(it, perm):
    if it["file"] not in B64: B64[it["file"]] = base64.b64encode(open(f"{AUD}/{it['file']}", "rb").read()).decode()
    q = Question(id=it["id"], text=it["q"], options=it["options"], lang=it["lang"] if it["lang"] in ("zh", "en") else "en")
    n = len(it["options"]); text = HEAD[q.lang] + "\n\n" + q.render(perm)
    for a in range(5):
        try:
            r = _post(cl.base_url + "/chat/completions", {"model": cl.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "allowed_token_ids": cl.letter_ids[:n], "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                    {"type": "input_audio", "input_audio": {"data": B64[it["file"]], "format": it["file"].rsplit(".", 1)[1]}}]}]}, cl.headers, 60)
            break
        except Exception as e:
            time.sleep(2 + 3 * a); err = e
    else: return dict(set=it["set"], id=it["id"], perm=perm, err=str(err))
    top = {t["token"]: t["logprob"] for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    pl = _softmax([top.get(LETTERS[i], -1e9) for i in range(n)])
    p = [0.0] * n
    for i in range(n): p[perm[i]] = pl[i]
    return dict(set=it["set"], id=it["id"], perm=perm, p=p, gold=it["gold"], n=n)
jobs = [(it, p) for it in items if 1 < len(it["options"]) <= 10 for p in orders(len(it["options"]), it["id"]) if (it["set"], it["id"], tuple(p)) not in done]
print("jobs", len(jobs), "already", len(done), flush=True)
t0 = time.time()
with open(OUT, "a") as f, ThreadPoolExecutor(int(os.environ.get("CONC", "8"))) as ex:
    for k, r in enumerate(ex.map(lambda a: job(*a), jobs)):
        f.write(json.dumps(r) + "\n")
        if k % 2000 == 0: print(k, round(time.time() - t0), "s", flush=True); f.flush()
print("DONE", len(jobs), round(time.time() - t0), "s", flush=True)
