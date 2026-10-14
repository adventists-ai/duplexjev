"""Closed-model eval (OpenAI chat audio input) on the same items as Qwen3-Omni / DuplexJev: one request per item, options in original order,
answer = first option letter in the reply. Resumable (skips ids already in the output jsonl).
usage: run_openai.py MODEL [WORKERS]"""
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import base64, subprocess, requests
MODEL = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 16
D = os.path.dirname(os.path.abspath(__file__)); AUDIO = os.environ.get("AUDIO_DIR", os.path.join(D, "audio"))
KEY = os.environ["OPENAI_API_KEY"]
L = "ABCDEFGHIJKLMNOP"
TAIL = {"zh": ("\n\n选项：\n", "\n\n请只回答正确选项的字母。"), "en": ("\n\nOptions:\n", "\n\nAnswer with only the letter of the correct option.")}
items = [json.loads(l) for l in open(f"{D}/items/items.jsonl")]
out_p = f"{D}/answers/{MODEL}.jsonl"; 
done = set()
if os.path.exists(out_p):
    for l in open(out_p):
        r = json.loads(l)
        if r.get("pred") is not None or r.get("final"): done.add((r["set"], r["id"]))
lock = threading.Lock(); fo = open(out_p, "a")
def one(it):
    o, t = TAIL[it["lang"]]
    text = it["q"] + o + "\n".join(f"{L[j]}. {x}" for j, x in enumerate(it["options"])) + t
    wav = subprocess.run(["ffmpeg", "-v", "quiet", "-i", os.path.join(AUDIO, it['file']), "-ar", "16000", "-ac", "1", "-f", "wav", "-"], capture_output=True).stdout
    b64 = base64.b64encode(wav).decode()
    ans, err = None, None
    for k in range(5):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": "Bearer " + KEY}, timeout=120,
                json={"model": MODEL, "modalities": ["text"], "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}, {"type": "text", "text": text}]}]})
            j = r.json()
            if "choices" not in j: raise RuntimeError(str(j.get("error"))[:200])
            ans = (j["choices"][0]["message"].get("content") or "").strip(); break
        except Exception as e:
            err = str(e)[:200]; time.sleep(2 * (k + 1) ** 2)
    n = len(it["options"])
    m = re.search(r"\b([A-%s])\b" % L[n - 1], ans or "") or re.search(r"[A-%s]" % L[n - 1], ans or "")
    pred = L.index(m.group(1) if m.groups() else m.group(0)) if m else None
    rec = dict(set=it["set"], id=it["id"], pred=pred, gold=it["gold"], ans=(ans or "")[:200], err=None if ans is not None else err, final=ans is not None)
    with lock: fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
seen = set(); todo = []
for it in items:
    k = (it["set"], it["id"])
    if k in done or k in seen: continue
    seen.add(k); todo.append(it)
if os.environ.get("LIMIT"): todo = todo[:int(os.environ["LIMIT"])]
print(MODEL, "todo", len(todo), flush=True)
with ThreadPoolExecutor(W) as ex: list(ex.map(one, todo))
print("DONE", MODEL, flush=True)
