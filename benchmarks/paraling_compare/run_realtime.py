"""OpenAI Realtime (websocket) eval on the same items: one session per item, text output, no VAD, options in original order,
answer = first option letter. Resumable. usage: run_realtime.py MODEL [CONCURRENCY]"""
import asyncio, json, base64, subprocess, os, re, sys, websockets
D = os.path.dirname(os.path.abspath(__file__)); AUDIO = os.environ.get("AUDIO_DIR", os.path.join(D, "audio")); MODEL = sys.argv[1]; C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
K = os.environ["OPENAI_API_KEY"]
L = "ABCDEFGHIJKLMNOP"
TAIL = {"zh": ("\n\n选项：\n", "\n\n请只回答正确选项的字母。"), "en": ("\n\nOptions:\n", "\n\nAnswer with only the letter of the correct option.")}
items = [json.loads(l) for l in open(f"{D}/items/items.jsonl")]
out_p = f"{D}/answers/{MODEL}.jsonl"; done = set()
if os.path.exists(out_p):
    for l in open(out_p):
        r = json.loads(l)
        if r.get("final"): done.add((r["set"], r["id"]))
fo = open(out_p, "a")
async def ask(it):
    o, t = TAIL[it["lang"]]
    text = it["q"] + o + "\n".join(f"{L[j]}. {x}" for j, x in enumerate(it["options"])) + t
    pcm = subprocess.run(["ffmpeg", "-v", "quiet", "-i", os.path.join(AUDIO, it['file']), "-ar", "24000", "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
    async with websockets.connect(f"wss://api.openai.com/v1/realtime?model={MODEL}", additional_headers={"Authorization": "Bearer " + K}, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "output_modalities": ["text"], "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None}}}}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": base64.b64encode(pcm).decode()}, {"type": "input_text", "text": text}]}}))
        await ws.send(json.dumps({"type": "response.create"}))
        out = ""
        while True:
            e = json.loads(await asyncio.wait_for(ws.recv(), 90)); ty = e["type"]
            if ty == "error": raise RuntimeError(str(e["error"])[:200])
            if ty.endswith("text.delta"): out += e["delta"]
            if ty == "response.done":
                if e["response"].get("status") != "completed": raise RuntimeError(str(e["response"].get("status_details"))[:200])
                return out.strip()
sem = asyncio.Semaphore(C)
async def one(it):
    async with sem:
        ans, err = None, None
        for k in range(5):
            try: ans = await ask(it); break
            except Exception as ex: err = str(ex)[:200]; await asyncio.sleep(2 * (k + 1) ** 2)
        n = len(it["options"])
        m = re.search(r"\b([A-%s])\b" % L[n - 1], ans or "") or re.search(r"[A-%s]" % L[n - 1], ans or "")
        pred = L.index(m.group(1) if m.groups() else m.group(0)) if m else None
        fo.write(json.dumps(dict(set=it["set"], id=it["id"], pred=pred, gold=it["gold"], ans=(ans or "")[:200], err=None if ans is not None else err, final=ans is not None), ensure_ascii=False) + "\n"); fo.flush()
async def main():
    seen = set(); todo = []
    for it in items:
        k = (it["set"], it["id"])
        if k in done or k in seen: continue
        seen.add(k); todo.append(it)
    if os.environ.get("LIMIT"): todo = todo[:int(os.environ["LIMIT"])]
    print(MODEL, "todo", len(todo), flush=True)
    await asyncio.gather(*(one(it) for it in todo))
    print("DONE", MODEL, flush=True)
asyncio.run(main())
