"""Gemini Live (bidiGenerateContent) eval on the same items: one session per item; the model only answers in audio, so we read
its output transcription. Question text first (client turn, not complete), then the audio streamed as realtime input inside one manual activity (VAD off), as a Live app would, options in original order. Resumable.
usage: run_glive.py MODEL [CONCURRENCY]"""
import asyncio, json, os, re, subprocess, sys
from google import genai
from google.genai import types
D = os.path.dirname(os.path.abspath(__file__)); AUDIO = os.environ.get("AUDIO_DIR", os.path.join(D, "audio")); MODEL = sys.argv[1]; C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
K = os.environ["GEMINI_API_KEY"]
cli = genai.Client(api_key=K); L = "ABCDEFGHIJKLMNOP"
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
    pcm = subprocess.run(["ffmpeg", "-v", "quiet", "-i", os.path.join(AUDIO, it['file']), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
    cfg = {"response_modalities": ["AUDIO"], "output_audio_transcription": {}, "realtime_input_config": {"automatic_activity_detection": {"disabled": True}}}
    async with cli.aio.live.connect(model="models/" + MODEL, config=cfg) as s:
        head, rest = text.split("\n\n", 1)
        await s.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=head + " (Listen to the audio that follows, then answer.)\n\n" + rest)]), turn_complete=False)
        await s.send_realtime_input(activity_start=types.ActivityStart())
        for i in range(0, len(pcm), 32000): await s.send_realtime_input(audio=types.Blob(data=pcm[i:i + 32000], mime_type="audio/pcm;rate=16000"))
        await s.send_realtime_input(activity_end=types.ActivityEnd())
        out = ""
        async def rx():
            nonlocal out
            async for r in s.receive():
                sc = r.server_content
                if sc and sc.output_transcription and sc.output_transcription.text: out += sc.output_transcription.text
                if sc and sc.turn_complete: return
        await asyncio.wait_for(rx(), 120)
        return out.strip()
sem = asyncio.Semaphore(C)
async def one(it):
    async with sem:
        ans, err = None, None
        for k in range(5):
            try: ans = await ask(it); break
            except Exception as ex: err = str(ex)[:200]; await asyncio.sleep(3 * (k + 1) ** 2)
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
