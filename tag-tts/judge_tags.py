"""Use DuplexJev as a judge for expressive TTS: did the tagged sound / stress actually happen?

This is the reward model behind tag-tts. Serve a DuplexJev model with vLLM
(`pip install duplexjev-vllm && vllm serve adventists-ai/DuplexJev-32B-Para --served-model-name dj`), then:

    python judge_tags.py items.jsonl scores.jsonl --base-url http://localhost:8000

items.jsonl, one per line:
    {"id": "a1", "audio": "a1.wav", "lang": "en", "kind": "stress", "options": ["lives", "hiding", "fear"], "target": 0}
    {"id": "a2", "audio": "a2.wav", "lang": "zh", "kind": "nonverbal", "target": "laughter"}

kind=stress     – "Which word does the speaker emphasise?"; "no word is emphasised" is appended as the last option.
kind=nonverbal  – "Besides speaking, which sound does the speaker make?" over sigh / laughter / cough / breath / none.

Each question is asked once per rotation of the options and the letter probabilities are averaged (removes
position bias). The prompt format must match DuplexJev's training format exactly: with a slightly different
template, stress accuracy drops from ~90% to near zero.
Output: {"id", "probs": [...], "p_target"} per line.
"""
import argparse, base64, io, json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf

LETTERS = "ABCDEFGH"
HEAD = {"en": "The user said: <|audio|>", "zh": "用户说：<|audio|>"}
TPL = {"en": ("Question: {q}", "Options:", "Answer with only the letter of the correct option."),
       "zh": ("问题：{q}", "选项：", "请只回答正确选项的字母。")}
Q_STRESS = {"en": "Does the speaker put special emphasis on any word in this sentence? If so, which one?",
            "zh": "说话人在这句话里有没有特别重读（强调）某个词？如果有，是哪一个？"}
NONE_STRESS = {"en": "No word is particularly emphasized", "zh": "没有特别强调哪个词"}
Q_NV = {"en": "Besides speaking, which of the following sounds does the speaker make in this recording?",
        "zh": "这段录音里，说话人除了说话，还发出了下面哪种声音？"}
NV = {"en": ["a sigh", "laughter", "a cough", "an audible breath", "none of these"],
      "zh": ["叹气", "笑声", "咳嗽", "明显的吸气声", "没有这些声音"]}
NV_KEYS = ["sigh", "laughter", "cough", "breath", "none"]


def wav16k_b64(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    b = io.BytesIO(); sf.write(b, a, 16000, format="WAV")
    return base64.b64encode(b.getvalue()).decode()


class Judge:
    def __init__(self, base_url, model="dj"):
        self.url, self.model = base_url.rstrip("/"), model
        self.ids = [self._tok(c) for c in LETTERS]

    def _post(self, path, body):
        r = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=120))

    def _tok(self, c):
        return self._post("/tokenize", {"model": self.model, "prompt": c, "add_special_tokens": False})["tokens"][0]

    def mcq(self, audio_b64, lang, question, options):
        n = len(options); acc = np.zeros(n)
        for s in range(n):                                   # rotate options, average letter probabilities
            order = [(j + s) % n for j in range(n)]
            q, o, t = TPL[lang]
            text = HEAD[lang] + "\n\n" + "\n".join([q.format(q=question), "", o]
                                                   + [f"{LETTERS[i]}. {options[k]}" for i, k in enumerate(order)] + ["", t])
            j = self._post("/v1/chat/completions", {
                "model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": n,
                "allowed_token_ids": self.ids[:n],
                "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                                                          {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}]}]})
            top = {x["token"].strip(): x["logprob"] for x in j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
            p = np.array([np.exp(top.get(LETTERS[i], -30)) for i in range(n)]); p /= max(p.sum(), 1e-9)
            for i, k in enumerate(order): acc[k] += p[i] / n
        return acc

    def score(self, item):
        lang, a = item.get("lang", "en"), wav16k_b64(item["audio"])
        if item["kind"] == "stress":
            opts = list(item["options"]) + [NONE_STRESS[lang]]
            p = self.mcq(a, lang, Q_STRESS[lang], opts); tgt = item["target"] if item.get("target") is not None else len(opts) - 1
        else:
            p = self.mcq(a, lang, Q_NV[lang], NV[lang]); tgt = NV_KEYS.index(item.get("target", "none"))
        return {"id": item["id"], "probs": [round(float(x), 4) for x in p], "p_target": round(float(p[tgt]), 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--base-url", default="http://localhost:8000"); ap.add_argument("--model", default="dj")
    ap.add_argument("--threads", type=int, default=8)
    A = ap.parse_args(); J = Judge(A.base_url, A.model)
    items = [json.loads(l) for l in open(A.items) if l.strip()]
    with open(A.out, "w") as f, ThreadPoolExecutor(A.threads) as ex:
        for r in ex.map(J.score, items): f.write(json.dumps(r, ensure_ascii=False) + "\n")
