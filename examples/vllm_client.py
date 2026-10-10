"""Ask several closed-set questions about one clip through a vLLM server running a DuplexJev model.

Server (one GPU):
    pip install duplexjev-vllm
    vllm serve adventists-ai/DuplexJev-4B

Client (this file, needs only `openai` and `numpy`):
    python vllm_client.py clip.wav

Every question is one request with max_tokens=1: the answer is the next-token distribution over the option letters,
so nothing is decoded. Requests about the same clip share their prefix (chat header + audio), which vLLM's prefix
cache computes once.
"""
import base64
import math
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

LETTERS = "ABCDEFGHIJ"
HEAD = {"en": "The user said: <|audio|>", "zh": "用户说：<|audio|>"}
TPL = {"en": ("Question: {q}", "Options:", "Answer with only the letter of the correct option."),
       "zh": ("问题：{q}", "选项：", "请只回答正确选项的字母。")}


def prompt(question: str, options: list, lang: str = "en") -> str:
    q, o, i = TPL[lang]
    lines = [q.format(q=question), "", o] + [f"{LETTERS[k]}. {opt}" for k, opt in enumerate(options)] + ["", i]
    return HEAD[lang] + "\n\n" + "\n".join(lines)


class DuplexJevClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str | None = None):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model or self.client.models.list().data[0].id
        # token ids of the option letters, from the server's own tokenizer
        root = base_url.rsplit("/v1", 1)[0]
        import json
        import urllib.request
        self.letter_ids = []
        for c in LETTERS:
            req = urllib.request.Request(f"{root}/tokenize", data=json.dumps(
                {"model": self.model, "prompt": c, "add_special_tokens": False}).encode(), headers={"Content-Type": "application/json"})
            ids = json.load(urllib.request.urlopen(req))["tokens"]
            assert len(ids) == 1, (c, ids)
            self.letter_ids.append(ids[0])

    def ask(self, wav_bytes: bytes, question: str, options: list, lang: str = "en") -> dict:
        audio = base64.b64encode(wav_bytes).decode()
        ids = self.letter_ids[:len(options)]
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt(question, options, lang)},
                {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}]}],
            extra_body={"allowed_token_ids": ids, "chat_template_kwargs": {"enable_thinking": False}})
        lp = {t.token: t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}
        z = [lp.get(LETTERS[k], -1e9) for k in range(len(options))]
        m = max(z)
        p = [math.exp(v - m) for v in z]
        s = sum(p)
        probs = {opt: p[k] / s for k, opt in enumerate(options)}
        best = max(probs, key=probs.get)
        return {"answer": best, "confidence": probs[best], "probs": probs}

    def decide(self, wav_bytes: bytes, questions: dict, lang: str = "en") -> dict:
        """questions: {name: (question, [options])} -> {name: {answer, confidence, probs}}, asked concurrently."""
        with ThreadPoolExecutor(len(questions)) as ex:
            futs = {k: ex.submit(self.ask, wav_bytes, q, o, lang) for k, (q, o) in questions.items()}
            return {k: f.result() for k, f in futs.items()}


if __name__ == "__main__":
    wav = open(sys.argv[1], "rb").read()
    dj = DuplexJevClient(sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1")
    out = dj.decide(wav, {
        "turn": ("Has the user finished speaking?", ["finished", "not finished"]),
        "gender": ("What is the perceived gender of the speaker?", ["female", "male"]),
        "emotion": ("What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"]),
        "intent": ("What does the user want?", ["climate", "media", "navigation", "phone", "chit-chat"]),
    })
    for k, v in out.items():
        print(f"{k:8s} {v['answer']:14s} {v['confidence']:.2f}")
