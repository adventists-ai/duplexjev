"""Clients that need no GPU and no torch: a vLLM server running a DuplexJev model, or the DuplexJev HTTP API.

    from duplexjev.remote import VLLMClient, ApiClient
    VLLMClient("http://localhost:8000/v1").decide("call.wav", questions)     # your own `vllm serve adventists-ai/DuplexJev-4B`
    ApiClient("https://<host>/duplexjev").decide("call.wav", questions)      # a `duplexjev gateway` in front of it

Both return ``{question_id: {"answer": option, "confidence": p, "probs": {option: p}}}``, like ``Decider.decide``.
Only the standard library is used (``soundfile`` too when the audio is given as an array).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .question import LETTERS, Question, as_questions

HEAD = {"en": "The user said: <|audio|>", "zh": "用户说：<|audio|>"}


def audio_bytes(audio: Any) -> tuple[bytes, str]:
    """(bytes, format) for a file path, raw file bytes, a 16 kHz float array or an (array, sample_rate) tuple."""
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio), _sniff(bytes(audio))
    if isinstance(audio, (str, os.PathLike)):
        with open(audio, "rb") as f:
            data = f.read()
        ext = os.path.splitext(str(audio))[1].lower().lstrip(".")
        return data, ext if ext in ("wav", "mp3", "flac", "ogg", "m4a", "webm") else _sniff(data)
    import soundfile as sf

    arr, sr = audio if isinstance(audio, tuple) else (audio, 16000)
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "wav"


def _sniff(b: bytes) -> str:
    if b[:4] == b"RIFF":
        return "wav"
    if b[:4] == b"fLaC":
        return "flac"
    if b[:4] == b"OggS":
        return "ogg"
    if b[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    return "mp3"


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _softmax(z: Sequence[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class VLLMClient:
    """Ask questions through an OpenAI-compatible vLLM server running a DuplexJev complete model.

    Each question is one chat request with ``max_tokens=1``; the answer is the next-token distribution over the
    option letters (``allowed_token_ids``), so nothing is decoded. Questions about one clip are sent concurrently and
    share the audio prefix in vLLM's prefix cache.
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str | None = None, api_key: str = "EMPTY",
                 timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        if model is None:
            req = urllib.request.Request(self.base_url + "/models", headers=self.headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                model = json.load(r)["data"][0]["id"]
        self.model = model
        root = self.base_url.rsplit("/v1", 1)[0]
        self.letter_ids = []
        for c in LETTERS[:10]:
            ids = _post(root + "/tokenize", {"model": model, "prompt": c, "add_special_tokens": False},
                        self.headers, timeout)["tokens"]
            if len(ids) != 1:
                raise RuntimeError(f"letter {c!r} is not a single token for {model}: {ids}")
            self.letter_ids.append(ids[0])

    def ask(self, audio_b64: str, fmt: str, q: Question, seed: int = 0) -> dict:
        perm = q.permutation(seed)
        n = len(q.options)
        if n > len(self.letter_ids):
            raise ValueError(f"{q.id}: at most {len(self.letter_ids)} options over the vLLM client")
        text = HEAD[q.lang] + "\n\n" + q.render(perm)
        r = _post(self.base_url + "/chat/completions", {
            "model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "allowed_token_ids": self.letter_ids[:n],
            # Training rendered Qwen3's empty <think></think> block; without it the next token is "<think>".
            # vLLM >= 0.2x reports top_logprobs before allowed_token_ids is applied, so ask for 20 and keep the letters.
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}}]}],
        }, self.headers, self.timeout)
        top = {t["token"]: t["logprob"] for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        p_letter = _softmax([top.get(LETTERS[i], -1e9) for i in range(n)])
        probs = {q.options[perm[i]]: p_letter[i] for i in range(n)}
        best = max(probs, key=probs.get)
        return {"answer": best, "confidence": probs[best], "probs": probs}

    def decide(self, audio: Any, questions, *, seed: int = 0) -> dict:
        qs = as_questions(questions if isinstance(questions, (list, tuple)) else [questions])
        data, fmt = audio_bytes(audio)
        b64 = base64.b64encode(data).decode()
        with ThreadPoolExecutor(min(16, len(qs))) as ex:
            futs = {q.id: ex.submit(self.ask, b64, fmt, q, seed) for q in qs}
            return {k: f.result() for k, f in futs.items()}


class ApiClient:
    """Client of a ``duplexjev gateway`` (for example our hosted trial API)."""

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 60):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def decide(self, audio: Any, questions=None, *, lang: str = "en") -> dict:
        data, fmt = audio_bytes(audio)
        payload = {"audio_b64": base64.b64encode(data).decode(), "format": fmt, "lang": lang}
        if questions is not None:
            payload["questions"] = [q.to_dict() for q in as_questions(questions)]
        r = _post(self.url + "/v1/decide", payload, self.headers, self.timeout)
        self.last_ms = r.get("ms")
        return r["answers"]
