"""Shared client for the DuplexJev benchmarks: P(finished) for one clip from a running ``duplexjev gateway``.

Start the model and the gateway first (one GPU, ~10 GB for the 4B model):
    vllm serve adventists-ai/DuplexJev-4B-Para --max-model-len 4096          # or DuplexJev-32B-Turn / -32B-Para
    duplexjev gateway --vllm http://localhost:8000/v1 --port 8420 --rate 1000000   # lift the 30/min limit
"""
from __future__ import annotations

import base64
import io
import os
import time
import wave

import numpy as np
import requests

SAMPLE_RATE = 16000
GATEWAY = os.environ.get("DUPLEXJEV_GATEWAY", "http://localhost:8420").rstrip("/")

# Same question and options as livekit-plugins-duplexjev / pipecat-duplexjev; P(finished) = P(first option).
TURN_QUESTION = {
    "en": {"id": "turn", "text": "Has the user finished speaking?",
           "options": ["finished, the assistant can reply", "not finished, still talking",
                       "just a backchannel, not taking the turn", "hesitating or asking to wait"], "lang": "en"},
    "zh": {"id": "turn", "text": "用户现在处于什么话轮状态？",
           "options": ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"],
           "lang": "zh"},
}


def resample(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr == SAMPLE_RATE:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), SAMPLE_RATE)
    return resample_poly(x, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)


def wav_bytes(x16k: np.ndarray) -> bytes:
    pcm = np.clip(np.asarray(x16k, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    return buf.getvalue()


_session = requests.Session()


def p_finished(x16k: np.ndarray, lang: str, *, retries: int = 3, timeout: float = 30.0) -> float:
    """P(finished) for a 16 kHz mono float clip. ``lang`` picks the question wording ("en" or "zh")."""
    q = TURN_QUESTION["zh" if lang == "zh" else "en"]
    if len(x16k) < SAMPLE_RATE // 10:  # under 0.1 s: pad, the gateway rejects empty audio
        x16k = np.pad(x16k, (SAMPLE_RATE // 10 - len(x16k), 0))
    payload = {"audio_b64": base64.b64encode(wav_bytes(x16k)).decode(), "format": "wav", "lang": q["lang"],
               "questions": [q]}
    for attempt in range(retries):
        try:
            r = _session.post(GATEWAY + "/v1/decide", json=payload, timeout=timeout)
            r.raise_for_status()
            probs = r.json()["answers"]["turn"]["probs"]
            # P(finished) + P(backchannel), as in the LiveKit / Pipecat plugins (>= 0.1.1 / 0.2.0)
            return float(probs.get(q["options"][0], 0.0)) + float(probs.get(q["options"][2], 0.0))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 + attempt)
    raise RuntimeError("unreachable")
