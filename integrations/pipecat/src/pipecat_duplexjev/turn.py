"""DuplexJev end-of-turn analyzer for Pipecat.

DuplexJev reads turn state (finished / still talking / backchannel / asking to wait) straight from the audio as one
constrained token of a speech LLM, so no transcript and no decoding are needed. Optional extra questions (emotion,
non-verbal sounds, intent, ...) are answered in the same forward pass and exposed on ``last_answers``.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
import wave
from typing import Any, Callable

import numpy as np
from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams, SmartTurnTimeoutException

DEFAULT_URL = "https://api.adventists.cn/duplexjev"

TURN_QUESTION = {
    "en": {"id": "turn", "text": "Has the user finished speaking?",
           "options": ["finished, the assistant can reply", "not finished, still talking",
                       "just a backchannel, not taking the turn", "hesitating or asking to wait"], "lang": "en"},
    "zh": {"id": "turn", "text": "用户现在处于什么话轮状态？",
           "options": ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"],
           "lang": "zh"},
}

PRESETS = {
    "emotion": {"en": ("What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"]),
                "zh": ("说话人当时的情绪状态是？", ["中性", "高兴", "生气", "伤心"])},
    "sound": {"en": ("Besides speech, which sound can be heard in this audio?",
                     ["laughter", "breathing", "coughing", "a sigh", "None of these"]),
              "zh": ("除了说话，这段音频里还有哪种声音？", ["笑声", "呼吸声", "咳嗽", "叹气", "都没有"])},
    "barge_in": {"en": ("Is the user trying to interrupt or take over the conversation?", ["yes", "no"]),
                 "zh": ("用户是不是想打断或者抢话？", ["是", "不是"])},
}


def _wav16k(audio: np.ndarray, sample_rate: int) -> bytes:
    """Mono float32 in [-1, 1] -> 16 kHz PCM16 WAV bytes."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != 16000 and len(x):
        n = max(1, int(round(len(x) * 16000 / sample_rate)))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


class DuplexJevTurnAnalyzer(BaseSmartTurn):
    """Audio end-of-turn detection with DuplexJev, via the hosted API or your own ``duplexjev gateway``.

    Args:
        url: base URL of a DuplexJev gateway. Defaults to the free hosted trial API (rate limited, served from China).
            For production run your own: ``vllm serve adventists-ai/DuplexJev-4B-Para`` + ``duplexjev gateway``.
        api_key: bearer key, if the gateway requires one.
        lang: ``"en"`` or ``"zh"``: language of the question wording (the model hears both languages either way).
        threshold: probability of "finished" at or above which the turn is complete.
        extra_questions: names from ``PRESETS`` (``"emotion"``, ``"sound"``, ``"barge_in"``) or question dicts
            ``{"id", "text", "options"}``; answered in the same pass, see ``last_answers``.
        on_answers: optional callback ``f(answers: dict)`` called after every prediction.
        timeout: HTTP timeout in seconds (defaults to ``params.stop_secs``).
    """

    def __init__(self, *, url: str = DEFAULT_URL, api_key: str | None = None, lang: str = "en",
                 threshold: float = 0.5, extra_questions: list | None = None,
                 on_answers: Callable[[dict], Any] | None = None, timeout: float | None = None,
                 sample_rate: int | None = None, params: SmartTurnParams | None = None):
        super().__init__(sample_rate=sample_rate, params=params)
        if lang not in TURN_QUESTION:
            raise ValueError("lang must be 'en' or 'zh'")
        self._url = url.rstrip("/") + "/v1/decide"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._lang = lang
        self._threshold = threshold
        self._timeout = timeout
        self._on_answers = on_answers
        self._questions = [TURN_QUESTION[lang]] + [self._as_question(q) for q in (extra_questions or [])]
        self.last_answers: dict = {}
        self.last_server_ms: float | None = None

    def _as_question(self, q) -> dict:
        if isinstance(q, str):
            if q not in PRESETS:
                raise ValueError(f"unknown preset {q!r}; choose from {sorted(PRESETS)} or pass a dict")
            text, options = PRESETS[q][self._lang]
            return {"id": q, "text": text, "options": options, "lang": self._lang}
        d = dict(q)
        d.setdefault("lang", self._lang)
        return d

    def _request(self, wav: bytes) -> dict:
        payload = {"audio_b64": base64.b64encode(wav).decode(), "format": "wav", "lang": self._lang,
                   "questions": self._questions}
        req = urllib.request.Request(self._url, data=json.dumps(payload).encode(), headers=self._headers)
        timeout = self._timeout if self._timeout is not None else self.params.stop_secs
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except TimeoutError as e:
            raise SmartTurnTimeoutException(str(e)) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise SmartTurnTimeoutException(str(e)) from e
            raise

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        try:
            r = self._request(_wav16k(audio_array, self.sample_rate or 16000))
        except SmartTurnTimeoutException:
            raise
        except Exception as e:  # network or server error: do not block the conversation
            logger.error(f"DuplexJev request failed: {e}")
            return {"prediction": 0, "probability": 0.0}
        answers = r.get("answers", {})
        self.last_answers = answers
        self.last_server_ms = r.get("ms")
        if self._on_answers:
            try:
                self._on_answers(answers)
            except Exception as e:
                logger.warning(f"on_answers callback raised: {e}")
        turn = answers.get("turn", {})
        p = float(turn.get("probs", {}).get(TURN_QUESTION[self._lang]["options"][0], 0.0))
        return {"prediction": 1 if p >= self._threshold else 0, "probability": p}
