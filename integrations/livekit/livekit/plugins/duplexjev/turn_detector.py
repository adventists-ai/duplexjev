"""DuplexJev audio turn detector for LiveKit Agents.

DuplexJev reads the turn state (finished / still talking / backchannel / asking to wait) straight from the user's
audio as one constrained token of a speech LLM, so the decision needs no transcript and no decoding. Extra questions
(emotion, non-verbal sounds, barge-in, your own multiple-choice questions) are answered in the same request and
exposed on ``last_answers`` / ``on_answers``.

This module implements LiveKit's *streaming* (audio) turn-detector protocol: ``AgentSession`` pushes every input
audio frame into the stream, calls ``predict()`` once the VAD has seen ~200 ms of silence, and waits up to
``prediction_timeout`` for the resulting ``TurnDetectionEvent``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
import os
import time
import wave
from typing import Any, Callable

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, utils
from livekit.agents.language import LanguageCode
from livekit.agents.voice.turn import TurnDetectionEvent

from .log import logger

DEFAULT_URL = "https://api.adventists.cn/duplexjev"
SAMPLE_RATE = 16000

TURN_QUESTION = {
    "en": {"id": "turn", "text": "Has the user finished speaking?",
           "options": ["finished, the assistant can reply", "not finished, still talking",
                       "just a backchannel, not taking the turn", "hesitating or asking to wait"], "lang": "en"},
    "zh": {"id": "turn", "text": "用户现在处于什么话轮状态？",
           "options": ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"],
           "lang": "zh"},
}

# "What should the assistant do now?" -- its "short acknowledgement" option is what LiveKit means by a backchannel
# opportunity (the *agent* says "mm-hmm"). Only asked when ``backchannel_threshold`` is set.
REPLY_QUESTION = {
    "en": {"id": "reply", "text": "What should the assistant do now?",
           "options": ["reply now", "keep listening", "give a short acknowledgement"], "lang": "en"},
    "zh": {"id": "reply", "text": "助手现在应该怎么做？",
           "options": ["马上回复", "继续听", "简短回应一下"], "lang": "zh"},
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

HOSTED_PREDICTION_TIMEOUT = 4.0
"""The free hosted API is served from mainland China: a round trip is often 2-3 s from elsewhere."""
SELF_HOSTED_PREDICTION_TIMEOUT = 1.0


def wav16k(pcm: np.ndarray) -> bytes:
    """16 kHz mono int16 samples -> WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return buf.getvalue()


def trim_leading_silence(pcm: np.ndarray, *, preroll_s: float = 0.3, block_s: float = 0.05) -> np.ndarray:
    """Drop quiet audio before the first loud block (keeping ``preroll_s``), so the window holds the utterance."""
    n = int(block_s * SAMPLE_RATE)
    if len(pcm) < 2 * n:
        return pcm
    blocks = pcm[: len(pcm) // n * n].astype(np.float32).reshape(-1, n)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    peak = float(rms.max())
    if peak < 100.0:  # nothing that looks like speech: send as is
        return pcm
    first = int(np.argmax(rms >= max(0.1 * peak, 60.0)))
    start = max(0, first * n - int(preroll_s * SAMPLE_RATE))
    return pcm[start:]


class TurnDetector:
    """Audio end-of-turn detection with DuplexJev, via the hosted API or your own ``duplexjev gateway``.

    Pass it as ``AgentSession(turn_detection=TurnDetector(), vad=silero.VAD.load(), ...)``. A VAD is required:
    LiveKit drops a streaming turn detector when the session has none.

    Args:
        api_url: base URL of a DuplexJev gateway. Defaults to the free hosted trial API (rate limited, served from
            mainland China). For production run your own: ``vllm serve adventists-ai/DuplexJev-4B-Para`` +
            ``duplexjev gateway``. Also read from ``DUPLEXJEV_API_URL``.
        api_key: bearer key, if the gateway requires one. Also read from ``DUPLEXJEV_API_KEY``.
        lang: ``"en"`` or ``"zh"``: wording of the questions (the model hears both languages either way).
        unlikely_threshold: LiveKit's knob. When P(finished) is below it, the session waits ``max_delay`` instead
            of ``min_delay`` before committing the turn.
        backchannel_threshold: if set, also asks "What should the assistant do now?" and reports P("give a short
            acknowledgement") as ``backchannel_probability``; LiveKit raises a backchannel opportunity at or above
            this value. ``None`` (default) disables it and keeps the request to one question.
        extra_questions: names from ``PRESETS`` (``"emotion"``, ``"sound"``, ``"barge_in"``) or question dicts
            ``{"id", "text", "options"}``, answered in the same request; see ``last_answers``.
        on_answers: optional callback ``f(answers: dict)`` called after every successful prediction.
        window_s: seconds of the current user turn sent with each request (the most recent ones).
        trim_silence: drop the quiet audio before the user started talking, keeping 0.3 s of pre-roll.
        prediction_timeout: how long LiveKit waits for a prediction before committing without one. Defaults to
            4 s for the hosted API and 1 s for your own gateway.
        fallback_probability: P(finished) reported when the gateway fails (network error, HTTP error, rate limit).
            ``1.0`` (like LiveKit's own detector) commits after ``min_delay``; ``0.0`` waits ``max_delay``.
        http_session: an ``aiohttp.ClientSession`` to reuse. By default the job's session is used, or one is
            created per stream.
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        lang: str = "en",
        unlikely_threshold: float = 0.5,
        backchannel_threshold: float | None = None,
        extra_questions: list[str | dict[str, Any]] | None = None,
        on_answers: Callable[[dict[str, Any]], Any] | None = None,
        window_s: float = 8.0,
        trim_silence: bool = True,
        prediction_timeout: float | None = None,
        fallback_probability: float = 1.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if lang not in TURN_QUESTION:
            raise ValueError("lang must be 'en' or 'zh'")
        api_url = api_url or os.environ.get("DUPLEXJEV_API_URL") or DEFAULT_URL
        self._base_url = api_url.rstrip("/")
        self._decide_url = self._base_url + "/v1/decide"
        api_key = api_key or os.environ.get("DUPLEXJEV_API_KEY")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._lang = lang
        self._unlikely_threshold = unlikely_threshold
        self._backchannel_threshold = backchannel_threshold
        self._on_answers = on_answers
        self._window_s = window_s
        self._trim_silence = trim_silence
        if prediction_timeout is None:
            prediction_timeout = (
                HOSTED_PREDICTION_TIMEOUT if self._base_url == DEFAULT_URL else SELF_HOSTED_PREDICTION_TIMEOUT
            )
        self._prediction_timeout = prediction_timeout
        self._fallback_probability = fallback_probability
        self._http_session = http_session

        self._questions: list[dict[str, Any]] = [TURN_QUESTION[lang]]
        if backchannel_threshold is not None:
            self._questions.append(REPLY_QUESTION[lang])
        self._questions += [self._as_question(q) for q in (extra_questions or [])]
        ids = [q["id"] for q in self._questions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate question ids: {ids}")

        self._model_name = "DuplexJev"
        self.last_answers: dict[str, Any] = {}
        """Answers of the latest successful request: ``{id: {"answer", "confidence", "probs"}}``."""
        self.last_server_ms: float | None = None
        """Server-side model time of the latest request, in milliseconds."""

    def _as_question(self, q: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(q, str):
            if q not in PRESETS:
                raise ValueError(f"unknown preset {q!r}; choose from {sorted(PRESETS)} or pass a dict")
            text, options = PRESETS[q][self._lang]
            return {"id": q, "text": text, "options": options, "lang": self._lang}
        d = dict(q)
        if not {"id", "text", "options"} <= d.keys():
            raise ValueError("a question dict needs 'id', 'text' and 'options'")
        d.setdefault("lang", self._lang)
        return d

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "duplexjev"

    @property
    def questions(self) -> list[dict[str, Any]]:
        return list(self._questions)

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> TurnDetectorStream:
        return TurnDetectorStream(detector=self, conn_options=conn_options)

    # the capability methods below are mirrored on the stream, which is what AudioRecognition queries
    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._unlikely_threshold

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return self._backchannel_threshold

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return True


class TurnDetectorStream:
    """Per-session audio buffer and inference requests. Created by ``TurnDetector.stream()``."""

    def __init__(self, *, detector: TurnDetector, conn_options: APIConnectOptions) -> None:
        self._detector = detector
        self._conn_options = conn_options
        self._chunks: list[np.ndarray] = []
        self._n_samples = 0
        self._max_samples = int(max(detector._window_s, 1.0) * SAMPLE_RATE * 2)
        self._in_rate: int | None = None
        self._resampler: rtc.AudioResampler | None = None
        self._last_audio_at: float | None = None
        self._closed = False

        self._ids = itertools.count(1)
        self._request_id: int | None = None
        self._request_fut: asyncio.Future[TurnDetectionEvent] | None = None
        self._request_task: asyncio.Task[None] | None = None
        self._session: aiohttp.ClientSession | None = detector._http_session
        self._owns_session = False
        self._warned = False

        self.last_answers: dict[str, Any] = {}
        self.last_server_ms: float | None = None

    # region: protocol properties
    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def prediction_timeout(self) -> float:
        return self._detector._prediction_timeout

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.unlikely_threshold(language)

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.backchannel_threshold(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self._detector.supports_language(language)

    # endregion

    # region: audio
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        if frame.sample_rate != self._in_rate:
            self._in_rate = frame.sample_rate
            self._resampler = (
                rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE, num_channels=1,
                                   quality=rtc.AudioResamplerQuality.QUICK)
                if frame.sample_rate != SAMPLE_RATE else None
            )
        if self._resampler is None:
            self._append(pcm.copy())
        else:
            mono = rtc.AudioFrame(pcm.tobytes(), frame.sample_rate, 1, len(pcm))
            for out in self._resampler.push(mono):
                self._append(np.frombuffer(out.data, dtype=np.int16).copy())
        self._last_audio_at = time.time()

    def _append(self, pcm: np.ndarray) -> None:
        if not len(pcm):
            return
        self._chunks.append(pcm)
        self._n_samples += len(pcm)
        while self._n_samples - len(self._chunks[0]) >= self._max_samples:
            self._n_samples -= len(self._chunks.pop(0))

    def _window(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, np.int16)
        pcm = np.concatenate(self._chunks)[-int(self._detector._window_s * SAMPLE_RATE):]
        return trim_leading_silence(pcm) if self._detector._trim_silence else pcm

    def _clear_audio(self) -> None:
        self._chunks = []
        self._n_samples = 0
        self._in_rate = None
        self._resampler = None

    # endregion

    # region: inference
    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        """Send the recent audio of the current user turn and return the future prediction."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[TurnDetectionEvent] = loop.create_future()
        if self._closed:
            fut.set_result(self._event(1.0))
            return fut
        self.cancel_inference()  # supersede any previous request
        pcm = self._window()
        request_id = next(self._ids)
        self._request_id, self._request_fut = request_id, fut
        if not len(pcm):
            self._resolve(request_id, self._event(self._detector._fallback_probability))
            return fut
        self._request_task = asyncio.create_task(self._run(request_id, wav16k(pcm), self._last_audio_at))
        return fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        """Abandon the in-flight request (new speech, turn boundary, prediction timeout, mode change)."""
        task, fut = self._request_task, self._request_fut
        self._request_id = self._request_fut = self._request_task = None
        if task is not None and not task.done():
            task.cancel()
        if fut is not None and not fut.done():
            fut.set_result(self._event(0.0))
        if timed_out:
            logger.debug("DuplexJev prediction timed out", extra={"timeout": self.prediction_timeout})

    def flush(self, reason: str | None = None) -> None:
        """Start a new user turn: forget the buffered audio and any in-flight request."""
        self.cancel_inference()
        self._clear_audio()

    def end_input(self) -> None:
        self.flush(reason="end_input")
        self._closed = True

    async def aclose(self) -> None:
        self.end_input()
        if self._owns_session and self._session is not None:
            await self._session.close()
        self._session = None

    def _http(self) -> aiohttp.ClientSession:
        if self._session is None:
            try:
                self._session = utils.http_context.http_session()
            except RuntimeError:
                self._session = aiohttp.ClientSession()
                self._owns_session = True
        return self._session

    async def _run(self, request_id: int, wav: bytes, last_audio_at: float | None) -> None:
        det = self._detector
        payload = {"audio_b64": base64.b64encode(wav).decode(), "format": "wav", "lang": det._lang,
                   "questions": det._questions}
        try:
            async with self._http().post(
                det._decide_url, json=payload, headers=det._headers,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                data = await resp.json()
            answers: dict[str, Any] = data["answers"]
            probs = answers["turn"]["probs"]
            p_finished = float(probs.get(TURN_QUESTION[det._lang]["options"][0], 0.0))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # network or server error: never block the conversation
            if not self._warned:
                logger.warning("DuplexJev request failed (%s); reporting P(finished)=%s",
                               e, det._fallback_probability)
                self._warned = True
            else:
                logger.debug("DuplexJev request failed: %s", e)
            self._resolve(request_id, self._event(det._fallback_probability))
            return

        if request_id != self._request_id:
            return  # superseded while the response was in flight
        server_ms = data.get("ms")
        if data.get("model"):
            det._model_name = str(data["model"])
        backchannel = None
        if "reply" in answers and det._backchannel_threshold is not None:
            backchannel = float(answers["reply"]["probs"].get(REPLY_QUESTION[det._lang]["options"][2], 0.0))
        now = time.time()
        self.last_answers = det.last_answers = answers
        self.last_server_ms = det.last_server_ms = server_ms
        event = TurnDetectionEvent(
            type="eot_prediction",
            end_of_turn_probability=p_finished,
            last_speaking_time=now,
            detection_delay=now - last_audio_at if last_audio_at is not None else None,
            inference_duration=float(server_ms) / 1000 if server_ms is not None else None,
            backchannel_probability=backchannel,
        )
        if det._on_answers is not None:
            try:
                det._on_answers(answers)
            except Exception as e:
                logger.warning("on_answers callback raised: %s", e)
        self._resolve(request_id, event)

    def _resolve(self, request_id: int, event: TurnDetectionEvent) -> None:
        if request_id != self._request_id:
            return  # stale: superseded or cancelled
        fut = self._request_fut
        self._request_id = self._request_fut = self._request_task = None
        if fut is not None and not fut.done():
            fut.set_result(event)

    @staticmethod
    def _event(probability: float) -> TurnDetectionEvent:
        return TurnDetectionEvent(type="eot_prediction", end_of_turn_probability=probability,
                                  last_speaking_time=time.time())

    # endregion
