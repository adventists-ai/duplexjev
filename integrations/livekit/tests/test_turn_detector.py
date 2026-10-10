import asyncio

import numpy as np
import pytest
from conftest import frames, read_wav

from livekit.agents.voice.turn import (
    _resolve_endpointing,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)
from livekit.plugins import duplexjev
from livekit.plugins.duplexjev.turn_detector import trim_leading_silence


def push(stream, *args, **kw):
    for f in frames(*args, **kw):
        stream.push_audio(f)


async def test_satisfies_livekit_streaming_protocol(gateway):
    det = duplexjev.TurnDetector(api_url=gateway.url)
    assert isinstance(det, _StreamingTurnDetector)
    stream = det.stream()
    assert isinstance(stream, _StreamingTurnDetectorStream)
    # AgentSession picks the tighter streaming endpointing defaults for it
    assert _resolve_endpointing(turn_detection=det)["min_delay"] == 0.3
    assert stream.prediction_timeout == 1.0  # self-hosted default
    assert duplexjev.TurnDetector().stream().prediction_timeout == 4.0  # hosted API default
    assert await stream.unlikely_threshold(None) == 0.5
    assert await stream.backchannel_threshold(None) is None
    assert await stream.supports_language(None)
    await stream.aclose()


async def test_predict_sends_16k_mono_and_returns_event(gateway):
    seen = []
    det = duplexjev.TurnDetector(api_url=gateway.url, extra_questions=["emotion", "sound"],
                                 on_answers=seen.append)
    stream = det.stream()
    push(stream, 1.0, sample_rate=48000, channels=2)
    push(stream, 0.2, sample_rate=48000, channels=2, amplitude=0.0)
    ev = await asyncio.wait_for(stream.predict(), 2)

    assert ev.type == "eot_prediction"
    assert ev.end_of_turn_probability == pytest.approx(0.9)
    assert ev.inference_duration == pytest.approx(0.042)
    assert ev.backchannel_probability is None
    assert ev.detection_delay is not None and ev.detection_delay >= 0

    body = gateway.requests[-1]
    assert [q["id"] for q in body["questions"]] == ["turn", "emotion", "sound"]
    assert body["questions"][0]["text"] == "Has the user finished speaking?"
    sr, ch, pcm = read_wav(__import__("base64").b64decode(body["audio_b64"]))
    assert (sr, ch) == (16000, 1)
    assert abs(len(pcm) - 1.2 * 16000) < 400  # resampler latency only
    assert np.abs(pcm).max() > 5000

    assert stream.last_answers["emotion"]["answer"] == "neutral"
    assert det.last_answers is stream.last_answers and det.last_server_ms == 42.0
    assert seen and seen[-1]["sound"]["answer"] == "laughter"
    assert det.model == "DuplexJev-Fake"

    gateway.p_finished = 0.1
    ev = await asyncio.wait_for(stream.predict(), 2)
    assert ev.end_of_turn_probability == pytest.approx(0.1)
    await stream.aclose()


async def test_backchannel_opportunity_uses_reply_question(gateway):
    det = duplexjev.TurnDetector(api_url=gateway.url, backchannel_threshold=0.6, lang="zh")
    stream = det.stream()
    push(stream, 0.5, sample_rate=16000)
    ev = await stream.predict()
    assert [q["id"] for q in gateway.requests[-1]["questions"]] == ["turn", "reply"]
    assert gateway.requests[-1]["lang"] == "zh"
    assert ev.backchannel_probability == pytest.approx(0.7)
    assert await stream.backchannel_threshold(None) == 0.6
    await stream.aclose()


async def test_window_keeps_only_recent_audio(gateway):
    stream = duplexjev.TurnDetector(api_url=gateway.url, window_s=3.0, trim_silence=False).stream()
    push(stream, 7.0, sample_rate=24000)
    await stream.predict()
    _, _, pcm = read_wav(__import__("base64").b64decode(gateway.requests[-1]["audio_b64"]))
    assert len(pcm) == 3 * 16000
    await stream.aclose()


async def test_cancel_inference_resolves_zero_and_ignores_late_reply(gateway):
    gateway.delay = 0.5
    stream = duplexjev.TurnDetector(api_url=gateway.url).stream()
    push(stream, 0.5)
    fut = stream.predict()
    await asyncio.sleep(0.1)
    stream.cancel_inference()
    assert fut.done() and fut.result().end_of_turn_probability == 0.0
    await asyncio.sleep(0.6)
    assert stream.last_answers == {}  # the late response was dropped
    await stream.aclose()


async def test_new_predict_supersedes_previous(gateway):
    gateway.delay = 0.2
    stream = duplexjev.TurnDetector(api_url=gateway.url).stream()
    push(stream, 0.5)
    first = stream.predict()
    second = stream.predict()
    assert first.done() and first.result().end_of_turn_probability == 0.0
    assert (await second).end_of_turn_probability == pytest.approx(0.9)
    await stream.aclose()


async def test_flush_starts_a_new_turn(gateway):
    stream = duplexjev.TurnDetector(api_url=gateway.url, fallback_probability=0.25).stream()
    push(stream, 2.0)
    stream.flush(reason="turn committed")
    ev = await stream.predict()  # nothing buffered: no request is sent
    assert ev.end_of_turn_probability == 0.25 and gateway.requests == []
    push(stream, 0.4)
    await stream.predict()
    _, _, pcm = read_wav(__import__("base64").b64decode(gateway.requests[-1]["audio_b64"]))
    assert len(pcm) < 0.5 * 16000
    await stream.aclose()


async def test_http_error_and_unreachable_gateway_fall_back(gateway):
    gateway.status = 429
    stream = duplexjev.TurnDetector(api_url=gateway.url).stream()
    push(stream, 0.5)
    assert (await stream.predict()).end_of_turn_probability == 1.0
    await stream.aclose()

    stream = duplexjev.TurnDetector(api_url="http://127.0.0.1:9", fallback_probability=0.0).stream()
    push(stream, 0.5)
    assert (await asyncio.wait_for(stream.predict(), 5)).end_of_turn_probability == 0.0
    await stream.aclose()


async def test_end_input_and_aclose(gateway):
    stream = duplexjev.TurnDetector(api_url=gateway.url).stream()
    push(stream, 0.5)
    stream.end_input()
    push(stream, 0.5)  # ignored
    assert (await stream.predict()).end_of_turn_probability == 1.0
    assert gateway.requests == []
    await stream.aclose()
    await stream.aclose()  # idempotent


def test_trim_leading_silence():
    sil = np.zeros(32000, np.int16)
    speech = (np.sin(np.arange(16000) / 5) * 8000).astype(np.int16)
    out = trim_leading_silence(np.concatenate([sil, speech]))
    assert abs(len(out) - (16000 + 0.3 * 16000)) <= 800
    assert len(trim_leading_silence(sil)) == len(sil)


def test_bad_arguments():
    with pytest.raises(ValueError):
        duplexjev.TurnDetector(lang="fr")
    with pytest.raises(ValueError):
        duplexjev.TurnDetector(extra_questions=["nope"])
    with pytest.raises(ValueError):
        duplexjev.TurnDetector(extra_questions=[{"id": "turn", "text": "x", "options": ["a", "b"]}])


async def test_agent_session_accepts_it(gateway):
    from livekit.agents import AgentSession

    det = duplexjev.TurnDetector(api_url=gateway.url)
    session = AgentSession(turn_handling={"turn_detection": det})
    assert session.turn_detection is det
