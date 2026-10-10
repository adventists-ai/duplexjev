"""Live check against the hosted DuplexJev API. Skipped unless DUPLEXJEV_LIVE=1.

Clips are read from DUPLEXJEV_CLIPS (default: the repo's docs/audio). ex4.wav is a finished sentence, ex3.wav an
unfinished one. Run with ``DUPLEXJEV_LIVE=1 pytest -s tests/test_live.py`` to see the probabilities.
"""
import os
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from livekit import rtc
from livekit.plugins import duplexjev

pytestmark = pytest.mark.skipif(os.environ.get("DUPLEXJEV_LIVE") != "1", reason="set DUPLEXJEV_LIVE=1")

CLIPS = Path(os.environ.get("DUPLEXJEV_CLIPS", Path(__file__).resolve().parents[3] / "docs" / "audio"))


def clip_frames(path: Path, frame_ms: int = 20):
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    n = sr * frame_ms // 1000
    x = np.concatenate([x, np.zeros(n * ch * 15, np.int16)])  # + 300 ms of silence, as when VAD fires
    for i in range(0, len(x) // (n * ch) * n * ch, n * ch):
        yield rtc.AudioFrame(x[i:i + n * ch].tobytes(), sr, ch, n)


@pytest.mark.parametrize("name,finished", [("ex4.wav", True), ("ex3.wav", False)])
async def test_live_clip(name, finished):
    path = CLIPS / name
    if not path.exists():
        pytest.skip(f"{path} not found; set DUPLEXJEV_CLIPS")
    det = duplexjev.TurnDetector(extra_questions=["emotion"])
    stream = det.stream()
    for f in clip_frames(path):
        stream.push_audio(f)
    t = time.time()
    ev = await stream.predict()
    rtt = time.time() - t
    await stream.aclose()
    print(f"\n{name}: P(finished)={ev.end_of_turn_probability:.3f} turn={det.last_answers['turn']['answer']!r} "
          f"emotion={det.last_answers['emotion']['answer']} server={ev.inference_duration * 1000:.1f} ms "
          f"round trip={rtt * 1000:.0f} ms model={det.model}")
    assert (ev.end_of_turn_probability >= 0.5) == finished
