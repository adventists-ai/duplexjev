import asyncio
import io
import wave

import numpy as np
import pytest
from aiohttp import web

from livekit import rtc


def frames(seconds: float, *, sample_rate: int = 48000, channels: int = 1, freq: float = 220.0,
           amplitude: float = 0.3, frame_ms: int = 10):
    """Fake microphone frames: a sine tone (or silence with amplitude=0), interleaved int16."""
    n = int(sample_rate * frame_ms / 1000)
    t0 = 0
    for _ in range(int(seconds * 1000 / frame_ms)):
        t = (np.arange(n) + t0) / sample_rate
        x = (amplitude * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
        t0 += n
        data = np.repeat(x, channels) if channels > 1 else x
        yield rtc.AudioFrame(data.tobytes(), sample_rate, channels, n)


def read_wav(b: bytes):
    with wave.open(io.BytesIO(b)) as w:
        return w.getframerate(), w.getnchannels(), np.frombuffer(w.readframes(w.getnframes()), np.int16)


class FakeGateway:
    """Mimics ``duplexjev gateway``'s POST /v1/decide."""

    def __init__(self):
        self.p_finished = 0.9
        self.p_ack = 0.7
        self.delay = 0.0
        self.status = 200
        self.requests: list[dict] = []

    async def decide(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.status != 200:
            return web.Response(status=self.status, text="rate limited")
        answers = {}
        for q in body["questions"]:
            opts = q["options"]
            probs = {o: 0.0 for o in opts}
            if q["id"] == "turn":
                probs[opts[0]] = self.p_finished
                probs[opts[1]] = 1 - self.p_finished
            elif q["id"] == "reply":
                probs[opts[2]] = self.p_ack
                probs[opts[1]] = 1 - self.p_ack
            else:
                probs[opts[0]] = 1.0
            best = max(probs, key=probs.get)
            answers[q["id"]] = {"answer": best, "confidence": probs[best], "probs": probs}
        return web.json_response({"answers": answers, "ms": 42.0, "model": "DuplexJev-Fake"})


@pytest.fixture()
async def gateway():
    gw = FakeGateway()
    app = web.Application()
    app.router.add_post("/v1/decide", gw.decide)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    gw.url = f"http://127.0.0.1:{port}"
    yield gw
    await runner.cleanup()
