"""HTTP server that batches every request arriving within one tick into a single decision pass.

    duplexjev serve --model Qwen/Qwen3-8B --tick-ms 160

Clients POST to ``/v1/decide``; the server collects all pending requests every ``tick_ms`` milliseconds (e.g. the
latest audio window of every live call), answers all of their questions in one batched pass, and returns each
request's own answers. Latency per request is at most one tick of waiting plus one pass.
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from .decider import Decider


def _decode_audio(b64: str):
    import soundfile as sf

    a, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    return (a, sr)


class TickBatcher:
    """Collect requests and run them together once per tick on a single worker thread."""

    def __init__(self, decider: Decider, tick_ms: float = 160.0, max_items: Optional[int] = None, mode: str = "packed"):
        self.d = decider
        self.tick = tick_ms / 1000.0
        self.max_items = max_items
        self.mode = mode
        self.pending: list[tuple[dict, asyncio.Future, float]] = []
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.ticks = 0
        self._task: Optional[asyncio.Task] = None

    def start(self):
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def submit(self, item: dict) -> dict:
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((item, fut, time.perf_counter()))
        return await fut

    async def _loop(self):
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        while True:
            nxt += self.tick
            await asyncio.sleep(max(0.0, nxt - loop.time()))
            if not self.pending:
                continue
            batch, self.pending = self.pending, []
            self.ticks += 1
            tick_id, t0 = self.ticks, time.perf_counter()
            try:
                res = await loop.run_in_executor(
                    self.pool, lambda: self.d.decide([b[0] for b in batch], mode=self.mode, max_items=self.max_items)
                )
                stats = dict(self.d.last_stats)
                for (item, fut, t_in), r in zip(batch, res):
                    if not fut.done():
                        fut.set_result({
                            "answers": r,
                            "tick": tick_id,
                            "batch_items": len(batch),
                            "wait_ms": round((t0 - t_in) * 1000, 1),
                            "pass_ms": round((time.perf_counter() - t0) * 1000, 1),
                            "pass": stats,
                        })
            except Exception as e:  # report the error to every request of this tick
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(e)
            if loop.time() > nxt:  # a long pass: start the next tick now instead of catching up
                nxt = loop.time()


def create_app(decider: Decider, tick_ms: float = 160.0, max_items: Optional[int] = None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class QuestionIn(BaseModel):
        id: str
        text: str
        options: List[str]
        lang: str = "en"

    class DecideIn(BaseModel):
        questions: List[QuestionIn]
        audio_b64: Optional[str] = None  # WAV/FLAC bytes, base64
        text: Optional[str] = None       # transcript (text-only models, or extra context for speech models)
        context: Optional[str] = None
        lang: Optional[str] = None

    app = FastAPI(title="DuplexJev decider", version="0.1.0")
    batcher = TickBatcher(decider, tick_ms=tick_ms, max_items=max_items)

    @app.on_event("startup")
    async def _start():
        batcher.start()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "speech": decider.is_audio, "tick_ms": tick_ms, "ticks": batcher.ticks}

    @app.post("/v1/decide")
    async def decide(req: DecideIn) -> dict[str, Any]:
        item: dict[str, Any] = {"questions": [q.model_dump() for q in req.questions]}
        if req.audio_b64:
            if not decider.is_audio:
                raise HTTPException(400, "this server runs a text-only model; send `text`")
            item["audio"] = _decode_audio(req.audio_b64)
        if req.text is not None:
            item["text"] = req.text
        if "audio" not in item and "text" not in item:
            raise HTTPException(400, "send `audio_b64` or `text`")
        if req.context:
            item["context"] = req.context
        if req.lang:
            item["lang"] = req.lang
        try:
            return await batcher.submit(item)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app
