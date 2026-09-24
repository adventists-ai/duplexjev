"""HTTP server that answers every request arriving within one tick in a single decision pass.

    duplexjev serve --model adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B --tick-ms 160

* ``POST /v1/decide``        one clip + option groups           -> ``{"answers": {question_id: ...}}``
* ``POST /v1/decide_batch``  many clips + option groups bound to clip ids -> ``{"answers": {clip_id: {...}}}``

Every ``tick_ms`` the server takes all pending requests (e.g. the latest audio window of every live call), runs
them as one batched pass, and returns each request its own answers. A request waits at most one tick plus one pass.
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from .decider import Decider
from .question import Question


def _decode_audio(b64: str):
    import soundfile as sf

    a, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    return (a, sr)


class TickBatcher:
    """Collect requests and run all of their clips together once per tick, on a single worker thread."""

    def __init__(self, decider: Decider, tick_ms: float = 160.0, max_items: Optional[int] = None,
                 max_tokens: Optional[int] = None, mode: str = "packed"):
        self.d = decider
        self.tick = tick_ms / 1000.0
        self.kw = {"mode": mode, "max_items": max_items, "max_tokens": max_tokens}
        self.pending: list[tuple[list, asyncio.Future, float]] = []
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.ticks = 0
        self._task: Optional[asyncio.Task] = None

    def start(self):
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def submit(self, items: list[dict]) -> dict:
        """Queue one request (a list of clip items); resolves to its answers plus tick info."""
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((items, fut, time.perf_counter()))
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
            flat = [it for items, _, _ in batch for it in items]
            try:
                res = await loop.run_in_executor(self.pool, lambda: self.d._decide_items(flat, **self.kw))
                stats = dict(self.d.last_stats)
                pass_ms = round((time.perf_counter() - t0) * 1000, 1)
                i = 0
                for items, fut, t_in in batch:
                    part, i = res[i:i + len(items)], i + len(items)
                    if not fut.done():
                        fut.set_result({"answers": part, "tick": tick_id, "requests_in_tick": len(batch),
                                        "clips_in_pass": len(flat), "wait_ms": round((t0 - t_in) * 1000, 1),
                                        "pass_ms": pass_ms, "pass": stats})
            except Exception as e:  # report the error to every request of this tick
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(e)
            if loop.time() > nxt:  # a long pass: start the next tick now instead of catching up
                nxt = loop.time()


def create_app(decider: Decider, tick_ms: float = 160.0, max_items: Optional[int] = None, max_tokens: Optional[int] = None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class QuestionIn(BaseModel):
        id: str
        text: str
        options: List[str]
        lang: str = "en"
        audio: Optional[Union[str, int, List[Union[str, int]]]] = None

    class DecideIn(BaseModel):
        audio_b64: str                   # WAV/FLAC bytes, base64
        questions: List[QuestionIn]
        context: Optional[str] = None
        lang: Optional[str] = None

    class DecideBatchIn(BaseModel):
        audios: Dict[str, str]           # clip id -> WAV/FLAC bytes, base64
        questions: List[QuestionIn]      # each with audio=<clip id>, [ids] or "*"
        context: Optional[Dict[str, str]] = None
        lang: Optional[str] = None

    app = FastAPI(title="DuplexJev decider", version="0.2.1")
    batcher = TickBatcher(decider, tick_ms=tick_ms, max_items=max_items, max_tokens=max_tokens)

    @app.on_event("startup")
    async def _start():
        batcher.start()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "tick_ms": tick_ms, "ticks": batcher.ticks}

    @app.post("/v1/decide")
    async def decide(req: DecideIn) -> dict[str, Any]:
        qs = [Question(q.id, q.text, q.options, q.lang) for q in req.questions]
        item = {"audio": _decode_audio(req.audio_b64), "questions": qs, "context": req.context, "lang": req.lang}
        try:
            out = await batcher.submit([item])
        except ValueError as e:
            raise HTTPException(400, str(e))
        out["answers"] = out["answers"][0]
        return out

    @app.post("/v1/decide_batch")
    async def decide_batch(req: DecideBatchIn) -> dict[str, Any]:
        ctx = req.context or {}
        per: dict[str, list] = {k: [] for k in req.audios}
        for q in req.questions:
            if q.audio is None:
                raise HTTPException(400, f"question {q.id!r}: set audio to a clip id, a list of ids, or '*'")
            targets = list(req.audios) if q.audio == "*" else (q.audio if isinstance(q.audio, list) else [q.audio])
            for t in map(str, targets):
                if t not in per:
                    raise HTTPException(400, f"question {q.id!r} refers to unknown audio {t!r}")
                per[t].append(Question(q.id, q.text, q.options, q.lang))
        ids = [k for k, v in per.items() if v]
        items = [{"audio": _decode_audio(req.audios[k]), "questions": per[k], "context": ctx.get(k), "lang": req.lang}
                 for k in ids]
        try:
            out = await batcher.submit(items)
        except ValueError as e:
            raise HTTPException(400, str(e))
        out["answers"] = dict(zip(ids, out["answers"]))
        return out

    return app
