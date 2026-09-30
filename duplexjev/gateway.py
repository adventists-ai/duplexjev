"""A small public-facing HTTP API + web demo in front of a vLLM server running a DuplexJev complete model.

    vllm serve adventists-ai/DuplexJev-4B --port 8010
    duplexjev gateway --vllm http://127.0.0.1:8010/v1 --port 8020

    GET  /                 web demo (record or upload a clip, see the decision table)
    GET  /v1/table?lang=zh the default decision table
    POST /v1/decide        {"audio_b64": ..., "format": "wav", "lang": "zh", "questions": [...optional]}
                           -> {"answers": {id: {answer, confidence, probs}}, "ms": 123.4, "model": "..."}

Limits: clip size (--max-mb), number of questions (--max-questions), requests per minute per client IP
(--rate); audio is never written to disk. Set DUPLEXJEV_KEYS=key1,key2 to require `Authorization: Bearer <key>`
on /v1/decide (the demo page then asks for a key). Put it behind a reverse proxy under any sub-path: the page uses
relative URLs.
"""

import base64
import os
import time
from collections import defaultdict, deque
from importlib import resources
from typing import List, Optional

from .question import Question, as_questions
from .remote import VLLMClient
from .table import LABELS, default_table


def create_app(vllm_url: str, *, max_mb: float = 4.0, max_questions: int = 16, rate: int = 30):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    client = VLLMClient(vllm_url)
    keys = {k.strip() for k in os.environ.get("DUPLEXJEV_KEYS", "").split(",") if k.strip()}
    hits: dict[str, deque] = defaultdict(deque)
    app = FastAPI(title="DuplexJev API", docs_url="/docs")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    class DecideRequest(BaseModel):
        audio_b64: str
        format: str = "wav"
        lang: str = "en"
        questions: Optional[List[dict]] = None

    def _limit(req: Request):
        ip = req.headers.get("x-real-ip") or req.headers.get("x-forwarded-for", "").split(",")[0].strip() or \
            (req.client.host if req.client else "?")
        now, q = time.time(), hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= rate:
            raise HTTPException(429, f"rate limit: {rate} requests per minute")
        q.append(now)

    @app.get("/", response_class=HTMLResponse)
    def page():
        return resources.files("duplexjev").joinpath("demo.html").read_text(encoding="utf-8")

    @app.get("/v1/table")
    def table(lang: str = "en"):
        if lang not in ("en", "zh"):
            raise HTTPException(400, "lang must be en or zh")
        return {"lang": lang, "labels": LABELS[lang], "questions": [q.to_dict() for q in default_table(lang)]}

    @app.get("/v1/health")
    def health():
        return {"ok": True, "model": client.model, "auth": bool(keys)}

    @app.post("/v1/decide")
    def decide(body: DecideRequest, req: Request):
        if keys:
            tok = req.headers.get("authorization", "")[len("Bearer "):].strip() if req.headers.get("authorization", "").startswith("Bearer ") else ""
            if tok not in keys:
                raise HTTPException(401, "missing or invalid API key")
        _limit(req)
        if body.lang not in ("en", "zh"):
            raise HTTPException(400, "lang must be en or zh")
        if len(body.audio_b64) > max_mb * 1.37e6:
            raise HTTPException(413, f"audio larger than {max_mb} MB")
        try:
            audio = base64.b64decode(body.audio_b64, validate=True)
        except Exception:
            raise HTTPException(400, "audio_b64 is not valid base64")
        try:
            qs = as_questions(body.questions) if body.questions else default_table(body.lang)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"bad questions: {e}")
        if len(qs) > max_questions:
            raise HTTPException(400, f"at most {max_questions} questions per request")
        qs = [Question(q.id, q.text, q.options, lang=q.lang) for q in qs]
        t = time.perf_counter()
        try:
            answers = client.decide(audio, qs)
        except Exception as e:  # decoding errors, too-long audio etc. come back from vLLM as HTTP 400
            raise HTTPException(400, f"model server rejected the request: {str(e)[:200]}")
        return {"answers": answers, "ms": round((time.perf_counter() - t) * 1000, 1), "model": client.model}

    return app
