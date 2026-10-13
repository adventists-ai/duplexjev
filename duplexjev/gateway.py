"""A small public-facing HTTP API + web demo in front of a vLLM server running a DuplexJev complete model.

    vllm serve adventists-ai/DuplexJev-4B --port 8010
    duplexjev gateway --vllm http://127.0.0.1:8010/v1 --port 8020

    GET  /                 web demo (record or upload a clip, see the decision table)
    GET  /v1/table?lang=zh the default decision table
    POST /v1/decide        {"audio_b64": ..., "format": "wav", "lang": "zh", "questions": [...optional],
                            "assistant_said": "...optional: the assistant's previous message"}
                           -> {"answers": {id: {answer, confidence, probs}}, "ms": 123.4, "model": "..."}

Limits: clip size (--max-mb), number of questions (--max-questions), requests per minute per client IP
(--rate); audio is never written to disk. With --log FILE, one JSON line per request is appended (time, salted
hash of the client IP, user agent, status, latency, language, question ids, custom question texts, clip length and
the answers; never the audio or the assistant's message text). Set DUPLEXJEV_KEYS=key1,key2 to require `Authorization: Bearer <key>`
on /v1/decide (the demo page then asks for a key). Put it behind a reverse proxy under any sub-path: the page uses
relative URLs.
"""

import base64
import hashlib
import io
import json
import os
import threading
import time
import wave
from collections import defaultdict, deque
from importlib import resources
from typing import List, Optional

from .question import Question, as_questions
from .remote import VLLMClient
from .table import LABELS, default_table


def _clip_seconds(audio: bytes, fmt: str):
    if fmt != "wav":
        return None
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        return None


class _RequestLog:
    """Append-only JSONL request log (no audio, no assistant text, client IP only as a salted hash)."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self.lock = threading.Lock()
        self.salt = os.environ.get("DUPLEXJEV_LOG_SALT", "") or (os.urandom(8).hex() if path else "")
        if path and not os.environ.get("DUPLEXJEV_LOG_SALT"):
            salt_file = path + ".salt"  # keep the salt across restarts so client counts stay comparable
            if os.path.exists(salt_file):
                self.salt = open(salt_file).read().strip()
            else:
                with open(salt_file, "w") as f:
                    f.write(self.salt)

    def client(self, ip: str) -> str:
        return hashlib.sha256((self.salt + ip).encode()).hexdigest()[:12]

    def write(self, rec: dict):
        if not self.path:
            return
        line = json.dumps(rec, ensure_ascii=False)
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def create_app(vllm_url: str, *, max_mb: float = 4.0, max_questions: int = 16, rate: int = 30,
               log_path: Optional[str] = None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    client = VLLMClient(vllm_url)
    keys = {k.strip() for k in os.environ.get("DUPLEXJEV_KEYS", "").split(",") if k.strip()}
    hits: dict[str, deque] = defaultdict(deque)
    rlog = _RequestLog(log_path)
    app = FastAPI(title="DuplexJev API", docs_url="/docs")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    class DecideRequest(BaseModel):
        audio_b64: str
        format: str = "wav"
        lang: str = "en"
        questions: Optional[List[dict]] = None
        assistant_said: Optional[str] = None  # the assistant's previous message (turn-taking context)

    def _ip(req: Request) -> str:
        return req.headers.get("x-real-ip") or req.headers.get("x-forwarded-for", "").split(",")[0].strip() or \
            (req.client.host if req.client else "?")

    def _base(req: Request, kind: str) -> dict:
        ip = _ip(req)
        return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, "client": rlog.client(ip),
                "net16": ".".join(ip.split(".")[:2]) if ip.count(".") == 3 else "",
                "ua": req.headers.get("user-agent", "")[:120], "origin": req.headers.get("origin", "")[:80],
                "dj_client": req.headers.get("x-duplexjev-client", "")[:60], "model": client.model}

    def _limit(req: Request):
        ip = _ip(req)
        now, q = time.time(), hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= rate:
            raise HTTPException(429, f"rate limit: {rate} requests per minute")
        q.append(now)

    @app.get("/", response_class=HTMLResponse)
    def page(req: Request):
        rlog.write(_base(req, "page"))
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
        rec = _base(req, "decide")
        rec.update({"lang": body.lang, "format": body.format, "audio_kb": round(len(body.audio_b64) * 0.75 / 1024, 1),
                    "custom_q": bool(body.questions), "said_len": len(body.assistant_said or "")})
        t0 = time.perf_counter()
        try:
            out = _decide(body, req, rec)
            rec.update({"status": 200, "ms": out["ms"],
                        "answers": {k: [v.get("answer"), round(float(v.get("confidence", 0)), 3)]
                                    for k, v in out["answers"].items()}})
            return out
        except HTTPException as e:
            rec.update({"status": e.status_code, "error": str(e.detail)[:200]})
            raise
        except Exception as e:
            rec.update({"status": 500, "error": repr(e)[:200]})
            raise
        finally:
            rec["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            try:
                rlog.write(rec)
            except Exception:
                pass

    def _decide(body, req, rec):
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
        rec["q_ids"] = [q.id for q in qs]
        if body.questions:
            rec["q_texts"] = [str(q.text)[:200] for q in qs]
        rec["clip_s"] = _clip_seconds(audio, body.format)
        t = time.perf_counter()
        try:
            answers = client.decide(audio, qs, assistant_said=(body.assistant_said or "")[-2000:] or None)
        except Exception as e:  # decoding errors, too-long audio etc. come back from vLLM as HTTP 400
            raise HTTPException(400, f"model server rejected the request: {str(e)[:200]}")
        return {"answers": answers, "ms": round((time.perf_counter() - t) * 1000, 1), "model": client.model}

    return app
