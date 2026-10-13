"""DuplexJev adapter for LiveKit's eot-bench, configured exactly like the LiveKit / Pipecat plugins.

- question: "Has the user finished speaking?" (4 options, same wording as the plugins)
- p_eot = P(finished) + P(backchannel): a short "ok" / "yeah" with nobody else talking also ends the turn
- context: the last assistant message in ``messages`` (last 300 characters) is sent as ``assistant_said``,
  as the LiveKit plugin does via ``detector.attach(session)``. The user's own words are NOT used:
  the model only hears the user's audio. Set ``DUPLEXJEV_CONTEXT=0`` for audio only.

    export DUPLEXJEV_GATEWAY=http://localhost:8420 DUPLEXJEV_MODEL_LABEL="DuplexJev-32B-Para v1.2"
    PYTHONPATH=/path/to/duplexjev/benchmarks/turn_detection eot-harness predict --path livekit/eot-bench-data \
        --name en --split validation --adapter eot_bench.duplexjev_ctx_adapter:DuplexJevContextAdapter \
        --batch-size 64 --output-dir output

Needs ``duplexjev gateway`` >= 0.6 (older gateways ignore ``assistant_said``).
"""
from __future__ import annotations

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import GATEWAY, SAMPLE_RATE, TURN_QUESTION, _session, resample, wav_bytes  # noqa: E402


def _assistant_said(messages) -> str:
    prev = [m.get("content") for m in (messages or []) if m.get("role") == "assistant" and m.get("content")]
    return str(prev[-1])[-300:] if prev else ""


class DuplexJevContextAdapter:
    score_point = 0.2
    max_audio_sec = 8.0

    def __init__(self) -> None:
        label = os.environ.get("DUPLEXJEV_MODEL_LABEL", "DuplexJev-32B-Para v1.2")
        self.display_name = label
        self.adapter_id = label.lower().replace(" ", "-")
        self.use_context = os.environ.get("DUPLEXJEV_CONTEXT", "1") != "0"
        self._pool = ThreadPoolExecutor(max_workers=int(os.environ.get("DUPLEXJEV_CONCURRENCY", "32")))

    def supports_language(self, lang_code: str) -> bool:
        return str(lang_code).lower() in ("en", "zh")

    def _one(self, item) -> float:
        lang = "zh" if str(item.get("language", "en")).lower() == "zh" else "en"
        q = TURN_QUESTION[lang]
        a = item["audio"]
        x = resample(np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"]))
        if len(x) < SAMPLE_RATE // 10:
            x = np.pad(x, (SAMPLE_RATE // 10 - len(x), 0))
        payload = {"audio_b64": base64.b64encode(wav_bytes(x)).decode(), "format": "wav", "lang": q["lang"],
                   "questions": [q]}
        said = _assistant_said(item.get("messages")) if self.use_context else ""
        if said:
            payload["assistant_said"] = said
        for attempt in range(4):
            try:
                r = _session.post(GATEWAY + "/v1/decide", json=payload, timeout=60)
                r.raise_for_status()
                p = r.json()["answers"]["turn"]["probs"]
                return float(p.get(q["options"][0], 0.0)) + float(p.get(q["options"][2], 0.0))
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.0 + attempt)
        raise RuntimeError("unreachable")

    def predict_batch(self, batch):
        return list(self._pool.map(self._one, batch))
