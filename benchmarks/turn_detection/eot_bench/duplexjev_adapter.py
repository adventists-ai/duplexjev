"""DuplexJev adapter for LiveKit's eot-bench (https://github.com/livekit/eot-bench).

The harness cuts the user's audio at each decision point (every 0.1 s inside every pause) and asks for p_eot.
DuplexJev answers "Has the user finished speaking?" from the audio alone (the transcript in ``messages`` is not
used), and p_eot is the probability of the "finished" option -- exactly what the LiveKit and Pipecat plugins use.

    pip install -e /path/to/eot-bench  &&  pip install requests scipy
    export DUPLEXJEV_GATEWAY=http://localhost:8420  DUPLEXJEV_MODEL_LABEL="DuplexJev-4B-Para"
    PYTHONPATH=/path/to/duplexjev/benchmarks/turn_detection eot-harness predict --path livekit/eot-bench-data --name en \
        --split validation --adapter eot_bench.duplexjev_adapter:DuplexJevAdapter --batch-size 32 --output-dir output

DuplexJev is trained on English and Chinese, so only ``en`` and ``zh`` are reported (``supports_language``).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import p_finished, resample  # noqa: E402  (benchmarks/turn_detection on PYTHONPATH)


class DuplexJevAdapter:
    score_point = 0.2      # same as the SmartTurn adapter: score the decision 200 ms into the pause
    max_audio_sec = 8.0    # the harness keeps the last 8 s before the decision point

    def __init__(self) -> None:
        label = os.environ.get("DUPLEXJEV_MODEL_LABEL", "DuplexJev-4B-Para")
        self.display_name = label
        self.adapter_id = label.lower().replace(" ", "-")
        self.workers = int(os.environ.get("DUPLEXJEV_CONCURRENCY", "16"))
        self._pool = ThreadPoolExecutor(max_workers=self.workers)

    def supports_language(self, lang_code: str) -> bool:
        return str(lang_code).lower() in ("en", "zh")

    def _one(self, item) -> float:
        audio = item["audio"]
        x = resample(np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"]))
        return p_finished(x, str(item.get("language", "en")).lower())

    def predict_batch(self, batch):
        return list(self._pool.map(self._one, batch))
