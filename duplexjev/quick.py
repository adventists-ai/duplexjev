"""One call from an audio clip to a table of decisions.

    from duplexjev import quick
    quick("call.wav")                                   # default table of 10 decisions, printed and returned
    quick("call_zh.wav", lang="zh")                     # Chinese speech: Chinese questions
    quick("call.wav", questions=my_table)               # your own fixed option groups

Where the model runs (first match wins):
    api="https://…/duplexjev"   a DuplexJev gateway (our hosted trial API or your own `duplexjev gateway`)
    vllm="http://host:8000/v1"  your own `vllm serve adventists-ai/DuplexJev-4B-Para`
    model="adventists-ai/…"     local PyTorch: a complete model (default DuplexJev-4B-Para) or a connector checkpoint
                                (needs `duplexjev[speech]` and a GPU)
    none of these               $DUPLEXJEV_API, then $DUPLEXJEV_VLLM, then the local default model
"""
from __future__ import annotations

import os
import time
from typing import Any

from .table import default_table, format_table

DEFAULT_LOCAL_MODEL = "adventists-ai/DuplexJev-4B-Para"
_cache: dict = {}


def _backend(api, vllm, model, api_key):
    api = api or (None if (vllm or model) else os.environ.get("DUPLEXJEV_API"))
    vllm = vllm or (None if (api or model) else os.environ.get("DUPLEXJEV_VLLM"))
    key = (api, vllm, model)
    if key in _cache:
        return _cache[key]
    if api:
        from .remote import ApiClient

        b = ("api", ApiClient(api, api_key=api_key or os.environ.get("DUPLEXJEV_API_KEY")))
    elif vllm:
        from .remote import VLLMClient

        b = ("vllm", VLLMClient(vllm))
    else:
        from .decider import Decider

        b = ("local", Decider.from_pretrained(model or DEFAULT_LOCAL_MODEL))
    _cache[key] = b
    return b


def quick(audio: Any, *, lang: str = "en", questions=None, api: str | None = None, vllm: str | None = None,
          model: str | None = None, api_key: str | None = None, show: bool = True) -> dict:
    """Answer the default decision table (or ``questions``) about one clip.

    Returns ``{id: {"answer", "confidence", "probs"}}`` and, with ``show=True``, prints it as a table.
    """
    kind, b = _backend(api, vllm, model, api_key)
    qs = questions if questions is not None else default_table(lang)
    t = time.perf_counter()
    if kind == "api":
        res = b.decide(audio, questions, lang=lang)  # the gateway applies its default table when questions is None
    else:
        res = b.decide(audio, qs)
    ms = (time.perf_counter() - t) * 1000
    if show:
        print(format_table(res, lang=lang, ms=ms))
    return res
