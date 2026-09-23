"""Qwen3-32B 续写旁路。训练时覆盖 parquet 里的 Llama-8B continuation。"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_DIR = "/data/exp01/exp03_train/artifacts/ml_asr/cont_qwen32b"
_CACHE: dict[str, str] | None = None


def store_dir() -> Path:
    return Path(os.environ.get("ML_ASR_CONT_DIR", DEFAULT_DIR))


def load() -> dict[str, str]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    out: dict[str, str] = {}
    root = store_dir()
    if root.is_dir():
        for p in sorted(root.glob("*.jsonl")):
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    path = rec.get("path") or rec.get("audio_path") or ""
                    text = (rec.get("continuation") or "").strip()
                    if path and text:
                        out[path] = text
    _CACHE = out
    return out


def get(path: str) -> str | None:
    if not path:
        return None
    store = load()
    if path in store:
        return store[path]
    # wenet jsonl 带 data/ 前缀，tar member 不带
    alt = path[5:] if path.startswith("data/") else "data/" + path
    return store.get(alt)
