"""Iterable over local WenetSpeech Lhotse shards (tar.gz + jsonl.gz).

Yields HuggingFace-style {audio: {array, sampling_rate, path}, text}.
Continuation 只来自 Qwen3-32B 旁路（continuation_store），不编造、不用 Llama 列。
"""
from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
import wave
from pathlib import Path

import numpy as np


def _find_pairs(root: str):
    root_p = Path(root)
    pairs = []
    for base in (root_p / "data", root_p):
        if not base.is_dir():
            continue
        for js in sorted(base.glob("cuts_L_fixed.*.jsonl.gz")):
            tar = js.with_name(js.name.replace(".jsonl.gz", ".tar.gz"))
            if tar.is_file():
                pairs.append((js, tar))
    return pairs


def is_wenet_lhotse(path: str) -> bool:
    return bool(_find_pairs(path))


def _pcm16(raw: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        buf = wf.readframes(wf.getnframes())
        audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
        if sr != 16000 and audio.size > 0:
            new_len = int(round(audio.shape[0] * 16000 / sr))
            if new_len > 0:
                audio = np.interp(
                    np.linspace(0.0, 1.0, new_len, endpoint=False),
                    np.linspace(0.0, 1.0, audio.shape[0], endpoint=False),
                    audio,
                ).astype(np.float32)
            sr = 16000
        return audio, sr


class WenetLhotseIterable:
    def __init__(self, path: str, seed: int = 42):
        self.path = path
        self.seed = seed
        self.pairs = _find_pairs(path)
        if not self.pairs:
            raise FileNotFoundError(f"no cuts_L_fixed jsonl+tar under {path}")
        self.n_shards = len(self.pairs)

    def __iter__(self):
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        pairs = list(self.pairs)
        rng = np.random.default_rng(self.seed + rank)
        rng.shuffle(pairs)
        mine = pairs[rank::world] if world > 1 else pairs
        for js, tar in mine:
            yield from self._iter_shard(js, tar)

    def _iter_shard(self, js: Path, tar: Path):
        by_full, by_base = self._load_texts(js)
        if not by_full and not by_base:
            return
        try:
            tf = tarfile.open(tar, "r:gz")
        except Exception as exc:
            print(f"wenet skip tar {tar}: {exc}", flush=True)
            return
        try:
            # Sequential gzip pass. Random extractfile on .tar.gz re-decompresses
            # from byte 0 and was ~50s/step.
            for member in tf:
                if not member.isfile():
                    continue
                text, src = self._lookup(member.name, by_full, by_base)
                if not text:
                    continue
                try:
                    handle = tf.extractfile(member)
                    if handle is None:
                        continue
                    raw = handle.read()
                    audio, sr = _pcm16(raw)
                except Exception:
                    continue
                if audio.size == 0:
                    continue
                if _c1_excluded(src):
                    continue
                row = {
                    "audio": {"array": audio, "sampling_rate": sr, "path": src},
                    "text": text,
                }
                cont = _qwen_cont(src)
                if cont:
                    row["continuation"] = cont
                yield row
        finally:
            tf.close()

    @staticmethod
    def _load_texts(js: Path):
        by_full = {}
        by_base = {}
        with gzip.open(js, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                text = ""
                for sup in rec.get("supervisions") or []:
                    text = (sup.get("text") or "").strip()
                    if text:
                        break
                if not text:
                    continue
                sources = (rec.get("recording") or {}).get("sources") or []
                if not sources:
                    continue
                src = sources[0].get("source") or ""
                if not src:
                    continue
                name = src[5:] if src.startswith("data/") else src
                by_full[src] = (text, src)
                by_full[name] = (text, src)
                by_base[os.path.basename(src)] = (text, src)
        return by_full, by_base

    @staticmethod
    def _lookup(member_name: str, by_full: dict, by_base: dict):
        hit = by_full.get(member_name) or by_full.get("data/" + member_name)
        if hit:
            return hit
        return by_base.get(os.path.basename(member_name), (None, member_name))


_C1_SEEN: set[str] | None = None


def _c1_excluded(src: str) -> bool:
    global _C1_SEEN
    path = os.environ.get(
        "ML_ASR_C1_EXCLUDE",
        "/data/exp01/exp03_train/artifacts/ml_asr/c1_wenet_seen_paths.txt",
    )
    if _C1_SEEN is None:
        p = Path(path)
        _C1_SEEN = set(p.read_text().splitlines()) if p.is_file() else set()
    if src in _C1_SEEN:
        return True
    alt = src[5:] if src.startswith("data/") else "data/" + src
    return alt in _C1_SEEN


def _qwen_cont(src: str):
    try:
        from continuation_store import get as _get
    except ImportError:
        try:
            from ultravox.data.configs.continuation_store import get as _get
        except ImportError:
            return None
    return _get(src)
