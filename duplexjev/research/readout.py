"""共享音频前缀上的闭集 1-token 读出。不调用 generate。

协议 L：沿用 bench 封闭题问法，闭集建在 gold 标签的首 token 上。
协议 S：schema 进 prompt，闭集建在置换后的字母/数字槽上。

批量：同一 forward 里堆 N 行。音频特征按行复制（prefix_sharing=batch_repeat）。
真正的 KV 展开留到延迟数字要写进论文时再做；本文件把口径写清楚，不装成已经共享 KV。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _add_bench_path():
    cands = [
        os.path.join(os.path.dirname(HERE), "bench"),
        os.path.join(os.path.dirname(HERE), "scripts_bench"),
        "/data/exp01/exp03_train/scripts_bench",
    ]
    for p in cands:
        if os.path.isdir(p):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
    raise RuntimeError(f"找不到 bench 脚本目录，试过 {cands}")


_add_bench_path()

import bench_questions as BQ  # noqa: E402
import run_bench as RB  # noqa: E402

from .render_letter import (  # noqa: E402
    assert_qid_absent,
    gate_single_tokens,
    gold_slot,
    option_ids_of,
    render_user_text,
    slots_for_question,
)

SAMPLE_RATE = 16000


def load_audio(path: str, t_ms: int | None) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if t_ms is not None:
        n = int(round(t_ms / 1000.0 * SAMPLE_RATE))
        audio = audio[: max(n, 1)]
    return audio


def protocol_l_question(item: dict, condition: str = "audio") -> str:
    task = item.get("task")
    if condition == "text_only":
        text = item.get("text") or ""
        if task in BQ.TEXT_ONLY:
            return BQ.TEXT_ONLY[task]["canonical"].format(text=text)
        q = next(iter(item["state"]["questions"].values()))
        return f"文字记录：「{text}」。{q['instructions']}"
    if task in BQ.AUDIO:
        return BQ.AUDIO[task]["canonical"]
    q = next(iter(item["state"]["questions"].values()))
    instr = q["instructions"].rstrip()
    if "<|audio|>" not in instr:
        instr = f"{instr} <|audio|>"
    return instr


def text_only_state(item: dict) -> dict:
    st = dict(item["state"]["state"])
    st["transcript"] = item.get("text") or ""
    return st


def protocol_l_options(item: dict) -> list[str]:
    q = next(iter(item["state"]["questions"].values()))
    return option_ids_of(q)


def protocol_s_pack(item: dict, seed: int, condition: str = "audio") -> tuple[str, dict[str, str], str]:
    qid, q = next(iter(item["state"]["questions"].items()))
    slots = slots_for_question(item["id"], qid, q, seed=seed)
    with_audio = condition != "text_only"
    st = text_only_state(item) if condition == "text_only" else item["state"]["state"]
    text = render_user_text(st, q, slots, with_audio=with_audio)
    assert_qid_absent(text, item["id"], qid)
    gslot = gold_slot(q, item["gold"], slots)
    return text, slots, gslot


def _to_device(batch: dict, model, device) -> dict:
    out = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    if "audio_values" in out:
        out["audio_values"] = out["audio_values"].to(model.dtype)
    return out


def encode_one(processor, tok, question: str, audio: np.ndarray | None):
    prompt = RB.build_prompt(tok, question)
    kw = {} if audio is None else {"audio": audio, "sampling_rate": SAMPLE_RATE}
    p = processor(text=prompt, **kw)
    return p, int(p["input_ids"].shape[1])


def pad_batch(encoded: list[dict], device, model) -> dict:
    keys = set(encoded[0])
    out = {}
    for k in keys:
        vs = [e[k] for e in encoded]
        if not hasattr(vs[0], "shape"):
            continue
        if k == "input_ids":
            pad_id = 0
            maxlen = max(v.shape[-1] for v in vs)
            rows = []
            masks = []
            for v in vs:
                t = v.view(-1)
                pad = maxlen - t.numel()
                # 右 pad：最后有效 token 位置 = 原长度-1
                rows.append(F.pad(t, (0, pad), value=pad_id))
                masks.append(torch.cat([torch.ones(t.numel(), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
            out["input_ids"] = torch.stack(rows)
            out["attention_mask"] = torch.stack(masks)
        elif k == "attention_mask":
            continue
        else:
            # 音频特征长度必须一致；延迟曲线用同一条音频，质量评测按条走
            shapes = {tuple(v.shape) for v in vs}
            if len(shapes) != 1:
                raise ValueError(f"{k} 形状不一致 {shapes}，不能进同一 batch")
            out[k] = torch.cat(vs, dim=0) if vs[0].dim() >= 1 and vs[0].shape[0] == 1 else torch.stack(vs)
    return _to_device(out, model, device)


@torch.no_grad()
def closed_set_logits(model, logits_last: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """logits_last: [B, V] → [B, K] softmax。"""
    idx = torch.tensor(token_ids, device=logits_last.device, dtype=torch.long)
    return F.softmax(logits_last.index_select(-1, idx).float(), dim=-1)


@torch.no_grad()
def forward_last_logits(model, batch: dict) -> torch.Tensor:
    out = model(**batch)
    logits = out.logits.float()
    if "attention_mask" in batch:
        last = batch["attention_mask"].sum(dim=1) - 1
    else:
        last = torch.full((logits.size(0),), logits.size(1) - 1, device=logits.device)
    b = torch.arange(logits.size(0), device=logits.device)
    return logits[b, last]


def option_token_ids(tok, labels: list[str], loc: str) -> list[int]:
    ids = []
    for lab in labels:
        enc = tok.encode(lab, add_special_tokens=False)
        if not enc:
            raise RuntimeError(f"{loc}: 空 encode {lab!r}")
        ids.append(int(enc[0]))
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{loc}: 选项首 token 碰撞 {list(zip(labels, ids))}")
    return ids


def read_item_l(model, processor, tok, item: dict, device: str, audio: np.ndarray | None, condition: str = "audio"):
    q = protocol_l_question(item, condition=condition)
    opts = protocol_l_options(item)
    enc, n_prompt = encode_one(processor, tok, q, audio)
    batch = _to_device(enc, model, device)
    last = forward_last_logits(model, batch)
    tids = option_token_ids(tok, opts, item["id"])
    probs = closed_set_logits(model, last, tids)[0]
    pred_i = int(probs.argmax())
    return {
        "id": item["id"],
        "protocol": "L",
        "gold": item["gold"],
        "pred": opts[pred_i],
        "correct": opts[pred_i] == item["gold"],
        "probs": {o: float(p) for o, p in zip(opts, probs.tolist())},
        "n_prompt": n_prompt,
        "skipped": False,
    }


def read_item_s(model, processor, tok, item: dict, device: str, audio: np.ndarray | None, seed: int, condition: str = "audio"):
    text, slots, gslot = protocol_s_pack(item, seed, condition=condition)
    labels = list(slots)
    gate_single_tokens(tok, labels, item["id"])
    enc, n_prompt = encode_one(processor, tok, text, audio)
    batch = _to_device(enc, model, device)
    last = forward_last_logits(model, batch)
    tids = option_token_ids(tok, labels, item["id"])
    probs = closed_set_logits(model, last, tids)[0]
    pred_i = int(probs.argmax())
    pred_slot = labels[pred_i]
    pred_opt = slots[pred_slot]
    gold_opt = item["gold"]
    if next(iter(item["state"]["questions"].values()))["type"] == "boolean":
        gold_opt = "true" if item["gold"] in (True, "true") else "false"
    elif next(iter(item["state"]["questions"].values()))["type"] == "score":
        gold_opt = str(int(item["gold"]))
    slot_hist = {s: float(p) for s, p in zip(labels, probs.tolist())}
    return {
        "id": item["id"],
        "protocol": "S",
        "gold": gold_opt,
        "gold_slot": gslot,
        "pred": pred_opt,
        "pred_slot": pred_slot,
        "correct": pred_opt == gold_opt,
        "probs": {slots[s]: slot_hist[s] for s in labels},
        "slot_probs": slot_hist,
        "slots": slots,
        "n_prompt": n_prompt,
        "skipped": False,
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("skipped")]
    n = len(ok)
    acc = sum(1.0 for r in ok if r["correct"]) / n if n else None
    slot_hist = {}
    for r in ok:
        if "pred_slot" in r:
            slot_hist[r["pred_slot"]] = slot_hist.get(r["pred_slot"], 0) + 1
    pred_hist = {}
    for r in ok:
        pred_hist[str(r["pred"])] = pred_hist.get(str(r["pred"]), 0) + 1
    return {
        "n": n,
        "skipped": sum(1 for r in rows if r.get("skipped")),
        "accuracy": None if acc is None else round(acc, 4),
        "pred_hist": pred_hist,
        "slot_hist": slot_hist,
        "majority_pred_rate": (
            round(max(pred_hist.values()) / sum(pred_hist.values()), 4) if pred_hist else None
        ),
    }


def time_batch_repeat(model, processor, tok, item: dict, audio: np.ndarray, device: str, n: int, seed: int, warmup: int = 2):
    """同一条音频、同一题重复 N 次，量 batch 维扩展。prefix_sharing=batch_repeat。"""
    text, slots, _ = protocol_s_pack(item, seed, condition="audio")
    labels = list(slots)
    gate_single_tokens(tok, labels, item["id"])
    enc, _ = encode_one(processor, tok, text, audio)
    encoded = [enc] * n
    batch = pad_batch(encoded, device, model)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    for _ in range(warmup):
        forward_last_logits(model, batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last = forward_last_logits(model, batch)
    _ = closed_set_logits(model, last, option_token_ids(tok, labels, item["id"]))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
    return {
        "n": n,
        "forward_passes": 1,
        "autoregressive_decode_steps": 0,
        "prefix_sharing": "batch_repeat",
        "wall_s": round(dt, 4),
        "questions_per_s": round(n / dt, 2) if dt > 0 else None,
        "peak_mem_bytes": peak,
        "note": "音频特征按行复制，不是 KV cache 共享；写论文延迟时必须保留这句。",
    }
