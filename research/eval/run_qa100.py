#!/usr/bin/env python3
"""qa100：同一 32B，文本 vs 语音，100 题一次 forward 批量出 1-token 字母答案。不 generate 决策。

条件：
  text          题干文字 + 选项，batch 100（baseline）+ 逐题单跑核对
  audio         题干换成 TTS 语音、选项仍是文字，batch 100（特征维 pad）+ 逐题单跑核对
  options_only  不给题干，只给选项：猜测下界
  asr           对 TTS 语音 generate 转写，CER：区分「TTS 念错」与「听不懂」

用法：
  CUDA_VISIBLE_DEVICES=1 venv/bin/python qa100/run_qa100.py --loader official \\
      --ckpt /data/exp01/models/ultravox-v0_6-qwen-3-32b --out artifacts/qa100/official.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = "/data/exp01/exp03_train"
for p in (ROOT, os.path.join(ROOT, "jev_qwen"), os.path.join(ROOT, "scripts_bench")):
    sys.path.insert(0, p)

import run_bench as RB  # noqa: E402
from jev_qwen.probe_batch_limit import pad_audio_tensors  # noqa: E402
from jev_qwen.readout import (  # noqa: E402
    _to_device,
    encode_one,
    forward_last_logits,
    load_audio,
    option_token_ids,
    pad_batch,
)
from jev_qwen.render_letter import gate_single_tokens  # noqa: E402

LETTERS = list("ABCD")
PROMPTS = {
    "zh": {
        "text": "问题：{q}\n\n选项：\n{opts}\n\n请只回答正确选项的字母（A、B、C 或 D）。",
        "audio": "<|audio|>\n\n请听上面这段录音里提出的问题。\n\n选项：\n{opts}\n\n请只回答正确选项的字母（A、B、C 或 D）。",
        "options_only": "下面是一道选择题的选项，题目没有给出。\n\n选项：\n{opts}\n\n请猜一个最可能正确的选项，只回答字母（A、B、C 或 D）。",
        "asr": "请把这段录音的内容逐字转写出来，只输出转写文字。<|audio|>",
    },
    "en": {
        "text": "Question: {q}\n\nOptions:\n{opts}\n\nAnswer with only the letter of the correct option (A, B, C, or D).",
        "audio": "<|audio|>\n\nListen to the question asked in the audio above.\n\nOptions:\n{opts}\n\nAnswer with only the letter of the correct option (A, B, C, or D).",
        "options_only": "Below are the options of a multiple-choice question; the question itself is not given.\n\nOptions:\n{opts}\n\nGuess the most likely correct option. Answer with only the letter (A, B, C, or D).",
        "asr": "Please transcribe this audio verbatim. Output only the transcript. <|audio|>",
    },
}


GENDER_PROMPTS = {
    "zh": {
        "text": "文字记录：「{q}」\n\n请判断说这句话的人的性别。\n\n选项：\n{opts}\n\n请只回答正确选项的字母（A 或 B）。",
        "audio": "<|audio|>\n\n请判断上面这段录音中说话人的性别。\n\n选项：\n{opts}\n\n请只回答正确选项的字母（A 或 B）。",
    },
    "en": {
        "text": "Transcript: \"{q}\"\n\nWhat is the gender of the speaker?\n\nOptions:\n{opts}\n\nAnswer with only the letter of the correct option (A or B).",
        "audio": "<|audio|>\n\nWhat is the gender of the speaker in the audio above?\n\nOptions:\n{opts}\n\nAnswer with only the letter of the correct option (A or B).",
    },
}


ZJU_PROMPT = ("Listen to the audio and identify the perceived gender of the main speaker's voice. "
              "Return exactly one lowercase label: male or female.")
ZJU_PROMPTS = {lang: {"audio": "<|audio|>\n\n" + ZJU_PROMPT,
                      "text": "Transcript: \"{q}\"\n\n" + ZJU_PROMPT.replace("Listen to the audio and identify", "Identify")}
               for lang in ("zh", "en")}


def use_zju_task():
    """浙大 audio-gender-benchmark 原版提示词（configs/local_python.json），闭集 = male / female 两个 token。"""
    global LETTERS, PROMPTS
    LETTERS = ["male", "female"]
    PROMPTS = ZJU_PROMPTS


def use_gender_task():
    """gender100：2 选项、文本条件只给转写（最小对，按构造≈0.5）、无 options_only / asr。"""
    global LETTERS, PROMPTS
    LETTERS = list("AB")
    PROMPTS = GENDER_PROMPTS


def prompt_of(it, cond):
    opts = "\n".join(f"{L}. {it['options'][L]}" for L in LETTERS)
    return PROMPTS[it["lang"]][cond].format(q=it["question"], opts=opts)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode(it, last_row, tids):
    probs = torch.softmax(last_row.index_select(-1, tids).float(), dim=-1)
    i = int(probs.argmax())
    return {"id": it["id"], "pred": LETTERS[i], "gold": it["gold"], "correct": LETTERS[i] == it["gold"],
            "p_pred": round(float(probs[i]), 4), "probs": {L: round(float(p), 4) for L, p in zip(LETTERS, probs.tolist())}}


def encode_all(processor, tok, items, cond, audios):
    enc = []
    for it in items:
        a = audios[it["id"]] if cond == "audio" else None
        e, _ = encode_one(processor, tok, prompt_of(it, cond), a)
        enc.append(dict(e))
    return enc


@torch.no_grad()
def _one_chunk_forward(model, processor, tok, chunk, cond, audios, device, tids, repeats):
    enc = encode_all(processor, tok, chunk, cond, audios)
    if cond == "audio":
        enc = pad_audio_tensors(enc)
    batch = pad_batch(enc, device, model)
    forward_last_logits(model, batch)  # warmup
    sync()
    walls, last = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = forward_last_logits(model, batch)
        sync()
        walls.append(time.perf_counter() - t0)
    rows = [decode(it, last[i], tids) for i, it in enumerate(chunk)]
    del batch
    return rows, float(np.median(walls))


def run_batch(model, processor, tok, items, cond, audios, device, tids, repeats=3, chunk_size=200):
    """100 题左右一次 forward；包更大时按 chunk_size 分块，仍是「一次 forward 读 N 题」的
    批量口径，只是大包拆成几次调用，避免变长真实语音 padding 后单次 forward 把显存打爆。"""
    rows = []
    walls = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        crows, wall = _one_chunk_forward(model, processor, tok, chunk, cond, audios, device, tids, repeats)
        rows.extend(crows)
        walls.append((len(chunk), wall))
        torch.cuda.empty_cache()
    n_chunks = len(walls)
    total_wall = sum(w for _, w in walls)
    timing = {
        "n": len(items), "forward_passes": n_chunks, "chunk_size": chunk_size,
        "wall_ms_median": round(total_wall / n_chunks * 1000, 1) if n_chunks else None,
        "ms_per_question": round(total_wall * 1000 / len(items), 2) if items else None,
        "repeats": repeats, "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "note": "包 n 大于 chunk_size 时拆成多次 forward；ms_per_question 是总墙钟/总题数",
    }
    return rows, timing


@torch.no_grad()
def run_seq(model, processor, tok, items, cond, audios, device, tids):
    rows, t0 = [], time.perf_counter()
    for it in items:
        a = audios[it["id"]] if cond == "audio" else None
        e, _ = encode_one(processor, tok, prompt_of(it, cond), a)
        last = forward_last_logits(model, _to_device(e, model, device))
        rows.append(decode(it, last[0], tids))
    sync()
    return rows, {"n": len(items), "forward_passes": len(items),
                  "wall_ms_total": round((time.perf_counter() - t0) * 1000, 1)}


def run_asr(model, processor, tok, items, audios, device):
    rows = []
    for it in items:
        gen = RB.generate(model, processor, tok, prompt_of(it, "asr"), audios[it["id"]], device, 96)
        rows.append({"id": it["id"], "lang": it["lang"], "ref": it["question"], "hyp": gen,
                     "cer": RB.err_rate(it["question"], gen, it["lang"])})
    return rows


def acc_table(items, rows):
    meta = {it["id"]: it for it in items}
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        it = meta[r["id"]]
        for k in ("all", it["lang"], it["category"], f"{it['lang']}-{it['category']}"):
            by[k][0] += int(r["correct"])
            by[k][1] += 1
    hist = defaultdict(int)
    for r in rows:
        hist[r["pred"]] += 1
    out = {k: {"ok": a, "n": n, "acc": round(a / n, 4)} for k, (a, n) in sorted(by.items())}
    out["pred_hist"] = dict(sorted(hist.items()))
    return out


def agree(a, b):
    m = {r["id"]: r["pred"] for r in b}
    same = sum(r["pred"] == m[r["id"]] for r in a)
    return {"same": same, "n": len(a), "diff_ids": [r["id"] for r in a if r["pred"] != m[r["id"]]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", choices=["official", "fdx"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", default=f"{ROOT}/artifacts/qa100/qa100.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-text", action="store_true", help="文本与模型无关（同一冻结 32B），第二个模型可跳过")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--task", choices=["qa", "gender", "zju"], default="qa")
    args = ap.parse_args()
    if args.task == "gender":
        use_gender_task()
        args.skip_asr = True
    elif args.task == "zju":
        use_zju_task()
        args.skip_asr = True

    doc = json.load(open(args.pack, encoding="utf-8"))
    items = doc["items"]
    assert all("audio" in it for it in items), "题包没有 audio 字段，先跑 synth_qa100.py"
    if args.loader == "official":
        from jev_qwen.load_official import load_official

        model, processor = load_official(args.ckpt, args.device)
    else:
        model, processor = RB.load_model(args.ckpt, args.device)
    tok = processor.tokenizer
    RB.assert_template_alignment(tok)
    gate_single_tokens(tok, LETTERS, "qa100")
    tids = torch.tensor(option_token_ids(tok, LETTERS, "qa100"), device=args.device)
    audios = {it["id"]: load_audio(it["audio"], None) for it in items}

    res = {"task": args.task, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "loader": args.loader,
           "ckpt": args.ckpt, "pack_sha256": doc.get("pack_sha256"), "tts": doc.get("tts"),
           "prompts": PROMPTS, "autoregressive_decode_steps": 0, "conditions": {}}
    conds = ([] if args.skip_text else (["text"] if args.task in ("gender", "zju") else ["text", "options_only"])) + ["audio"]
    for cond in conds:
        print(f"== {cond} batch", flush=True)
        brows, timing = run_batch(model, processor, tok, items, cond, audios, args.device, tids)
        blk = {"batch": {"timing": timing, "score": acc_table(items, brows), "rows": brows}}
        if cond != "options_only":
            print(f"== {cond} sequential", flush=True)
            srows, stiming = run_seq(model, processor, tok, items, cond, audios, args.device, tids)
            blk["sequential"] = {"timing": stiming, "score": acc_table(items, srows), "rows": srows}
            blk["batch_vs_sequential"] = agree(brows, srows)
        res["conditions"][cond] = blk
        s = blk["batch"]["score"]
        cat_bits = "  ".join(f"{k} {v['acc']}" for k, v in s.items()
                             if k not in ("all", "pred_hist") and "-" not in k)
        print(f"   {cond} batch acc {s['all']['acc']}  {cat_bits}  {timing}  "
              f"agree {blk.get('batch_vs_sequential', {}).get('same')}", flush=True)
    if not args.skip_asr:
        print("== asr", flush=True)
        rows = run_asr(model, processor, tok, items, audios, args.device)
        cer = defaultdict(list)
        for r in rows:
            if r["cer"] is not None:
                cer[r["lang"]].append(r["cer"])
                cer["all"].append(r["cer"])
        res["asr"] = {"cer_mean": {k: round(float(np.mean(v)), 4) for k, v in cer.items()},
                      "n_cer_gt_0.3": sum(1 for r in rows if (r["cer"] or 0) > 0.3), "rows": rows}
        print("   asr", res["asr"]["cer_mean"], "bad", res["asr"]["n_cer_gt_0.3"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
