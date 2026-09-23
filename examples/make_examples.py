#!/usr/bin/env python3
"""Generate docs/examples.json (+ docs/audio/*.wav) for the demo page from a released checkpoint.

Every question for a clip is answered as a single constrained token; all questions of one clip run in ONE
batched forward pass (batch-repeat; the paper's packed prefix-sharing gives the same answers).

Usage (on the training server, 1 GPU):
  python make_examples.py --ckpt <checkpoint dir> --clips examples_config.json --out ../../docs \
         [--engine "PyTorch eager, 1x H200, Qwen3-32B"]

examples_config.json:
  {"clips": [{"id": "ex1", "label": "EN · pause mid-request", "lang": "en",
              "audio": "/path/to/clip.wav", "transcript": "...",
              "source": "LibriSpeech test-clean (CC-BY-4.0)"}, ...],
   "questions": [{"en": ["question", [options]], "zh": ["问题", [选项]]}, ...]}   # optional; defaults below

Only use clips whose license allows redistribution (LibriSpeech CC-BY-4.0, Common Voice CC0, AISHELL-1 Apache-2.0,
our own recordings with consent). Never use production call audio or ODSQA audio.
"""
import argparse, json, os, random, shutil, sys, time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "duplexjev"))
import run_bench as RB  # noqa: E402  (load_model, build_prompt)

LETTERS = "ABCDEFGH"
# Each question: text shown/asked in English and Chinese. The model is asked in the clip's language;
# the page shows the question in the reader's UI language.
DEFAULT_Q = [
    {"en": ("Has the user finished the turn?", ["finished", "not finished"]),
     "zh": ("用户这句话说完了吗？", ["说完了", "没说完"])},
    {"en": ("What does the user want?", ["climate", "media", "navigation", "phone", "vehicle info", "chit-chat"]),
     "zh": ("用户想做什么？", ["空调", "媒体", "导航", "电话", "车辆信息", "闲聊"])},
    {"en": ("Which filler fits?", ["“Sure —”", "“One moment —”", "“On it, searching now —”", "(stay silent)"]),
     "zh": ("先说哪句垫话？", ["“好的，”", "“稍等，”", "“马上为您查找，”", "（不说话）"])},
    {"en": ("Speaker gender", ["female", "male"]), "zh": ("说话人性别", ["女性", "男性"])},
    {"en": ("Language", ["English", "Chinese", "other"]), "zh": ("语言", ["英语", "中文", "其他"])},
    {"en": ("Urgency (1–5)", ["1", "2", "3", "4", "5"]), "zh": ("紧急程度（1–5）", ["1", "2", "3", "4", "5"])},
    {"en": ("Sentiment", ["positive", "neutral", "negative"]), "zh": ("情绪", ["正面", "中性", "负面"])},
    {"en": ("Needs a human agent?", ["yes", "no"]), "zh": ("需要转人工吗？", ["需要", "不需要"])},
]
TEMPLATE = {
    "en": "<|audio|>\n\nQuestion: {q}\n\nOptions:\n{opts}\n\nAnswer with only the letter of the correct option.",
    "zh": "<|audio|>\n\n问题：{q}\n\n选项：\n{opts}\n\n请只回答正确选项的字母。",
}


def load16(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


@torch.no_grad()
def answer_all(model, processor, tok, audio, lang, questions, device, seed):
    rng = random.Random(seed)
    encs, perms, letter_ids = [], [], []
    for q in questions:
        k = len(q["options"])
        perm = list(range(k)); rng.shuffle(perm)            # letter i shows option perm[i]
        opts = "\n".join(f"{LETTERS[i]}. {q['options'][perm[i]]}" for i in range(k))
        text = TEMPLATE[lang].format(q=q["q"], opts=opts)
        enc = processor(text=RB.build_prompt(tok, text), audio=audio, sampling_rate=16000)
        encs.append(enc); perms.append(perm)
        letter_ids.append([tok.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(k)])
    # one batched forward pass (right padding)
    L = max(e["input_ids"].shape[1] for e in encs)
    batch = {}
    for key in encs[0]:
        vals = [e[key] for e in encs]
        if not hasattr(vals[0], "shape"):
            continue
        if key in ("input_ids", "attention_mask"):
            pad = 0
            batch[key] = torch.cat([F.pad(v, (0, L - v.shape[1]), value=pad) for v in vals])
        else:
            batch[key] = torch.cat(vals)
    batch = {k: v.to(device) for k, v in batch.items()}
    if "audio_values" in batch:
        batch["audio_values"] = batch["audio_values"].to(model.dtype)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    logits = model(**batch).logits.float()
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
    last = batch["attention_mask"].sum(1) - 1
    out = []
    for j, q in enumerate(questions):
        z = logits[j, last[j], letter_ids[j]]
        p_letter = torch.softmax(z, -1).tolist()
        p = [0.0] * len(q["options"])
        for i, opt_idx in enumerate(perms[j]):
            p[opt_idx] = round(p_letter[i], 4)
        out.append(p)
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True, help="docs/ folder")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--engine", default="PyTorch eager, 1x H200, Qwen3-32B")
    args = ap.parse_args()
    cfg = json.load(open(args.clips, encoding="utf-8"))
    Q = cfg.get("questions") or DEFAULT_Q
    model, processor = RB.load_model(args.ckpt, args.device)
    tok = processor.tokenizer
    os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
    examples, lat = [], []
    for n, c in enumerate(cfg["clips"]):
        a = load16(c["audio"])
        qs = [{"q": q[c["lang"]][0], "options": q[c["lang"]][1]} for q in Q]
        answer_all(model, processor, tok, a, c["lang"], qs, args.device, n)  # warm-up
        ps, ms = answer_all(model, processor, tok, a, c["lang"], qs, args.device, n)
        lat.append(ms)
        dst = f"audio/{c['id']}.wav"
        sf.write(os.path.join(args.out, dst), a, 16000, subtype="PCM_16")
        examples.append({
            "id": c["id"], "label": c["label"], "lang": c["lang"], "source": c["source"],
            "transcript": c["transcript"], "label_zh": c.get("label_zh", c["label"]), "audio": dst, "duration_s": round(len(a) / 16000, 2), "seed": n + 1,
            "questions": [{"q_en": Q[k]["en"][0], "q_zh": Q[k]["zh"][0],
                           "options_en": Q[k]["en"][1], "options_zh": Q[k]["zh"][1], "p": p}
                          for k, p in enumerate(ps)],
            "forward_ms": round(ms, 1),
        })
        print(c["id"], round(ms, 1), "ms", flush=True)
    doc = {"placeholder": False, "checkpoint": os.path.basename(os.path.normpath(args.ckpt)),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "examples": examples,
           "pass": {"forward_ms": round(float(np.median(lat)), 1), "decode_steps": 0, "engine": args.engine}}
    json.dump(doc, open(os.path.join(args.out, "examples.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", os.path.join(args.out, "examples.json"))


if __name__ == "__main__":
    main()
