#!/usr/bin/env python3
"""冻结 Qwen3-32B 按官方模板写 continuation。不用 Llama-8B 列。

官方：Continue the following text using less than 50 words:
Qwen3 关 thinking。多卡：每张卡一份 --rank/--world。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

USER = "Continue the following text using less than 50 words:\n\n{text}"
MODEL = "/data/exp01/models/Qwen3-32B"
OUT_DIR = Path("/data/exp01/exp03_train/artifacts/ml_asr/cont_qwen32b")


def build_prompt(tok, text: str) -> str:
    msg = USER.format(text=text.strip())
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": msg}], **kwargs)


def strip_cont(text: str) -> str:
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[-1].strip()
    return " ".join(s.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()
    man = Path(args.manifest)
    if args.out:
        out = Path(args.out)
    elif args.world > 1:
        out = OUT_DIR / f"{man.stem}.r{args.rank}.jsonl"
    else:
        out = OUT_DIR / (man.stem + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with man.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % args.world != args.rank:
                continue
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            path = rec.get("path") or ""
            if text and path:
                rows.append(rec)
            if args.limit and len(rows) >= args.limit:
                break

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    n = 0
    with out.open("w", encoding="utf-8") as wo:
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            prompts = [build_prompt(tok, r["text"]) for r in chunk]
            inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=tok.pad_token_id,
                )
            in_len = inputs["input_ids"].shape[1]
            for rec, seq in zip(chunk, gen):
                text = tok.decode(seq[in_len:], skip_special_tokens=True)
                wo.write(
                    json.dumps(
                        {
                            "path": rec["path"],
                            "text": rec["text"],
                            "continuation": strip_cont(text),
                            "lang": rec.get("lang") or "",
                            "source": rec.get("source") or "",
                            "generator": "Qwen3-32B",
                            "prompt": "official-continuation-lt50",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
            if n % 64 == 0 or start == 0:
                print(f"rank {args.rank} n {n}/{len(rows)}", flush=True)
    print("wrote", out, "n", n)


if __name__ == "__main__":
    main()
