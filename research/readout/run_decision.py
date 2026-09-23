#!/usr/bin/env python3
"""脚手架 / 正式 100 题的决策入口。不 generate。

用法（机器上）：
  python scripts/jev_qwen/run_decision.py \\
    --ckpt .../fdx-p3-.../checkpoint-30994 \\
    --pack artifacts/jev_qwen/scaffold.json \\
    --protocol L --device cuda:0 \\
    --out artifacts/jev_qwen/scaffold_L.json

  再跑 --protocol S。延迟曲线加 --latency-n 1,4,16,64。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
for _p in (
    os.path.join(os.path.dirname(HERE), "bench"),
    os.path.join(os.path.dirname(HERE), "scripts_bench"),
    "/data/exp01/exp03_train/scripts_bench",
):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

import run_bench as RB  # noqa: E402

from jev_qwen.contract import SCHEMA_VERSION, empty_execution, sha256_json, validate_pack  # noqa: E402
from jev_qwen.readout import (  # noqa: E402
    load_audio,
    read_item_l,
    read_item_s,
    summarize,
    time_batch_repeat,
)
from jev_qwen.render_letter import gate_single_tokens  # noqa: E402
from jev_qwen.render_path import count_tokens, path_texts  # noqa: E402


def load_pack(path: str) -> dict:
    doc = json.load(open(path, encoding="utf-8"))
    packed = validate_pack(doc)
    packed["meta"] = {k: v for k, v in doc.items() if k != "items"}
    return packed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=["L", "S", "both"], default="both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="协议 S 字母置换盐")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--latency-n", default="1,4,16", help="逗号分隔；空则不测延迟")
    ap.add_argument(
        "--condition",
        choices=["audio", "text_only"],
        default="audio",
        help="audio=听录音；text_only=只给转写、不给音频（主语言先验）",
    )
    ap.add_argument(
        "--loader",
        choices=["fdx", "official"],
        default="fdx",
        help="fdx=本线 Qwen3-ASR+fusion ckpt；official=C 路官方 Ultravox v0.6（whisper-turbo+Qwen3-32B）",
    )
    ap.add_argument("--cost-diag", action="store_true", help="只数 path vs letter token，可先于质量跑")
    args = ap.parse_args()

    pack = load_pack(args.pack)
    items = pack["items"]
    if args.limit:
        items = items[: args.limit]
    print(f"pack n={len(items)} sha={sha256_json({'items': pack['items']})[:16]}…", flush=True)

    if args.loader == "official":
        from jev_qwen.load_official import load_official

        model, processor = load_official(args.ckpt, args.device)
    else:
        model, processor = RB.load_model(args.ckpt, args.device)
    tok = processor.tokenizer
    prompt, rest = RB.assert_template_alignment(tok)
    print(f"模板对齐：prompt 尾 {prompt[-20:]!r}  答案头 {rest[:8]!r}", flush=True)
    gate_single_tokens(tok, list("ABCDYNY12345"), "startup")

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "ckpt": args.ckpt,
        "loader": args.loader,
        "pack": os.path.abspath(args.pack),
        "pack_meta": pack.get("meta"),
        "seed": args.seed,
        "device": args.device,
        "condition": args.condition,
        "template_ok": True,
        "protocols": {},
        "latency": [],
        "cost_diag": None,
        "execution": empty_execution(),
    }

    if args.cost_diag and items:
        q = next(iter(items[0]["state"]["questions"].values()))
        paths = [t for _, t in path_texts(items[0]["state"]["state"], q)]
        result["cost_diag"] = {
            "item": items[0]["id"],
            "path": count_tokens(tok, paths),
            "letter_suffix_tokens": len(
                tok.encode(
                    # 只比「一题一份」对「一题 K 份」
                    paths[0].split("Candidate:")[0],
                    add_special_tokens=False,
                )
            ),
            "note": "path 把 state+question 复制 K 次；letter 只编一次。这是布局成本，不是准确率。",
        }
        print("cost_diag", json.dumps(result["cost_diag"], ensure_ascii=False), flush=True)

    want = ["L", "S"] if args.protocol == "both" else [args.protocol]
    for proto in want:
        rows = []
        t0 = datetime.now(timezone.utc)
        for i, it in enumerate(items, 1):
            audio = None if args.condition == "text_only" else load_audio(it["audio"], it["t_ms"])
            print(f"[{args.condition} {proto} {i}/{len(items)}] {it['id']} {it.get('task','')}", flush=True)
            if proto == "L":
                rows.append(
                    read_item_l(model, processor, tok, it, args.device, audio, condition=args.condition)
                )
            else:
                rows.append(
                    read_item_s(
                        model, processor, tok, it, args.device, audio, args.seed, condition=args.condition
                    )
                )
        summary = summarize(rows)
        result["protocols"][proto] = {
            "summary": summary,
            "rows": rows,
            "started_utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        print(f"  {proto} accuracy={summary['accuracy']} n={summary['n']} hist={summary['pred_hist']}", flush=True)

    ns = [int(x) for x in args.latency_n.split(",") if x.strip()]
    if ns and items and args.condition == "audio":
        audio = load_audio(items[0]["audio"], items[0]["t_ms"])
        for n in ns:
            print(f"[latency] n={n}", flush=True)
            result["latency"].append(
                time_batch_repeat(model, processor, tok, items[0], audio, args.device, n, args.seed)
            )
        print(json.dumps(result["latency"], ensure_ascii=False, indent=2), flush=True)

    result["execution"].update(
        {
            "states": len(items),
            "questions": len(items),
            "candidate_paths": 0,
            "forward_passes": sum(
                len(result["protocols"][p]["rows"]) for p in result["protocols"]
            )
            + len(result["latency"]),
            "autoregressive_decode_steps": 0,
            "prefix_sharing": "batch_repeat" if result["latency"] else False,
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
