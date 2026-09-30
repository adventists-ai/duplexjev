"""Command line.

    duplexjev decide --model M call.wav --q "turn|Has the user finished?|finished,not finished"
    duplexjev batch  --model M --audio car1=a.wav --audio car2=b.wav --questions groups.json
    duplexjev serve  --model M --tick-ms 160
    duplexjev quick  call.wav [--lang zh] [--api URL | --vllm URL | --model M] [--questions table.json]
    duplexjev gateway --vllm http://127.0.0.1:8010/v1 --port 8020      # public API + web demo in front of vLLM
"""
from __future__ import annotations

import argparse
import json
import sys

from .question import Question


def _questions(args, need_audio=False) -> list[Question]:
    qs = []
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            qs += [Question.from_dict(q) for q in json.load(f)]
    for spec in args.q or []:
        # "id|question text|opt1,opt2,..." or, for batch, "clip|id|question text|opt1,opt2,..."
        parts = spec.split("|")
        clip = parts.pop(0) if need_audio and len(parts) == 4 else None
        qid, text, opts = parts[0], parts[1], parts[2]
        qs.append(Question(qid, text, [o.strip() for o in opts.split(",")], lang=args.lang or "en", audio=clip))
    if not qs:
        sys.exit("give --questions FILE.json or at least one --q")
    return qs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="duplexjev")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="speech checkpoint (Ultravox format), HF id or path")
    common.add_argument("--text-model", help="override the LLM the checkpoint points to (e.g. a local path)")
    common.add_argument("--audio-model", help="override the encoder the checkpoint points to")
    common.add_argument("--device")

    q = argparse.ArgumentParser(add_help=False)
    q.add_argument("--questions", help="JSON list of option groups {id, text, options, lang[, audio]}")
    q.add_argument("--q", action="append", help="inline option group 'id|text|opt1,opt2' (batch: 'clip|id|text|opts')")
    q.add_argument("--lang", choices=["en", "zh"])
    q.add_argument("--mode", default="packed", choices=["packed", "batch"])
    q.add_argument("--n-perm", type=int, default=1)

    d = sub.add_parser("decide", parents=[common, q], help="option groups about one audio clip")
    d.add_argument("audio")
    d.add_argument("--context")

    b = sub.add_parser("batch", parents=[common, q], help="option groups about many clips (groups name their clip)")
    b.add_argument("--audio", action="append", required=True, help="clip as id=path (repeatable)")

    s = sub.add_parser("serve", parents=[common], help="HTTP server; batches all requests of each tick")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--tick-ms", type=float, default=160.0)
    s.add_argument("--max-items", type=int)
    s.add_argument("--max-tokens", type=int)

    k = sub.add_parser("quick", help="default decision table (or --questions) about one clip, printed as a table")
    k.add_argument("audio")
    k.add_argument("--lang", default="en", choices=["en", "zh"])
    k.add_argument("--questions", help="JSON list of option groups {id, text, options[, lang]}")
    k.add_argument("--api", help="DuplexJev gateway URL (default: $DUPLEXJEV_API)")
    k.add_argument("--vllm", help="vLLM server URL, e.g. http://localhost:8000/v1 (default: $DUPLEXJEV_VLLM)")
    k.add_argument("--model", help="local connector checkpoint (PyTorch)")
    k.add_argument("--json", action="store_true", help="print JSON instead of the table")

    g = sub.add_parser("gateway", help="public HTTP API + web demo in front of a vLLM server")
    g.add_argument("--vllm", required=True, help="vLLM server URL, e.g. http://127.0.0.1:8010/v1")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8020)
    g.add_argument("--root-path", default="", help="sub-path when served behind a reverse proxy, e.g. /duplexjev")
    g.add_argument("--max-mb", type=float, default=4.0)
    g.add_argument("--max-questions", type=int, default=16)
    g.add_argument("--rate", type=int, default=30, help="requests per minute per client IP")

    args = ap.parse_args(argv)
    if args.cmd == "quick":
        from .quick import quick

        qs = None
        if args.questions:
            with open(args.questions, encoding="utf-8") as f:
                qs = [Question.from_dict({"lang": args.lang, **q}) for q in json.load(f)]
        res = quick(args.audio, lang=args.lang, questions=qs, api=args.api, vllm=args.vllm, model=args.model,
                    show=not args.json)
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    if args.cmd == "gateway":
        import uvicorn

        from .gateway import create_app

        uvicorn.run(create_app(args.vllm, max_mb=args.max_mb, max_questions=args.max_questions, rate=args.rate),
                    host=args.host, port=args.port, root_path=args.root_path, proxy_headers=True,
                    forwarded_allow_ips="*")
        return
    from .decider import Decider

    dec = Decider.from_pretrained(args.model, device=args.device, text_model=args.text_model, audio_model=args.audio_model)
    if args.cmd == "decide":
        res = dec.decide(args.audio, _questions(args), context=args.context, lang=args.lang, mode=args.mode, n_perm=args.n_perm)
        print(json.dumps({"audio": args.audio, "answers": res, "pass": dec.last_stats}, ensure_ascii=False, indent=1))
    elif args.cmd == "batch":
        audios = dict(a.split("=", 1) for a in args.audio)
        res = dec.decide_batch(audios, _questions(args, need_audio=True), lang=args.lang, mode=args.mode, n_perm=args.n_perm)
        print(json.dumps({"answers": res, "pass": dec.last_stats}, ensure_ascii=False, indent=1))
    else:
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(dec, tick_ms=args.tick_ms, max_items=args.max_items, max_tokens=args.max_tokens),
                    host=args.host, port=args.port)


if __name__ == "__main__":
    main()
