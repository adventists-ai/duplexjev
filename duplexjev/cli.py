"""Command line: ``duplexjev decide`` for one-off runs, ``duplexjev serve`` for the tick-batched HTTP server."""
from __future__ import annotations

import argparse
import json
import sys

from .question import Question


def _questions(args) -> list[Question]:
    qs = []
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            qs += [Question.from_dict(q) for q in json.load(f)]
    for spec in args.q or []:
        # "id|question text|opt1,opt2,..."
        qid, text, opts = spec.split("|", 2)
        qs.append(Question(qid, text, [o.strip() for o in opts.split(",")], lang=args.lang or "en"))
    if not qs:
        sys.exit("give --questions FILE.json or at least one --q 'id|text|opt1,opt2'")
    return qs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="duplexjev")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="HF id or path: any causal LM, or an Ultravox/DuplexJev speech checkpoint")
    common.add_argument("--text-model", help="override the LLM a speech checkpoint points to (e.g. a local path)")
    common.add_argument("--audio-model", help="override the encoder a speech checkpoint points to")
    common.add_argument("--device")

    d = sub.add_parser("decide", parents=[common], help="answer questions about audio files or transcripts")
    d.add_argument("--audio", nargs="*", default=[], help="audio files (one item each)")
    d.add_argument("--text", nargs="*", default=[], help="transcripts (one item each)")
    d.add_argument("--context")
    d.add_argument("--questions", help="JSON list of {id, text, options, lang}")
    d.add_argument("--q", action="append", help="inline question: 'id|text|opt1,opt2,...' (repeatable)")
    d.add_argument("--lang", choices=["en", "zh"])
    d.add_argument("--mode", default="packed", choices=["packed", "batch"])
    d.add_argument("--n-perm", type=int, default=1)

    s = sub.add_parser("serve", parents=[common], help="HTTP server; batches all requests of each tick")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--tick-ms", type=float, default=160.0)
    s.add_argument("--max-items", type=int, help="max items per forward pass")

    args = ap.parse_args(argv)
    from .decider import Decider

    dec = Decider.from_pretrained(args.model, device=args.device, text_model=args.text_model, audio_model=args.audio_model)
    if args.cmd == "decide":
        items = [{"audio": a} for a in args.audio] + [{"text": t} for t in args.text]
        if not items:
            sys.exit("give --audio and/or --text")
        for it in items:
            if args.context:
                it["context"] = args.context
            if args.lang:
                it["lang"] = args.lang
        res = dec.decide(items, _questions(args), mode=args.mode, n_perm=args.n_perm)
        names = args.audio + args.text
        print(json.dumps({"results": [{"input": n, "answers": r} for n, r in zip(names, res)], "pass": dec.last_stats},
                         ensure_ascii=False, indent=1))
    else:
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(dec, tick_ms=args.tick_ms, max_items=args.max_items), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
