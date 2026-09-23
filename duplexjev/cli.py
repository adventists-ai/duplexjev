"""Command line.

    duplexjev decide --model M call.wav --q "turn|Has the user finished?|finished,not finished"
    duplexjev batch  --model M --audio car1=a.wav --audio car2=b.wav --questions groups.json
    duplexjev serve  --model M --tick-ms 160
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

    args = ap.parse_args(argv)
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
