"""Send many concurrent requests to `duplexjev serve`; requests of the same tick are answered in one pass.

    duplexjev serve --model Qwen/Qwen3-8B --tick-ms 160      # in another shell
    python examples/server_client.py --n 50
"""
import argparse
import asyncio
import base64

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/v1/decide")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--audio", help="optional WAV file (speech servers)")
args = ap.parse_args()

Q = [{"id": "turn", "text": "Has the user finished the turn?", "options": ["finished", "not finished"]},
     {"id": "intent", "text": "What does the user want?", "options": ["climate", "media", "navigation", "phone"]}]


async def one(c, i):
    body = {"questions": Q}
    if args.audio:
        body["audio_b64"] = base64.b64encode(open(args.audio, "rb").read()).decode()
    else:
        body["text"] = ["Turn on the heater", "Play some music", "Call my", "Navigate home please"][i % 4]
    r = await c.post(args.url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


async def main():
    async with httpx.AsyncClient() as c:
        res = await asyncio.gather(*[one(c, i) for i in range(args.n)])
    ticks = sorted({r["tick"] for r in res})
    print(f"{args.n} requests answered in {len(ticks)} tick(s); items per pass: {sorted({r['batch_items'] for r in res})}")
    print(res[0]["answers"])


asyncio.run(main())
