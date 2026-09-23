"""Concurrent requests to `duplexjev serve`: all requests of one tick are answered in one pass.

    duplexjev serve --model fixie-ai/ultravox-v0_6-qwen-3-32b --tick-ms 160      # in another shell
    python examples/server_client.py call.wav --n 50
"""
import argparse
import asyncio
import base64

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=50)
args = ap.parse_args()

B64 = base64.b64encode(open(args.audio, "rb").read()).decode()
G = [{"id": "turn", "text": "Has the user finished the turn?", "options": ["finished", "not finished"]},
     {"id": "intent", "text": "What does the user want?", "options": ["climate", "media", "navigation", "phone"]}]


async def main():
    async with httpx.AsyncClient(timeout=60) as c:
        one = [c.post(f"{args.url}/v1/decide", json={"audio_b64": B64, "questions": G}) for _ in range(args.n)]
        batch = c.post(f"{args.url}/v1/decide_batch", json={
            "audios": {"car1": B64, "car2": B64},
            "questions": [dict(G[0], audio="*"), dict(G[1], audio="car2")]})
        res = [r.json() for r in await asyncio.gather(*one, batch)]
    print(f"{len(res)} requests in {len({r['tick'] for r in res})} tick(s); clips per pass: {sorted({r['clips_in_pass'] for r in res})}")
    print("single:", res[0]["answers"])
    print("batch :", res[-1]["answers"])


asyncio.run(main())
