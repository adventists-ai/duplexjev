"""One tick of a batched decision loop: the latest window of every live call, all option groups, one pass.

    python examples/tick_batch.py --model fixie-ai/ultravox-v0_6-qwen-3-32b --audio clip.wav --calls 1 8 32 64
"""
import argparse
import time

import numpy as np

from duplexjev import Decider, Question, load_audio

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--text-model")
ap.add_argument("--audio", required=True, help="a clip used as every call's current window (shifted per call)")
ap.add_argument("--calls", type=int, nargs="+", default=[1, 8, 32, 64])
ap.add_argument("--window-s", type=float, default=4.0)
args = ap.parse_args()

d = Decider.from_pretrained(args.model, text_model=args.text_model)
G = [
    Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
    Question("filler", "Which filler fits?", ["“Sure —”", "“One moment —”", "“Let me check —”", "(stay silent)"], audio="*"),
    Question("intent", "What does the user want?", ["climate", "media", "navigation", "phone", "chit-chat"], audio="*"),
    Question("interrupt", "Is the user interrupting the agent?", ["yes", "no"], audio="*"),
]
a = load_audio(args.audio)
W = int(args.window_s * 16000)
a = np.concatenate([a, np.zeros(max(0, W - len(a)), dtype=np.float32)])


def window(i):
    off = (i * 1600) % max(1, len(a) - W + 1)
    return a[off:off + W]


d.decide_batch([window(0)], G)  # warm-up
for n in args.calls:
    calls = {f"call{i}": window(i) for i in range(n)}
    t = time.perf_counter()
    d.decide_batch(calls, G, max_tokens=24000)
    ms = (time.perf_counter() - t) * 1000
    print(f"calls={n:4d} groups/call={len(G)} pass={ms:8.1f} ms  per call={ms / n:6.1f} ms  "
          f"decode_steps={d.last_stats['decode_steps']}")
