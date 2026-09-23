"""Many clips, many option groups; each group names the clip(s) it is about.

    python examples/batch.py --model fixie-ai/ultravox-v0_6-qwen-3-32b car1=a.wav car2=b.wav car3=c.wav
"""
import argparse
import json

from duplexjev import Decider, Question

ap = argparse.ArgumentParser()
ap.add_argument("clips", nargs="+", help="id=path")
ap.add_argument("--model", default="fixie-ai/ultravox-v0_6-qwen-3-32b")
ap.add_argument("--text-model")
args = ap.parse_args()

audios = dict(c.split("=", 1) for c in args.clips)
first = next(iter(audios))
d = Decider.from_pretrained(args.model, text_model=args.text_model)
groups = [
    Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),     # every clip
    Question("intent", "What does the user want?", ["climate", "media", "navigation", "phone"], audio="*"),
    Question("human", "Does the user need a human agent?", ["yes", "no"], audio=first),               # one clip
]
answers = d.decide_batch(audios, groups)
for clip, a in answers.items():
    print(clip, json.dumps({k: (v["answer"], round(v["confidence"], 3)) for k, v in a.items()}, ensure_ascii=False))
print(d.last_stats)
