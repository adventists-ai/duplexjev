"""One audio clip, several option groups, one forward pass.

    pip install "duplexjev[speech]"
    python examples/quickstart.py call.wav --model fixie-ai/ultravox-v0_6-qwen-3-32b
"""
import argparse
import json

from duplexjev import Decider, Question

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--model", default="fixie-ai/ultravox-v0_6-qwen-3-32b")
ap.add_argument("--text-model", help="local copy of the checkpoint's LLM, e.g. /models/Qwen3-32B")
args = ap.parse_args()

d = Decider.from_pretrained(args.model, text_model=args.text_model)
groups = [
    Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
    Question("filler", "Which filler fits before the full answer?", ["“Sure —”", "“One moment —”", "(stay silent)"]),
    Question("gender", "Speaker gender", ["female", "male"]),
    Question("language", "Which language is spoken?", ["English", "Chinese", "other"]),
]
answers = d.decide(args.audio, groups)
print(json.dumps({k: (v["answer"], round(v["confidence"], 3)) for k, v in answers.items()}, ensure_ascii=False))
print(d.last_stats)  # decode_steps is always 0
