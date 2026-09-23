"""Typed decisions over transcripts with any open LLM (no speech model needed).

    pip install duplexjev
    python examples/quickstart_text.py --model Qwen/Qwen3-8B
"""
import argparse
import json

from duplexjev import Decider, Question

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
args = ap.parse_args()

d = Decider.from_pretrained(args.model)
questions = [
    Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
    Question("intent", "What does the user want?", ["climate", "media", "navigation", "phone", "vehicle info", "chit-chat"]),
    Question("filler", "Which filler fits before the full answer?", ["“Sure —”", "“One moment —”", "(stay silent)"]),
    Question("human", "Does the user need a human agent?", ["yes", "no"]),
]
calls = [
    "Turn on the air conditioning please.",
    {"text": "Can you navigate to the", "context": "Driver is on the highway."},
    {"text": "帮我打电话给妈妈", "lang": "zh"},
]
for call, answers in zip(calls, d.decide(calls, questions)):
    print(json.dumps({"input": call, "answers": {k: (v["answer"], round(v["confidence"], 3)) for k, v in answers.items()}},
                     ensure_ascii=False))
print(d.last_stats)  # decode_steps is always 0
