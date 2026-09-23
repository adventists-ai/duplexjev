"""Typed decisions straight from audio with a speech checkpoint (Ultravox format).

    pip install "duplexjev[speech]"
    python examples/quickstart_speech.py --model fixie-ai/ultravox-v0_6-qwen-3-32b a.wav b.wav

DuplexJev adapters (adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B, ...) load the same way once released.
Use --text-model /path/to/Qwen3-32B to point a checkpoint at a local copy of its LLM.
"""
import argparse
import json

from duplexjev import Decider, Question

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="fixie-ai/ultravox-v0_6-qwen-3-32b")
ap.add_argument("--text-model")
ap.add_argument("audio", nargs="+")
args = ap.parse_args()

d = Decider.from_pretrained(args.model, text_model=args.text_model)
questions = [
    Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
    Question("gender", "Speaker gender", ["female", "male"]),
    Question("language", "Which language is spoken?", ["English", "Chinese", "other"]),
    Question("sentiment", "Sentiment of the speaker", ["positive", "neutral", "negative"]),
]
for path, answers in zip(args.audio, d.decide(args.audio, questions)):
    print(json.dumps({"audio": path, "answers": {k: (v["answer"], round(v["confidence"], 3)) for k, v in answers.items()}},
                     ensure_ascii=False))
print(d.last_stats)
