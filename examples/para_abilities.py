"""Try the abilities added in the Para models: non-verbal sounds, speaking style, stress, same speaker or not.

    pip install "duplexjev[speech]"                                 # local GPU, default model DuplexJev-4B-Para
    python examples/para_abilities.py call.wav
    python examples/para_abilities.py call_zh.wav --lang zh
    python examples/para_abilities.py call.wav --words I say money  # which of these words is stressed?
    python examples/para_abilities.py call.wav --sample voice.wav   # same speaker as in voice.wav?
    python examples/para_abilities.py call.wav --vllm http://localhost:8000/v1   # your `vllm serve adventists-ai/DuplexJev-4B-Para`

The question wordings below are the ones the models were trained on. Keep the "none" options.
"""
import argparse

import numpy as np

from duplexjev import Question, quick

Q = {
    "en": {
        "sound": ("Besides speech, which sound can be heard in this audio?",
                  ["laughter", "breathing", "coughing", "a sigh", "None of these"]),
        "laugh": ("Is there laughter in this audio?", ["Yes", "No"]),
        "breath": ("Is there audible breathing or panting in this audio?", ["Yes", "No"]),
        "style": ("How is the speaker talking?",
                  ["whispering", "speaking very loudly / shouting", "speaking very fast", "a normal speaking voice"]),
        "stress": ("Does the speaker stress one of these words, or none of them?", "None of these words is emphasized"),
        "same": ("Before the silence is a voice sample of someone; after it is another clip. Are they the same speaker?",
                 ["different speakers", "same speaker"]),
    },
    "zh": {
        "sound": ("除了说话，这段音频里还有哪种声音？", ["笑声", "呼吸声", "咳嗽", "叹气", "都没有"]),
        "laugh": ("这段话里有没有笑声？", ["有", "没有"]),
        "breath": ("这段话里有没有呼吸声？", ["有", "没有"]),
        "style": ("说话人是用什么方式在说话？", ["小声耳语", "大声喊着说", "语速很快", "正常音量说话"]),
        "stress": ("说话人是否重读（强调）了某个词？", "没有强调任何词"),
        "same": ("静音前是某人的一段声音样本，静音后是另一段语音。两段是同一个人吗？", ["是不同的人", "是同一个人"]),
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--lang", default="en", choices=["en", "zh"])
ap.add_argument("--words", nargs="+", help="candidate words for the stress question")
ap.add_argument("--sample", help="a voice sample: is the clip the same speaker?")
ap.add_argument("--vllm", help="OpenAI-compatible URL of your vLLM server")
ap.add_argument("--model", help="local model (default adventists-ai/DuplexJev-4B-Para)")
a = ap.parse_args()
q = Q[a.lang]

table = [Question(k, *q[k], lang=a.lang) for k in ("sound", "laugh", "breath", "style")]
if a.words:
    text, none = q["stress"]
    table.append(Question("stress", text, a.words + [none], lang=a.lang))
quick(a.audio, lang=a.lang, questions=table, vllm=a.vllm, model=a.model)

def load16k(path):  # mono float32 at 16 kHz (linear resampling is enough here)
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    if sr != 16000:
        x = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype(np.float32)
    return x


if a.sample:  # voice sample + 0.8 s silence + the clip, as in training
    joined = np.concatenate([load16k(a.sample), np.zeros(12800, np.float32), load16k(a.audio)])
    quick(joined, lang=a.lang, questions=[Question("same", *q["same"], lang=a.lang)], vllm=a.vllm, model=a.model)
