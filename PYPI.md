# duplexjev

Jev-style typed decisions about speech with **zero decode steps**. Give it audio and one or more option groups
(*has the user finished? which filler? which intent? speaker gender?*); every group comes back as a probability over
its options from a single forward pass. Many groups about many calls share one pass, with exact prefix sharing.

**Quick start, no GPU** — the default decisions about any clip through our trial API (rate-limited):

```bash
pip install duplexjev
duplexjev quick call.wav --api https://api.adventists.cn/duplexjev              # --lang zh for Chinese speech
```

```python
from duplexjev import quick, Question
quick("call.wav", api="https://api.adventists.cn/duplexjev")                   # prints and returns the table
quick("call.wav", vllm="http://localhost:8000/v1",                              # your own `vllm serve adventists-ai/DuplexJev-4B-Para`
      questions=[Question("product", "Which product is the user asking about?", ["phone", "laptop", "other"])])
```

`duplexjev gateway --vllm http://localhost:8000/v1` serves the same web demo and HTTP API in front of your own vLLM
server. Web demo: https://api.adventists.cn/duplexjev/

**Local PyTorch** — the default model is **DuplexJev-4B-Para** (~10 GB GPU memory), which also hears laughter, breathing,
coughs and sighs, speaking style (whisper, shouting) and stress; `quick("call.wav")` adds *non-verbal sound* and
*speaking style* to the default table (ten decisions). Complete models and connector checkpoints both load with
`Decider.from_pretrained` (one forward pass for many option groups and many clips):

```bash
pip install "duplexjev[speech]"      # add [all] for the HTTP server
```

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("adventists-ai/DuplexJev-4B-Para")      # or DuplexJev-32B-Para with device="auto"
groups = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
          Question("gender", "What is the perceived gender of the speaker?", ["female", "male"]),
          Question("sound", "Besides speech, which sound can be heard in this audio?",
                   ["laughter", "breathing", "coughing", "a sigh", "None of these"])]

d.decide("call.wav", groups)
# {'turn': {'answer': 'finished', 'confidence': 0.97, 'probs': {...}}, 'gender': {...}}

# many clips: each option group names its clip(s) with audio=<id> | [ids] | "*"
d.decide_batch({"car1": "a.wav", "car2": "b.wav"},
               [Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
                Question("gender", "Speaker gender", ["female", "male"], audio="car2")])
# {'car1': {'turn': ...}, 'car2': {'turn': ..., 'gender': ...}}
```

Ask in the language of the clip (`Question(..., lang="zh")` with Chinese options for Chinese speech): the DuplexJev
connectors were trained with language-matched prompts. Released weights, results and licences:
https://huggingface.co/adventists-ai. Local copies of the encoder and LLM: `from_pretrained(..., text_model=..., audio_model=...)`,
also offline.

Serve with tick batching: `duplexjev serve --model <checkpoint> --tick-ms 160`.

**0.4.0**: default model `adventists-ai/DuplexJev-4B-Para`; complete DuplexJev models (encoder + connector + LLM in one
repo) load in PyTorch with vendored model code; default table gains *non-verbal sound* and *speaking style* (10 rows);
`examples/para_abilities.py`.

**0.3.0**: `quick` / `duplexjev quick` with a default table of eight decisions; clients for vLLM servers and the
DuplexJev HTTP API that need no torch (torch moved to the `speech` extra); `duplexjev gateway` (web demo + API).

**0.2.1**: audio is padded to whole audio tokens at the end instead of the start (start padding cost up to 6 points on
emotion); `text_model` / `audio_model` overrides work offline.

Project: https://github.com/adventists-ai/duplexjev · License: Apache-2.0
