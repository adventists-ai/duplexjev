# duplexjev

Jev-style typed decisions about speech with **zero decode steps**. Give it audio and one or more option groups
(*has the user finished? which filler? which intent? speaker gender?*); every group comes back as a probability over
its options from a single forward pass. Many groups about many calls share one pass, with exact prefix sharing.

```bash
pip install "duplexjev[speech]"      # add [all] for the HTTP server
```

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B", device="auto")
groups = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
          Question("gender", "What is the perceived gender of the speaker?", ["female", "male"])]

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

**0.2.1**: audio is padded to whole audio tokens at the end instead of the start (start padding cost up to 6 points on
emotion); `text_model` / `audio_model` overrides work offline.

Project: https://github.com/adventists-ai/duplexjev · License: Apache-2.0
