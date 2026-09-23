# duplexjev

Jev-style typed decisions about speech with **zero decode steps**. Give it audio and one or more option groups
(*has the user finished? which filler? which intent? speaker gender?*); every group comes back as a probability over
its options from a single forward pass. Many groups about many calls share one pass, with exact prefix sharing.

```bash
pip install "duplexjev[speech]"      # add [all] for the HTTP server
```

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("fixie-ai/ultravox-v0_6-qwen-3-32b")   # or a DuplexJev adapter
groups = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
          Question("gender", "Speaker gender", ["female", "male"])]

d.decide("call.wav", groups)
# {'turn': {'answer': 'finished', 'confidence': 0.97, 'probs': {...}}, 'gender': {...}}

# many clips: each option group names its clip(s) with audio=<id> | [ids] | "*"
d.decide_batch({"car1": "a.wav", "car2": "b.wav"},
               [Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
                Question("gender", "Speaker gender", ["female", "male"], audio="car2")])
# {'car1': {'turn': ...}, 'car2': {'turn': ..., 'gender': ...}}
```

Serve with tick batching: `duplexjev serve --model <checkpoint> --tick-ms 160`.

Project: https://github.com/adventists-ai/duplexjev · License: Apache-2.0
