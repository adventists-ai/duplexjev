# duplexjev

Batched typed speech decisions without decoding. Every question — *has the user finished? which filler? which intent?*
— is answered as a probability over its options from a **single forward pass**, with no ASR or text decoding. Many
questions about many calls share one pass, with exact prefix sharing.

```bash
pip install duplexjev            # text models
pip install "duplexjev[all]"     # + speech checkpoints and the HTTP server
```

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("Qwen/Qwen3-8B")        # any causal LM; or a speech checkpoint (see below)
qs = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
      Question("intent", "What does the user want?", ["climate", "media", "navigation", "phone"])]
d.decide(["Turn on the air conditioning", "Navigate to the"], qs)
# [{'turn': {'answer': 'finished', 'confidence': 0.97, 'probs': {...}}, 'intent': {...}}, {...}]
```

Speech checkpoints (Ultravox format, including the DuplexJev adapters) take audio directly:
`Decider.from_pretrained("fixie-ai/ultravox-v0_6-qwen-3-32b").decide(["call1.wav", "call2.wav"], qs)`.

Serve with tick batching: `duplexjev serve --model <id> --tick-ms 160`.

Project: https://github.com/adventists-ai/duplexjev · License: Apache-2.0
