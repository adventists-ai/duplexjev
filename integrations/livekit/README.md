# livekit-plugins-duplexjev

**Audio end-of-turn detection for [LiveKit Agents](https://github.com/livekit/agents), powered by
[DuplexJev](https://github.com/adventists-ai/duplexjev).** It listens to *how* the user speaks, not only what they
say, and tells your agent whether the user is **finished**, **still talking**, **just backchanneling** ("mm-hmm") or
**asking you to wait**. The same request can also return **emotion**, **laughter / breathing / coughs / sighs** and
**barge-in**.

- **Reads the audio, not the transcript.** Every answer is one constrained token of a speech LLM: nothing is
  decoded, so the decision does not wait for STT.
- **Fast.** Ten decisions about one utterance take **92 ms** on one GPU, against **1,978 ms** for ASR → LLM with
  the same LLM.
- **Accurate.** On the CoDeTT turn-taking benchmark (English, zero-shot), DuplexJev-32B-Turn scores **70.0**,
  against **51.4** for Smart Turn v3. See the [paper](https://arxiv.org/abs/2610.02638).
- **More than turns, same request.** Ask for emotion, non-verbal sounds or your own multiple-choice questions.
- **Chinese and English.** CPU only on the agent side: the plugin calls an HTTP API.

Tested with `livekit-agents` 1.8.6. Maintained by Adventists.ai (AI降临派).

## Install

```bash
pip install livekit-plugins-duplexjev
```

## Use

```python
from livekit.agents import AgentSession
from livekit.plugins import duplexjev, silero

session = AgentSession(
    stt=..., llm=..., tts=...,
    vad=silero.VAD.load(),
    turn_handling={"turn_detection": duplexjev.TurnDetector(
        api_url="http://localhost:8420",       # your own gateway; omit for the free hosted trial API
        extra_questions=["emotion", "sound"],
        on_answers=lambda a: print({k: v["answer"] for k, v in a.items()}),
    )},
)
```

`AgentSession(turn_detection=duplexjev.TurnDetector(), ...)` works too (LiveKit marks that argument as deprecated).
A complete agent is in [`examples/agent.py`](examples/agent.py).

**A VAD is required.** LiveKit ignores a streaming turn detector when the session has no VAD. The plugin buffers the
user's audio, and after about 200 ms of VAD silence LiveKit asks it for a prediction: the last 8 s of the current
user turn go to DuplexJev, and P(finished) becomes LiveKit's `end_of_turn_probability`. Below `unlikely_threshold`
the session waits `max_delay` instead of `min_delay` before replying. The VAD's `min_silence_duration` must be at
least 0.25 s (Silero's default is 0.55 s).

**STT is optional for the decision.** If the session has an STT, LiveKit still waits for its transcript before it
commits the turn, as with any turn detector, because the LLM needs the text.

## Run the model yourself (recommended for real-time)

The default URL is our free trial API (rate limited to 30 requests per minute; the round trip from outside mainland China adds network latency). For a live agent, serve the model next to it (DuplexJev-4B-Para needs about 10 GB of GPU memory):

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm "duplexjev[server]>=0.4.1"
vllm serve adventists-ai/DuplexJev-4B-Para --max-model-len 4096 --port 8000 &
duplexjev gateway --vllm http://localhost:8000/v1 --port 8420
```

Then pass `api_url="http://localhost:8420"` (or set `DUPLEXJEV_API_URL`).

## Options

| argument | default | |
|---|---|---|
| `api_url` | hosted trial API | base URL of a `duplexjev gateway` (or `DUPLEXJEV_API_URL`) |
| `api_key` | `None` | bearer key, if your gateway requires one (or `DUPLEXJEV_API_KEY`) |
| `lang` | `"en"` | `"en"` or `"zh"`: wording of the questions (the model hears both languages either way) |
| `unlikely_threshold` | `0.5` | below this P(finished), wait `max_delay` instead of `min_delay` |
| `backchannel_threshold` | `None` | if set, also asks "What should the assistant do now?" and reports P(short acknowledgement) as LiveKit's backchannel probability |
| `extra_questions` | `[]` | `"emotion"`, `"sound"`, `"barge_in"`, or dicts `{"id", "text", "options"}` |
| `on_answers` | `None` | callback with every answer dict; the latest is also on `detector.last_answers` |
| `window_s` | `8.0` | seconds of the current user turn sent with each request |
| `trim_silence` | `True` | drop the silence before the user started talking (keeps 0.3 s) |
| `prediction_timeout` | `4.0` hosted, `1.0` own gateway | how long LiveKit waits for a prediction |
| `fallback_probability` | `1.0` | P(finished) reported if the gateway fails, so the conversation never blocks |

## License

The plugin code is BSD-2-Clause. DuplexJev model weights are CC BY-NC 4.0 (non-commercial); see the
[model cards](https://huggingface.co/adventists-ai). For commercial use, contact jiejin@adventists.ai.
