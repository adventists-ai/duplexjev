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

> **Getting the published turn-taking numbers.** LiveKit eot-bench (English, false cut-offs at 300 ms): **20.0%** for
> DuplexJev-32B-Para v1.2 with the agent's previous message, 23.5% audio only, vs 35.2% for Smart Turn v3.2. That needs
> DuplexJev-32B-Para **v1.2 on your own GPU** behind `duplexjev gateway` >= 0.6 (see *Run the model yourself*), the
> default `readout="turn"`, and the agent's previous message (below). Since 2026-10-13 the free trial API serves the same
> DuplexJev-32B-Para v1.2, but it is rate limited (30 requests per minute) and served from mainland China: fine for trying
> the plugin, self-host for a live agent or a benchmark run.

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

**Give it the agent's last message** (new in 0.2.0). Turn-taking is clearly better when the model knows what the
agent had just said. Keep a reference to the detector and attach it to the session once:

```python
detector = duplexjev.TurnDetector(api_url="http://localhost:8420")
session = AgentSession(..., turn_handling={"turn_detection": detector})
detector.attach(session)        # follows conversation_item_added; or detector.set_assistant_said(text)
```

It needs a gateway from `duplexjev>=0.6`. On LiveKit eot-bench (English, DuplexJev-32B-Para v1.2, full run) this cuts
false cut-offs at a 300 ms budget from 23.5% to 20.0%.

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

The default URL is our free trial API (rate limited to 30 requests per minute; the round trip from outside mainland China adds network latency; since 2026-10-13 it serves DuplexJev-32B-Para v1.2). For a live agent, serve the model next to it. For turn-taking, use DuplexJev-32B-Para v1.2 on one 80 GB GPU (the eot-bench numbers above):

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm "duplexjev[server]>=0.6"
vllm serve adventists-ai/DuplexJev-32B-Para --revision v1.2 --max-model-len 4096 --port 8000 &
duplexjev gateway --vllm http://localhost:8000/v1 --port 8420 --rate 1000000
```

On a smaller GPU (about 10 GB), serve `adventists-ai/DuplexJev-4B-Para-LiveTurn`, the 4B with the same turn-taking
update: eot-bench false cut-offs at 300 ms 25.5% English / 38.8% Chinese with the agent's previous message (Smart Turn
v3.2: 35.2% / 42.9%). `--rate 1000000` lifts the gateway's default limit of 30 requests per
minute, which a live agent exceeds quickly.

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
| `readout` | `"turn"` | `"act"` asks "What should the assistant do now?" (P(reply) + P(short acknowledgement)); slightly better on eot-bench with context |

After each prediction `detector.last_turn_class` is `finished`, `not_finished`, `backchannel` or `hesitating`
(`reply`, `keep_listening`, `acknowledge` with `readout="act"`), for your own routing.

## License

The plugin code is BSD-2-Clause. DuplexJev model weights are CC BY-NC 4.0 (non-commercial); see the
[model cards](https://huggingface.co/adventists-ai). For commercial use, contact jiejin@adventists.ai.
