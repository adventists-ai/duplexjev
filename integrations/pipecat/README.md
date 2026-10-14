# pipecat-duplexjev

**Audio end-of-turn detection for [Pipecat](https://github.com/pipecat-ai/pipecat), powered by
[DuplexJev](https://github.com/adventists-ai/duplexjev).** It listens to *how* the user speaks, not only what they say,
and tells your bot whether the user is **finished**, **still talking**, **just backchanneling** ("mm-hmm") or
**asking you to wait**. The same request can also return **emotion**, **laughter / breathing / coughs / sighs** and
**barge-in** at no extra cost.

- **Four turn states, not a yes/no.** A backchannel or a "hold on" is not a reason to start talking.
- **Hears beyond the transcript.** Every answer is read from the audio as one constrained token of a speech LLM.
  Nothing is decoded and no STT is needed for the decision.
- **More than turns, same forward pass.** Ask for emotion, non-verbal sounds or your own multiple-choice questions.
  They come back in `last_answers` with the turn decision.
- **~55 ms on the server** for a turn decision with DuplexJev-4B-Para on one GPU.
- **Chinese and English.**
- On the CoDeTT turn-taking benchmark (English, zero-shot), DuplexJev-32B-Turn scores **70.0**, against **51.4** for
  Smart Turn v3. See the [paper](https://arxiv.org/abs/2610.02638) and the
  [project page](https://adventists-ai.github.io/duplexjev/).

Tested with `pipecat-ai` 1.12.0. Maintained by Adventists.ai (AI降临派).

> **Getting the published turn-taking numbers.** LiveKit eot-bench (English, false cut-offs at 300 ms): **20.0%** for
> DuplexJev-32B-Para v1.2 with the agent's previous message, 23.5% audio only, vs 35.2% for Smart Turn v3.2. That needs
> DuplexJev-32B-Para **v1.2 on your own GPU** behind `duplexjev gateway` >= 0.6 (see *Run the model yourself*), the
> default `readout="turn"`, and the agent's previous message (below). Since 2026-10-13 the free trial API serves the same
> DuplexJev-32B-Para v1.2, but it is rate limited (30 requests per minute) and served from mainland China: fine for trying
> the plugin, self-host for a live bot or a benchmark run.

## Install

```bash
pip install pipecat-duplexjev
```

## Use

Drop it in wherever Pipecat takes a turn analyzer (Pipecat >= 1.0):

```python
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat_duplexjev import DuplexJevTurnAnalyzer

turn = DuplexJevTurnAnalyzer(
    url="http://localhost:8420",          # your own gateway; omit to use the free hosted trial API
    extra_questions=["emotion", "sound"],
    on_answers=lambda a: print({k: v["answer"] for k, v in a.items()}),
)

aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
    vad_analyzer=SileroVADAnalyzer(),
    user_turn_strategies=UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn)]),
))
```

**Give it the assistant's last message** (new in 0.3.0). Turn-taking is clearly better when the model knows what the
bot had just said (a question invites a short answer; "ok" after a question ends the turn). Put the tracker after
your LLM service:

```python
from pipecat_duplexjev import DuplexJevContextTracker

pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm,
                     DuplexJevContextTracker(turn),       # tells the analyzer what the bot is saying
                     tts, transport.output(), aggregators.assistant()])
```

or call `turn.set_assistant_said(text)` yourself. It needs a gateway from `duplexjev>=0.6`. On LiveKit eot-bench
(English, DuplexJev-32B-Para v1.2, full run) this cuts false cut-offs at a 300 ms budget from 23.5% to 20.0%.

A complete snippet is in [`examples/wiring.py`](examples/wiring.py). [`examples/live_check.py`](examples/live_check.py)
feeds real clips through the analyzer, frame by frame, the way a transport does:

```text
ex4.wav: COMPLETE   p(finished)=1.00  turn='finished, the assistant can reply'   emotion=neutral  server 59.6 ms
ex3.wav: INCOMPLETE p(finished)=0.11  turn='not finished, still talking'          emotion=sad      server 53.2 ms
ex5.wav: INCOMPLETE p(finished)=0.28  turn='hesitating or asking to wait'         emotion=angry    server 56.0 ms
```

## Run the model yourself (recommended for real-time)

The default URL is our free trial API. It is rate limited (30 requests per minute), served from mainland China
(the network round trip from elsewhere can be over a second); since 2026-10-13 it serves DuplexJev-32B-Para v1.2.
For a live bot, serve the model next to your agent. For turn-taking, use
DuplexJev-32B-Para v1.2 on one 80 GB GPU (the eot-bench numbers above):

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm "duplexjev[server]>=0.6"
vllm serve adventists-ai/DuplexJev-32B-Para --revision v1.2 --max-model-len 4096 --port 8000 &
duplexjev gateway --vllm http://localhost:8000/v1 --port 8420 --rate 1000000
```

On a smaller GPU (about 10 GB), serve `adventists-ai/DuplexJev-4B-Para-LiveTurn`, the 4B with the same turn-taking
update: eot-bench false cut-offs at 300 ms 25.5% English / 38.8% Chinese with the agent's previous message (Smart Turn
v3.2: 35.2% / 42.9%). `--rate 1000000` lifts the gateway's default limit of 30 requests per
minute, which a live bot exceeds quickly.

Then pass `url="http://localhost:8420"`.

## Options

| argument | default | |
|---|---|---|
| `url` | hosted trial API | base URL of a `duplexjev gateway` |
| `api_key` | `None` | bearer key, if your gateway requires one |
| `lang` | `"en"` | `"en"` or `"zh"`: wording of the questions (the model hears both languages either way) |
| `threshold` | `0.5` | probability of "finished" from which the turn is complete |
| `extra_questions` | `[]` | `"emotion"`, `"sound"`, `"barge_in"`, or dicts `{"id", "text", "options"}` |
| `on_answers` | `None` | callback with every answer dict |
| `timeout` | `params.stop_secs` | HTTP timeout; on timeout Pipecat treats the turn as complete |
| `params` | `SmartTurnParams()` | Pipecat's usual `stop_secs`, `pre_speech_ms`, `max_duration_secs` |
| `readout` | `"turn"` | `"act"` asks "What should the assistant do now?" (P(reply) + P(short acknowledgement)); slightly better on eot-bench with context |
| `wait_stop_secs` | `None` | if the user sounds like they are hesitating, wait up to this long in silence before forcing the turn to end |

After each prediction `turn.last_turn_class` is `finished`, `not_finished`, `backchannel` or `hesitating`
(`reply`, `keep_listening`, `acknowledge` with `readout="act"`), for your own routing.

If the gateway can't be reached, the analyzer logs the error and reports "not finished", so Pipecat falls back to its
silence timeout and the conversation never blocks.

## License

The plugin code is BSD-2-Clause. DuplexJev model weights are CC BY-NC 4.0 (non-commercial); see the
[model cards](https://huggingface.co/adventists-ai). For commercial use, contact jiejin@adventists.ai.
