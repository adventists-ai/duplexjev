<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.svg">
    <img src="docs/assets/banner_light.svg" alt="DuplexJev — batched speech decisions without decoding" width="880">
  </picture>
</p>

<p align="center">
  <b>English</b> &nbsp;|&nbsp; <a href="README_CN.md">中文</a>
</p>

<h1 align="center">Stop transcribing to decide.</h1>

<p align="center">
  <b>DuplexJev is a System One model for speech: raw audio in, typed decisions out. The ears of a voice agent.</b><br>
  Has the user finished? Should I stop talking? Are they angry? Which filler fits?<br>
  Every answer straight from raw speech, in <b>one forward pass</b>: <b>10 decisions in 92 ms</b>, <b>21× faster</b> than ASR → LLM,
  under <b>$0.50 per million decisions</b>, and a new kind of decision is <b>one line of text</b>.<br>
  🏆 <b>State of the art in paralinguistic understanding</b> on public benchmarks (<a href="https://arxiv.org/abs/2510.10444">LISTEN</a>, StressTest and MSPB combined):
  ahead of <b>Gemini 3.8 Flash, Gemini 3.8 Live, GPT-realtime-2.1, GPT-audio-1.5 and Qwen3-Omni</b> on the same questions.<br>
  👁️ <b>It also sees.</b> <a href="https://huggingface.co/adventists-ai/DuplexJev-32B-Vision">DuplexJev-32B-Vision</a> checks the voice against a picture in the same pass (preview) — <a href="https://huggingface.co/spaces/BitKnight/DuplexJev-Vision">try it with a photo + your voice</a>.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/BitKnight/DuplexJev"><img src="https://img.shields.io/badge/%F0%9F%8E%A7%20Try%20it-in%20your%20browser-22D3A6?style=for-the-badge" alt="Try it in your browser"></a>
  <a href="https://huggingface.co/spaces/BitKnight/DuplexJev-Vision"><img src="https://img.shields.io/badge/%F0%9F%91%81%EF%B8%8F%20Try%20Vision-photo%20%2B%20voice-0EA5E9?style=for-the-badge" alt="Try the vision demo"></a>
  <a href="https://github.com/adventists-ai/duplexjev/stargazers"><img src="https://img.shields.io/badge/%E2%AD%90-Star%20on%20GitHub-181717?style=for-the-badge&logo=github" alt="Star on GitHub"></a>
  <a href="integrations/pipecat"><img src="https://img.shields.io/badge/Pipecat-plugin-6c47ff?style=for-the-badge" alt="Pipecat plugin"></a>
  <a href="integrations/livekit"><img src="https://img.shields.io/badge/LiveKit-plugin-1f6feb?style=for-the-badge" alt="LiveKit plugin"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02638"><img src="https://img.shields.io/badge/arXiv-2610.02638-b31b1b" alt="arXiv"></a>
  <a href="https://huggingface.co/adventists-ai"><img src="https://img.shields.io/badge/%F0%9F%A4%97-open%20weights%204B%20%C2%B7%2032B-yellow" alt="Hugging Face models"></a>
  <a href="https://www.modelscope.cn/organization/AdventistsAI"><img src="https://img.shields.io/badge/ModelScope-mirror-624aff" alt="ModelScope"></a>
  <a href="https://pypi.org/project/duplexjev/"><img src="https://img.shields.io/pypi/v/duplexjev" alt="PyPI"></a>
  <a href="https://adventists-ai.github.io/duplexjev/"><img src="https://img.shields.io/badge/project-page-0b7285" alt="Project page"></a>
</p>

<p align="center"><img src="docs/assets/race.gif" alt="Ten decisions: DuplexJev 92 ms vs ASR → LLM 1,978 ms on the same H200" width="820"></p>
<p align="center"><sub>▶ <a href="docs/assets/duplexjev_launch.mp4">Watch the 2-minute video (with sound)</a> · answers shown are real model outputs</sub></p>

## ⚡ Speech-to-Decision: fast, cheap, any decision

DuplexJev brings the System One idea (decision models, popularized by Jev) to raw speech: a text decision model only sees the transcript, and the transcript is where tone, stress and hesitation disappear.

A voice agent has to decide something at every pause. The usual way transcribes the speech and asks an LLM to write an
answer, one generation per question. DuplexJev listens to the raw audio and answers **all** of your questions as
multiple choice in **one forward pass**, reading a single token per question: no transcription, no text decoding.
Each answer comes with its probability, so you also get a confidence score.

| | **DuplexJev** | ASR → LLM, same GPU and LLM |
|---|:---:|---|
| ten decisions about one utterance (one H200) | **92 ms** | 1,978 ms |
| throughput, every request under 0.25 s | **~170 decisions / s** on one H200 | one generation per question |
| 50 questions over a 5k-token context | **5.5 s** | 111.5 s |
| cost | under **$0.50 per million decisions** (4B) | |
| a new kind of decision | **one line of text at runtime** | new data, new model |
| hardware | **~10 GB** (4B) · one 80 GB GPU (32B) | |

**What it can decide, from one clip:**

| | decisions |
|---|---|
| 🔁 **conversation flow** | turn state (finished, mid-sentence, backchannel, asking to wait) · real barge-in or just a backchannel · speech not meant for the assistant · end of turn right after a pause |
| 💬 **what is said** | intent · which filler fits · wants a human · any multiple-choice question about the content (spoken QA 90, against 91 for the same LLM reading the transcript) |
| 🎭 **how it is said** | emotion · which word is stressed · pauses and phrasing · speaking style (whisper, shouting, fast, slow) · laughter, breathing, coughs, sighs |
| 🧑 **who is speaking** | gender · same speaker or not · accent and dialect region · language |
| 👁️ **what it sees** (preview) | voice against a picture: stressed word vs. the red word 94.1, face vs. voice 93.5 (picture alone 50) |

**One clip, eight decisions.** 🔊 *"Wait, wait — stop. That's not the address I asked for."* — said while the assistant is talking
([listen](docs/audio/ex1.wav), synthetic clip) — one call to DuplexJev-32B-Turn, 0.2 s on one H200:

| question | answer | probability |
|---|---|---:|
| What is the user's turn state? | asking to wait | 0.92 |
| What does the user want? | navigation | 1.00 |
| Which filler fits? | “One moment —” | 0.73 |
| The assistant is speaking when this comes in. What should it do? | stop and listen | 1.00 |
| Speaker gender | male | 0.99 |
| Language | English | 1.00 |
| Emotion | angry | 1.00 |
| Can the assistant handle this by itself? | ask a follow-up question first | 0.51 |

<details>
<summary><b>What it hears beyond the words: one example per ability</b></summary>

Real outputs of DuplexJev-32B-Stress (vLLM, both option orders averaged) on short clips; the same cards, with a player, are on the
[project page](https://adventists-ai.github.io/duplexjev/#abilities). Real-speech stress and phrasing clips: MSPB (CC BY 4.0). The last three rows (Para models) give benchmark scores instead of examples.

| ability | what it decides | examples → model answer (DuplexJev-32B-Stress) |
|---|---|---|
| Turn state | Has the user finished, stopped mid-sentence, only made a backchannel, or asked the assistant to wait? Intonation and rhythm decide it, not only the words. | 🔊 [帮我导航到最近的充电站。](docs/audio/ex4.wav) → **finished** 100%<br>🔊 [Could you turn the air conditioning down a little and, um —](docs/audio/ex3.wav) → **not finished** 99%<br>🔊 [Mm-hmm, yeah.](docs/audio/ex2.wav) → **just a backchannel** 100%<br>🔊 [等一下，我想想……](docs/audio/ex5.wav) → **asking to wait** 100% |
| Barge-in while speaking | While the assistant is talking, tell a real interruption from a backchannel: stop and listen, or keep talking. | 🔊 [Wait, wait — stop. That's not the address I asked for.](docs/audio/ex1.wav) → **stop and listen** 100%<br>🔊 [Mm-hmm, yeah.](docs/audio/ex2.wav) → **keep talking** 100% |
| Voices not meant for the assistant | People chatting nearby, a TV, or noise should neither trigger a reply nor stop the assistant. | 🔊 [哎你晚上吃什么？要不我们去吃火锅吧。——行啊，那我先订个位子。](docs/audio/ab_b1.wav) → **stay silent and keep waiting (not addressed to the assistant)** 100% |
| Emotion | Neutral, happy, angry or sad, heard in the voice rather than read from the words. | 🔊 [Wait, wait — stop. That's not the address I asked for.](docs/audio/ex1.wav) → **angry** 100%<br>🔊 [太好了，谢谢你！](docs/audio/ex6.wav) → **happy** 93%<br>🔊 [帮我导航到最近的充电站。](docs/audio/ex4.wav) → **neutral** 100% |
| Speaker gender | Perceived gender of the voice, a cue a transcript does not carry. | 🔊 [帮我导航到最近的充电站。](docs/audio/ex4.wav) → **male** 100%<br>🔊 [太好了，谢谢你！](docs/audio/ex6.wav) → **female** 100% |
| Stress (word focus) 🆕 | Same words, different stressed word, different meaning. The model tells which word carries the stress, and also when no word is stressed. | 🔊 [**张昊**昨晚做烤肉。](docs/audio/ab_m1.wav) → **张昊 (Zhang Hao)** 100%<br>🔊 [张昊**昨晚**做烤肉。](docs/audio/ab_m2.wav) → **昨晚 (last night)** 100%<br>🔊 [张昊昨晚**做烤肉**。](docs/audio/ab_m3.wav) → **做烤肉 (made barbecue)** 97%<br>🔊 [我明天去北京开会。](docs/audio/ab_n0.wav) → **no word is particularly emphasized** 96% |
| Pauses and phrasing 🆕 | Where the speaker pauses can change the structure of a sentence: one item or two, who did what to whom. | 🔊 [我买了巧克力雪糕｜和果汁。](docs/audio/ab_q1.wav) → **two (chocolate ice cream, juice)** 96%<br>🔊 [我买了巧克力｜雪糕｜和果汁。](docs/audio/ab_q2.wav) → **three (chocolate, ice cream, juice)** 98% |
| Non-verbal sounds 🆕 (Para) | Laughter, breathing, coughs, sighs, sniffing, throat clearing, sneezes, crying, with a "none" option. | Unseen real speech (NonverbalTTS test), AUC laughter / breathing / cough / sigh: 32B-Para 0.88 / 0.81 / 0.85 / 0.80, 4B-Para 0.80 / 0.76 / 0.75 / 0.81 (before: 0.40–0.73; an AudioSet AST tagger: 0.71–0.85). |
| Speaking style 🆕 (Para) | Whispering, shouting, very fast or slow, unusually high or low pitch, or a normal voice. | 7-way on held-out EARS speakers: 82% (32B-Para) / 67% (4B-Para), from 45–55%. |
| Who is speaking 🆕 (Para) | Same speaker or not, given a voice sample. A coarse, human-like sense of "same voice", far behind dedicated speaker models; not for identifying people. | Same-gender AISHELL-1 pairs: EER 18% (32B-Para) / 9% (4B-Para); official VoxCeleb1-H list: 40% / 37% (dedicated x-vector model: ~2–4%). |

</details>

<details>
<summary><b>How it works</b></summary>

**DuplexJev** turns a frozen speech LLM into a *typed decision engine* for full-duplex spoken agents
(product name: **Speech-to-Decision**). Hidden states of an off-the-shelf ASR encoder are projected into a frozen
LLM, and every runtime-declared question — *is the turn complete? which filler to play? who is speaking?* — is read
as a **single-token, closed-set distribution**: no ASR decoding, no text decoding. Many questions, about one call or
across many calls, share one forward pass. All released models are ready to serve with vLLM ([Models](#models)); we
recommend **[DuplexJev-32B-Para](https://huggingface.co/adventists-ai/DuplexJev-32B-Para)**, or
**[DuplexJev-4B-Para](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)** on smaller GPUs.

```
audio ──► Qwen3-ASR-0.6B encoder (frozen)
                 ▼  last layer
          connector: stack 2 frames → 6.25 tokens/s, MLP → d_LLM        ← trained (Turn models: + rank-16 LoRA on the LLM)
                 ▼
state + N typed questions ──► frozen LLM, ONE forward pass
                              (DuplexJev-32B: language model of Qwen3-VL-32B · DuplexJev-4B: Qwen3-4B)
                 ▼
          p(A|q1) … p(D|q1),  …,  p(A|qN) … p(D|qN)      — 0 decode steps
```

- **Typed single-token readout.** Each question lists its options under permuted letters; the answer is the next-token
  softmax restricted to those letters. The output is always valid, and `max p` is a confidence score.
- **Hears the speaker, not only the words.** Transcript distillation leaves speech LLMs deaf to gender and emotion;
  supervising the single answer token lifts both to about 90% while content understanding moves by about 1 point.
- **Prefix sharing.** Context and audio are encoded once and all question suffixes are packed into one row under a
  block-diagonal mask; answers match one-by-one runs up to bf16 noise.

</details>

## 🏆 Paralinguistic understanding: state of the art

Paralinguistic understanding means hearing *how* something is said, not only *what*. Three public benchmarks combined:
the four audio-only conditions of [LISTEN](https://arxiv.org/abs/2510.10444) (EACL 2026, built to test whether audio models
*listen* or just read the words), StressTest (word stress) and MSPB (prosody); 1,035 questions, none from a corpus in our
training data, and every model answered the same items.

| model | **combined** | LISTEN, 4-condition mean | MSPB |
|---|---:|---:|---:|
| 🤗 **DuplexJev-32B-Para v1.3** (open) | **50.2** | **34.5** | **77.3** |
| Gemini 3.8 Flash | 47.8 | 31.9 | 67.2 |
| Qwen3-Omni-30B | 45.0 | 32.5 | 71.9 |
| Gemini 3.1 Pro | 41.0 | 29.0 | 54.7 |
| GPT-audio-1.5 | 40.9 | 30.2 | 57.8 |
| Gemini 3.8 Live (realtime) | 40.0 | 30.5 | 46.9 |
| GPT-realtime-2.1 (realtime) | 35.6 | 24.8 | 47.7 |

**How far ahead:**

- **Combined:** +2.4 over Gemini 3.8 Flash, +5.2 over Qwen3-Omni, +10.2 over Gemini 3.8 Live and +14.6 over GPT-realtime-2.1.
- **First or tied first on 8 of 11 corpora** against Gemini 3.8 Flash and Qwen3-Omni.
- **Emotion the words do not carry, on voices it has never heard** (LISTEN neutral text, TESS): **38.3**, against 25.5 for Qwen3-Omni and 17.0 for Gemini 3.8 Flash.
- **Emotion with no words at all** (LISTEN laughs, sighs, breaths, full set): **28.1** against 21.0 for the best published result (Qwen3-Omni).
- **Prosody (MSPB): 77.3**, the highest of the seven models.
- **The realtime flagships trail their own offline models** (Gemini 3.8 Live below 3.8 Flash, GPT-realtime-2.1 below GPT-audio-1.5),
  yet hearing tone in real time is exactly what a voice agent needs.

<sub>Gemini 3.8 Flash leads StressTest (89.0 vs 86.5), and general models lead when the words themselves carry the emotion
(MELD dialogue: 59.6 for Qwen3-Omni vs 38.5); dialogue emotion data is in our next training round. Thanks to Dipankar Sarkar for
suggesting the split: LISTEN's ESD and CREMA-D items (in our training data) are left out of the headline; with them, 1,128 items,
DuplexJev scores 50.9. Per-test table, protocol and every model's raw answer: [Detailed results](#against-closed-models-realtime-flagships-included),
[`benchmarks/paraling_compare`](benchmarks/paraling_compare).</sub>

## 🥇 Task by task

Ahead of dedicated models and closed flagships on most decision tasks we measured (same items for every model in a row):

| task | benchmark | **DuplexJev** | next best |
|---|---|---:|---|
| which word is stressed, zh / en | held-out sentences† | **95.5 / 96.0** | 71.5 / 88.5 Gemini 3.8 Flash |
| what the stress means | DEBATE held-out† | **93.0** | 79.5 Gemini 3.8 Flash |
| emotion from the voice, 4-way | ESD + CREMA-D† | **92.0** | 46.0 Qwen3-Omni-30B |
| speaking style (whisper, shout, fast …) | EARS held-out speakers† | **79.5** | 68.5 Gemini 3.1 Pro |
| prosody | MSPB | **77.3** | 71.9 Qwen3-Omni-30B |
| emotion, neutral words | LISTEN, full set | **50.6** | 34.9 Gemini 2.5 Pro (published) |
| laughter / breathing / cough / sigh, AUC | NonverbalTTS test (unseen) | **0.88 / 0.81 / 0.85 / 0.80** | 0.71 / 0.75 / 0.85 / 0.76 AudioSet AST tagger |
| turn action, zero-shot, zh / en | CoDeTT | **69.2 / 70.0** | 59.5 / 65.4 best dedicated turn model (NAMO-Turn / FireRedChat) |
| live end of turn, 14 languages, 300 ms | LiveKit [eot-bench](https://github.com/livekit/eot-bench) | **fewer false cut-offs than Smart Turn v3.2 in all 14** | English 20.0% vs 35.2%, Japanese 14.1% vs 34.9% |
| gender | AISHELL-1, LibriSpeech | **93.8** | ~55 speech LLMs trained on transcripts |
| voice vs. picture (preview) | held-out synthetic sets | **94.1 / 93.5** | 50 picture alone |

<sub>† Test sets from corpora in our training data (held-out speakers or sentences); the other models are zero-shot.
Full tables: [Detailed results](#detailed-results).</sub>

## Quick start

**Get a result in 30 seconds:**

```bash
pip install duplexjev
python -c "from duplexjev.remote import ApiClient; print(ApiClient('https://api.adventists.cn/duplexjev').decide('your.wav'))"
```

**Self-host:** `vllm serve adventists-ai/DuplexJev-4B-Para` (~10 GB GPU) with the `duplexjev-vllm` plugin ([Quick start](#quick-start)) ·
**Pipecat:** `pip install pipecat-duplexjev` · **LiveKit Agents:** `pip install livekit-plugins-duplexjev` ([integrations/livekit](integrations/livekit)) — audio end-of-turn detection, plus emotion and non-verbal sounds from the same pass.

> 🏢 **Building a product?** Our full-duplex voice API already runs on **300+ AI hardware products and 100,000+ devices**.
> Contact centres, in-car assistants, devices, voice-agent platforms: weights are CC BY-NC 4.0, and we offer
> **commercial licences and a hosted Speech-to-Decision API**. Email **jiejin@adventists.ai**.

⭐ **Star the repo** to follow new abilities — image + voice and an open full-duplex pipeline are next.

**Without a GPU** — ask our trial API (rate-limited, audio is not stored):

```bash
pip install duplexjev
duplexjev quick call.wav --api https://api.adventists.cn/duplexjev            # English speech
duplexjev quick call_zh.wav --api https://api.adventists.cn/duplexjev --lang zh
```

```
decision       answer                             confidence
------------------------------------------------------------
turn state     finished, the assistant can reply  0.74
what to do     reply now                          0.50
...
8 decisions in 80 ms
```

Your own questions: `--questions table.json` (a list of `{"id", "text", "options"}`), or in Python
`quick("call.wav", questions=[Question("product", "Which product is the user asking about?", ["phone", "laptop", "other"])])`.
The same page with a recorder is at [api.adventists.cn/duplexjev](https://api.adventists.cn/duplexjev/).

**On your own GPU** with vLLM:

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-4B-Para --max-model-len 4096   # or adventists-ai/DuplexJev-32B-Para
duplexjev quick call.wav --vllm http://localhost:8000/v1           # the default table against your server
duplexjev gateway --vllm http://localhost:8000/v1 --port 8020      # this web demo + HTTP API in front of it
```

Or ask your own questions from Python (one request per question, `max_tokens=1`, OpenAI-compatible API):

```python
from vllm_client import DuplexJevClient      # examples/vllm_client.py: openai + standard library only

dj = DuplexJevClient("http://localhost:8000/v1")
dj.decide(open("call_017.wav", "rb").read(), {
    "turn":    ("Has the user finished speaking?", ["finished", "not finished"]),
    "gender":  ("What is the perceived gender of the speaker?", ["female", "male"]),
    "emotion": ("What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"]),
})
# {'turn': {'answer': 'finished', 'confidence': 0.88, 'probs': {...}}, 'gender': {...}, 'emotion': {...}}

# Chinese speech: ask in Chinese
dj.decide(open("call_018.wav", "rb").read(), {"gender": ("说话人的性别是？", ["男性", "女性"])}, lang="zh")
```

The questions about one clip are sent concurrently and share the audio prefix in vLLM's prefix cache. Ask in the
language of the clip. The prompt format and the plugin are described on the
[model card](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) and in [`vllm_plugin/`](vllm_plugin).
To use a connector checkpoint with the PyTorch package instead, see [docs/connectors.md](docs/connectors.md#pytorch-package).

### Try the new abilities

With a Para model (`vllm serve adventists-ai/DuplexJev-4B-Para`, or locally with `pip install "duplexjev[speech]"`), the
default table of `duplexjev quick` has two more rows, *non-verbal sound* and *speaking style*. Or ask directly
(these are the wordings the models were trained on; always keep a "none" option):

```python
dj.decide(clip, {
    "laugh":  ("Is there laughter in this audio?", ["Yes", "No"]),
    "sound":  ("Besides speech, which sound can be heard in this audio?", ["laughter", "breathing", "coughing", "a sigh", "None of these"]),
    "style":  ("How is the speaker talking?", ["whispering", "speaking very loudly / shouting", "a normal speaking voice"]),
    "stress": ("Does the speaker stress one of these words, or none of them?", ["I", "say", "money", "None of these words is emphasized"]),
})
dj.decide(clip_zh, {"sound": ("除了说话，这段音频里还有哪种声音？", ["笑声", "呼吸声", "咳嗽", "叹气", "都没有"])}, lang="zh")
```

Same speaker or not: put a voice sample and the new clip in one file with about 0.8 s of silence between them and ask
`"Before the silence is a voice sample of someone; after it is another clip. Are they the same speaker?"` with options
`["different speakers", "same speaker"]` (Chinese: `"静音前是某人的一段声音样本，静音后是另一段语音。两段是同一个人吗？"`, `["是不同的人", "是同一个人"]`).
Ready-made scripts: [`examples/para_abilities.py`](examples/para_abilities.py).

### Speech + image together (research preview)

[DuplexJev-32B-Vision](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision) puts Qwen3-VL's vision encoder back
in front of the DuplexJev-32B-Para language model, so one question can use what it **hears** and what it **sees**
(one 80 GB GPU). Try it in the browser: [DuplexJev-Vision Space](https://huggingface.co/spaces/BitKnight/DuplexJev-Vision).

```python
# pip install "duplexjev[vision]>=0.5"
from duplexjev.vision import VisionDecider
vd = VisionDecider.from_pretrained("adventists-ai/DuplexJev-32B-Vision")
vd.decide(audio="clip.wav", image="face.jpg", questions=[
    {"id": "match", "text": "Does the face in the picture show the same emotion as the speaker's voice?", "options": ["Yes", "No"]}])
vd.chat(image="photo.jpg", text="What is written in the picture?")
```

Or serve it with vLLM (`pip install "duplexjev-vllm>=0.3"`), about 0.05 s per question with a picture and a 3 s clip:

```bash
vllm serve adventists-ai/DuplexJev-32B-Vision --max-model-len 8192 --limit-mm-per-prompt '{"image": 1, "audio": 1, "video": 0}'
python examples/vllm_vision_client.py clip.wav face.jpg
```

Held-out results: stressed word vs. the red word 94.1, face vs. voice 93.5, sound vs. caption 97.5 (picture alone: 50);
real photos of unseen speakers with their own voice 64.4, happy voices 81.7. Examples with real photos are on the
[project page](https://adventists-ai.github.io/duplexjev/#ab-vision).

## Models

Four models, each **one repository with everything needed** (audio encoder + connector + LLM), served with
[vLLM](https://github.com/vllm-project/vllm) and a small plugin ([Quick start](#quick-start)). Same prompts and plugin for all.

| model | use it for | GPU | highlights |
|---|---|---|---|
| ⭐ 🤗 [**DuplexJev-32B-Para**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) v1.3 | **best overall**: every ability, best paralinguistics, best live turn-taking | one 80 GB GPU | public paralinguistic benchmarks **50.2** · gender 93.8 · emotion 92.1 · qa100 96 |
| ⭐ 🤗 [**DuplexJev-4B-Para**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) v1.2 | **small GPU**, every ability, also holds a normal conversation; Japanese / Korean / Vietnamese / Thai speech; `duplexjev` default | ~10 GB | FLEURS error ja / ko 35.8 / 20.3% (v1.1: 59.6 / 61.5) · Mandarin breath / laughter 91 / 82% |
| 🤗 [**DuplexJev-4B-Para-LiveTurn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para-LiveTurn) | small GPU, **live end-of-turn** in LiveKit / Pipecat agents | ~10 GB | eot-bench false cut-offs at 300 ms: en **25.5%** / zh **38.8%** (Smart Turn v3.2: 35.2 / 42.9) |
| 🤗 [**DuplexJev-32B-Vision**](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision) (preview) | **voice + picture** in one question | one 80 GB GPU | stressed word vs red word 94.1 · face vs voice 93.5 (picture alone 50) · [try it](https://huggingface.co/spaces/BitKnight/DuplexJev-Vision) |

All on [ModelScope](https://www.modelscope.cn/organization/AdventistsAI) too (same weights):
`modelscope download --model AdventistsAI/DuplexJev-32B-Para --local_dir ./DuplexJev-32B-Para`, then `vllm serve ./DuplexJev-32B-Para`.
Weights: CC BY-NC 4.0; commercial licences: jiejin@adventists.ai.

**Version history, per-version scores, specialised and archived models** (Stress, Turn, Gemma-31B) and the research
connector checkpoints: **[docs/models.md](docs/models.md)**.

## News

- **2026-10-21** · 🤗 [**DuplexJev-4B-Para v1.2**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para): hears Japanese, Korean, Vietnamese and Thai (FLEURS error ja 59.6 → 35.8%, ko 61.5 → 20.3%) and much more Mandarin paralinguistics (breath, lip smack, laughter, sniff, stressed word: 57–65% → 81–95% on held-out speakers). Paralinguistic benchmarks combined unchanged (47.6 vs 47.8). [Release note](research/2026-10-weak-languages.md).
- **2026-10-15** · 🤗 [**DuplexJev-32B-Para v1.3**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para): much better at hearing stress (StressTest 71.5 → 86.5, MSPB 68.0 → 77.3). Public paralinguistic benchmarks combined **50.2**, against 47.8 for Gemini 3.8 Flash.
- **2026-10-15** · 🏆 Same questions for the realtime flagships (Gemini 3.8 Live, GPT-realtime-2.1) and GPT-audio-1.5, plus LiveKit eot-bench in all 14 languages.
- **2026-10-14** · 🤗 [**DuplexJev-4B-Para-LiveTurn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para-LiveTurn): live end-of-turn on a ~10 GB GPU.
- **Coming next:** an open-source full-duplex Jev dialogue pipeline.

All releases: [docs/news.md](docs/news.md).

## Detailed results

### Against closed models, realtime flagships included

Same items again (200 per test; 107–174 where a set is smaller), each
model asked once per item with the options in their original order and its first option letter taken as the answer:
`gemini-3.1-pro-preview` and `gemini-3.8-flash` (default thinking), `gpt-audio-1.5` (chat completions with audio input),
and the two realtime flagships through their live APIs: `gemini-3.8-live` (the question first, then the audio streamed as
realtime input in one manual activity; it can only answer by voice, so we read its output transcription) and
`gpt-realtime-2.1` (websocket session, text output, audio and question in one user message, no VAD).

| test | DuplexJev-32B-Para v1.3 | Qwen3-Omni-30B | Gemini 3.1 Pro | Gemini 3.8 Flash | Gemini 3.8 Live | GPT-audio-1.5 | GPT-realtime-2.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **public benchmarks combined** (LISTEN ×4, StressTest, MSPB; 1,035 items, no training corpora) | **50.2** | 45.0 | 41.0 | 47.8 | 40.0 | 40.9 | 35.6 |
| … with LISTEN's ESD + CREMA-D items (1,128 items) | **50.9** | 44.6 | 40.1 | 46.9 | 38.9 | 40.1 | 34.2 |
| **LISTEN, mean of 4 audio-only conditions** (no training corpora) | **34.5** | 32.5 | 29.0 | 31.9 | 30.5 | 30.2 | 24.8 |
| LISTEN neutral words | **43.0** | 33.5 | 25.0 | 31.0 | 25.0 | 24.5 | 15.5 |
| … unseen corpora only (TESS, SAVEE, RAVDESS) | **29.9** | 28.0 | 20.6 | 26.2 | 23.4 | 18.7 | 12.1 |
| LISTEN no words | **26.0** | 18.0 | 15.5 | 19.5 | 14.5 | **25.0** | 13.5 |
| LISTEN sarcastic words | **44.0** | 41.5 | 42.5 | 42.0 | 42.5 | 41.5 | 38.5 |
| LISTEN words match | 38.0 | **42.5** | 37.5 | 40.0 | 41.5 | 35.5 | 35.0 |
| emotion, ESD + CREMA-D† | **92.0** | 46.0 | 32.5 | 40.5 | 27.5 | 35.0 | 31.5 |
| which word is stressed, zh / en† | **95.5 / 96.0** | 62.0 / 73.6 | 58.5 / 52.9 | 71.5 / 88.5 | 43.0 / 56.9 | 28.0 / 42.0 | 14.0 / 30.5 |
| meaning from stress, DEBATE† | **93.0** | 36.5 | 51.5 | 79.5 | 55.0 | 27.5 | 26.0 |
| StressTest | 86.5 | 70.0 | 70.5 | **89.0** | 66.0 | 62.5 | 60.0 |
| speaking style, EARS† | **79.5** | 51.0 | 68.5 | 45.0 | 56.5 | 54.0 | 48.0 |
| MSPB (statement / question, …) | **77.3** | 71.9 | 54.7 | 67.2 | 46.9 | 57.8 | 47.7 |

† Test sets from corpora that are also in our training data (held-out speakers or sentences); every other model is zero-shot. The other rows are public benchmarks; of the LISTEN corpora only ESD and CREMA-D are in our emotion training data.

Over the six public benchmarks together, leaving out every item from a corpus in our training data, v1.3 scores 50.2 against 47.8 for Gemini 3.8 Flash, the best other model, and 45.0 for Qwen3-Omni; v1.3 is
ahead on MSPB and on LISTEN, Gemini 3.8 Flash on StressTest (89.0 vs 86.5). Averaged over LISTEN's four audio-only
conditions, DuplexJev scores 38.9 on the full set (37.8 on these items), ahead of
every published and tested model, closed realtime flagships included: state of the art in paralinguistic understanding on
LISTEN. Its largest lead is on emotion the words do not carry; on the full LISTEN neutral-text set it scores 50.6, against
34.9 for the best published result (Gemini 2.5 Pro, LISTEN paper). The realtime models trail their offline siblings
(Gemini 3.8 Live below 3.8 Flash, GPT-realtime-2.1 below GPT-audio-1.5), and the GPT models and Gemini 3.8 Live mostly
answer "neutral" to emotion questions (GPT-realtime-2.1: 150 of the 200 neutral-words items); we score them as asked
and do not know whether this is a design choice or a limit of what they hear. Gemini 3.8 Flash is ahead
on StressTest, and the general models are ahead when the words themselves carry the emotion. Unparsable answers count
as wrong (GPT-audio-1.5 2%, Gemini 3.8 Live 0.5%, the others ~0). Scripts, the item list and every model's raw answer:
[`benchmarks/paraling_compare`](benchmarks/paraling_compare).

**Shuffling the options** (all 24 orders of every 4-option item, 33,508 requests): the answer changes in 3.3% of orders,
only on uncertain items (0.01% when the answer's probability is ≥ 0.8), with no letter-position bias and the same
accuracy as the average over orders. Details: [`benchmarks/option_order`](benchmarks/option_order).

### Live end of turn: LiveKit eot-bench and Pipecat Smart Turn

Both benchmarks ask the hard live question: the user has *just* gone quiet — finished, or only pausing?

| model | eot-bench en: false cut-offs at 300 / 600 ms | eot-bench zh | Smart Turn v3.2 test, accuracy en / zh |
|---|---:|---:|---:|
| LiveKit Turn Detector v1 | 9.9 / 4.5% | 20.6 / 8.7% | – |
| **DuplexJev-32B-Para v1.2**, with the assistant's previous message (plugin setting) | **20.0 / 9.4%** | **37.7 / 12.2%** | – |
| **DuplexJev-32B-Para v1.2**, audio only | 23.5 / 11.1% | 40.0 / 12.8% | **94.0 / 91.1**° |
| **DuplexJev-4B-Para-LiveTurn** (~10 GB), with the assistant's previous message (plugin setting) | **25.5 / 12.3%** | **38.8 / 12.0%** | **93.3 / 89.6**° |
| Smart Turn v3.2 | 35.2 / 14.8% | 42.9 / 13.2% | 94.7 / 90.5 |

- eot-bench: validation set, decision 0.2 s into each pause, official `compute-metrics` / `compare-models`; its audio is
  not in our training data. Other rows are from the eot-bench repository. The context row is a full eot-bench run with
  the plugins' setting: `The assistant had just said: "..."` before the audio (the new plugins add it automatically),
  score P(finished) + P(backchannel). The audio-only DuplexJev rows use fast mode (one call per pause, at the 0.2 s score
  point).
- ° The Smart Turn v3.2 *training* set is in the v1.2 training mix, so this column is in-distribution for v1.2 (as it
  is for Smart Turn itself).
- **All 14 eot-bench languages** (DuplexJev-32B-Para v1.2 with the assistant's previous message, full run; trained on
  English and Chinese only, the question is asked in English for the other 12). False cut-offs at 300 / 600 ms:

  | | ar | de | es | fr | hi | id | it | ja | ko | nl | pt | tr |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | DuplexJev-32B-Para v1.2 | **34.7** / **11.5** | **21.3** / **8.3** | **33.4** / **13.6** | **23.1** / **9.1** | **29.7** / **5.3** | **31.9** / **14.0** | **28.6** / **10.9** | **14.1** / **3.7** | **23.8** / 9.7 | **25.3** / **10.5** | **32.5** / **10.8** | **29.8** / **13.2** |
  | Smart Turn v3.2 | 39.2 / 13.2 | 35.5 / 14.2 | 38.7 / 17.3 | 38.9 / 10.6 | 42.6 / 6.5 | 46.2 / 15.9 | 44.1 / 15.2 | 34.9 / 11.2 | 30.1 / 9.7 | 42.3 / 13.4 | 42.7 / 13.1 | 43.3 / 17.9 |
  | LiveKit Turn Detector v1 | 16.2 / 6.9 | 11.8 / 5.7 | 15.5 / 6.0 | 12.7 / 5.4 | 10.5 / 2.0 | 22.3 / 8.8 | 16.9 / 7.9 | 5.6 / 1.3 | 5.5 / 2.1 | 12.5 / 5.1 | 20.1 / 7.3 | 15.7 / 6.9 |

  Bold: fewer false cut-offs than Smart Turn v3.2. LiveKit's own detector is ahead in every language.
- Code: [`benchmarks/turn_detection/`](benchmarks/turn_detection); the context row uses
  [`eot_bench/duplexjev_ctx_adapter.py`](benchmarks/turn_detection/eot_bench/duplexjev_ctx_adapter.py).

> **Reproducing these numbers.** They come from **DuplexJev-32B-Para v1.2**. Since 2026-10-13 the free trial API
> serves the same model, so the plugin defaults already use it, but it is rate limited (30 requests per minute) and
> served from mainland China, so for live agents and benchmark runs serve it yourself:
>
> ```bash
> pip install "vllm[audio]>=0.29" duplexjev-vllm "duplexjev[server]>=0.6"
> vllm serve adventists-ai/DuplexJev-32B-Para --revision v1.2 --max-model-len 4096 --port 8000 &   # one 80 GB GPU
> duplexjev gateway --vllm http://localhost:8000/v1 --port 8420 --rate 1000000
> ```
>
> then point the plugin at it (`api_url=` / `url="http://localhost:8420"`), keep the default `readout="turn"`, and
> give it the agent's previous message (`detector.attach(session)` in LiveKit, `DuplexJevContextTracker` in
> Pipecat). Without the agent's message expect the audio-only row. Full eot-bench command:
> [`benchmarks/turn_detection/README.md`](benchmarks/turn_detection/README.md).

### Speed and spoken knowledge

| | DuplexJev-32B(-Turn) | DuplexJev-4B(-Turn) |
|---|---|---|
| one decision event (ten questions about one clip, sent as ten concurrent requests to the vLLM OpenAI server) | median **240 ms** on one H200 | about **80 ms** for the eight default questions on one H100 (our trial API) |
| GPU memory | one 80 GB GPU | ~10 GB |
| spoken knowledge, VoiceBench OBQA / MMSU (speech in) | **83.7 / 70.8** | 47.3 / 41.3 |

For reference, the same LLM answering from text reaches 95.4 / 79.3 (Qwen3-VL-32B). Against the usual cascade
(transcribe, then let the LLM generate JSON), the single-token readout answers ten decisions in 92 ms instead of
1,567 ms + 411 ms ASR (paper measurement: Qwen3-32B on one H200, vLLM, all ten questions in one packed request). Full tables:
[docs/paper_results.md](docs/paper_results.md).

<details>
<summary><b>Turn-taking vs. dedicated detectors</b> (Easy-Turn, CoDeTT)</summary>

Measured with DuplexJev-32B-Turn, the turn-taking base of the Para models (32B-Para v1.3: Easy-Turn 94.4). It decides turn-taking as well as dedicated detectors do, and in the same forward pass it also answers
every other question you declare (intent, emotion, gender, barge-in, …). It does not decode any tokens, so ten
decisions about one clip cost about as much as one.

| model | Easy-Turn test (complete / incomplete / backchannel / wait) | Easy-Turn overall | CoDeTT zh / en | other decisions in the same call | latency |
|---|---|---:|---:|---|---|
| Smart Turn v2 (95 MB) | 78.7 / 62.0 / – / – | – | – / 51.4 (v3) | no | 27 ms |
| TEN Turn Detection (7 B) | 86.7 / 89.3 / – / 91.0 | – | – | no | 204 ms |
| Easy-Turn (0.85 B) | 96.3 / 97.7 / 91.0 / 98.0 | 96.4 | 37.9 / – | no | 263 ms |
| NAMO-Turn · FireRedChat | – | – | 59.5 / – · – / 65.4 | no | – |
| GPT-4o-audio · Qwen3-Omni | – | – | 66.6 / 71.9 · 70.4 / 70.9 | yes, by generating text | seconds |
| **DuplexJev-32B-Turn** (ours) | **98.7** / 94.3 / 88.0 / 95.0 | **95.3** | **69.2 / 70.0** | **yes, any number, one pass** | **~0.24 s for 10 decisions** (one H200) |
| DuplexJev-4B-Turn (ours) | 92.3 / 93.0 / 90.0 / 94.0 | 92.5 | 67.0 / 68.6 | yes | ~0.08 s for 8 decisions (one H100) |

- **Easy-Turn**: accuracy (%) on the 800-item test set. Easy-Turn itself and our Turn models were both trained on its
  training split. Other rows are from the Easy-Turn paper (Table 2, their hardware); their latency is per single decision.
- **CoDeTT** ([arXiv:2603.25434](https://arxiv.org/abs/2603.25434)): 18 k items, system state given, mean of four
  action accuracies. Zero-shot for us: no CoDeTT data in training. Other rows are from the CoDeTT paper. Our protocol gives
  the current utterance as audio without history; the official one also plays earlier user turns.
- On our TurnBench-dev clip protocol (not the official leaderboard), the Turn models reach 88.7 (32B) and 87.3 (4B),
  against 60.7 and 49.4 before the upgrade.
- **Nothing else got worse.** For 32B: qa100 90 → 96, gender 89.4 → 91.5, emotion 90.6 → 91.1, VoiceBench MMSU
  70.8 → 72.1. How it was done (public data only, a rank-16 LoRA, one recipe for 4B and 32B):
  [research note](research/2026-10-turn-taking-lora.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench_turn_dark.svg">
    <img src="docs/assets/bench_turn_light.svg" alt="Turn-taking accuracy of DuplexJev against dedicated detectors and speech LLMs on CoDeTT and Easy-Turn" width="880">
  </picture>
</p>

</details>

<details>
<summary><b>Common decision tasks vs. dedicated models</b> (gender, emotion, speaker, non-verbal sounds, spoken QA)</summary>

| task | benchmark | DuplexJev | dedicated and other models |
|---|---|---:|---|
| turn state: finished / unfinished / backchannel / wait | Easy-Turn test (800) | **95.3** | Easy-Turn detector 96.4 · TEN Turn Detection and Smart Turn v2 cover only 3 / 2 of the 4 states |
| what to do, system state given | CoDeTT zh / en (18 k, zero-shot) | **69.2 / 70.0** | Easy-Turn 37.9 (zh) · Smart-Turn-v3 51.4 (en) · NAMO-Turn 59.5 (zh) · FireRedChat 65.4 (en) · GPT-4o-audio 66.6 / 71.9 · Qwen3-Omni 70.4 / 70.9 · Gemini-3-Pro 80.8 / 81.9 |
| gender | 800 real utterances (AISHELL-1, LibriSpeech) | **91.5** | speech LLMs trained on transcripts: ~55 (chance) |
| emotion (4-way) | 800 utterances (ESD, CREMA-D) | **91.1** | speech LLMs trained on transcripts: ~28 (chance) · Qwen3-Omni-30B 45.2 (same questions, 400 of the items)† |
| emotion, neutral words | LISTEN neutral-text, audio only (TESS, SAVEE, RAVDESS, ESD, CREMA-D) | **50.6** (v1.3, all 4,286; v1.2 on the 400 compared: 50.7) | Qwen3-Omni-30B 32.8 (same 400 items) · paper: Gemini 2.5 Pro 34.9, Qwen3-Omni 29.3 |
| emotion, no words at all | LISTEN paralinguistic (IEMOCAP laughs, sighs, breaths) | **28.1** (v1.3, all 975; v1.2 on the 400 compared: 26.0) | Qwen3-Omni-30B 18.2 (same 400 items) · paper: Qwen3-Omni 21.0, Gemini 2.5 Pro 15.7 |
| same speaker? | official VoxCeleb1-H list: different speakers share gender and nationality (2,000 trials) | **EER 36.9%** (4B-Para) / 40.3% (32B-Para); research checkpoints‡ 29.6% / 31.2% | gender alone: chance (50); dedicated speaker models (ECAPA-TDNN): EER ≈ 2% |
| laughter / breathing / cough / sigh | NonverbalTTS test (real speech, unseen), AUC | **0.88 / 0.81 / 0.85 / 0.80** (32B-Para) | AudioSet AST tagger 0.71 / 0.75 / 0.85 / 0.76 |
| spoken knowledge QA | VoiceBench OBQA / MMSU | **85.5 / 72.1** | same LLM reading the transcript: 95.4 / 79.3 |

Other systems' numbers are from their papers (Easy-Turn Table 2, CoDeTT, Dynamic-SUPERB Phase-2). Easy-Turn is in-domain
for both the Easy-Turn detector and our model. ‡ Research checkpoints trained only for speaker verification
(no VoxCeleb1 speakers), not released; the Para models trade some of that for keeping every other ability; see the [research note](research/2026-10-speaker-identity.md). On random pairs about half of the different-speaker pairs also differ in
gender, so a gender-only guess already reaches 74%; we therefore report hard pairs only.

</details>

Version-by-version scores and the earlier head-to-head with Qwen3-Omni on more items: [docs/models.md](docs/models.md).

## Benchmarks

| benchmark | task | size | link |
|---|---|---:|---|
| qa100 | spoken multiple-choice, synthetic speech (zh + en) | 100 | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| ZJU-ML | main-language part of the ZJU audio benchmark v2.0.0: spoken questions, half real speech | 100 | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| Easy-Turn | four-way turn state (complete / incomplete / backchannel / wait); the complete models are trained on its training split | 800 | Easy-Turn test set ([arXiv:2509.23938](https://arxiv.org/abs/2509.23938)) |
| gender | speaker gender, real speech (AISHELL-1, Common Voice, LibriSpeech), zh + en | 800 | [`evaluation/`](evaluation) |
| emotion | neutral / happy / angry / sad, acted speech (ESD, CREMA-D), zh + en | 800 | [`evaluation/`](evaluation) |

**Main language** = mean of qa100, ZJU-ML and Easy-Turn; **paralinguistics** = mean of gender and emotion;
**total** = mean of the two. Chance: 25% on the four-way tasks, 50% on gender. The [connector leaderboard](docs/connectors.md#leaderboard) is generated by
[`evaluation/build_leaderboard.py`](evaluation/build_leaderboard.py) from the result files next to it.

## Training

Frozen encoder and frozen LLM; only the connector is trained, with a patched
[Ultravox](https://github.com/fixie-ai/ultravox). Both complete models follow the same three stages:

1. **Content, R1 and R2.** Transcript distillation on the Ultravox v0.6 mixture (WenetSpeech, GigaSpeech, Common Voice,
   CoVoST 2, People's Speech, LibriSpeech, MLS, MUSAN), split into 100 disjoint packs; R1 and R2 use one pack each.
2. **Decisions.** Cross-entropy on the single option letter at the readout position: turn state (Easy-Turn training
   split), gender (AISHELL-1, LibriSpeech) and emotion (ESD, CREMA-D).
3. **Mixed run.** One run from R2 over content and decision samples (distillation for content, answer-token
   cross-entropy for decisions); 4,000 steps, learning rate 1e-4, with content weighted up (×10) and gender and emotion
   weighted down (×0.5) so that content understanding is kept while turn state, gender and emotion are learned.

Recipe, data links and scripts: [`training/`](training). The emotion corpora are not redistributed. The research
checkpoints in [docs/connectors.md](docs/connectors.md) use earlier versions of this recipe.

## Repository layout

| path | contents |
|---|---|
| [`vllm_plugin/`](vllm_plugin) | `duplexjev-vllm`: the vLLM plugin that serves DuplexJev-32B, DuplexJev-4B and DuplexJev-Gemma-31B |
| [`duplexjev/`](duplexjev) | the `duplexjev` package: `quick()` and the default decision table, vLLM and API clients, the web demo gateway; the PyTorch `Decider` for connector checkpoints |
| [`examples/`](examples) | [vLLM client](examples/vllm_client.py), quick starts, server client |
| [`evaluation/`](evaluation) | benchmark runners and results |
| [`training/`](training) | Ultravox configs and patches, data recipe, 100-pack split, encoder ports |
| [`research/`](research) | research notes: experiments and design decisions, with numbers and noise levels |
| [`docs/`](docs) | project page, [connector checkpoints](docs/connectors.md), [paper results](docs/paper_results.md), [package details](docs/package.md) |
| [`duplexjev/research/`](duplexjev/research), [`tests/`](tests), [`benchmarks/`](benchmarks) | paper code (readout, question contract, fused encoder), equivalence tests, latency and capacity experiments |

The paper code still contains paths from our cluster; see [docs/PATHS.md](docs/PATHS.md).

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). Complete models (DuplexJev-32B, DuplexJev-4B, DuplexJev-Gemma-31B): CC BY-NC 4.0, because their
connectors were trained on emotion data. Connector weights: Apache-2.0, except those trained on emotion data
(`-Emotion-` and `-Para-`), which are CC BY-NC 4.0 because ESD is licensed for research only. qa100: CC-BY-4.0.
The frozen encoders and LLMs keep their own licences (Falcon-H1: TII Falcon License; SenseVoice: FunASR Model License).
Some training corpora (e.g. WenetSpeech, CoVoST 2) have non-commercial terms. See [NOTICE](NOTICE).

## Citation

```bibtex
@misc{jin2026duplexjev,
  title         = {Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen {LLM} Hear Beyond the Transcript},
  author        = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Song, Haigang and Pang, Zhikun and Zhang, Xiaowen},
  year          = {2026},
  eprint        = {2610.02638},
  archivePrefix = {arXiv},
  note          = {Submitted to IEEE ICASSP 2027}
}
```

## Acknowledgements

DuplexJev-32B and DuplexJev-4B are built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[Qwen3](https://github.com/QwenLM/Qwen3) and Qwen3-ASR (DuplexJev-Gemma-31B on [Gemma 4](https://huggingface.co/google/gemma-4-31B-it)), trained with [Ultravox](https://github.com/fixie-ai/ultravox)
and served with [vLLM](https://github.com/vllm-project/vllm). The research checkpoints also use
[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize), Whisper, SenseVoice, SmolLM3
and Falcon-H1. We thank the ZJU team for the [audio benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark).
Claude (Anthropic) assisted with code. DuplexJev is an independent project and is not affiliated with TypeSafe (Jev).
Questions and issues: [GitHub Issues](https://github.com/adventists-ai/duplexjev/issues).
