<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.svg">
    <img src="docs/assets/banner_light.svg" alt="DuplexJev — batched speech decisions without decoding" width="880">
  </picture>
</p>

<p align="center">
  <b>English</b> &nbsp;|&nbsp; <a href="README_CN.md">中文</a>
</p>

<p align="center">
  🎧 <a href="https://api.adventists.cn/duplexjev/">Try it online</a> &nbsp;|&nbsp;
  🌐 <a href="https://adventists-ai.github.io/duplexjev/">Project page</a> &nbsp;|&nbsp;
  📄 <a href="https://arxiv.org/abs/2610.02638">Paper (arXiv:2610.02638)</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/adventists-ai/DuplexJev-32B-Turn">DuplexJev-32B-Turn</a> · <a href="https://huggingface.co/adventists-ai/DuplexJev-4B-Turn">4B-Turn</a> &nbsp;|&nbsp;
  📦 <a href="https://pypi.org/project/duplexjev/">PyPI</a> &nbsp;|&nbsp;
  📊 <a href="https://huggingface.co/datasets/adventists-ai/qa100">qa100</a> &nbsp;|&nbsp;
  📝 <a href="#10-citation">Citation</a>
</p>

---

**DuplexJev answers the closed questions a full-duplex voice agent asks about every user turn — *finished? reply or keep
listening? backchannel? emotion, gender, intent?* — as one token from an LLM that hears the audio: no decoding, many
questions in one pass.**

| [DuplexJev-32B-Turn](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) | score | reference |
|---|---:|---|
| **Speed** | | |
| ten decisions about one clip | **≈ 0.24 s** | one H200, all questions in one pass; the Easy-Turn detector needs 0.26 s for one decision |
| ten decisions in one packed request, vs. a cascade (ASR, then the LLM writes JSON) | **92 ms** | cascade: 1,567 ms + 411 ms ASR (paper measurement, Qwen3-32B) |
| **Main language** | | |
| turn-taking, Easy-Turn test set | **95.3** | dedicated Easy-Turn detector 96.4 |
| turn-taking, CoDeTT zh / en (zero-shot) | **69.2 / 70.0** | Qwen3-Omni 70.4 / 70.9; dedicated turn models 37.9–65.4 |
| spoken QA, VoiceBench OBQA / MMSU (question spoken) | **85.5 / 72.1** | same LLM reading the transcript as text: 95.4 / 79.3 |
| **Paralinguistics** | | |
| gender / emotion | **91.5 / 91.1** | chance (~55 / ~28) for speech LLMs trained on transcripts only |
| speaker verification, VCTK / LibriSpeech test-clean / VoxCeleb1 *(preview‡)* | **88.5 / 88.5 / 71.2** | speech LLMs zero-shot: ~50 (chance) |

‡ Research preview: a 4B checkpoint trained on public speaker-verification pairs, not yet part of the released models.

Models: **DuplexJev-32B-Turn** (one 80 GB GPU) and **DuplexJev-4B-Turn** (~10 GB) ([§3](#3-models)). Full comparison with dedicated detectors: [§5](#5-performance).

## 1. Introduction

**DuplexJev** turns a frozen speech LLM into a *typed decision engine* for full-duplex spoken agents
(product name: **Speech-to-Decision**). Hidden states of an off-the-shelf ASR encoder are projected into a frozen
LLM, and every runtime-declared question — *is the turn complete? which filler to play? who is speaking?* — is read
as a **single-token, closed-set distribution**: no ASR decoding, no text decoding. Many questions, about one call or
across many calls, share one forward pass. All released models are ready to serve with vLLM ([§3](#3-models)); we
recommend **[DuplexJev-32B-Turn](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn)**, or
**[DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn)** on smaller GPUs.

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

## 2. News

- **2026-10-05** — 📄 Paper on arXiv: [arXiv:2610.02638](https://arxiv.org/abs/2610.02638). 🤗 [**DuplexJev-32B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn)
  and [**DuplexJev-4B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn): a turn-taking upgrade (connector + rank-16
  LoRA on public turn-taking data). 32B-Turn: Easy-Turn 95.3, CoDeTT 69.2 / 70.0 zero-shot, with qa100, gender and emotion
  all slightly up. [Research note](research/2026-10-turn-taking-lora.md).
- **2026-10-03** — 🤗 [**DuplexJev-Gemma-31B**](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B), the first
  non-Qwen model: Qwen3-ASR-0.6B encoder + connector + Gemma-4-31B-it with a merged LoRA. In vLLM: qa100 97, gender
  94.4, emotion 90.6 (no turn-taking training yet). Needs [`duplexjev-vllm`](https://pypi.org/project/duplexjev-vllm/) ≥ 0.2.0.
- **2026-10-01** — 🎧 [**Online demo and trial API**](https://api.adventists.cn/duplexjev/) (DuplexJev-4B) and
  [`duplexjev` 0.3](https://pypi.org/project/duplexjev/): `duplexjev quick call.wav` prints a table of eight default
  decisions (turn state, what to do, barge-in, intent, emotion, gender, language, wants a human) for any clip.
- **2026-09-30** — 🤗 [**DuplexJev-32B**](https://huggingface.co/adventists-ai/DuplexJev-32B), our new recommended model:
  Qwen3-ASR-0.6B encoder + connector + the language model of Qwen3-VL-32B, one repository, served with vLLM.
  Best total score so far (87.7): qa100 90, Easy-Turn 79.1, gender 89.4, emotion 90.6; spoken VoiceBench MMSU 70.8
  (earlier 32B connectors: 56–60). DuplexJev-4B updated with the same recipe (total 75.9 → 78.6). Connector checkpoints moved to [their own page](docs/connectors.md).
- **2026-09-28** — New [research notes](research) section. First note: [does cross-layer fusion help a speech connector?](research/2026-09-connector-a-vs-b.md)
- **2026-09-28** — First complete model, 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B), and the
  vLLM plugin [`duplexjev-vllm`](https://pypi.org/project/duplexjev-vllm/).

Coming next: Speech-to-Decision commercial API (2026-10-10), open-source full-duplex Jev dialogue pipeline (2026-10-15).

## 3. Models

Each model is **one repository with everything needed** (audio encoder + trained connector + LLM) and runs on
[vLLM](https://github.com/vllm-project/vllm) with a small plugin ([§4](#4-quick-start)).

| model | for | built on | params | GPU memory | qa100 | ZJU-ML | Easy-Turn | gender | emotion | total | licence |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ⭐ 🤗 [**DuplexJev-32B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) | server, best quality | Qwen3-VL-32B (language model, + rank-16 LoRA) + Qwen3-ASR-0.6B encoder | 33.0 B | one 80 GB GPU | **96** | 87 | **95.3**\* | **91.5** | **91.1** | **92.2** | CC BY-NC 4.0 |
| 🤗 [DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn) | small GPU, edge | Qwen3-4B (+ rank-16 LoRA) + Qwen3-ASR-0.6B encoder | 4.2 B | ~10 GB | 74 | 52 | 92.5\* | 89.4 | 92.0 | 80.0 | CC BY-NC 4.0 |

Earlier versions without the turn-taking upgrade, [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B) (total 87.7)
and [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) (78.6), stay online but are superseded: the Turn models
are equal or better on every benchmark we run.

% under the paper protocol (single-token readout); total = mean of main language (qa100, ZJU-ML, Easy-Turn) and
paralinguistics (gender, emotion), see [Benchmarks](#6-benchmarks). \* Both models were trained with the Easy-Turn
training split (disjoint from the test set), so their Easy-Turn scores are in-domain. Both answer ten questions about one clip in about
0.1–0.25 s on one GPU.

**Also available: 🤗 [DuplexJev-Gemma-31B](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B)** (Gemma-4-31B-it +
Qwen3-ASR-0.6B encoder, ~58 GB, one 80 GB GPU, CC BY-NC 4.0). No turn-taking training yet; like the Turn models, its LLM carries a merged LoRA. Served by vLLM in the prompt wording of [§4](#4-quick-start): qa100 97, ZJU-ML 87, gender 94.4, emotion
90.6; Easy-Turn 68.0 zero-shot (no turn-taking training). It is sensitive to question wording (paper-protocol wording:
gender 78.5, emotion 60.9), so it is not in the table above; details on its model card.

**Which one to use.** `DuplexJev-32B-Turn` wherever an 80 GB GPU is available; `DuplexJev-4B-Turn` on smaller GPUs. The Turn
models are drop-in replacements (same prompts, same plugin) and equal or better on every benchmark we run (within noise). The 32B models
use only the language model of Qwen3-VL-32B: their inputs are audio and text for now (image input is planned).

> **Research checkpoints.** Behind these models are several dozen connector checkpoints for other encoders (MOSS, Whisper,
> SenseVoice), LLMs (Qwen3 0.6B–32B, Falcon-H1, SmolLM3) and both connector types, used with the `duplexjev` PyTorch
> package. Leaderboard, details and package usage: **[docs/connectors.md](docs/connectors.md)**.

## 4. Quick start

**Without a GPU** — ask our trial API (DuplexJev-4B; rate-limited, audio is not stored):

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
vllm serve adventists-ai/DuplexJev-32B --max-model-len 4096      # or adventists-ai/DuplexJev-4B
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
[model card](https://huggingface.co/adventists-ai/DuplexJev-32B) and in [`vllm_plugin/`](vllm_plugin).
To use a connector checkpoint with the PyTorch package instead, see [docs/connectors.md](docs/connectors.md#pytorch-package).

## 5. Performance

### Turn-taking vs. dedicated detectors

DuplexJev-32B-Turn decides turn-taking as well as dedicated detectors do, and in the same forward pass it also answers
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

## 6. Benchmarks

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

## 7. Training

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

## 8. Repository layout

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

## 9. License

Code: Apache-2.0 ([LICENSE](LICENSE)). Complete models (DuplexJev-32B, DuplexJev-4B, DuplexJev-Gemma-31B): CC BY-NC 4.0, because their
connectors were trained on emotion data. Connector weights: Apache-2.0, except those trained on emotion data
(`-Emotion-` and `-Para-`), which are CC BY-NC 4.0 because ESD is licensed for research only. qa100: CC-BY-4.0.
The frozen encoders and LLMs keep their own licences (Falcon-H1: TII Falcon License; SenseVoice: FunASR Model License).
Some training corpora (e.g. WenetSpeech, CoVoST 2) have non-commercial terms. See [NOTICE](NOTICE).

## 10. Citation

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

## 11. Acknowledgements

DuplexJev-32B and DuplexJev-4B are built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[Qwen3](https://github.com/QwenLM/Qwen3) and Qwen3-ASR (DuplexJev-Gemma-31B on [Gemma 4](https://huggingface.co/google/gemma-4-31B-it)), trained with [Ultravox](https://github.com/fixie-ai/ultravox)
and served with [vLLM](https://github.com/vllm-project/vllm). The research checkpoints also use
[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize), Whisper, SenseVoice, SmolLM3
and Falcon-H1. We thank the ZJU team for the [audio benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark).
Claude (Anthropic) assisted with code.
Questions and issues: [GitHub Issues](https://github.com/adventists-ai/duplexjev/issues).
