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
  📄 Paper (arXiv, coming soon) &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/adventists-ai">Weights</a> &nbsp;|&nbsp;
  📦 <a href="https://pypi.org/project/duplexjev/">PyPI</a> &nbsp;|&nbsp;
  📊 <a href="https://huggingface.co/datasets/adventists-ai/qa100">qa100</a> &nbsp;|&nbsp;
  📝 <a href="#10-citation">Citation</a>
</p>

---

## 1. Introduction

**DuplexJev** turns a frozen speech LLM into a *typed decision engine* for full-duplex spoken agents
(product name: **Speech-to-Decision**). Hidden states of an off-the-shelf ASR encoder are projected into a frozen
LLM, and every runtime-declared question — *is the turn complete? which filler to play? who is speaking?* — is read
as a **single-token, closed-set distribution**: no ASR decoding, no text decoding. Many questions, about one call or
across many calls, share one forward pass.

```
audio ──► frozen ASR encoder (Qwen3-ASR, MOSS-Transcribe, Whisper, SenseVoice, …)
            ├─ B: last layer ──────────────────────────────────┐
            └─ A: cross-attention fusion over three layers ────┤  (zero-init, residual)
                                                               ▼
                              connector (frame stacking → 6.25 tokens/s, MLP → d_LLM)
                                                               ▼
state + N typed questions ──► frozen LLM (0.6B – 32B), ONE forward pass
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
| ⭐ 🤗 [**DuplexJev-32B**](https://huggingface.co/adventists-ai/DuplexJev-32B) | server, best quality | Qwen3-VL-32B (language model) + Qwen3-ASR-0.6B encoder | 33.0 B | one 80 GB GPU | 90 | 87 | 79.1\* | 89.4 | 90.6 | **87.7** | CC BY-NC 4.0 |
| 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) | small GPU, edge | Qwen3-4B + Qwen3-ASR-0.6B encoder | 4.2 B | ~10 GB | 74 | 50 | 78.1\* | 88.6 | 91.2 | 78.6 | CC BY-NC 4.0 |

% under the paper protocol (single-token readout); total = mean of main language (qa100, ZJU-ML, Easy-Turn) and
paralinguistics (gender, emotion), see [Benchmarks](#6-benchmarks). \* Both models were trained with the Easy-Turn
training split (disjoint from the test set), so their Easy-Turn scores are in-domain. Both answer ten questions about one clip in about
0.1–0.25 s on one GPU.

**Which one to use.** `DuplexJev-32B` wherever an 80 GB GPU is available; `DuplexJev-4B` on smaller GPUs. DuplexJev-32B
uses only the language model of Qwen3-VL-32B: its inputs are audio and text for now (image input is planned).

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

Ten decisions about one call, Qwen3-32B on one H200 (vLLM, bf16): **92 ms** with the single-token readout, against
**1,567 ms + 411 ms ASR** for the usual cascade (transcribe, then let the LLM generate JSON). Within a 0.25 s budget the
readout serves 17 such events per second; the cascade serves none. Full tables: [docs/paper_results.md](docs/paper_results.md).

## 6. Benchmarks

| benchmark | task | size | link |
|---|---|---:|---|
| qa100 | spoken multiple-choice, synthetic speech (zh + en) | 100 | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| ZJU-ML | main-language part of the ZJU audio benchmark v2.0.0: spoken questions, half real speech | 100 | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| Easy-Turn | four-way turn state (complete / incomplete / backchannel / wait), zero-shot | 800 | Easy-Turn test set ([arXiv:2509.23938](https://arxiv.org/abs/2509.23938)) |
| gender | speaker gender, real speech (AISHELL-1, Common Voice, LibriSpeech), zh + en | 800 | [`evaluation/`](evaluation) |
| emotion | neutral / happy / angry / sad, acted speech (ESD, CREMA-D), zh + en | 800 | [`evaluation/`](evaluation) |

**Main language** = mean of qa100, ZJU-ML and Easy-Turn; **paralinguistics** = mean of gender and emotion;
**total** = mean of the two. Chance: 25% on the four-way tasks, 50% on gender. The [connector leaderboard](docs/connectors.md#leaderboard) is generated by
[`evaluation/build_leaderboard.py`](evaluation/build_leaderboard.py) from the result files next to it.

## 7. Training

Frozen encoder and frozen LLM; only the connector is trained, with a patched [Ultravox](https://github.com/fixie-ai/ultravox).

1. **Content (R1–R2).** Transcript distillation on the Ultravox v0.6 mixture (WenetSpeech, GigaSpeech, Common Voice,
   CoVoST 2, People's Speech, LibriSpeech, MLS, MUSAN), split into 100 disjoint packs; R1 and R2 use one pack each.
2. **Decisions.** Cross-entropy on the single option letter at the readout position. Gender: AISHELL-1 and LibriSpeech;
   emotion: ESD and CREMA-D.
3. **Mixed objective (`-Para`).** One run over content + gender + emotion: distillation for content samples,
   answer-token cross-entropy for decision samples.

Recipe, data links and scripts: [`training/`](training). The emotion corpora are not redistributed.

## 8. Repository layout

| path | contents |
|---|---|
| [`duplexjev/`](duplexjev) | the installable package: `Decider`, `Question`, CLI and tick-batched server |
| [`duplexjev/research/`](duplexjev/research) | paper code: readout, question contract, encoder with fusion |
| [`vllm_plugin/`](vllm_plugin) | `duplexjev-vllm`: vLLM plugin for the complete models |
| [`research/`](research) | research notes: experiments and design decisions, with numbers and noise levels |
| [`tests/`](tests) | equivalence and batch-invariance tests |
| [`examples/`](examples) | quick starts, tick-batch timing, server client, [vLLM client](examples/vllm_client.py) |
| [`evaluation/`](evaluation) | benchmark runners, connector leaderboard script and results |
| [`training/`](training) | Ultravox configs and patches, data recipe, 100-pack split, encoder ports |
| [`benchmarks/`](benchmarks) | latency, SLO capacity and prefix-sharing experiments |
| [`docs/`](docs) | project page, [connector checkpoints](docs/connectors.md), [paper results](docs/paper_results.md), [package details](docs/package.md) |

The paper code still contains paths from our cluster; see [docs/PATHS.md](docs/PATHS.md).

## 9. License

Code: Apache-2.0 ([LICENSE](LICENSE)). Complete models (DuplexJev-32B, DuplexJev-4B): CC BY-NC 4.0, because their
connectors were trained on emotion data. Connector weights: Apache-2.0, except those trained on emotion data
(`-Emotion-` and `-Para-`), which are CC BY-NC 4.0 because ESD is licensed for research only. qa100: CC-BY-4.0.
The frozen encoders and LLMs keep their own licences (Falcon-H1: TII Falcon License; SenseVoice: FunASR Model License).
Some training corpora (e.g. WenetSpeech, CoVoST 2) have non-commercial terms. See [NOTICE](NOTICE).

## 10. Citation

```bibtex
@inproceedings{jin2027duplexjev,
  title     = {Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen {LLM} Hear Beyond the Transcript},
  author    = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Song, Haigang and Pang, Zhikun and Zhang, Xiaowen},
  booktitle = {Submitted to IEEE ICASSP},
  year      = {2027}
}
```

## 11. Acknowledgements

Built on [Ultravox](https://github.com/fixie-ai/ultravox), [Qwen3](https://github.com/QwenLM/Qwen3), Qwen3-ASR,
[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize), Whisper, SenseVoice,
SmolLM3 and Falcon-H1. We thank the ZJU team for the [audio benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark).
Claude (Anthropic) assisted with code.
Questions and issues: [GitHub Issues](https://github.com/adventists-ai/duplexjev/issues).
