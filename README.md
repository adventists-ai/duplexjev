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

- **2026-09-28** — New [research notes](research) section. First note: [does cross-layer fusion help a speech connector?](research/2026-09-connector-a-vs-b.md)
- **2026-09-28** — First **complete model**, 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B)
  (Qwen3-ASR-0.6B encoder + connector + Qwen3-4B in one repository), deployable with **vLLM**:
  `pip install duplexjev-vllm`, then `vllm serve adventists-ai/DuplexJev-4B`
  ([`duplexjev-vllm` 0.1.0](https://pypi.org/project/duplexjev-vllm/0.1.0/)). Same accuracy as the PyTorch package on
  all five benchmarks.
- **2026-09-27** — **16 new connectors** with the MOSS-Transcribe-Diarize and Whisper-small encoders on
  Qwen3-4B-2507, SmolLM3-3B, Falcon-H1-1.5B and Falcon-H1-3B, the
  [MOSS encoder](https://huggingface.co/adventists-ai/MOSS-Transcribe-Diarize-Whisper-Encoder), and
  [`duplexjev` 0.2.2](https://pypi.org/project/duplexjev/0.2.2/) (automatic `mode="batch"` for recurrent LLMs such as
  Falcon-H1). Qwen3-ASR-0.6B → **Qwen2.5-72B** connectors (A and B) are training; results soon.
- **2026-09-25** — 20 small connectors on Qwen3-0.6B / 1.7B / 4B with the Qwen3-ASR and SenseVoice-Small encoders (and Whisper-small with Qwen3-1.7B).
- **2026-09-24** — 7 Qwen3-32B connectors and [`duplexjev` 0.2.1](https://pypi.org/project/duplexjev/0.2.1/); paper submitted to ICASSP 2027.

Coming next: Speech-to-Decision commercial API (2026-10-10), open-source full-duplex Jev dialogue pipeline (2026-10-15).

## 3. Models

### Complete models (vLLM)

| model | built on | params | qa100 | ZJU-ML | Easy-Turn | gender | emotion | licence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) | Qwen3-4B + Qwen3-ASR-0.6B encoder | 4.2 B | 72 | 51 | 60.2 | 89.4 | 91.9 | CC BY-NC 4.0 |

One repository holds the encoder, the connector and the LLM; serve it with vLLM ([§4](#4-quick-start)). More sizes
will follow. The connectors below are the research checkpoints; `DuplexJev-4B` contains the weights of
`DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B`.

### Connectors

Each checkpoint is a **connector for one specific encoder and LLM**: it contains only the trained connector; the frozen
encoder and LLM are fetched automatically by `Decider.from_pretrained("adventists-ai/<repo>")`. It does not work with
other encoders or LLMs, including other sizes of the same family.

### Leaderboard

Total = (main language + paralinguistics) / 2. Main language = mean of qa100, ZJU-ML and Easy-Turn; paralinguistics = mean of gender and emotion (all %). One checkpoint per encoder × LLM × connector: the one with the best total.

| rank | encoder | LLM | connector | checkpoint | total | main language | paralinguistics |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Qwen3-ASR-0.6B | Qwen3-32B | A · cross-attention | 🤗 [DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B) | **80.6** | 71.2 | 90.0 |
| 2 | Qwen3-ASR-0.6B | Qwen3-4B | B · native | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B) | **75.9** | 61.1 | 90.7 |
| 3 | Qwen3-ASR-0.6B | Qwen3-32B | B · native | 🤗 [DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | **72.7** | 76.7 | 68.7 |
| 4 | MOSS-Transcribe-Diarize | Qwen3-4B-2507 | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507) | **70.0** | 49.6 | 90.5 |
| 5 | MOSS-Transcribe-Diarize | Falcon-H1-3B | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B) | **67.7** | 43.8 | 91.6 |
| 6 | SenseVoice-Small | Qwen3-4B | B · native | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B) | **66.5** | 50.0 | 82.9 |
| 7 | Qwen3-ASR-0.6B | Qwen3-1.7B | B · native | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B) | **66.3** | 45.8 | 86.9 |
| 8 | MOSS-Transcribe-Diarize | SmolLM3-3B | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B) | **65.6** | 40.0 | 91.2 |
| 9 | Whisper-small | Falcon-H1-3B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Falcon-H1-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Falcon-H1-3B) | **65.3** | 45.7 | 85.0 |
| 10 | MOSS-Transcribe-Diarize | Falcon-H1-1.5B | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B) | **63.2** | 37.1 | 89.3 |
| 11 | Whisper-small | Qwen3-4B-2507 | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507) | **63.1** | 45.7 | 80.6 |
| 12 | Qwen3-ASR-0.6B | Qwen3-0.6B | B · native | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B) | **62.8** | 36.2 | 89.3 |
| 13 | Whisper-small | SmolLM3-3B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-SmolLM3-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-SmolLM3-3B) | **62.6** | 39.5 | 85.7 |
| 14 | Whisper-small | Falcon-H1-1.5B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B) | **56.8** | 36.8 | 76.7 |
| 15 | SenseVoice-Small | Qwen3-0.6B | B · native | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B) | **55.2** | 33.8 | 76.7 |
| 16 | SenseVoice-Small | Qwen3-1.7B | B · native | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B) | **54.9** | 37.0 | 72.7 |
| 17 | Whisper-small | Qwen3-1.7B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-1.7B) | **54.2** | 37.9 | 70.6 |

#### Details

| checkpoint | qa100 | ZJU-ML | Easy-Turn | gender | emotion | params | CPU (8 threads) | licence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B | 82 | 61 | 70.5 | 89.9 | 90 | 33.4 B | – | CC BY-NC 4.0 |
| DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B | 72 | 51 | 60.2 | 89.4 | 91.9 | 4.2 B | 5.4 s | CC BY-NC 4.0 |
| DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B | 89 | 69 | 72.2 | 51.8 | 85.5 | 33.4 B | – | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507 | 61 | 42 | 45.8 | 95.5 | 85.4 | 4.4 B | 6.2 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B | 51 | 37 | 43.5 | 98.1 | 85.1 | 3.5 B | 171.3 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B | 61 | 37 | 52 | 76.8 | 89 | 4.3 B | 6.1 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B | 59 | 39 | 39.4 | 84.8 | 89.1 | 1.9 B | 2.1 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B | 33 | 41 | 46.1 | 97.9 | 84.6 | 3.4 B | 5.8 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Falcon-H1-3B | 50 | 45 | 42.1 | 90.4 | 79.5 | 3.3 B | 169.7 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B | 44 | 38 | 29.2 | 94 | 84.6 | 1.9 B | 100.3 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507 | 43 | 45 | 49.2 | 91.1 | 70.1 | 4.1 B | 5.2 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B | 38 | 26 | 44.5 | 89.4 | 89.2 | 0.8 B | 0.9 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-SmolLM3-3B | 35 | 42 | 41.5 | 93.5 | 77.8 | 3.2 B | 6.5 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B | 43 | 35 | 32.4 | 83.2 | 70.2 | 1.7 B | 85.3 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B | 34 | 27 | 40.4 | 67.5 | 85.8 | 0.8 B | 1.0 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B | 47 | 30 | 34.1 | 66.9 | 78.4 | 2.0 B | 2.4 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Qwen3-1.7B | 48 | 28 | 37.8 | 78.8 | 62.4 | 1.8 B | 2.4 s | CC BY-NC 4.0 |

Scores are % under the paper protocol (single-token readout, one H200); see [Benchmarks](#6-benchmarks). CPU = one
decision event (10 questions, a 4.5 s clip) on 8 CPU threads, fp32, not quantized; on one H200 every model answers in
about 0.05–0.2 s.

**Which one to use.**
- *Server, best overall:* `DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B` (one 80 GB GPU).
- *Edge / small GPU:* `DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B`; on CPU, `…-Qwen3-0.6B` answers in under 1 s.
- *Content decisions only, commercial use:* the content versions without `-Para` (Apache-2.0; gender and emotion at
  chance), e.g. `DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B` (qa100 90).
- *Falcon-H1 connectors* need a GPU (no fast CPU path for their Mamba layers).

Every trained variant (content-only, gender-only, emotion-only, all encoders) stays available on
🤗 [Hugging Face](https://huggingface.co/adventists-ai); the paper's full comparison of the Qwen3-32B variants is in
[docs/paper_results.md](docs/paper_results.md).

## 4. Quick start

**Serve a complete model with vLLM** (OpenAI-compatible API; one request per question, `max_tokens=1`):

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-4B --max-model-len 4096
```

```python
from vllm_client import DuplexJevClient      # examples/vllm_client.py: openai + standard library only

dj = DuplexJevClient("http://localhost:8000/v1")
dj.decide(open("call_017.wav", "rb").read(), {
    "turn":    ("Has the user finished speaking?", ["finished", "not finished"]),
    "gender":  ("What is the perceived gender of the speaker?", ["female", "male"]),
    "emotion": ("What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"]),
})
# {'turn': {'answer': 'finished', 'confidence': 0.88, 'probs': {...}}, 'gender': {...}, 'emotion': {...}}
```

The questions about one clip are sent concurrently and share the audio prefix in vLLM's prefix cache. The prompt
format and the plugin are described on the [model card](https://huggingface.co/adventists-ai/DuplexJev-4B) and in
[`vllm_plugin/`](vllm_plugin).

**PyTorch package** (every connector checkpoint; batching and prefix sharing in one forward pass):

```bash
pip install "duplexjev[speech]>=0.2.2"      # add [all] for the HTTP server
```

**One clip, several option groups:**

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B", device="auto")
groups = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
          Question("filler", "Which filler fits?", ["Sure —", "One moment —", "(stay silent)"]),
          Question("gender", "What is the perceived gender of the speaker?", ["female", "male"]),
          Question("emotion", "What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"])]

d.decide("call_017.wav", groups)
# {'turn': {'answer': 'finished', 'confidence': 0.97, 'probs': {...}}, 'filler': {...}, 'gender': {...}, 'emotion': {...}}

# Chinese speech: ask in Chinese, with Chinese options
d.decide("call_018.wav", [Question("gender", "说话人的性别是？", ["男性", "女性"], lang="zh")], lang="zh")
```

Ask in the language of the clip: the connectors were trained with language-matched prompts. Each model card lists the
question wordings used in training.

**Many clips, many option groups** — each group names the clip(s) it is about; everything runs in one batched pass:

```python
d.decide_batch(
    {"car1": "a.wav", "car2": "b.wav", "car3": "c.wav"},
    [Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
     Question("gender", "What is the perceived gender of the speaker?", ["female", "male"], audio=["car2", "car3"]),
     Question("human", "Does the user need a human agent?", ["yes", "no"], audio="car1")],
    context={"car1": "Driver asked to call home twice."})
```

**Batched server** — every request that arrives within one tick is answered in one pass:

```bash
duplexjev serve --model adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B --tick-ms 160
# POST /v1/decide        {"audio_b64": ..., "questions": [...]}
# POST /v1/decide_batch  {"audios": {"car1": ..., "car2": ...}, "questions": [{..., "audio": "car1"}]}
```

Local model copies, multi-GPU sharding, other Ultravox checkpoints and the package's guarantees:
[docs/package.md](docs/package.md).

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
**total** = mean of the two. Chance: 25% on the four-way tasks, 50% on gender. The leaderboard is generated by
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
| [`evaluation/`](evaluation) | benchmark runners, leaderboard script and results |
| [`training/`](training) | Ultravox configs and patches, data recipe, 100-pack split, encoder ports |
| [`benchmarks/`](benchmarks) | latency, SLO capacity and prefix-sharing experiments |
| [`docs/`](docs) | project page, [paper results](docs/paper_results.md), [package details](docs/package.md) |

The paper code still contains paths from our cluster; see [docs/PATHS.md](docs/PATHS.md).

## 9. License

Code: Apache-2.0 ([LICENSE](LICENSE)). Complete models follow the licence of the connector they contain. Connector weights: Apache-2.0, except those trained on emotion data
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
