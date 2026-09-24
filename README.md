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
  📝 <a href="#9-citation">Citation</a>
</p>

---

## 1. Introduction

**DuplexJev** turns a frozen speech LLM into a *typed decision engine* for full-duplex spoken agents
(product name: **Speech-to-Decision**). Hidden states of an off-the-shelf ASR encoder are projected into a frozen
LLM, and every runtime-declared question — *is the turn complete? which filler to play? who is speaking?* — is read
as a **single-token, closed-set distribution**: no ASR decoding, no text decoding. Many questions, about one call or
across many calls, share one forward pass.

Paper: *Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen LLM Hear Beyond the
Transcript* (ICASSP 2027 submission).

```
audio ──► frozen ASR encoder (Qwen3-ASR-0.6B, 12.5 Hz)
            ├─ B: last layer h18 ─────────────────────────────┐
            └─ A: cross-attention fusion  Q=h18, K=h14, V=h9 ─┤  (zero-init, residual)
                                                              ▼
                              projector (stack 2 frames → 6.25 tokens/s, MLP → d_LLM)
                                                              ▼
state + N typed questions ──► frozen LLM (Qwen3-32B), ONE forward pass
                                                              ▼
            p(A|q1) … p(D|q1),  …,  p(A|qN) … p(D|qN)      — 0 decode steps
```

- **Typed single-token readout.** Each question lists its options under permuted letters; the answer is the
  next-token softmax restricted to those letters. The output is always valid, and `max p` is a confidence score.
- **Hears the speaker, not only the words.** Speech LLMs trained by transcript distillation are deaf to gender and
  emotion, because their teacher only reads the transcript. Supervising the single answer token instead
  (cross-entropy on the option letter) lifts gender and emotion accuracy to 90%, while content understanding drops by
  1 point.
- **Prefix sharing.** Template, dialogue context and audio are encoded once; all question suffixes are packed into one
  row under a block-diagonal 4-D mask, with position ids restarting at the prefix length. The KV cache holds
  `P + ΣL_i` positions instead of `N(P + L)`, and answers match one-by-one runs up to bf16 noise.
- **Modular.** Any ASR encoder, a small trainable connector (17.8 M parameters; +3.2 M for the fusion block in
  variant A) and any frozen LLM that accepts input embeddings.

## 2. News

- **2026-09-25** — Released **8 small connectors** (Qwen3-0.6B / 1.7B / 4B with the Qwen3-ASR encoder, Qwen3-1.7B with Whisper-small), content and gender + emotion versions; see [Small connectors](#small-connectors-edge-sized).
- **2026-09-24** — Released **7 connectors and 2 encoder repositories** on 🤗 [Hugging Face](https://huggingface.co/adventists-ai) and [`duplexjev` 0.2.1](https://pypi.org/project/duplexjev/0.2.1/) (audio is now padded at the end: padding at the start cost up to 6 points on emotion; `text_model` / `audio_model` work offline).
- **2026-09** — Paper submitted to ICASSP 2027. Released the `duplexjev` package, research code, latency benchmarks and project page.
- **2026-10-10 (planned)** — Speech-to-Decision commercial API.
- **2026-10-15 (planned)** — Open-source full-duplex Jev dialogue pipeline.

## 3. Model download

Each checkpoint is a **connector for one specific pair of models**: it contains only the trained projector (and, for A,
the fusion block) that joins the frozen ASR encoder to the frozen LLM. It does not work with other encoders or LLMs,
including other sizes of the same family. `Decider.from_pretrained("adventists-ai/<repo>")` fetches the encoder and the
LLM automatically.

All connectors use the frozen **Qwen3-ASR-0.6B** encoder and the frozen **[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)** LLM.

| repository | connector | trained for | trainable params | license |
|---|---|---|---:|---|
| 🤗 [DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B) | A · cross-attention fusion | content (R2) | 21.1 M | Apache-2.0 |
| 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B) | B · last layer | content (R2), best spoken QA | 17.8 M | Apache-2.0 |
| 🤗 [DuplexJev-A-Gender-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Gender-Qwen3-ASR-0.6B-Qwen3-32B) | A | + speaker gender | 21.1 M | Apache-2.0 |
| 🤗 [DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B) | B | + speaker gender | 17.8 M | Apache-2.0 |
| 🤗 [DuplexJev-A-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | A | + emotion (4-way) | 21.1 M | CC BY-NC 4.0 |
| 🤗 [DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | B | + emotion (4-way) | 17.8 M | CC BY-NC 4.0 |
| 🤗 [DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B) | A | gender + emotion + content, mixed objective (best paralinguistic) | 21.1 M | CC BY-NC 4.0 |

Encoders (fetched automatically): 🤗 [Qwen3-ASR-0.6B-Encoder](https://huggingface.co/adventists-ai/Qwen3-ASR-0.6B-Encoder)
(for B) and 🤗 [Qwen3-ASR-0.6B-Encoder-XAttn](https://huggingface.co/adventists-ai/Qwen3-ASR-0.6B-Encoder-XAttn) (for A;
same weights, plus the fusion-block code). Both are the audio encoder of Qwen3-ASR-0.6B, unchanged, Apache-2.0.

The emotion connectors were trained on ESD, which is licensed for research only, so they are released for
non-commercial research use.

### Small connectors (edge-sized)

The same recipe (R1 → R2 → MIX-KD, last-layer connector B) on small frozen LLMs and two encoders. Scores are the
MIX-KD connectors under the paper protocol (%); CPU = one decision event of 10 questions on a 4.5 s clip, fp32,
not quantized (one H200 GPU: 40–80 ms for all four). Small LLMs answer knowledge questions poorly even from the
transcript, so use them for short decisions and speaker cues. Model cards list the `duplexjev` numbers too.

| encoder | LLM | content connector (Apache-2.0) | + gender & emotion, MIX-KD (CC BY-NC 4.0) | trainable | total | qa100 (speech / transcript) | gender | emotion | CPU, 8 threads |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR-0.6B | Qwen3-0.6B | 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-0.6B) | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B) | 9.4 M | 0.8 B | 38 / 45 | 89.4 | 89.2 | 0.9 s |
| Qwen3-ASR-0.6B | Qwen3-1.7B | 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-1.7B) | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B) | 11.5 M | 1.9 B | 59 / 66 | 84.8 | 89.1 | 2.1 s |
| Whisper-small | Qwen3-1.7B | 🤗 [DuplexJev-B-Whisper-small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Whisper-small-Qwen3-1.7B) | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-1.7B) | 29.4 M | 1.8 B | 48 / 66 | 78.8 | 62.4 | 2.4 s |
| Qwen3-ASR-0.6B | Qwen3-4B | 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-4B) | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B) | 12.6 M | 4.2 B | 72 / 82 | 89.4 | 91.9 | 5.4 s |

With the Qwen3-ASR encoder, even Qwen3-0.6B hears gender and emotion about as well as Qwen3-32B (89 / 89 vs. 90 / 90).

## 4. Quick start

```bash
pip install "duplexjev[speech]>=0.2.1"      # add [all] for the HTTP server
```

**One clip, several option groups** (the default):

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B", device="auto")
groups = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
          Question("filler", "Which filler fits?", ["Sure —", "One moment —", "(stay silent)"]),
          Question("gender", "What is the perceived gender of the speaker?", ["female", "male"])]

d.decide("call_017.wav", groups)
# {'turn': {'answer': 'finished', 'confidence': 0.97, 'probs': {...}}, 'filler': {...}, 'gender': {...}}

# Chinese speech: ask in Chinese, with Chinese options
d.decide("call_018.wav", [Question("gender", "说话人的性别是？", ["男性", "女性"], lang="zh")], lang="zh")
```

Ask in the language of the clip: the connectors were trained with language-matched prompts (Chinese questions and
options for Chinese speech). Each model card lists the question wordings used in training.

**Many clips, many option groups** (advanced): each group names the clip(s) it is about with `audio=<id>`, a list
of ids, or `"*"` for every clip. Everything runs in one batched pass.

```python
d.decide_batch(
    {"car1": "a.wav", "car2": "b.wav", "car3": "c.wav"},
    [Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
     Question("gender", "What is the perceived gender of the speaker?", ["female", "male"], audio=["car2", "car3"]),
     Question("human", "Does the user need a human agent?", ["yes", "no"], audio="car1")],
    context={"car1": "Driver asked to call home twice."})
# {'car1': {'turn': ..., 'human': ...}, 'car2': {'turn': ..., 'gender': ...}, 'car3': {...}}
```

**Batched server:** every request that arrives within one tick is answered in one pass.

```bash
duplexjev serve --model adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B --tick-ms 160
# POST /v1/decide        {"audio_b64": ..., "questions": [...]}
# POST /v1/decide_batch  {"audios": {"car1": ..., "car2": ...}, "questions": [{..., "audio": "car1"}]}
```

Command line: `duplexjev decide --model M call.wav --q "turn|Has the user finished?|finished,not finished"`.

### Choosing a model

`Decider.from_pretrained(...)` (CLI: `--model`) takes a Hugging Face repo id or a local path of a **speech checkpoint in
Ultravox format**. A checkpoint is an ASR encoder plus the connector trained for it, and it names the frozen LLM it was
trained against. So you pick one name; the encoder comes with it, and the LLM is fetched from the id in its config.

```python
d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B")      # download everything
d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B",
                            text_model="/data/models/Qwen3-32B")                        # reuse a local LLM copy
d = Decider.from_pretrained("/data/models/my-checkpoint", device="auto")                # local path, shard over GPUs
```

| checkpoint | encoder (inside) | frozen LLM | status |
|---|---|---|---|
| `adventists-ai/DuplexJev-{A,B}[-Gender,-Emotion,-Para]-Qwen3-ASR-0.6B-Qwen3-32B` | Qwen3-ASR-0.6B (A: + fusion) | Qwen3-32B | released, tested with 0.2.1 |
| `fixie-ai/ultravox-v0_6-qwen-3-32b` | Whisper-large-v3-turbo | Qwen3-32B | tested (qa100 0.89) |
| `fixie-ai/ultravox-v0_6-gemma-3-27b` | Whisper-large-v3-turbo | Gemma-3-27B (licence acceptance needed) | same format |

- **The ASR encoder is not a separate choice.** The connector only works with the encoder it was trained on.
  `audio_model=` / `text_model=` exist only to point at a *local copy of the same* encoder or LLM (offline machines,
  shared model folders); a different encoder or LLM loads without error but gives meaningless answers.
- Only the encoder's hidden states are used: nothing is transcribed.
- `device="cuda:0"` (default: first GPU, bf16) or `device="auto"` to shard a large LLM over several GPUs.
  A 32B LLM in bf16 needs one 80 GB GPU.
- Standard Hugging Face variables apply: `HF_TOKEN` (gated models), `HF_HOME` (cache location),
  `HF_ENDPOINT` (mirror), `HF_HUB_OFFLINE=1` (use the local cache only).

What the package guarantees (unit tests in `tests/`, checked on Qwen3-32B):

- **0 decode steps.** Each answer is the next-token softmax over its option letters; options appear under permuted letters (`n_perm` averages several orders).
- **Exact prefix sharing** (`mode="packed"`, default): the context and audio of an item are encoded once and all its questions are packed into one row under a block-diagonal mask. The packing engine, checked in fp32 on Qwen3-32B, matches one-row-per-question to within 2e-5; in bf16 the mean difference is 0.001–0.002, with at most 1/100 answers changed.
- **Batch invariance.** An item's answers do not depend on which other items share the pass (fp32: max difference 1e-5). For speech this requires aligning every clip to whole audio tokens; the package pads silence at the end of each clip (batched Ultravox inference otherwise leaks padding into the last audio token of shorter clips). In bf16, GPU kernels depend on batch shape, so running an item alone vs. in a batch changed 1/100 answers.
- **Checked accuracy.** Through the package (0.2.1, default prompts), every released connector is within 0–5 points of the paper's numbers (see each model card), e.g. A-Gender: gender 89.2 (paper 89.4); A-Para: emotion 88.2 (paper 90.0).
- **Speed (current).** Plain PyTorch, one A800, bf16: 8 questions for 64 calls in 9 s (about 140 ms per call); 48 concurrent server requests are answered in one tick. The paper's latency numbers use a vLLM engine; a vLLM backend for the package is on the roadmap.

Speech checkpoints in Ultravox format need transformers 4.51–4.55, which the `speech` extra installs.

More in [`examples/`](examples): tick-batch timing, server client, speech quick start.

## 5. Evaluation results

**Accuracy** (%, paper Tables 2–3; single-token readout, one H200):

| connector | qa100 | ZJU-ML | Easy-Turn | gender (800 real) | ZJU gender | emotion (800, 4-way) |
|---|---:|---:|---:|---:|---:|---:|
| reading the oracle transcript | 91 | – | – | – | – | – |
| A (R2) | 83 | 77 | 77.1 | 55 | – | 28 |
| B (R2) | **90** | 79 | 76.1 | 54 | – | 27 |
| A-Gender | 86 | 79 | – | 89.4 | 73 | – |
| B-Gender | 87 | 81 | – | 87.9 | 60 | – |
| A-Emotion | 84 | 71 | – | – | – | 71.8 |
| B-Emotion | 89 | 69 | – | – | – | 85.5 |
| A-Para (mixed objective) | 82 | 61 | – | **89.9** | **90** | **90.0** |

qa100: 100 bilingual spoken multiple-choice questions (TTS stems). ZJU-ML: main-language part of the ZJU audio
benchmark v2.0.0; half of it is real-speech factual questions, and emotion data lower it by 6–17 points, almost all on
that half. Gender: 800 real utterances (AISHELL-1, Common Voice, LibriSpeech); 50% is chance. Emotion: 800 acted
utterances (ESD, CREMA-D), neutral / happy / angry / sad; 25% is chance.

**Latency and capacity** (1× H200, bf16, Qwen3-32B; ten decisions per event, 30 real utterances of 2–12 s, the same
vLLM engine for both methods):

| context | method | decode steps | 1 event | events/s within 0.25 s | 0.5 s | 2 s |
|---|---|---:|---:|---:|---:|---:|
| none | ASR → LLM generates JSON | 12.5 + 86 | 1,567 ms (+411 ms ASR) | 0 | 0 | 16.9 |
| none | single-token readout | 0 | 92 ms | 17.3 | 20.4 | 33.5 |
| 1.5k tokens | ASR → LLM generates JSON | 12.5 + 87 | 1,598 ms (+ASR) | 0 | 0 | 8.4 |
| 1.5k tokens | single-token readout | 0 | 144 ms | 9.7 | 12.0 | 18.4 |

On one 8×H200 node (one engine per GPU), eight events (80 decisions) are answered in about 0.1 s.

**Prefix sharing** (one packed row vs sequential calls): 7× cheaper for 100 questions over a 1.5k-token context and
20× for 50 questions over a 5k-token context, with identical answers on 40/40 items.

| benchmark | task | link |
|---|---|---|
| qa100 | spoken multiple-choice, synthetic speech (zh + en) | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| ZJU audio benchmark · `gender` | speaker gender (zh + en, real + TTS) | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| ZJU audio benchmark · `main_language` | spoken multiple-choice, real speech + TTS | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |

Runners: [`evaluation/`](evaluation).

## 6. Training

Frozen Qwen3-ASR-0.6B encoder and frozen Qwen3-32B; only the connector is trained, with a patched
[Ultravox](https://github.com/fixie-ai/ultravox).

1. **Content (R1–R2).** Transcript distillation (token-level KL to the LLM's transcript-conditioned output) on the
   Ultravox v0.6 mixture (WenetSpeech, GigaSpeech, Common Voice, CoVoST 2, People's Speech, LibriSpeech, MLS, MUSAN),
   split into 100 disjoint packs that each keep the official ratio; R1 and R2 use one pack each.
2. **Decisions.** Answer-token supervision: cross-entropy on the single option letter at the readout position.
   Gender: real speech from AISHELL-1 and LibriSpeech with corpus speaker labels. Emotion: ESD and CREMA-D.
3. **Mixed objective (A-Para).** One run over content + gender + emotion samples: distillation for content samples,
   answer-token cross-entropy for decision samples.

Recipe, data links and scripts: [`training/`](training). The emotion corpora are not redistributed.

## 7. Repository layout

| path | contents |
|---|---|
| [`duplexjev/`](duplexjev) | the installable package: `Decider`, `Question`, CLI and tick-batched server |
| [`duplexjev/research/`](duplexjev/research) | paper code: readout, question contract, encoder with fusion |
| [`tests/`](tests) | equivalence and batch-invariance tests |
| [`examples/`](examples) | quick starts, tick-batch timing, server client |
| [`evaluation/`](evaluation) | benchmark runners and where to get each benchmark |
| [`training/`](training) | Ultravox configs and patches, data recipe, 100-pack split, continuation generation |
| [`benchmarks/`](benchmarks) | latency, SLO capacity and prefix-sharing experiments |
| [`docs/`](docs) | project page |

The paper code still contains paths from our cluster; see [docs/PATHS.md](docs/PATHS.md).

## 8. License

Code: Apache-2.0 ([LICENSE](LICENSE)). Model weights: Apache-2.0, except the connectors trained on emotion data (A-Emotion,
B-Emotion, A-Para and every `-Para-` small connector), which are CC BY-NC 4.0 because ESD is licensed for research only. qa100: CC-BY-4.0.
Some training corpora (e.g. WenetSpeech, CoVoST 2) have non-commercial terms; check them before commercial use.
Third-party models and datasets keep their own licenses; see [NOTICE](NOTICE).

## 9. Citation

```bibtex
@inproceedings{jin2027duplexjev,
  title     = {Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen {LLM} Hear Beyond the Transcript},
  author    = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Song, Haigang and Pang, Zhikun and Zhang, Xiaowen},
  booktitle = {Submitted to IEEE ICASSP},
  year      = {2027}
}
```

## 10. Acknowledgements

Built on [Ultravox](https://github.com/fixie-ai/ultravox), [Qwen3](https://github.com/QwenLM/Qwen3) and Qwen3-ASR.
We thank the ZJU team for the [audio benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark).
Claude (Anthropic) assisted with code.
Questions and issues: [GitHub Issues](https://github.com/adventists-ai/duplexjev/issues).
