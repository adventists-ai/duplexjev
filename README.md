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
  🤗 Weights (coming soon) &nbsp;|&nbsp;
  📊 <a href="https://huggingface.co/datasets/adventists-ai/qa100">qa100</a> &nbsp;|&nbsp;
  📝 <a href="#9-citation">Citation</a>
</p>

---

## 1. Introduction

**DuplexJev** turns a frozen speech LLM into a *typed decision engine* for full-duplex spoken agents
(product name: **Speech-to-Decision**). Hidden states of an off-the-shelf ASR encoder are projected into a frozen
LLM, and every runtime-declared question — *is the turn complete? which filler to play? which flow step?* — is read
as a **single-token, closed-set distribution**: no ASR decoding, no text decoding. Many questions, about one call or
across many calls, share one forward pass.

Paper: *Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents*
(ICASSP 2027 submission).

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
- **Prefix sharing.** Template, dialogue context and audio are encoded once; all question suffixes are packed into one
  row under a block-diagonal 4-D mask, with position ids restarting at the prefix length. The KV cache holds
  `P + ΣL_i` positions instead of `N(P + L)`, and answers match one-by-one runs up to bf16 noise.
- **Modular.** Any ASR encoder, a small trainable projector (39.9 M parameters; +3.2 M for the fusion in variant A)
  and any frozen LLM that accepts input embeddings.

## 2. News

- **2026-09** — Paper submitted to ICASSP 2027. Released the [`duplexjev`](https://pypi.org/project/duplexjev/) package on PyPI (batched decider, tick server), research code, latency benchmarks and project page.
- **2026-10-10 (planned)** — Speech-to-Decision commercial API.
- **2026-10-15 (planned)** — Open-source inference pipeline and model weights.

## 3. Model download

Each checkpoint is an **adapter for one specific pair of models**: it contains only the trained projector (and, for A,
the fusion block) that connects the frozen ASR encoder below to the frozen LLM below. It does not work with other
encoders or LLMs, including other sizes of the same family; supporting another pair means training a new adapter.

| model | ASR encoder (frozen) | LLM (frozen) | encoder read-out | trainable params | download |
|---|---|---|---|---:|---|
| DuplexJev-A | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | cross-attention fusion (h18 / h14 / h9) | 43.1 M | 🤗 `DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B` (coming soon) |
| DuplexJev-B | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | last layer (h18) | 39.9 M | 🤗 `DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B` (coming soon) |

Download the encoder and the LLM from their original repositories and load the adapter on top. Adapter repositories
are named `DuplexJev-<variant>-<ASR encoder>-<LLM>`, so adapters for other model pairs can be added later.

## 4. Quick start

```bash
pip install "duplexjev[speech]"      # add [all] for the HTTP server
```

**One clip, several option groups** (the default):

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("fixie-ai/ultravox-v0_6-qwen-3-32b")   # DuplexJev adapters load the same way
groups = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
          Question("filler", "Which filler fits?", ["Sure —", "One moment —", "(stay silent)"]),
          Question("gender", "Speaker gender", ["female", "male"])]

d.decide("call_017.wav", groups)
# {'turn': {'answer': 'finished', 'confidence': 0.97, 'probs': {...}}, 'filler': {...}, 'gender': {...}}
```

**Many clips, many option groups** (advanced): each group names the clip(s) it is about with `audio=<id>`, a list
of ids, or `"*"` for every clip. Everything runs in one batched pass.

```python
d.decide_batch(
    {"car1": "a.wav", "car2": "b.wav", "car3": "c.wav"},
    [Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
     Question("gender", "Speaker gender", ["female", "male"], audio=["car2", "car3"]),
     Question("human", "Does the user need a human agent?", ["yes", "no"], audio="car1")],
    context={"car1": "Driver asked to call home twice."})
# {'car1': {'turn': ..., 'human': ...}, 'car2': {'turn': ..., 'gender': ...}, 'car3': {...}}
```

**Batched server:** every request that arrives within one tick is answered in one pass.

```bash
duplexjev serve --model fixie-ai/ultravox-v0_6-qwen-3-32b --tick-ms 160
# POST /v1/decide        {"audio_b64": ..., "questions": [...]}
# POST /v1/decide_batch  {"audios": {"car1": ..., "car2": ...}, "questions": [{..., "audio": "car1"}]}
```

Command line: `duplexjev decide --model M call.wav --q "turn|Has the user finished?|finished,not finished"`.

### Choosing a model

`Decider.from_pretrained(...)` (CLI: `--model`) takes a Hugging Face repo id or a local path of a **speech checkpoint in
Ultravox format**. A checkpoint is an ASR encoder plus the connector trained for it, and it names the frozen LLM it was
trained against. So you pick one name; the encoder comes with it, and the LLM is fetched from the id in its config.

```python
d = Decider.from_pretrained("fixie-ai/ultravox-v0_6-qwen-3-32b")                      # download everything
d = Decider.from_pretrained("fixie-ai/ultravox-v0_6-qwen-3-32b",
                            text_model="/data/models/Qwen3-32B")                        # reuse a local LLM copy
d = Decider.from_pretrained("/data/models/my-checkpoint", device="auto")                # local path, shard over GPUs
```

| checkpoint | encoder (inside) | frozen LLM | status |
|---|---|---|---|
| `fixie-ai/ultravox-v0_6-qwen-3-32b` | Whisper-large-v3-turbo | Qwen3-32B | tested (qa100 0.89) |
| `fixie-ai/ultravox-v0_6-gemma-3-27b` | Whisper-large-v3-turbo | Gemma-3-27B (licence acceptance needed) | same format, not yet tested |
| `adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B` | Qwen3-ASR-0.6B + cross-attention fusion | Qwen3-32B | coming soon |
| `adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B` | Qwen3-ASR-0.6B (last layer) | Qwen3-32B | coming soon |

- **The ASR encoder is not a separate choice.** The connector only works with the encoder it was trained on.
  `audio_model=` / `text_model=` exist only to point at a *local copy of the same* encoder or LLM (offline machines,
  shared model folders); a different encoder or LLM loads without error but gives meaningless answers.
- Only the encoder's hidden states are used: nothing is transcribed.
- `device="cuda:0"` (default: first GPU, bf16) or `device="auto"` to shard a large LLM over several GPUs.
  A 32B LLM in bf16 needs one 80 GB GPU.
- Standard Hugging Face variables apply: `HF_TOKEN` (gated models), `HF_HOME` (cache location),
  `HF_ENDPOINT` (mirror), `HF_HUB_OFFLINE=1` (use the local cache only).

What the package guarantees (unit tests in `tests/`, checked on Qwen3-32B and Ultravox v0.6 on an A800):

- **0 decode steps.** Each answer is the next-token softmax over its option letters; options appear under permuted letters (`n_perm` averages several orders).
- **Exact prefix sharing** (`mode="packed"`, default): the context and audio of an item are encoded once and all its questions are packed into one row under a block-diagonal mask. The packing engine, checked in fp32 on Qwen3-32B, matches one-row-per-question to within 2e-5; in bf16 the mean difference is 0.001–0.002, with at most 1/100 answers changed.
- **Batch invariance.** An item's answers do not depend on which other items share the pass (fp32: max difference 1e-5). For speech this requires aligning every clip to whole audio tokens, which the package does (batched Ultravox inference otherwise leaks padding into the last audio token of shorter clips). In bf16, GPU kernels depend on batch shape, so running an item alone vs. in a batch changed 1/100 answers.
- **Checked accuracy.** qa100 through the package: 0.88–0.90 with Ultravox v0.6.
- **Speed (current).** Plain PyTorch, one A800, bf16: 8 questions for 64 calls in 9 s (about 140 ms per call); 48 concurrent server requests are answered in one tick. The paper's latency numbers use a vLLM engine; a vLLM backend for the package is on the roadmap.

Speech checkpoints in Ultravox format need transformers 4.51–4.55, which the `speech` extra installs.

More in [`examples/`](examples): tick-batch timing, server client, speech quick start.

## 5. Evaluation results

**Latency and capacity** (1× H200, bf16, Qwen3-32B; ten decisions per event, 30 real utterances of 2–12 s, the same
vLLM engine for both methods):

| context | method | decode steps | 1 event | events/s within 0.25 s | 0.5 s | 2 s |
|---|---|---:|---:|---:|---:|---:|
| none | ASR → LLM generates JSON | 12.5 + 86 | 1,567 ms (+411 ms ASR) | 0 | 0 | 16.9 |
| none | single-token readout | 0 | 92 ms | 17.3 | 20.4 | 33.5 |
| 1.5k tokens | ASR → LLM generates JSON | 12.5 + 87 | 1,598 ms (+ASR) | 0 | 0 | 8.4 |
| 1.5k tokens | single-token readout | 0 | 144 ms | 9.7 | 12.0 | 18.4 |

**Prefix sharing** (one packed row vs sequential calls): 7× cheaper for 100 questions over a 1.5k-token context and
20× for 50 questions over a 5k-token context, with identical answers on 40/40 items.

**Accuracy** on the benchmarks below will be added with the final checkpoints; see [`evaluation/`](evaluation).

| benchmark | task | link |
|---|---|---|
| qa100 | spoken multiple-choice, synthetic speech (zh + en) | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| ZJU audio-gender-benchmark · `gender` | speaker gender (zh + en, real + TTS) | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| ZJU audio-gender-benchmark · `main_language` | spoken multiple-choice, real speech | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |

## 6. Training

Frozen Qwen3-ASR-0.6B encoder and frozen Qwen3-32B; only the projector (and fusion block) is trained, with a patched
[Ultravox](https://github.com/fixie-ai/ultravox) on the Ultravox v0.6 mixture (WenetSpeech, GigaSpeech, Common Voice,
CoVoST 2, People's Speech, LibriSpeech, MLS, MUSAN). The mixture is split into 100 disjoint packs that each keep the
official ratio. Recipe, data links and scripts: [`training/`](training).

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

Code and model weights: Apache-2.0 ([LICENSE](LICENSE)). qa100: CC-BY-4.0. Third-party models and datasets keep their
own licenses; see [NOTICE](NOTICE).

## 9. Citation

```bibtex
@misc{jin2026duplexjev,
  title  = {Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents},
  author = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Pang, Zhikun and Zhang, Xiaowen},
  year   = {2026},
  note   = {ICASSP 2027 submission}
}
```

## 10. Acknowledgements

Built on [Ultravox](https://github.com/fixie-ai/ultravox), [Qwen3](https://github.com/QwenLM/Qwen3) and Qwen3-ASR.
We thank the ZJU team for the [audio-gender-benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark).
Questions and issues: [GitHub Issues](https://github.com/adventists-ai/duplexjev/issues).
