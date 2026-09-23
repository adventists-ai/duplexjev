# DuplexJev — Batched Speech Decisions Without Decoding

**DuplexJev** turns a frozen speech LLM into a *typed decision engine* for full-duplex spoken agents
(product name: **Speech-to-Decision**). Hidden states of an off-the-shelf ASR encoder are projected into a
frozen LLM, and every runtime-declared question (*is the turn complete? which filler to play? which flow step?*)
is read as a **single-token, closed-set distribution** — no ASR decoding, no text decoding. Many questions,
over one call or across calls, share one forward pass.

📄 Paper: *Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents*
(ICASSP 2027 submission) — arXiv: _coming soon_ · 🌐 Demo: https://adventists-ai.github.io/duplexjev/ ·
🤗 Weights: _coming soon_

> Status: research release in preparation. `research/` contains the exact code used for the paper; a cleaned,
> installable package and model weights will follow.

## How it works

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

* **Typed single-token readout.** Each question lists its options under permuted letters (A, B, …); the answer is
  the next-token softmax restricted to those letters. Output is always valid; `max p` is a confidence score.
* **Prefix sharing.** The shared prefix (template, dialogue context, audio) is encoded once; all question suffixes
  are packed into a single row under a block-diagonal 4-D mask with position ids restarting at the prefix length.
  The KV cache holds `P + ΣL_i` positions instead of `N(P + L)`. Answers match one-by-one runs up to bf16 noise.
* **Modular.** Any ASR encoder, a small trainable projector (39.9 M params; +3.2 M for the fusion in variant A),
  any frozen LLM that accepts input embeddings. Encoder and LLM stay frozen.

## Measured results (1× H200, bf16, Qwen3-32B)

Ten decisions per event, 30 real utterances (2–12 s), same vLLM engine for both methods:

| context | method | decode steps | 1 event | events/s within 0.25 s | 0.5 s | 2 s |
|---|---|---:|---:|---:|---:|---:|
| none | ASR → LLM generates JSON | 12.5 + 86 | 1,567 ms (+411 ms ASR) | 0 | 0 | 16.9 |
| none | single-token readout | 0 | 92 ms | 17.3 | 20.4 | 33.5 |
| 1.5k tokens | ASR → LLM generates JSON | 12.5 + 87 | 1,598 ms (+ASR) | 0 | 0 | 8.4 |
| 1.5k tokens | single-token readout | 0 | 144 ms | 9.7 | 12.0 | 18.4 |

Prefix sharing (packed row) vs sequential calls: 7× cheaper for 100 questions over a 1.5k-token context,
20× for 50 questions over a 5k-token context; identical answers on 40/40 items across contexts.
Accuracy tables (qa100, ZJU audio-gender-benchmark, ZJU main-language) will be added with the final checkpoints.

## Repository layout

| path | contents |
|---|---|
| `research/readout/` | typed single-token readout, letter rendering, decision contract, model loading |
| `research/encoder/` | Qwen3-ASR encoder with cross-attention fusion (variant A) and configs for A/B |
| `research/cost/` | prefix-sharing experiments (M1–M3), cascade baseline, same-engine latency and SLO capacity |
| `research/eval/` | qa100 / gender / ZJU benchmark runner |
| `research/data_pipeline/` | 100-pack split of the Ultravox v0.6 mixture, verification, Qwen3-32B continuation store |
| `research/train_configs/` | training configs for variants A and B (stack factor 2) |
| `data/qa100/` | **qa100** bilingual spoken multiple-choice test set (CC-BY-4.0) |
| `docs/` | demo page (GitHub Pages) |

`research/` is published as used in the paper and still contains absolute paths from our cluster;
see [research/PATHS.md](research/PATHS.md).

## qa100

100 four-way multiple-choice questions: 50 Chinese + 50 English, each split into 25 logic and 25 factual items,
gold letters balanced over A–D. Question stems are synthesised with VoxCPM2 and verified by an independent ASR
(exact normalised match, re-synthesised otherwise); options are given as text. Three conditions share one prompt:
*audio*, *text* (oracle transcript; upper bound) and *options-only* (lower bound). See
[data/qa100/DATASHEET.md](data/qa100/DATASHEET.md).

## License

Code and model weights: Apache-2.0 ([LICENSE](LICENSE)). qa100: CC-BY-4.0. Third-party components and datasets keep
their own licenses; see [NOTICE](NOTICE).

## Citation

```bibtex
@misc{jin2026duplexjev,
  title  = {Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents},
  author = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Pang, Zhikun and Zhang, Xiaowen},
  year   = {2026},
  note   = {ICASSP 2027 submission}
}
```

## Acknowledgements

Built on [Ultravox](https://github.com/fixie-ai/ultravox) (training code and projector design),
[Qwen3](https://github.com/QwenLM/Qwen3) and Qwen3-ASR, and the ZJU
[audio-gender-benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark). Typed single-token decisions
follow the "System-One" line of work popularised by Jev.
