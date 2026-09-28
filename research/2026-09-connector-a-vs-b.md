# Does cross-layer fusion help a speech connector? A vs. B on Qwen3-32B and Qwen2.5-72B

*Adventists.ai research note, 2026-09-28. [中文版](2026-09-connector-a-vs-b_zh.md)*

**Short answer: not reliably.** We compared reading the ASR encoder's last layer (connector B) with fusing three
encoder layers by cross-attention (connector A) in five matched pairs. With Qwen3-32B, B was clearly better on a
public spoken benchmark (VoiceBench) in all three pairs; with Qwen2.5-72B, A was ahead by 1–3 points, inside the
noise. Since B is simpler and smaller, and it is the only one that standard serving stacks
support as-is, we build new DuplexJev models with B only.

## The two connectors

Both join a frozen [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) encoder (18 layers, 12.5 frames/s)
to a frozen LLM; only the connector is trained.

```
            ┌─ B: layer 18 (last) ──────────────────────────────────────────────┐
encoder ────┤                                                                   ├─► stack 2 frames → SwiGLU MLP → LLM
            └─ A: layer 18 + CrossAttn(Q = layer 18, K = layer 14, V = layer 9) ┘   (6.25 audio tokens/s)
                  zero-initialised, added residually (3.2 M extra parameters)
```

The idea behind A: intermediate encoder layers keep more of *how* something was said (voice, prosody), which the
last layer of an ASR encoder tends to discard, so letting the connector look back at them should help speaker and
emotion decisions. Because the fusion block starts at zero, A begins training as exactly B.

Everything else was identical within each pair: same data, same training schedule, same evaluation code. One run per
configuration.

## Results

### Paper protocol (single-token readout; % correct)

| LLM | stage | connector | qa100 | ZJU-ML | Easy-Turn | gender | emotion |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-32B | content (R2) | A | 83 | 77 | 76.2 | 55 | 28 |
| | | B | **90** | 79 | 76.6 | 54 | 27 |
| | + gender | A | 86 | 79 | 72.8 | 89.4 | 27.1 |
| | | B | 87 | 81 | 75.2 | 87.9 | 27.4 |
| | + emotion | A | 84 | 71 | 68.4 | 48.8 | 71.8 |
| | | B | 89 | 69 | 72.2 | 51.8 | **85.5** |
| Qwen2.5-72B | content (R2) | A | 90 | 74 | 73.1 | 55.9 | 27.1 |
| | | B | 90 | 75 | 73.0 | 55.4 | 27.8 |
| | gender + emotion (Para) | A | 85 | 71 | 66.4 | 89.9 | 92.0 |
| | | B | 89 | 74 | 63.1 | 90.5 | 91.9 |

Gender and emotion are at chance (50 % and 25 %) until they are trained. Approximate standard errors: ±4 points on
qa100 and ZJU-ML (100 items each), ±1.5 on Easy-Turn, gender and emotion (800 each). The only difference that is
clearly outside the noise is emotion-only training on Qwen3-32B, where B is 13.7 points better.

### Public spoken benchmark: VoiceBench (question and options spoken; options also given as text)

| LLM | stage | A: OBQA / MMSU | B: OBQA / MMSU | B − A |
|---|---|---:|---:|---:|
| Qwen3-32B | content (R2) | 70.1 / 57.7 | 76.3 / 60.1 | **+6.2 / +2.4** |
| | + gender | 67.0 / 55.5 | 77.6 / 60.6 | **+10.6 / +5.1** |
| | + emotion | 66.8 / 56.5 | 73.2 / 56.9 | **+6.4** / +0.4 |
| Qwen2.5-72B | content (R2) | 77.4 / 59.2 | 74.5 / 58.2 | −2.9 / −1.0 |
| | gender + emotion (Para) | 73.2 / 57.6 | 72.5 / 56.3 | −0.7 / −1.3 |

[VoiceBench](https://github.com/matthewcym/voicebench) OpenBookQA (455 items, ±2 points) and MMSU (3,073 items, ±0.9).
We read the answer as a single token over the option letters, the same way as in the table above.

## What we take from it

1. **Fusion is not a free win.** On Qwen3-32B it cost 6–11 points of spoken QA and 14 points of emotion. On
   Qwen2.5-72B it was slightly ahead, but never by more than the noise.
2. **The paralinguistic information is already in the last layer.** Both connectors reach about 90 % gender and
   90 % emotion after decision training. Linear probes on the Qwen3-ASR encoder output agree: gender 99 %, emotion
   91 %.
3. **The bigger loss is elsewhere.** Given the transcript as text, Qwen3-32B scores 76.5 on MMSU and Qwen2.5-72B 79.5;
   through speech, both connectors land at 56–60. About 20 points are lost between the audio and the LLM, whatever
   the connector. That, not cross-layer fusion, is where connector research should go.
4. **Simplicity matters in deployment.** B is the standard Ultravox layout; it runs on vLLM with a small plugin
   ([`duplexjev-vllm`](https://pypi.org/project/duplexjev-vllm/)). A needs custom encoder code in every runtime.

## Limitations

One seed per configuration; one fusion design (layer choice 18/14/9, one cross-attention block); one encoder
(Qwen3-ASR-0.6B); the LLM is always frozen. A different fusion design, a trainable LLM or a much longer schedule
could change the picture. The 32B and 72B pairs were trained a few days apart with the same code; their absolute
numbers are not comparable across LLMs, only within each pair.

## Reproduce

The Qwen3-32B checkpoints are public: `adventists-ai/DuplexJev-{A,B}[-Gender|-Emotion]-Qwen3-ASR-0.6B-Qwen3-32B` and
`DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B` on [Hugging Face](https://huggingface.co/adventists-ai). The four
Qwen2.5-72B checkpoints were research runs and are not released. Training recipe: [`training/`](../training); evaluation:
[`evaluation/`](../evaluation).
