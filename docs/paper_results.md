# Paper results

The numbers reported in the paper (*Batched Speech Decisions Without Decoding*, ICASSP 2027 submission), with every
Qwen3-32B connector variant that was compared. The [README](../README.md) keeps only the best checkpoint per encoder ×
LLM × connector; this page keeps the full comparison.

## Qwen3-32B connectors

All use the frozen **Qwen3-ASR-0.6B** encoder and the frozen **[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)**.

| repository | connector | trained for | trainable params | license |
|---|---|---|---:|---|
| 🤗 [DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B) | A · cross-attention fusion | content (R2) | 21.1 M | Apache-2.0 |
| 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B) | B · last layer | content (R2), best spoken QA | 17.8 M | Apache-2.0 |
| 🤗 [DuplexJev-A-Gender-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Gender-Qwen3-ASR-0.6B-Qwen3-32B) | A | + speaker gender | 21.1 M | Apache-2.0 |
| 🤗 [DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B) | B | + speaker gender | 17.8 M | Apache-2.0 |
| 🤗 [DuplexJev-A-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | A | + emotion (4-way) | 21.1 M | CC BY-NC 4.0 |
| 🤗 [DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | B | + emotion (4-way) | 17.8 M | CC BY-NC 4.0 |
| 🤗 [DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B) | A | gender + emotion + content, mixed objective | 21.1 M | CC BY-NC 4.0 |

## Accuracy (paper Tables 2–3)

%, single-token readout, one H200:

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

For the README leaderboard every released checkpoint was re-measured on the cells the paper left empty, and Easy-Turn
was re-run for all of them with the released package (duplexjev ≥ 0.2.1 pads audio at the end, which moves Easy-Turn by
up to 1 point: A 76.2 vs. 77.1, B 76.6 vs. 76.1).

## Latency and capacity

1× H200, bf16, Qwen3-32B; ten decisions per event, 30 real utterances of 2–12 s, the same vLLM engine for both methods:

| context | method | decode steps | 1 event | events/s within 0.25 s | 0.5 s | 2 s |
|---|---|---:|---:|---:|---:|---:|
| none | ASR → LLM generates JSON | 12.5 + 86 | 1,567 ms (+411 ms ASR) | 0 | 0 | 16.9 |
| none | single-token readout | 0 | 92 ms | 17.3 | 20.4 | 33.5 |
| 1.5k tokens | ASR → LLM generates JSON | 12.5 + 87 | 1,598 ms (+ASR) | 0 | 0 | 8.4 |
| 1.5k tokens | single-token readout | 0 | 144 ms | 9.7 | 12.0 | 18.4 |

On one 8×H200 node (one engine per GPU), eight events (80 decisions) are answered in about 0.1 s.

## Prefix sharing

One packed row vs. sequential calls: 7× cheaper for 100 questions over a 1.5k-token context and 20× for 50 questions
over a 5k-token context, with identical answers on 40/40 items. Scripts: [`benchmarks/`](../benchmarks).
