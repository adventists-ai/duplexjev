# DuplexJev models

[← back to the README](../README.md) · [中文](models_zh.md)

The README recommends four models: **DuplexJev-32B-Para**, **DuplexJev-4B-Para**, **DuplexJev-4B-Para-LiveTurn** and
**DuplexJev-32B-Vision** (preview). This page lists every released model, the scores of each version, and the comparisons
between our own versions.

## All released models

| model | for | built on | params | GPU memory | qa100 | ZJU-ML | Easy-Turn | gender | emotion | total | licence |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ⭐ 🤗 [**DuplexJev-32B-Para**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) 🆕 v1.3 | server, widest set of abilities (now with much better stress and accents), best live turn-taking, best overall | as 32B-Turn (+ a second rank-16 LoRA) | 33.0 B | one 80 GB GPU | 96 | 87 | 94.4\* | **93.8** | **92.1** | **92.7** | CC BY-NC 4.0 |
| ⭐ 🤗 [**DuplexJev-4B-Para**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) 🆕 | small GPU, widest set of abilities, **and normal conversation** (v1.1); `duplexjev` default | as 4B-Turn (+ a second rank-16 LoRA) | 4.2 B | ~10 GB | 77 | **58** | 91.0\* | 89.2 | 90.0 | **82.5** | CC BY-NC 4.0 |
| 🤗 [**DuplexJev-4B-Para-LiveTurn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para-LiveTurn) 🆕 | small GPU, **live end-of-turn** for LiveKit / Pipecat agents (+ accents); 4B-Para stays the general 4B | as 4B-Para (+ accent and pause-level turn-taking data) | 4.2 B | ~10 GB | 76 | 62 | 92.0\* | 89.5 | 90.5 | — | CC BY-NC 4.0 |
| 🤗 [DuplexJev-32B-Vision](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision) (preview) | voice + picture in one question | 32B-Para + Qwen3-VL vision encoder + one more LoRA | — | one 80 GB GPU | | | | | | | CC BY-NC 4.0 |
| 🤗 [DuplexJev-32B-Stress](https://huggingface.co/adventists-ai/DuplexJev-32B-Stress) 🆕 | server, + stress and pauses | as 32B-Turn (+ a second rank-16 LoRA) | 33.0 B | one 80 GB GPU | **96** | 87 | 94.4\* | 92.1 | 91.4 | — | CC BY-NC 4.0 |
| 🤗 [DuplexJev-4B-Stress](https://huggingface.co/adventists-ai/DuplexJev-4B-Stress) 🆕 | small GPU, + stress and pauses | as 4B-Turn (+ a second rank-16 LoRA) | 4.2 B | ~10 GB | 73 | 51 | 91.8\* | 89.1 | 91.4 | — | CC BY-NC 4.0 |

% under the paper protocol (single-token readout); total = mean of main language (qa100, ZJU-ML, Easy-Turn) and
paralinguistics (gender, emotion), see [Benchmarks](../README.md#benchmarks). \* Both models were trained with the Easy-Turn
training split (disjoint from the test set), so their Easy-Turn scores are in-domain. Both answer ten questions about one clip in about
0.1–0.25 s on one GPU.

**Which one to use.** The **Para** models if you want everything the voice carries (non-verbal sounds, speaking style,
speaker, stress, emotion, gender, turn-taking) in one model — `DuplexJev-32B-Para` on an 80 GB GPU, `DuplexJev-4B-Para`
on smaller GPUs. `DuplexJev-32B-Para` v1.2 is also the best for live turn-taking; `DuplexJev-4B-Para-LiveTurn` for live end-of-turn on a small GPU;
`DuplexJev-32B-Stress` / `4B-Stress` if stress and pauses are the main thing. All models
use the same prompts and the same plugin. The 32B models
use only the language model of Qwen3-VL-32B: their inputs are audio and text. For speech + image, use
[DuplexJev-32B-Vision](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision) (research preview).

## What the Para models add, by version

**What the Para models add** (AUC, 0.5 = chance; unseen real speech from the NonverbalTTS test set and a third-party
sample of natural Mandarin conversation; speaker EER on same-gender AISHELL-1 pairs):

| | laughter | breathing | cough | sigh | laughter / breathing in conversation | speaking style (7 classes) | same speaker? EER |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32B-Stress | 0.73 | 0.56 | 0.40 | 0.70 | 0.61 / 0.44 | 45% | — |
| **32B-Para** v1.2 | 0.87 | **0.81** | 0.84 | 0.75 | 0.80 / 0.56 | 79% | — |
| 32B-Para v1.1 | **0.88** | **0.81** | **0.85** | 0.80 | 0.83 / 0.65 | **82%** | ≈ v1.0 (18%)² |
| 32B-Para v1.0 | 0.85 | 0.73 | 0.70 | 0.74 | 0.80 / 0.78 | 69% | 18% |
| **4B-Para** v1.1 | 0.81 | 0.77 | 0.77 | 0.71 | 0.85 / 0.74 | 67% | ≈ v1.0 (9%)¹ |
| 4B-Para v1.0 | 0.80 | 0.76 | 0.75 | 0.81 | 0.84 / 0.84 | 67% | 9% |

¹ v1.1 was not re-run on the AISHELL-1 pairs; on VoxCeleb1 hard pairs its EER is 42.0% vs 41.8% for v1.0.
² 32B-Para v1.1 was not re-run on the AISHELL-1 pairs; on VoxCeleb1 random / hard pairs its EER is 28.2% / 39.6% (v1.0: 37.4% / 45.7%).

The costs are small and listed on the model cards (32B-Para v1.2: emotion −1 vs 32B-Turn and slightly lower English stress, sighs and speaker pairs than v1.1;
4B-Para: ZJU-ML −2, Easy-Turn −1 vs 4B-Turn). Speaker judgments are coarse, human-like, and must not be used to
identify people.

**DuplexJev-4B-Para v1.1 (2026-10-10)** is trained with conversation replay, so the same weights also hold a normal conversation. Decision scores are within 0.5–2.6 points of v1.0 (still available as revision `v1.0`); see the [research note](../research/2026-10-chat-replay.md). The 32B version is training.

The Stress models add stress and pause understanding (which word is stressed: 92 / 95 on held-out Mandarin sentences for 32B / 4B; see the [research note](../research/2026-10-stress-understanding.md)). 32B-Stress matches 32B-Turn on every other benchmark within noise. **4B-Stress trades about 4 points of VoiceBench** (42.0 / 38.2 vs 46.4 / 41.7) for it, so 4B-Para stays the default small model.

## DuplexJev-32B-Para: v1.3 against v1.2

Same items and protocol as the [closed-model comparison](../README.md#against-closed-models-realtime-flagships-included).
v1.3 is much better at stress; v1.2 is a little better on emotion where the words are neutral. v1.2 stays available as
revision `v1.2` (`vllm serve adventists-ai/DuplexJev-32B-Para --revision v1.2`).

| test | DuplexJev-32B-Para v1.3 | v1.2 |
| --- | ---: | ---: |
| **public benchmarks combined** (LISTEN ×4, StressTest, MSPB; 1,128 items) | **50.9** | 46.8 |
| **LISTEN, mean of 4 audio-only conditions** | **37.8** | 37.3 |
| LISTEN neutral words | 43.0 | **45.0** |
| … unseen corpora only (TESS, SAVEE, RAVDESS) | 29.9 | **34.6** |
| LISTEN no words | **26.0** | 25.0 |
| LISTEN sarcastic words | **44.0** | 43.0 |
| LISTEN words match | 38.0 | 36.0 |
| emotion, ESD + CREMA-D† | **92.0** | 91.0 |
| which word is stressed, zh / en† | **95.5 / 96.0** | 93.5 / 91.4 |
| meaning from stress, DEBATE† | **93.0** | 90.5 |
| StressTest | 86.5 | 71.5 |
| speaking style, EARS† | 79.5 | **80.0** |
| MSPB (statement / question, …) | **77.3** | 68.0 |

† Held-out sets from corpora in our training data.

Other v1.3 changes: which meaning a stressed word implies 90.5 → 93.0; emotion a little better; gender and Easy-Turn about
one point lower. LISTEN 4-condition mean on the full set: 38.9 (v1.2: 38.4).

## Live end of turn, all versions

LiveKit eot-bench (validation set) and the Smart Turn v3.2 test set; protocol as in the
[README](../README.md#live-end-of-turn-livekit-eot-bench-and-pipecat-smart-turn).

| model | eot-bench en: false cut-offs at 300 / 600 ms | eot-bench zh | Smart Turn v3.2 test, accuracy en / zh |
|---|---:|---:|---:|
| LiveKit Turn Detector v1 | 9.9 / 4.5% | 20.6 / 8.7% | – |
| **DuplexJev-32B-Para v1.2**, with the assistant's previous message (plugin setting) | **20.0 / 9.4%** | **37.7 / 12.2%** | – |
| **DuplexJev-32B-Para v1.2**, audio only | 23.5 / 11.1% | 40.0 / 12.8% | **94.0 / 91.1**° |
| DuplexJev-32B-Turn, audio only | 29.8 / 13.9% | 44.9 / 14.5% | 65.2 / 84.6 |
| **DuplexJev-4B-Para-LiveTurn** (~10 GB), with the assistant's previous message (plugin setting) | **25.5 / 12.3%** | **38.8 / 12.0%** | **93.3 / 89.6**° |
| DuplexJev-4B-Para v1.1, audio only | 46.4 / 17.7% | 48.5 / 15.6% | 67.9 / 78.9 |
| Smart Turn v3.2 | 35.2 / 14.8% | 42.9 / 13.2% | 94.7 / 90.5 |

## Head-to-head with Qwen3-Omni (v1.2, more items)

Same items, same prompt, options rotated and averaged, highest-probability letter, for both models
(DuplexJev-32B-Para v1.2 and Qwen3-Omni-30B-A3B-Instruct, bf16, one H200).

| ability | test | DuplexJev-32B-Para | Qwen3-Omni-30B |
|---|---|---:|---:|
| emotion, 4-way | ESD + CREMA-D (400)† | **90.5** | 45.2 |
| emotion, neutral words | LISTEN neutral-text (400) · unseen corpora only: TESS, SAVEE, RAVDESS (213) | **50.7** · **39.9** | 32.8 · 27.2 |
| emotion, no words | LISTEN paralinguistic (400) | **26.0** | 18.2 |
| emotion, words match | LISTEN matched (IEMOCAP, MELD, MSP-Podcast, MOSEI, OMG; 400) | 34.8 | **43.0** |
| emotion vs. sarcastic words | LISTEN mismatched (MUStARD, 400) | 39.2 | 40.5 |
| which word is stressed, zh / en | held-out sentences (400 / 187)† | **91.5 / 91.4** | 62.5 / 73.3 |
| meaning from stress | DEBATE held-out (400)† | **88.5** | 43.8 |
| meaning from stress | StressTest (218) | **72.0** | 69.7 |
| speaking style (whisper, shout, fast, …) | EARS held-out speakers (220)† | **78.6** | 49.5 |
| laughter / breathing / cough / sigh, AUC | NonverbalTTS test (400) | 0.93 / **0.86** / 0.86 / 0.85 | **0.96** / 0.65 / **0.87** / **0.87** |

- † Test sets from corpora that are also in our training data (held-out speakers or sentences); Qwen3-Omni is zero-shot.
  LISTEN, StressTest and NonverbalTTS are public benchmarks; LISTEN corpora are not in our emotion training data except
  ESD and CREMA-D.
- NonverbalTTS AUCs here are on the 400 items both models answered, so they differ from the full-test-set row above.
- LISTEN asks up to 16 emotion labels; DuplexJev was trained on four (neutral, happy, angry, sad).
- Readout check: asking Qwen3-Omni to answer in free text instead (600 items) moves its scores by −9 to +4, so the gaps
  are not an artefact of the single-token readout.
- On natural Mandarin conversation (MagicData samples: laughter, breathing, stress), the two are level. Where the words
  themselves carry the emotion (LISTEN matched), Qwen3-Omni is ahead: it reads the words, DuplexJev answers from the voice.

## Archived models

Superseded by the Para models; still downloadable, not recommended for new projects.

| model | replaced by | qa100 | ZJU-ML | Easy-Turn | gender | emotion | total |
|---|---|---:|---:|---:|---:|---:|---:|
| 🤗 [DuplexJev-32B-Turn](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) | 32B-Para | 96 | 87 | 95.3\* | 91.5 | 91.1 | 92.2 |
| 🤗 [DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn) | 4B-Para / 4B-Para-LiveTurn | 74 | 52 | 92.5\* | 89.4 | 92.0 | 80.0 |
| 🤗 [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B) | 32B-Para | | | | | | 87.7 |
| 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) | 4B-Para | | | | | | 78.6 |

The Turn models keep a small edge on a few single scores (32B-Turn: qa100 +1, emotion +1; 4B-Turn: emotion +2), but have
none of the Para abilities (non-verbal sounds, speaking style, speaker, stress, accents). DuplexJev-32B / 4B are the
versions before the turn-taking upgrade (paper models).

🤗 [DuplexJev-Gemma-31B](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B) (Gemma-4-31B-it + Qwen3-ASR-0.6B
encoder, ~58 GB): no turn-taking training. In the prompt wording of [§4](../README.md#quick-start): qa100 97, ZJU-ML 87, gender 94.4,
emotion 90.6, Easy-Turn 68.0 zero-shot; sensitive to question wording (paper-protocol wording: gender 78.5, emotion 60.9).

## Research checkpoints

> **Research checkpoints.** Behind these models are several dozen connector checkpoints for other encoders (MOSS, Whisper,
> SenseVoice), LLMs (Qwen3 0.6B–32B, Falcon-H1, SmolLM3) and both connector types, used with the `duplexjev` PyTorch
> package. They live in a separate organization, 🤗 [adventists-ai-research](https://huggingface.co/adventists-ai-research)
> (moved 2026-10-14; old links redirect). Leaderboard, details and package usage: **[docs/connectors.md](connectors.md)**.
