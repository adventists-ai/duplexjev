# Japanese as a third language: 4B-Para v1.3 and 32B-Para v1.4

*2026-10-22 · DuplexJev-4B-Para v1.3, DuplexJev-32B-Para v1.4 · [4B](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) · [32B](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) · [中文](2026-10-japanese_zh.md)*

**Short version.** After the weak-language round ([previous note](2026-10-weak-languages.md)) Japanese was still at
36 % (4B) and 18 % (32B) error on FLEURS. Scoring by kana reading instead of characters changed little (33 % on 4B),
so the errors were real, not a scoring artefact. A dedicated Japanese stage with 358 k re-checked utterances took
FLEURS Japanese to **17.2 %** on 4B and **11.6 %** on 32B, and conversational Japanese to 13.1 % / 11.1 % (encoder
ceiling 8.3 %). Chinese and English did not move.

## 1. Data

- 358 k Japanese utterances: Emilia JA, Emilia-YODAS JA, Common Voice ja and other Japanese speech data.
- Every label re-transcribed with Qwen3-ASR-0.6B. Clips were dropped when the detected language was not Japanese or
  the label disagreed: CER > 15 % in general, > 25 % for noisier sources, > 40 % for Common Voice's human labels.
- 20 k Japanese continuation items, text teacher Qwen3-32B.
- Held out: one Emilia JA shard (conversational) and the Common Voice ja and FLEURS ja test splits.

## 2. 4B: three variants, then a second stage

All variants were continued from v1.2 for 4,000 steps with the full v1.2 mix replayed and Japanese weighted 15,000.
Character error in %, lower is better.

| | LoRA rank | extra | FLEURS ja | Emilia ja | CV ja |
|---|---:|---|---:|---:|---:|
| v1.2 | — | — | 35.8 | 27.5 | 39.7 |
| A | 16 | — | 22.8 | 18.1 | 25.7 |
| C | 16 | emotion-class data | 23.3 | 17.7 | 26.7 |
| **B** | **64** | — | **19.4** | **15.0** | **25.0** |
| **B → D (v1.3)** | 64 | Japanese ×2 | **17.2** | **13.1** | **20.3** |

- Rank 64 beat rank 16 by about 3 points. A small LoRA was the limit, not the data.
- Emotion-class data (C) did not hurt Japanese.

## 3. 32B: Japanese stage, then paralinguistic stages

| | FLEURS ja | Emilia ja | CV ja |
|---|---:|---:|---:|
| v1.3 | 32.7 | 38.2 | 33.3 |
| weak-language round | 17.9 | — | — |
| + Japanese stage | 14.5 | 12.1 | 16.7 |
| + MagicData paralinguistic stage | 12.9 | 11.1 | 16.0 |
| **+ emotion / stress replay ×1.5 (v1.4)** | **11.6** | **11.1** | **13.0** |

- The paralinguistic stages kept improving Japanese, because a small Japanese replay stayed in the mix.
- They brought the Mandarin paralinguistic test up (drawl 71 → 79–83, laughter 85 → 89, stressed word 84 → 85–86).

## 4. What we learned

- When the encoder is good and the output is wrong, look at the alignment data share first.
- LoRA rank matters for a new language on a 4B model.
- A small replay of the new language inside later stages keeps it improving instead of fading.

## Limitations

- Each language test is 150 utterances; differences under 2 points are within noise.
- Turn-taking, accent and same-speaker scores were not re-run for 32B v1.4.
