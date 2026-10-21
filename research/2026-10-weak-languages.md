# Why our 4B could not hear Japanese, and how v1.2 fixed it without losing the rest

*2026-10-21 · DuplexJev-4B-Para v1.2 · [model](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) · [中文](2026-10-weak-languages_zh.md)*

**Short version.** DuplexJev-4B-Para v1.1 transcribed Japanese and Korean with about 60 % error, though its frozen
Qwen3-ASR-0.6B encoder handles both at 3–7 %. The cause was not the encoder but the alignment data: Japanese was
0.05 % of it and Korean was absent. Adding re-checked speech in six weak languages, plus licensed Mandarin
paralinguistic data and a little extra replay, took Japanese to 35.8 % and Korean to 20.3 % while the 1,035-item
paralinguistic benchmark stayed at 47.6 (v1.1: 47.8).

## 1. Diagnosis

FLEURS test, 150 utterances per language (never trained on), median character error (ja, ko, th, zh) or word error
(others), prompt = the alignment template "Repeat the following text, which is written in X…":

| | ja | ko | vi | th | ar | ru | es | fr | de | zh | en |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR-0.6B (same encoder, its own decoder) | 7.0 | 2.7 | 7.8 | 5.1 | 16.7 | 7.8 | 3.4 | 6.2 | 5.6 | 2.0 | 1.3 |
| DuplexJev-32B-Para v1.3 | 32.7 | 31.8 | 41.7 | 29.1 | 27.0 | 10.6 | 4.4 | 8.5 | 5.3 | 3.8 | 1.5 |
| DuplexJev-4B-Para v1.1 | 59.6 | 61.5 | 70.8 | 48.7 | 68.9 | 39.2 | 11.1 | 21.4 | 21.4 | 9.6 | 6.1 |

- The encoder is fine in every language; the gap opens in the connector + LLM.
- The weak languages are exactly the ones with little alignment data: Japanese ~11 k utterances out of 18.9 M
  (0.05 %), Vietnamese ~2 k, Korean none. Spanish / French / German had 0.4–0.8 M and are near the ceiling on 32B.
- The errors sound right but read wrong ("父は銀行を持っていた" → "地は金 kobu まっていた"): the model hears the
  sounds but has no words for them, so understanding fails too.

## 2. Round 1: more data, too much weight

Continue v1.1 with 88 k weak-language utterances at about a third of all samples. Japanese and Korean improved a lot
(59.6 → 39.0, 61.5 → 24.3), but other skills paid: the paralinguistic benchmark fell to 47.3 and English neutral-tone
judgement dropped 5.5 points. Not released.

## 3. Round 2 (released as v1.2)

Three changes:

1. **Cleaner labels.** 280 k candidate clips (Emilia / Emilia-YODAS, YODAS2, Zeroth-Korean, VIVOS, FLEURS train,
   Common Voice) were re-transcribed with Qwen3-ASR-0.6B. Clips whose label disagreed (CER > 15 % for ja / ko / th,
   WER > 25–30 % for the others) or whose detected language was wrong were dropped; unlabelled YODAS2 speech in
   Vietnamese, Thai and Arabic got the Qwen3-ASR transcript as its label. 162 k clips kept.
2. **Understanding, not just repeating.** 36 k "continue this text" items in the same languages, targets written by a
   text LLM (Qwen3-32B) from the transcript.
3. **Half the weight, more replay.** The new data got half the round-1 weight; the whole v1.1 mix was replayed, with
   non-verbal sounds ×1.3 and stress ×1.2, plus 18 k new Mandarin paralinguistic questions from a licensed
   conversational corpus (training speakers only).

2,500 steps, one H200, about two hours.

| | v1.1 | round 1 | **v1.2** |
|---|---:|---:|---:|
| FLEURS ja / ko / vi / th / ar | 59.6 / 61.5 / 70.8 / 48.7 / 68.9 | 39.0 / 24.3 / 52.8 / 32.4 / 57.5 | **35.8 / 20.3 / 41.3** / 33.3 / 58.6 |
| FLEURS es / fr / de / zh / en | 11.1 / 21.4 / 21.4 / 9.6 / 6.1 | 10.8 / 20.0 / 20.0 / 8.8 / 5.6 | 11.1 / 19.3 / 19.0 / 9.4 / 5.6 |
| paralinguistic benchmark, 1,035 items | 47.8 | 47.3 | **47.6** |
| StressTest / MSPB | 81.5 / 79.7 | 81.0 / 76.6 | **84.0** / 77.3 |
| Mandarin paralinguistics, held-out speakers (8-task mean) | 56.5 | — | **82.5** |
| older Mandarin conversation set (NV-magic5) | 52.7 | 52.9 | **82.5** |
| qa100 / gender / emotion / Easy-Turn (vLLM) | 76 / 87.4 / 87.9 / 91.4 | 76 / 86.9 / 87.5 / 93.1 | 73 / 87.1 / 88.2 / 90.9 |

With half the weight, the re-checked data did more than round 1's larger, noisier set for Japanese, Korean and
Vietnamese. Thai and Arabic, which got the least new everyday speech, barely moved.

## 4. Limitations

- Still 3–6× the encoder's own error in the weak languages; the 4B LLM is a real limit (even well-covered European
  languages stay 3–4× above the ceiling on 4B, about 1× on 32B).
- Arabic and Thai need more conversational data.
- qa100 (−3), StressPresso (−4.5), English stress position (−3.8) and MSPB (−2.4) are below v1.1.
- FLEURS is read speech; conversational accuracy in these languages is not measured here.
