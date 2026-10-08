# One model, more of the voice: non-verbal sounds, speaking style and speaker on top of stress (DuplexJev-*-Para)

*2026-10-09 · models: [DuplexJev-32B-Para](https://huggingface.co/adventists-ai/DuplexJev-32B-Para), [DuplexJev-4B-Para](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) · [中文](2026-10-para-abilities_zh.md)*

**Summary.** DuplexJev-Turn and -Stress hear little beyond words, gender, emotion, turn-taking and stress: on unseen
real speech they detect laughter at AUC 0.71–0.73, and breathing, coughs and sighs at 0.40–0.70 (0.5 = chance).
Adding about 108 k detection questions built from public data lifts these to 0.70–0.85, at or above an AudioSet AST
tagger, and adds speaking style (whisper, shouting …) and a coarse same-speaker judgment. Three things decided
whether the rest of the model survived:

1. **Spoken multiple-choice replay.** Questions whose options do not depend on the content of the speech (speaker
   pairs, "which meaning does the stress imply") teach the model to stop reading the content when answering a
   multiple-choice question; VoiceBench fell 4–6 points. Replaying about 11 k spoken content questions (ARC,
   OpenBookQA and CommonsenseQA train splits, read by TTS, kept only if a text teacher answers them correctly) brought
   it back and above.
2. **Negatives from the same recording.** For NonVerbalSpeech-38K, every positive window (containing the event) has
   a negative cut from the same clip away from the event: same voice, same room, same microphone. Otherwise the model
   can learn which corpus a clip comes from instead of the sound.
3. **Weight.** At weight 3000 the non-verbal data crowds out stress (4B −4.5, 32B −8 on DEBATE meaning); at 800–1500
   most of the non-verbal gain stays and stress returns to within 2–3 points.

## Data

| source | licence | what it gives | negatives | held out |
|---|---|---|---|---|
| VocalSound | CC BY-SA 4.0 | isolated laughter, cough, sigh, sneeze, sniff, throat clearing (≤2,000 per class) | ask about an absent class | speakers with id % 10 = 0 |
| NonVerbalSpeech-38K (zh, en) | CC BY-NC 4.0 | radio drama / animation with an event region | same clip, away from the region | md5(id) % 20 = 0 |
| NonverbalTTS train + other | — | real speech (VoxCeleb, Expresso) with emoji tags | — | official test + dev for evaluation only |
| Expresso read | CC BY-NC 4.0 | laughing (→ laughter), whisper, default | — | speaker ex04 |
| EARS | CC BY-NC 4.0 | style sentences whose text is unique to one style (whisper, loud, fast, slow, high, low, regular) | — | speakers with id % 10 = 0 |

Question types: *is there X* (yes/no), *besides speech, which sound* (two distractors + "none of these"), and
*how is the speaker talking* (4-way). Chinese clips are asked in Chinese 80% of the time. The Hugging Face mirror of
EARS lost its file names, so only sentences whose text identifies the style could be labelled.

## Results

AUC on clips never seen in training (0.5 = chance): NonverbalTTS test (real speech) and a third-party sample of natural
Mandarin conversation (MagicData, 5.6 h; its labels agree with AST at AUC 0.81 laughter / 0.89 breathing).

| model | laughter | breathing | cough | sigh | conversation: laughter / breathing | style (EARS, 7-way) |
|---|---:|---:|---:|---:|---:|---:|
| 32B-Stress | 0.73 | 0.56 | 0.40 | 0.70 | 0.61 / 0.44 | 45% |
| **32B-Para** | **0.85** | **0.73** | **0.70** | **0.74** | **0.80 / 0.78** | **69%** |
| 4B, same recipe without non-verbal data | 0.71 | 0.51 | 0.53 | 0.55 | 0.70 / 0.50 | 55% |
| **4B-Para** | 0.80 | 0.76 | 0.75 | 0.81 | 0.84 / 0.84 | 67% |
| AudioSet AST tagger (reference) | 0.71 | 0.75 | 0.85 | 0.76 | — | — |

The rest of the model (paper protocol):

| | qa100 | ZJU-ML | gender | emotion | Easy-Turn* | VoiceBench OBQA‡ / MMSU | DEBATE stress meaning | same speaker EER (AISHELL-1 same gender / VoxCeleb1-O) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32B-Turn | 96 | 87 | 91.5 | 91.1 | 95.2 | 85.5 / 72.1 | 35 | — |
| 32B-Stress | 96 | 87 | 92.1 | 91.4 | 94.4 | 86.2 / 71.4 | 83 | — |
| **32B-Para** | 95 | 87 | 92.1 | 89.5 | 94.0 | 90.1 / 73.4 | 81 | 18.2% / 38.5% |
| 4B-Turn | 74 | 52 | 89.4 | 92.0 | 92.5 | 46.4 / 41.7 | — | — |
| 4B-Stress | 73 | 51 | 89.1 | 91.4 | 91.8 | 42.0 / 38.2 | 94 | — |
| **4B-Para** | 79 | 50 | 89.5 | 91.5 | 91.5 | 57.8 / 41.8 | 89 | 8.7% / 32.4% |

\* in-domain. ‡ OpenBookQA's train split is in the replay data, so the OBQA gain is partly in-domain; MMSU is not.
vLLM acceptance of 32B-Para (package wording): qa100 95, ZJU-ML 90, gender 91.9, emotion 89.5, Easy-Turn 94.1.

### Weight sweep (non-verbal data weight; everything else fixed)

| run | nv weight | DEBATE stress meaning | VoiceBench MMSU | emotion | NonverbalTTS breathing AUC |
|---|---:|---:|---:|---:|---:|
| 4B | 0 | 91.2 | 41.8 | 91.6 | 0.51 |
| 4B | 800 (released) | 89.0 | 41.8 | 91.5 | 0.76 |
| 4B | 1500 | 89.0 | 41.7 | 89.4 | 0.82 |
| 4B | 3000 | 86.7 | 40.2 | 90.2 | 0.86 |
| 4B | 6000 | 82.2 | 40.8 | 90.8 | — |
| 32B (stress 2000) | 0 | 79.1 | 73.8 | 91.8 | 0.54 |
| 32B (stress 3000, mcq 4000; released) | 1500 | 80.6 | 73.4 | 89.5 | 0.73 |
| 32B (stress 2000) | 3000 | 71.2 | 71.7 | 91.0 | 0.74 |

## Limitations

- Non-verbal training audio is mostly acted; natural conversation scores are lower.
- Coughs in natural conversation are rare in every corpus we have, so cough is the least tested sound.
- Speaker judgments are coarse (dedicated models reach ~1–2% EER) and must not be used to identify people.
- Each 4B/32B number is one run; differences under 1–2 points are within noise.
