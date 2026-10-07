# Teaching a speech-decision LLM to hear stress: data, two gates, negatives and the "source shortcut"

*Adventists.ai research note, 2026-10-08. [中文版](2026-10-stress-understanding_zh.md)*

The same words mean different things depending on which one is stressed ("**I** didn't say she took the money" vs. "I
didn't say she took the **money**"). Mandarin uses stress heavily too, and where a speaker pauses can change the parse
("咬死了｜猎人的狗" *bit the hunter's dog to death* vs. "咬死了猎人的｜狗" *the dog that killed the hunter*).
DuplexJev reads closed-set decisions from speech as one token of an LLM that hears the audio. We asked whether it can
learn to use stress and pauses.

**Takeaways:**

- The released DuplexJev models ignore stress entirely: they are at chance in both languages. They partly use pauses.
- A LoRA trained on public data teaches the 32B model to find the stressed word on sentences it never saw in training,
  in Chinese and English. The full recipe reaches 94 % on Mandarin stress and 91 % on pauses.
- Training on stressed examples only makes the model "hear" stress in neutral speech, so negatives are needed. Once we
  added them, a subtler failure appeared: the model judged the **source of the audio** (real speech vs. a given TTS)
  instead of the stress. Negatives have to be paired with positives inside every source.
- Questions that ask for the *meaning* of a stressed sentence (same words, choose by prosody) cost 2–4 points on
  spoken multiple-choice QA (VoiceBench). Questions about *which word is stressed* do not, and they still transfer
  part of the way to meaning questions. The released model uses that second recipe.

| | DuplexJev-32B-Turn | **DuplexJev-32B-Stress** |
|---|---:|---:|
| qa100 / ZJU (spoken QA) | 96 / 87 | 96 / 87 |
| gender / emotion | 91.5 / 91.1 | 92.1 / 91.4 |
| Easy-Turn / CoDeTT zh / CoDeTT en / TurnBench-dev | 95.2 / 69.2 / 70.0 / 88.7 | 94.4 / 69.7 / 70.5 / 88.3 |
| VoiceBench OBQA / MMSU | 85.5 / 72.1 | 86.2 / 71.4 |
| which word is stressed: Mandarin DEBATE / English StressPresso ("none" offered) | 35 / 1 | **92 / 88** |
| neutral speech answered "no word is stressed": AISHELL / LibriSpeech | 92 / 50 | **99 / 99** |
| meaning from stress: DEBATE stress (held-out sentences) / StressTest | 35 / 52 | 83 / 67 |
| meaning from pause: DEBATE pause (held-out sentences) | 71 | 75 |

At 4B we release **DuplexJev-4B-Stress** next to 4B-Turn, as a second version:

| | DuplexJev-4B-Turn | **DuplexJev-4B-Stress** |
|---|---:|---:|
| qa100 / ZJU (spoken QA) | 74 / 52 | 73 / 51 |
| gender / emotion | 89.4 / 92.0 | 89.1 / 91.4 |
| Easy-Turn / CoDeTT zh / CoDeTT en / TurnBench-dev | 92.5 / 67.0 / 68.6 / 87.3 | 91.8 / 67.0 / 68.9 / 86.9 |
| VoiceBench OBQA / MMSU | **46.4 / 41.7** | 42.0 / 38.2 |
| which word is stressed: DEBATE / StressPresso ("none" offered) | 39 / 26 | **95 / 95** |
| neutral speech answered "no word is stressed": AISHELL / LibriSpeech | 78 / 37 | **99 / 99** |
| meaning from stress: DEBATE stress (held-out) / StressTest | 36 / 49 | **94 / 86** |
| meaning from pause: DEBATE pause (held-out) | 67 | **85** |

4B-Stress is about 4 points below 4B-Turn on VoiceBench (section 6), so 4B-Turn stays the default small model.

## 1. Starting point: stress unused, pauses partly used

All evaluations are single-token multiple choice, with probabilities averaged over cyclic shifts of the options. As a
control, the transcript without stress marks replaces the audio; this control should be at chance.

| zero-shot | DEBATE stress (chance 33.5) | DEBATE pause (chance 49.4) | StressTest (50) | StressPresso (50) |
|---|---:|---:|---:|---:|
| DuplexJev-4B-Turn, audio | 35.5 | **67.3** | 48.6 | 50.5 |
| DuplexJev-4B-Turn, transcript only | 33.5 | 49.4 | 50.9 | 50.5 |
| DuplexJev-32B-Turn, audio | 34.7 | 70.6* | 52.3 | 53.5 |

\* Measured on the held-out sentences.

For reference, GPT-4o-audio scores 68.8 on StressTest, and the three models in the DEBATE paper score 58–68 % on its
stress task.

## 2. Data and benchmarks

| data | lang | audio | use | licence |
|---|---|---|---|---|
| DEBATE Task_Stres / Task_Pause | zh | 10 real speakers; 120 stress sets, 198 pause sets | split by **sentence**: 2/3 train, 1/3 held-out test | CC BY-NC 4.0 |
| MSPB | zh | 1 real speaker, 160 items | test | CC BY 4.0 |
| StressTest, StressPresso | en | real | test | — |
| Stress-17K (WhiStress-consistent part), TinyStress-15K (4 k) | en | synthetic | train | CC BY-NC 4.0 |
| Expresso emphasis style (StressPresso clips removed) | en | real | train | CC BY-NC 4.0 |
| AISHELL-1, LibriSpeech | zh / en | real neutral read speech | negatives | Apache 2.0 / CC BY 4.0 |

**The split matters.** Our first version trained on all DEBATE speakers and sentences and scored 89.3 % on stress. With
a sentence-level split it scores 88.5 % on held-out sentences, but its pause score falls from 70.6 to 54.8 (section 4).
Every Mandarin number after that is on sentences the model never saw in training.

## 3. Synthetic Mandarin: two gates or nothing

Public Mandarin stress data is scarce. We had Qwen3-32B write 5,000 minimal sets ("same sentence, different stressed
word, different meaning") and synthesised them with CosyVoice2, using `<strong>` for stress and commas for pauses. One
voice reads every variant of a sentence. Unchecked synthetic items would put wrong answers into training, so every
clip has to pass two gates:

- **Text gate.** A second LLM (Qwen2.5-72B) reads the sentence with the stress marked and must pick the written
  meaning among all of the sentence's meanings, with p ≥ 0.6 under every option order. This checks that the stress
  position really implies that meaning.
- **Acoustic gate.** Character alignment with MFA. Compared with the same voice's other variants, the stressed span
  must be more prominent on at least two of three measures (duration > 1.08×, energy +1 dB, F0 peak +0.8 semitones),
  and no measure may point clearly the other way. For pauses, the gap at the intended break must be ≥ 0.10 s and the
  longest in the sentence.

| | text gate | acoustic gate | both |
|---|---:|---:|---:|
| stress | 39 % | 31 % | 1,586 clips |
| pause | 5.4 % | 28.7 % | 111 clips → dropped |

LLMs write poor pause ambiguities: switching models, thinking mode and a self-check still left 2.6 % passing. The TTS
actually realised the stress only about a third of the time. Clips that passed both gates added little (DEBATE stress
93.9 → 94.7, pause 90.2 → 89.7), so later versions use them only to balance speech sources (section 5).

## 4. Mixing: English first, plus real Mandarin training sentences

32B, LoRA r16, 4,000 steps, mixed into the original content-distillation / paralinguistic / turn-taking data:

| version | DEBATE stress (held-out) | DEBATE pause (held-out) | StressTest | StressPresso | qa100 | VB OBQA |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot | 34.6 | 70.6 | 52.3 | 53.5 | 96 | 85.5 |
| v1: all DEBATE + unchecked synthetic | 88.5 | **54.8** | 88.5 | 94.1 | 94 | 83.3 |
| v3 h1: English first + DEBATE training sentences | 93.9 | 90.2 | 88.1 | 92.1 | 95 | 83.5 |
| v3 h2: h1 + synthetic that passed both gates | 94.7 | 89.7 | 87.2 | 90.1 | 93 | 81.8 |

v1 was trained on stress items only, and its pause score dropped from 70.6 to 54.8. The model learned that every
sentence has a stressed word and lost the pause cues it already used. Adding DEBATE pause items in v3 brought pauses
back to 90.

## 5. Negatives and the "source shortcut"

Every evaluation item also offers "no word is particularly emphasized". This measures two things: whether the model
answers "none" on neutral speech, and whether it still picks the stressed word when there is one. Training uses
different wording for both the question and the "none" option.

**Positives only → hallucinated stress.** v3 h1 answers "none" on only 33 % of neutral Mandarin (AISHELL) and 45 % of
neutral English (LibriSpeech). The zero-shot model scores 92 % and 50 %.

**v4: one third negatives, balanced by language.** Neutral speech becomes almost perfect (99.7 / 99.5), but real
English stress with a "none" option drops to **9.6 %**. Every negative was real read speech, while nearly every
positive with a "none" option was synthetic, so the model learned "real English → none". A same-sentence,
same-voice check exposed the Mandarin version of the same problem. CosyVoice reads a sentence once with stress and
once neutrally, and the stressed version is answered correctly only **1.1 %** of the time: "CosyVoice voice → none".

**v5: balance within every source.**

- Real English: all Expresso stress positives carry the "none" option and are upsampled ×4; real negatives are capped
  at the same count.
- Synthetic English: CosyVoice reads the same Stress-17K / TinyStress sentences neutrally (LibriSpeech voices as
  prompts) as synthetic negatives.
- Mandarin: gated CosyVoice stress positives, paired with CosyVoice neutral readings.

| with a "none" option | v3 h1 | v4 (1/3 negatives) | v5 p33 | v5 p20 |
|---|---:|---:|---:|---:|
| neutral → none: AISHELL | 33.3 | 99.7 | 98.7 | 99.0 |
| neutral → none: LibriSpeech | 44.5 | 99.5 | 99.5 | 98.5 |
| stressed → right word: DEBATE | 94.8 | 96.0 | 96.5 | 95.7 |
| stressed → right word: StressPresso | 96.8 | **9.6** | 95.2 | 93.6 |
| same sentence and voice (zh): neutral / stressed | 65.7 / 29.7 | 100 / **1.1** | 98.6 / 51.6 | 97.1 / 57.1 |
| synthetic English: neutral / stressed | 28.2 / 97.1 | 43.6 / 100 | 99.2 / 99.0 | 99.2 / 99.0 |

Lesson: **"match the distribution" has to hold inside every source, not just overall.** Aggregate numbers hid the
problem: v4 looked nearly perfect on neutral speech and on Mandarin stress. Only same-sentence pairs and per-source
breakdowns revealed it.

**An adjustable threshold.** Subtracting a bias b from log p(none) trades misses against false alarms. On the Mandarin
pairs, p20 reaches 92.9 neutral / 68.1 stressed at b = 2.5. On real Mandarin data both stay above 95 for any b in 0–4.
A deployment can expose b as a knob.

## 6. The cost, and the recipe we release

**Meaning-choice items hurt spoken multiple-choice QA.** Every model trained with the full item mix was 2–4 points
below the release on VoiceBench (spoken multiple choice). We ruled out the causes one at a time:

| 32B | stress data | VB OBQA / MMSU | DEBATE stress, meaning choice |
|---|---|---:|---:|
| released 32B-Turn | — | 85.5 / 72.1 | 34.6 |
| control: same continued training, (almost) no stress data | — | **86.8 / 72.8** | 34.1 |
| all item types, weight 3600 | v5 | 81.5 / 67.8 | 94.3 |
| all item types, weight 1500 / 800 | v5 | 83.1 / 70.5 · 84.6 / 70.3 | 93.6 · 92.0 |
| all item types + content distillation ×1.6 | v5 | 78.5 / 67.4 | 94.3 |
| meaning-choice items kept at 1/4 | v5e | 83.5 / 69.9 | 91.1 |
| **detection items only (which word is stressed / where the pause is / is anything stressed)** | v5d | **86.2 / 71.4** | 83.3 |

- Continued training on its own costs nothing (the control). More content distillation does not bring the score back.
- The loss comes from meaning-choice items, where the options share the same words and only prosody decides. They
  teach "in a multiple-choice question, don't look at the content", and that habit carries over to spoken
  multiple-choice benchmarks.
- Without those items VoiceBench is back at release level. Learning only *which word is stressed* still transfers part
  of the way to meaning questions (35 → 83).

DuplexJev-32B-Stress uses the detection-only recipe at weight 1200. It matches 32B-Turn on every spoken-QA,
paralinguistic and turn-taking benchmark within statistical noise (MMSU −0.7, Easy-Turn −0.8, TurnBench −0.4).
Pushing meaning-from-stress above 90 currently costs VoiceBench points (the full-mix model: 94 on meaning, 81.5 on
OBQA), so we do not release that recipe.

**4B is more sensitive, and the cost did not go away.** The same approach at 4B left VoiceBench 4–10 points down in every
configuration. Its control does not drop either (46.2 / 41.9), so the cause is again the stress data itself:

| 4B | VB OBQA / MMSU | qa100 | DEBATE stress, meaning choice |
|---|---:|---:|---:|
| released 4B-Turn / control | 46.4 / 41.7 · 46.2 / 41.9 | 74 · 76 | 35.7 · 34.6 |
| all item types, weight 3000 / 2200 / 1500 / 800 | 38.9 / 38.0 · 40.7 / 36.1 · 36.0 / 36.2 · 40.9 / 37.3 | 73 · 73 · 72 · 71 | 94.1 · 92.5 · 92.2 · 90.0 |
| detection only, weight 2200 / 1300 / 600 | 42.4 / 38.1 · 40.2 / 38.0 · 40.4 / 37.3 | 70 · 72 · 75 | 81.9 · 79.8 · 64.7 |
| detection only, 2,000 steps | 37.1 / 37.0 · 38.9 / 37.2 | 71 · 73 | 78.1 · 72.2 |
| **all item types + content distillation ×2.5 (released)** | 42.0 / 38.2 | 73 | 93.6 |
| all item types + content distillation ×4, weight 3000 / 2000 | 42.9 / 38.0 · 40.7 / 36.4 | 71 · 69 | 92.2 · 92.2 |

- At 4B, dropping meaning-choice items did not help VoiceBench (42.4 / 38.1, about the same as the full mix) and cost 12
  points on meaning from stress.
- More content distillation recovers a little.
- The 4B release therefore uses the full mix with content distillation ×2.5 and sits next to 4B-Turn: use 4B-Stress if
  stress matters more than about 4 points of spoken multiple choice, 4B-Turn otherwise.
- VoiceBench differences between weights are not monotonic, which suggests noise of about ±2 points at 4B (OBQA has 455
  items and 4B scores about 45 %).

## 7. Limitations

- The only Mandarin stress recordings are DEBATE (10 readers). All of it is read speech, not conversation; stress in
  spontaneous speech is weaker and more varied, and false-alarm rates may differ there.
- CosyVoice's synthetic stress is weak: on the same-sentence pairs the stressed version is answered correctly 52–57 %
  of the time (91 clips), and only 29.7 % by the model without negatives.
- The "emphasis-style but unmarked" Expresso sentences have doubtful labels (67–71 % answered "none"). We do not use
  them as the main neutral test.
- All tests are closed-set questions. They measure whether the model *can* use stress, not whether it volunteers it
  in open conversation.
- Our own mistake: the first build of the English "stressed + none option" test parsed the stress annotation wrongly,
  so the gold answers were garbage. All models were re-scored after the fix and only corrected numbers appear here;
  the v4 conclusion is unchanged (9.6).

## 8. Licence

DEBATE, Stress-17K, TinyStress-15K and Expresso are all CC BY-NC 4.0, so the stress-capable weights are released
under CC BY-NC 4.0 and cannot be used commercially. DuplexJev itself is already CC BY-NC because of its emotion data.
A commercial version would need replacements for these four datasets and for the emotion data.

## Reproduction

Under `exp03_train/scripts_ml_asr/` on our training server:

- test sets: build_stress_items.py, build_neutral_items.py, build_pair_items.py
- synthesis and gates: gen_zh_stress_text.py, synth_zh_stress.py, text_check.py, acoustic_check.py
- training sets: build_stress_train_v3/v4/v5.py (v5d / v5e are filtered copies of v5b)
- training: stress_pipe3–14.sh with spk2.sh (SPK_DS, SPK_W, CONT_MULT)
- evaluation: stress_eval.py (writes per-option probabilities) and stress_summary.py
