# Can a speech-decision LLM tell who is speaking? Shortcuts, probes and ten ablations

*Adventists.ai research note, 2026-10-06. [中文版](2026-10-speaker-identity_zh.md)*

*DuplexJev reads closed-set decisions from speech as one token of an LLM that hears the audio.* We asked whether the
same structure can verify a speaker: given a reference clip and a test clip, is it the same person? **Short answer:
it learns a coarse version of the skill, but it stays far behind dedicated speaker models, and model size does not
help.** The best configuration reaches an EER of 20.5 % on VoxCeleb1-O and 29.6 % on VoxCeleb1-H (a dedicated
x-vector model scores 4.4 % on the same trials), and 4.2 % on same-gender Mandarin pairs. We do not release a
speaker version.

| best of each family (EER %, lower is better) | Vox1-O | Vox1-O same-gender | Vox1-H | AISHELL-1 same-gender | qa100 | Easy-Turn |
|---|---:|---:|---:|---:|---:|---:|
| DuplexJev-4B-Turn, no speaker training | 48.7 | 48.7 | 48.0 | 53.0 | 74 | 92.5 |
| + speaker LoRA, current connector (F3) | 28.4 | 30.3 | 36.2 | 6.1 | 76 | 90.4 |
| + speaker LoRA, cross-attention connector (F1) | **20.5** | 26.9 | **29.6** | **4.2** | 73 | 90.6 |
| + frozen x-vector branch, gain 10 (E7) | 19.9 | **22.5** | 31.6 | 8.2 | 70 | 91.1 |
| DuplexJev-32B-Turn + cross-attention connector (G1) | 22.1 | 28.1 | 31.2 | 7.1 | 92 (was 96) | 93.5 (was 95.3) |
| WavLM-base-plus-sv x-vector alone (cosine) | 4.4 | — | — | — | — | — |

## 1. The first numbers were a gender shortcut

A first pilot (4B, LoRA + connector, 45k pairs from Dynamic-SUPERB VoxCeleb1/AISHELL-1, LibriSpeech and synthetic
two-speaker dialogues) went from chance to 85–97 % accuracy on VCTK and LibriSpeech pairs and 79 % on VoxCeleb1 for
the 32B model. Those numbers do not measure speaker identity:

- In randomly drawn VoxCeleb1 pairs, 48.5 % of the different-speaker pairs differ in gender. Answering "same" iff
  the genders match already scores **74.2 %**.
- On different-speaker pairs of the **same** gender the 32B model was right only 46.2 % of the time (cross-gender:
  99.6 %).

From then on we report only **EER on hard trials**: the official VoxCeleb1-O (cleaned) and VoxCeleb1-H lists
(non-target pairs share gender and nationality), EER restricted to same-gender pairs, and a Mandarin set built from
AISHELL-1 test speakers (never in training) where every non-target pair has the same gender. 1,000 target + 1,000
non-target trials each; the probability of the "same person" option is the score, averaged over both option orders.
The pilot's training data contained VoxCeleb1 dev speakers, so its VoxCeleb1-H number is contaminated; all later
runs use data with no VoxCeleb1 speakers.

## 2. Where the identity information is lost

Before training, we mean-pooled each layer's output for both clips and scored the pair by cosine similarity:

| EER % (untrained, mean-pool cosine) | enc. layer 1 | 6 | 9 | 13 | 14 | 18 (last) | connector output |
|---|---:|---:|---:|---:|---:|---:|---:|
| LibriSpeech clean | 11 | 12 | 11 | 18 | 20 | 18 | 35 |
| VoxCeleb1 test | 28 | 21 | 18 | 14 | 16 | 22 | 35 |

The Qwen3-ASR encoder carries identity, most of it in the middle layers (8–14) on real-world audio, but the
connector, trained only for content, gender and emotion, throws most of it away. This motivated two directions:
read the middle layers (cross-attention connector) or add a dedicated speaker encoder.

## 3. Training data

`spk_v3`: reference clip (≤ 6 s) + 0.8 s silence + test clip (≤ 8 s), one A/B token. 51k items:

- VoxCeleb2 dev: 30k pairs + 6k "is the reference speaker in this two-person conversation?" items;
- AISHELL-1 train speakers: 8k pairs;
- AISHELL-3 train: 8k pairs.

Target pairs come from different videos and sessions; 85 % of non-target pairs share the gender. Every run starts
from DuplexJev-4B-Turn (or 32B-Turn) with a new rank-16 LoRA, the connector trainable, and the turn-taking mix
replayed so the other skills are kept. 4,000 steps.

An engineering note: about a third of VoxCeleb2's m4a files cannot be decoded from a pipe (the moov atom sits at
the end). Our first build (`spk_v2`) silently lost half the English pairs; decoding through a temporary file fixed
it. Going from v2 to v3 lowered EER by about 1–2 points.

## 4. Architecture ablations (4B)

All new parts start at zero, so step 0 is exactly the base model (checked: max |Δlogit| = 0).

| run | change | Vox1-O | same-gender | Vox1-H | AISHELL-1 SG | qa100 | ZJU | VB OBQA / MMSU | Easy-Turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 (v2 data) | current connector | 32.4 | 35.1 | 35.0 | 6.3 | 77 | 55 | 45.1 / 41.9 | 90.5 |
| E2 | cross-attention connector | 22.7 | 28.3 | 30.5 | 4.6 | 76 | 59 | 47.3 / 40.4 | 93.0 |
| E3 | + WavLM hidden states, own projector | 29.9 | 32.8 | 34.5 | 6.8 | 76 | 54 | 46.8 / 41.8 | 91.3 |
| E4 | + WavLM hidden states, fused into projector | 29.3 | 32.6 | 34.0 | 5.9 | 76 | 57 | 47.3 / 41.2 | 91.9 |
| E5 | + x-vector per token, own projector | 27.7 | 30.4 | 34.8 | 6.2 | 75 | 50 | 45.7 / 41.6 | 91.4 |
| E5 | + x-vector per token, fused | 31.4 | 33.5 | 35.8 | 6.0 | 76 | 55 | 46.6 / 41.1 | 91.3 |
| E6 | cross-attention + x-vector fused | 23.0 | 28.4 | 31.1 | 4.7 | 73 | 56 | 44.8 / 41.1 | 91.4 |
| E7 | x-vector, own projector, gain 10 | 22.9 | 24.2 | 32.0 | 6.9 | 74 | 53 | 45.1 / 39.9 | 91.6 |
| E7 | x-vector, fused, gain 10 | 19.9 | 22.5 | 31.6 | 8.2 | 70 | 53 | 42.6 / 39.6 | 91.1 |
| E7 | cross-attention + x-vector fused, gain 10 | 22.3 | 28.8 | 31.0 | 4.4 | **36** | 37 | 31.6 / 33.7 | 76.3 |
| F3 (v3 data) | current connector | 28.4 | 30.3 | 36.2 | 6.1 | 76 | 55 | 44.6 / 40.7 | 90.4 |
| **F1** | cross-attention connector | **20.5** | 26.9 | **29.6** | **4.2** | 73 | 59 | 45.7 / 40.9 | 90.6 |
| F2 | cross-attention + x-vector own projector, gain 3 | 20.9 | 26.9 | 30.3 | 4.2 | 73 | 52 | 43.3 / 38.1 | 92.1 |

The rows above use 1,000 + 1,000 trials; a difference of about 2 EER points is within noise, and qa100 has 100
questions (±4 points).

What the ablations show:

1. **The cross-attention connector is the most useful single change.** Letting the last encoder layer attend to
   layers 14 and 9 cuts EER by about 8 points on Vox1-O and 5–6 on Vox1-H, with no clear cost elsewhere at 4B. This
   matches the probe: the identity is in the middle layers.
2. **A dedicated speaker encoder helps only once it is actually trained.** Feeding WavLM hidden states (E3/E4)
   gained 2–3 points. These states are a weak identity signal on their own (mean-pool cosine EER 24–36 % by layer);
   the identity of WavLM-sv lives in its x-vector head (EER 4.4 %). Feeding per-token x-vectors (1.5 s windows, E5)
   still gained little, because the zero-initialised output layer barely grew in 4,000 steps (weight norm about
   1/20 of the main projector's). Scaling the branch output by 10 (an effective learning-rate boost, E7) made it
   work on Vox1-O (19.9 %, same-gender 22.5 %, the best so far) but **pushed out content**: qa100 fell to 70, and
   combined with the cross-attention connector it collapsed (qa100 36).
3. **VoxCeleb1-H does not move below about 30 %** in any configuration: when the impostor shares gender and
   nationality, the LLM cannot exploit even a strong embedding in this format.

## 5. Scaling to 32B does not help

| 32B-Turn + speaker LoRA (v3 data) | Vox1-O | same-gender | Vox1-H | AISHELL-1 SG | qa100 | ZJU | gender | emotion | Easy-Turn | VB OBQA / MMSU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32B-Turn as released | ≈47 | — | 49.0 | — | 96 | 87 | 91.5 | 91.1 | 95.3 | 85.5 / 72.1 |
| current connector (G2) | 25.3 | 31.8 | 33.8 | 7.5 | 97 | 87 | 93.8 | 89.9 | 95.1 | 84.8 / 71.8 |
| cross-attention connector (G1) | 22.1 | 28.1 | 31.2 | 7.1 | 92 | 86 | 98.0 | 86.0 | 93.5 | 85.9 / 71.6 |

The 32B model is no better than the 4B one at speaker verification (best Vox1-H 31.2 vs 29.6): the bottleneck is how
much voice information the audio front-end passes on, not the size of the LLM. At 32B the cross-attention connector
also costs content and turn-taking (qa100 97 → 92, Easy-Turn 95.1 → 93.5); gender rises to 98.0, a sign that it
passes more timbre.

## 6. Conclusions and limitations

- Speaker verification is learnable in this structure (from chance to 20–30 % EER on hard English trials, about 4 %
  on clean Mandarin), but it is not competitive: dedicated models are at 1–4 %. For "is this the enrolled person?",
  use a dedicated speaker model; DuplexJev is suited to coarse questions ("did the speaker change?", gender, age
  group).
- Random-pair speaker benchmarks (Dynamic-SUPERB VoxCeleb/VCTK/LibriSpeech) measure gender. Report hard trials and
  same-gender EER.
- If speaker skill matters, the cross-attention connector is the change to make; it is not supported by the current
  vLLM plugin.
- Limitations: one seed per configuration; 4,000 steps; the two-clip "reference + silence + test" format is
  only one way to pose the task (separate audio segments or an enrolment-embedding token were not tried); CN-Celeb
  could not be downloaded at usable speed, so no multi-genre Mandarin test.

Speaker verification is a biometric capability; any future release will state its intended use and limits.

Scripts (H200, `exp03_train/scripts_ml_asr/`): `probe_layers.py`, `spk_eval_hard.py`, `spk_bench.py`,
`build_spk_v3.py`, `patch_spk_branch.py` (speaker branch: modes sep / fuse / xsep / xfuse, gain), `spk2.sh`,
`xvec_ceiling.py`.
