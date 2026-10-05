# A rank-16 LoRA makes a speech-decision LLM a competitive turn-taking detector

*Adventists.ai research note, 2026-10-05. [中文版](2026-10-turn-taking-lora_zh.md)*

*DuplexJev reads the decisions a voice agent needs (turn-taking, emotion, intent, …) as one token from an LLM that
hears the audio.* **Adding a rank-16 LoRA trained on public turn-taking data puts it level with dedicated detectors:**
DuplexJev-32B-Turn scores 95.3 on Easy-Turn (dedicated detector 96.4) and 69.2 / 70.0 on CoDeTT zero-shot (Qwen3-Omni
70.4 / 70.9; dedicated turn models 37.9–65.4), and nothing else got worse.

| | Easy-Turn | CoDeTT zh / en | TurnBench-dev clips† | qa100 | ZJU-ML | gender | emotion | VoiceBench OBQA / MMSU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DuplexJev-32B | 79.1 | 64.7 / 62.9 | 60.7 | 90 | 87 | 89.4 | 90.6 | 83.7 / 70.8 |
| **DuplexJev-32B-Turn** | **95.3** | **69.2 / 70.0** | **88.7** | **96** | 87 | **91.5** | **91.1** | **85.5 / 72.1** |
| DuplexJev-4B | 78.1 | 62.0 / 53.5 | 49.4 | 74 | 50 | 88.6 | 91.2 | 47.3 / 41.3 |
| **DuplexJev-4B-Turn** | **92.5** | **67.0 / 68.6** | **87.3** | 74 | 52 | 89.4 | 92.0 | 46.4 / 41.7 |

For reference on CoDeTT (published numbers): Easy-Turn 37.9 (zh), Smart-Turn-v3 51.4 (en), NAMO-Turn 59.5 (zh),
FireRedChat 65.4 (en), MiniCPM-o-4.5 64.4 / 65.1, GPT-4o-audio 66.6 / 71.9, Qwen3-Omni 70.4 / 70.9, Gemini-3-Pro 80.8 / 81.9.

† Our own clip-level protocol on the TurnBench development conversations (the official test labels are withheld); not
comparable with the official leaderboard.

Unlike a dedicated detector, it stays a general decision engine: the same forward pass answers any other closed-set
question (intent, emotion, gender, barge-in, …) with no decoding, about 0.24 s for ten decisions on one H200.

## Starting point

DuplexJev reads closed-set decisions from speech as a single-token distribution of an LLM (encoder → trained
connector → LLM). The released DuplexJev-32B / 4B were trained for content, gender and emotion, with only the
Easy-Turn training split for turn-taking. Zero-shot on CoDeTT they had one dominant failure: **they almost always
react**. While the assistant is speaking they chose "stop and listen" for backchannels and noise (Maintain 14–28%);
when idle they took the turn even when the user had not finished.

## What we changed

1. **Data: 110 k turn-taking questions rewritten from public corpora** (nothing synthesised from scratch except
   truncations and noise mixes):

   | source | clips | what it teaches |
   |---|---:|---|
   | Easy-Turn training set, synthetic and real parts | 44 k | Chinese: finished / unfinished / backchannel / "wait" |
   | SmoothConv (Chinese multi-party, per-segment turn labels) | 30 k | real conversation, 1–5 turns of text history |
   | otoSpeech two-channel + DualTurn frame labels | 12.5 k | English: end of turn, pause, backchannel, interruption |
   | DuplexGen Spoken | 14 k | English human–AI dialogues with history; truncated turns as "unfinished" |
   | MUSAN noise; distant, attenuated background voices | 9 k | sounds to ignore in both states |

   60% of the questions give the **system state** and ask for an action (speaking → keep talking / stop and listen;
   idle → reply / keep waiting); 40% ask for the user's turn state. Wording differs from every evaluation script.
2. **Model: train the connector together with a rank-16 LoRA on the LLM** (q/k/v/o, 0.6% of the parameters), mixed
   with the original content-distillation and paralinguistic data. In a pilot, training the connector alone also helped
   CoDeTT but cost 9 points on Easy-Turn and 4 on VoiceBench: one small module had to carry both alignment and
   turn-taking. The LoRA gives the LLM room to use what it already hears.
3. **One detail that matters:** the content-distillation teacher (the LLM reading the transcript) must run **with the
   adapter disabled**. Otherwise the LoRA changes teacher and student together, the KL loss collapses toward zero and
   content understanding is destroyed.
4. LoRA merged into the weights for release: same size, same vLLM plugin, same prompts.

Recipe: batch 16, lr 1e-4, 8,000 steps; 4B on one H200 in 85 minutes, 32B on four H200s in about 2 hours.

## Ablations (4B)

| variant | Easy-Turn | CoDeTT zh / en | TurnBench-dev | qa100 |
|---|---:|---:|---:|---:|
| data weight 847, 4,000 steps | 92.0 | 66.7 / 69.5 | 86.6 | 73 |
| data weight 2,000 | 91.5 | 65.8 / 68.8 | 86.5 | 76 |
| LoRA rank 64 | 91.8 | 66.5 / 68.1 | 85.4 | 77 |
| 8,000 steps (released) | 92.5 | 67.0 / 68.6 | 87.3 | 74 |

The recipe is insensitive to data weight, rank and length within the noise (±4 on qa100, ±1.5 on Easy-Turn).

## What is still weak

- **A pause inside a long turn** is taken as the end of the turn about half the time (TurnBench-dev "hold" 54–57%).
- **Staying silent when idle** (the user has not finished, cancels, or talks to someone else): CoDeTT Dismiss 37% (zh) /
  46% (en) for 32B-Turn.
- CoDeTT protocol difference: we give only the current utterance as audio; the official protocol also plays earlier
  user turns. Gemini-3-Pro (~81) remains well ahead.

## Limitations

One seed per configuration. Easy-Turn is in-domain for our models (as for the Easy-Turn detector itself). TurnBench
numbers use our own clip protocol. Latency numbers of other systems come from their papers and hardware.

Models: [DuplexJev-32B-Turn](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) ·
[DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn). Paper: [arXiv:2610.02638](https://arxiv.org/abs/2610.02638).
