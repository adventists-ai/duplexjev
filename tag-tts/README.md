# tag-tts: DuplexJev as the judge, CosyVoice3 as the voice

[🤗 Hugging Face](https://huggingface.co/adventists-ai/tag-tts) · [ModelScope](https://www.modelscope.cn/models/AdventistsAI/tag-tts) · [中文](#中文)

DuplexJev hears what a voice does besides the words: laughter, coughs, sighs, which word is stressed.
**tag-tts** turns that ear into a teacher. We used DuplexJev-32B-Para as the reward model in a reinforcement-learning
loop and trained the language model of [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
until its inline tags actually happen in real-time (bi-streaming) synthesis.

| Tag (bi-streaming, cloned voice, no instruction) | CosyVoice3 | tag-tts v1.0 | **tag-tts v1.1** |
|---|---|---|---|
| `[laughter]` – laugh heard (DuplexJev) | 62% | 100% | **100%** |
| `<laughter>…</laughter>` | 12% | 88% | **92%** |
| `[cough]` | 42% | 79% | **71%** |
| `[sigh]` (held-out) | 4.3% | 15.0% | **17.9%** |
| `<strong>word</strong>` – stress on the tagged word (held-out) | 28.6% | 35.0% | **40.7%** |
| Seed-TTS-eval zh CER / en WER / zh-hard CER (same setting) | 1.53 / 2.58 / 6.00 | 1.29 / 2.02 / 6.42 | **1.27 / 1.96 / 6.24** |

**v1.1 (2026-10-30)**: two more rounds with GRPO. For sentences where the model never stressed the tagged word,
we sampled with a short pause before the word as a hint, kept the candidates DuplexJev judged as stressed, and trained
on the original text, so the model learns to stress from `<strong>` alone.

Full numbers, audio examples and usage: see the [model card](https://huggingface.co/adventists-ai/tag-tts).

## The loop

```
sentence + tag ──► CosyVoice3 LM (policy) ──► 8 candidates, bi-streaming, cloned voices
                                                   │
           DuplexJev-32B-Para: "which word is stressed?" / "which sound besides speech?"
           + AudioSet-AST, ASR error, speaker similarity, speaking-rate checks
                                                   │
                 best vs worst in each group ──► DPO on the 0.5B LM ──► next round (3 rounds, v1.0)
                 group-normalised reward + guided positives ──► GRPO (2 rounds, v1.1)
```

Only `llm.pt` changes; flow matching and vocoder are untouched. Training uses model-generated audio only.

## Use DuplexJev as a judge for your own TTS

[`judge_tags.py`](judge_tags.py) asks DuplexJev the two questions we used as rewards and returns the probability of
the intended answer (options rotated, letter probabilities averaged):

```bash
pip install duplexjev-vllm
vllm serve adventists-ai/DuplexJev-32B-Para --served-model-name dj    # or DuplexJev-4B-Para on ~10 GB
python judge_tags.py items.jsonl scores.jsonl --base-url http://localhost:8000
```

```json
{"id": "a1", "audio": "a1.wav", "lang": "en", "kind": "stress", "options": ["lives", "hiding", "fear"], "target": 0}
{"id": "a2", "audio": "a2.wav", "lang": "zh", "kind": "nonverbal", "target": "laughter"}
```

The prompt must match DuplexJev's training format exactly. With a slightly different template the stress question
drops from ~90% to near zero, so reuse the template in the script.

[`tagbench.json`](tagbench.json) is the listening bench we used: 14 CosyVoice3 tags × sentence start / middle / end,
Chinese and English, plus untagged controls.

## Things we learned about CosyVoice3 tags

- With no style instruction the prefix must be exactly `You are a helpful assistant.<|endofprompt|>`. One extra space
  before `<|endofprompt|>` silently disables most tags.
- In bi-streaming, send a tag as one complete text chunk. A split tag (`[laugh` + `ter]`) is read out loud.
- Unknown or malformed tags (`[laugh]`, `(laughs)`, `[happy]`, `【笑声】`) are read out loud. Whitelist your tags.

## 中文

DuplexJev 能听出文字之外的东西：笑、咳嗽、叹气，重音落在哪个词上。**tag-tts** 把这对“耳朵”当成老师：
用 DuplexJev-32B-Para 做强化学习的奖励模型，训练 Fun-CosyVoice3-0.5B 的语言模型，
让它在实时逐字双流合成里真正执行文本里的标签。

- 笑声命中 62% → **100%**，成对写法 `<laughter>` 12% → **92%**，咳嗽 42% → **71%**，叹气（留出句）4.3% → **17.9%**
- 重音落在 `<strong>` 标注的词上：28.6% → **40.7%**（留出句；v1.0 为 35.0%）
- Seed-TTS-eval 中文字错率 1.53 → **1.27**，英文 2.58 → **1.96**，绕口令 6.00 → 6.24（同一流式设置）
- v1.1（10-30）：再做两轮 GRPO；对模型从来读不出重音的句子，采样时在目标词前加停顿作引导，挑出 DuplexJev 判为重读的样本，训练时换回原文本

模型：[Hugging Face](https://huggingface.co/adventists-ai/tag-tts) · [ModelScope](https://www.modelscope.cn/models/AdventistsAI/tag-tts)（Apache-2.0）。
`judge_tags.py` 可以直接拿 DuplexJev 给你自己的 TTS 打分。
