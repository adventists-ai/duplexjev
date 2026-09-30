# Connector checkpoints and the PyTorch package

*[中文](connectors_zh.md) · [back to README](../README.md)*

Most users should start with a **complete model** (encoder + connector + LLM in one repository, served with vLLM):
🤗 [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B) or
🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B); see the [README](../README.md#3-models).

This page is for research use: the **connector checkpoints** behind the complete models and the paper, across
encoders, LLMs and connector types, and the `duplexjev` PyTorch package that loads them.

Each checkpoint is a **connector for one specific encoder and LLM**: it contains only the trained connector; the frozen
encoder and LLM are fetched automatically by `Decider.from_pretrained("adventists-ai/<repo>")`. It does not work with
other encoders or LLMs, including other sizes of the same family. `DuplexJev-4B` contains the weights of
`DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B`; `DuplexJev-32B` (Qwen3-VL-32B) is published as a complete model only.

## Leaderboard

Total = (main language + paralinguistics) / 2. Main language = mean of qa100, ZJU-ML and Easy-Turn; paralinguistics = mean of gender and emotion (all %). One checkpoint per encoder × LLM × connector: the one with the best total.

| rank | encoder | LLM | connector | checkpoint | total | main language | paralinguistics |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Qwen3-ASR-0.6B | Qwen3-32B | A · cross-attention | 🤗 [DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B) | **80.6** | 71.2 | 90.0 |
| 2 | Qwen3-ASR-0.6B | Qwen3-4B | B · native | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B) | **75.9** | 61.1 | 90.7 |
| 3 | Qwen3-ASR-0.6B | Qwen3-32B | B · native | 🤗 [DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | **72.7** | 76.7 | 68.7 |
| 4 | MOSS-Transcribe-Diarize | Qwen3-4B-2507 | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507) | **70.0** | 49.6 | 90.5 |
| 5 | MOSS-Transcribe-Diarize | Falcon-H1-3B | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B) | **67.7** | 43.8 | 91.6 |
| 6 | SenseVoice-Small | Qwen3-4B | B · native | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B) | **66.5** | 50.0 | 82.9 |
| 7 | Qwen3-ASR-0.6B | Qwen3-1.7B | B · native | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B) | **66.3** | 45.8 | 86.9 |
| 8 | MOSS-Transcribe-Diarize | SmolLM3-3B | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B) | **65.6** | 40.0 | 91.2 |
| 9 | Whisper-small | Falcon-H1-3B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Falcon-H1-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Falcon-H1-3B) | **65.3** | 45.7 | 85.0 |
| 10 | MOSS-Transcribe-Diarize | Falcon-H1-1.5B | B · native | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B) | **63.2** | 37.1 | 89.3 |
| 11 | Whisper-small | Qwen3-4B-2507 | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507) | **63.1** | 45.7 | 80.6 |
| 12 | Qwen3-ASR-0.6B | Qwen3-0.6B | B · native | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B) | **62.8** | 36.2 | 89.3 |
| 13 | Whisper-small | SmolLM3-3B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-SmolLM3-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-SmolLM3-3B) | **62.6** | 39.5 | 85.7 |
| 14 | Whisper-small | Falcon-H1-1.5B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B) | **56.8** | 36.8 | 76.7 |
| 15 | SenseVoice-Small | Qwen3-0.6B | B · native | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B) | **55.2** | 33.8 | 76.7 |
| 16 | SenseVoice-Small | Qwen3-1.7B | B · native | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B) | **54.9** | 37.0 | 72.7 |
| 17 | Whisper-small | Qwen3-1.7B | B · native | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-1.7B) | **54.2** | 37.9 | 70.6 |

#### Details

| checkpoint | qa100 | ZJU-ML | Easy-Turn | gender | emotion | params | CPU (8 threads) | licence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B | 82 | 61 | 70.5 | 89.9 | 90 | 33.4 B | – | CC BY-NC 4.0 |
| DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B | 72 | 51 | 60.2 | 89.4 | 91.9 | 4.2 B | 5.4 s | CC BY-NC 4.0 |
| DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B | 89 | 69 | 72.2 | 51.8 | 85.5 | 33.4 B | – | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507 | 61 | 42 | 45.8 | 95.5 | 85.4 | 4.4 B | 6.2 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B | 51 | 37 | 43.5 | 98.1 | 85.1 | 3.5 B | 171.3 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B | 61 | 37 | 52 | 76.8 | 89 | 4.3 B | 6.1 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B | 59 | 39 | 39.4 | 84.8 | 89.1 | 1.9 B | 2.1 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B | 33 | 41 | 46.1 | 97.9 | 84.6 | 3.4 B | 5.8 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Falcon-H1-3B | 50 | 45 | 42.1 | 90.4 | 79.5 | 3.3 B | 169.7 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B | 44 | 38 | 29.2 | 94 | 84.6 | 1.9 B | 100.3 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507 | 43 | 45 | 49.2 | 91.1 | 70.1 | 4.1 B | 5.2 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B | 38 | 26 | 44.5 | 89.4 | 89.2 | 0.8 B | 0.9 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-SmolLM3-3B | 35 | 42 | 41.5 | 93.5 | 77.8 | 3.2 B | 6.5 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B | 43 | 35 | 32.4 | 83.2 | 70.2 | 1.7 B | 85.3 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B | 34 | 27 | 40.4 | 67.5 | 85.8 | 0.8 B | 1.0 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B | 47 | 30 | 34.1 | 66.9 | 78.4 | 2.0 B | 2.4 s | CC BY-NC 4.0 |
| DuplexJev-B-Para-Whisper-small-Qwen3-1.7B | 48 | 28 | 37.8 | 78.8 | 62.4 | 1.8 B | 2.4 s | CC BY-NC 4.0 |

Scores are % under the paper protocol (single-token readout, one H200); see [Benchmarks](../README.md#6-benchmarks). CPU = one
decision event (10 questions, a 4.5 s clip) on 8 CPU threads, fp32, not quantized; on one H200 every model answers in
about 0.05–0.2 s.

**Which connector to use.** For serving, prefer the complete models above.
- *Best connector on the leaderboard:* `DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B` (one 80 GB GPU).
- *Edge / small GPU:* `DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B`; on CPU, `…-Qwen3-0.6B` answers in under 1 s.
- *Content decisions only, commercial use:* the content versions without `-Para` (Apache-2.0; gender and emotion at
  chance), e.g. `DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B` (qa100 90).
- *Falcon-H1 connectors* need a GPU (no fast CPU path for their Mamba layers).

Every trained variant (content-only, gender-only, emotion-only, all encoders) stays available on
🤗 [Hugging Face](https://huggingface.co/adventists-ai); the paper's full comparison of the Qwen3-32B variants is in
[paper_results.md](paper_results.md).

## PyTorch package

The package runs every connector checkpoint and packs many questions, about one clip or many clips, into one forward
pass with a shared prefix.

```bash
pip install "duplexjev[speech]>=0.2.2"      # add [all] for the HTTP server
```

**One clip, several option groups:**

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B", device="auto")
groups = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
          Question("filler", "Which filler fits?", ["Sure —", "One moment —", "(stay silent)"]),
          Question("gender", "What is the perceived gender of the speaker?", ["female", "male"]),
          Question("emotion", "What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"])]

d.decide("call_017.wav", groups)
# {'turn': {'answer': 'finished', 'confidence': 0.97, 'probs': {...}}, 'filler': {...}, 'gender': {...}, 'emotion': {...}}

# Chinese speech: ask in Chinese, with Chinese options
d.decide("call_018.wav", [Question("gender", "说话人的性别是？", ["男性", "女性"], lang="zh")], lang="zh")
```

Ask in the language of the clip: the connectors were trained with language-matched prompts. Each model card lists the
question wordings used in training.

**Many clips, many option groups** — each group names the clip(s) it is about; everything runs in one batched pass:

```python
d.decide_batch(
    {"car1": "a.wav", "car2": "b.wav", "car3": "c.wav"},
    [Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
     Question("gender", "What is the perceived gender of the speaker?", ["female", "male"], audio=["car2", "car3"]),
     Question("human", "Does the user need a human agent?", ["yes", "no"], audio="car1")],
    context={"car1": "Driver asked to call home twice."})
```

**Batched server** — every request that arrives within one tick is answered in one pass:

```bash
duplexjev serve --model adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B --tick-ms 160
# POST /v1/decide        {"audio_b64": ..., "questions": [...]}
# POST /v1/decide_batch  {"audios": {"car1": ..., "car2": ...}, "questions": [{..., "audio": "car1"}]}
```

Local model copies, multi-GPU sharding, other Ultravox checkpoints and the package's guarantees:
[package.md](package.md).
