# 连接器 checkpoint 与 PyTorch 包

*[English](connectors.md) · [返回 README](../README_CN.md)*

大多数用户请直接用**完整模型**（编码器 + 连接器 + 大模型在一个仓库里，用 vLLM 部署）：
🤗 [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B) 或
🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B)，见 [README](../README_CN.md#3-模型)。

本页面向研究用途：完整模型和论文背后的**连接器 checkpoint**（覆盖不同编码器、大模型和连接器类型），以及加载它们的
`duplexjev` PyTorch 包。

每个 checkpoint 都是**某一对编码器和大模型专用的连接器**：只包含训练好的连接器，冻结的编码器和大模型由
`Decider.from_pretrained("adventists-ai/<仓库名>")` 自动下载。它不能换用其他编码器或大模型，同系列的其他尺寸也不行。
`DuplexJev-4B` 的权重就是 `DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B`；`DuplexJev-32B`（Qwen3-VL-32B）只以完整模型形式发布。

## 排行榜

总分 =（主语言 + 副语言）/ 2。主语言 = qa100、ZJU-ML、Easy-Turn 三项平均；副语言 = 性别、情绪两项平均（均为百分制）。每个 编码器 × LLM × 连接器 组合只列总分最高的一个模型。

| 排名 | 编码器 | LLM | 连接器 | 模型 | 总分 | 主语言 | 副语言 |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Qwen3-ASR-0.6B | Qwen3-32B | A · 交叉注意力 | 🤗 [DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B) | **80.6** | 71.2 | 90.0 |
| 2 | Qwen3-ASR-0.6B | Qwen3-4B | B · 原生 | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B) | **75.9** | 61.1 | 90.7 |
| 3 | Qwen3-ASR-0.6B | Qwen3-32B | B · 原生 | 🤗 [DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | **72.7** | 76.7 | 68.7 |
| 4 | MOSS-Transcribe-Diarize | Qwen3-4B-2507 | B · 原生 | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Qwen3-4B-2507) | **70.0** | 49.6 | 90.5 |
| 5 | MOSS-Transcribe-Diarize | Falcon-H1-3B | B · 原生 | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-3B) | **67.7** | 43.8 | 91.6 |
| 6 | SenseVoice-Small | Qwen3-4B | B · 原生 | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B) | **66.5** | 50.0 | 82.9 |
| 7 | Qwen3-ASR-0.6B | Qwen3-1.7B | B · 原生 | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B) | **66.3** | 45.8 | 86.9 |
| 8 | MOSS-Transcribe-Diarize | SmolLM3-3B | B · 原生 | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-SmolLM3-3B) | **65.6** | 40.0 | 91.2 |
| 9 | Whisper-small | Falcon-H1-3B | B · 原生 | 🤗 [DuplexJev-B-Para-Whisper-small-Falcon-H1-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Falcon-H1-3B) | **65.3** | 45.7 | 85.0 |
| 10 | MOSS-Transcribe-Diarize | Falcon-H1-1.5B | B · 原生 | 🤗 [DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-MOSS-Transcribe-Falcon-H1-1.5B) | **63.2** | 37.1 | 89.3 |
| 11 | Whisper-small | Qwen3-4B-2507 | B · 原生 | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-4B-2507) | **63.1** | 45.7 | 80.6 |
| 12 | Qwen3-ASR-0.6B | Qwen3-0.6B | B · 原生 | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B) | **62.8** | 36.2 | 89.3 |
| 13 | Whisper-small | SmolLM3-3B | B · 原生 | 🤗 [DuplexJev-B-Para-Whisper-small-SmolLM3-3B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-SmolLM3-3B) | **62.6** | 39.5 | 85.7 |
| 14 | Whisper-small | Falcon-H1-1.5B | B · 原生 | 🤗 [DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Falcon-H1-1.5B) | **56.8** | 36.8 | 76.7 |
| 15 | SenseVoice-Small | Qwen3-0.6B | B · 原生 | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B) | **55.2** | 33.8 | 76.7 |
| 16 | SenseVoice-Small | Qwen3-1.7B | B · 原生 | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B) | **54.9** | 37.0 | 72.7 |
| 17 | Whisper-small | Qwen3-1.7B | B · 原生 | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-1.7B) | **54.2** | 37.9 | 70.6 |

#### 分项成绩

| 模型 | qa100 | ZJU-ML | Easy-Turn | 性别 | 情绪 | 总参数 | CPU（8 线程） | 许可 |
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

分数为论文口径的百分比（单 token 读出，单张 H200），见[评测基准](../README_CN.md#6-评测基准)。CPU 列为一次判断（10 个问题、4.5 秒音频）
在 8 个 CPU 线程上的耗时，fp32、未量化；在单张 H200 上所有模型都在约 0.05–0.2 秒内完成。

**选哪个连接器。** 部署服务请优先用上面的完整模型。
- *排行榜上最好的连接器：* `DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B`（一张 80 GB 显卡）。
- *端侧 / 小显卡：* `DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B`；纯 CPU 用 `…-Qwen3-0.6B`，1 秒以内。
- *只做内容判断、需要商用：* 用不带 `-Para` 的内容版（Apache-2.0；性别和情绪为随机水平），例如
  `DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B`（qa100 90 分）。
- *Falcon-H1 系列* 需要 GPU（其中的 Mamba 层在 CPU 上没有快速实现）。

所有训练过的版本（只训内容、只训性别、只训情绪、各种编码器）都保留在 🤗 [Hugging Face](https://huggingface.co/adventists-ai)；
论文中 Qwen3-32B 各版本的完整对比见 [paper_results.md](paper_results.md)。

## PyTorch 包

这个包能运行所有连接器 checkpoint，并把同一段或多段音频的多个问题打包进一次前向计算、共享前缀。

```bash
pip install "duplexjev[speech]>=0.2.2"      # 需要 HTTP 服务再加 [all]
```

**一段音频，多组选项：**

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B", device="auto")
groups = [Question("turn", "用户说完了吗？", ["说完了", "还没说完"], lang="zh"),
          Question("filler", "先说哪句垫话？", ["好的——", "请稍等——", "（不说话）"], lang="zh"),
          Question("gender", "说话人的性别是？", ["男性", "女性"], lang="zh"),
          Question("emotion", "说话人当时的情绪状态是？", ["中性", "高兴", "生气", "伤心"], lang="zh")]

d.decide("call_018.wav", groups, lang="zh")
# {'turn': {'answer': '说完了', 'confidence': 0.97, 'probs': {...}}, 'filler': {...}, 'gender': {...}, 'emotion': {...}}
```

用音频的语言提问：连接器训练时问题和选项的语言与音频一致（中文语音用中文问题和选项）。每个模型卡都列出了训练时用的问法。

**多段音频，多组选项** —— 每组选项注明针对哪段音频，全部在一次批量计算中完成：

```python
d.decide_batch(
    {"car1": "a.wav", "car2": "b.wav", "car3": "c.wav"},
    [Question("turn", "用户说完了吗？", ["说完了", "还没说完"], lang="zh", audio="*"),
     Question("gender", "说话人的性别是？", ["男性", "女性"], lang="zh", audio=["car2", "car3"]),
     Question("human", "用户需要转人工吗？", ["需要", "不需要"], lang="zh", audio="car1")],
    context={"car1": "司机两次要求给家里打电话。"}, lang="zh")
```

**批量服务** —— 同一个时间片内到达的请求在一次计算里一起回答：

```bash
duplexjev serve --model adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B --tick-ms 160
# POST /v1/decide        {"audio_b64": ..., "questions": [...]}
# POST /v1/decide_batch  {"audios": {"car1": ..., "car2": ...}, "questions": [{..., "audio": "car1"}]}
```

使用本地模型副本、多卡切分、其他 Ultravox 格式模型，以及包的各项保证：见 [package.md](package.md)。
