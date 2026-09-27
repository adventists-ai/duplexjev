<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.svg">
    <img src="docs/assets/banner_light.svg" alt="DuplexJev —— 不解码的批量语音判断" width="880">
  </picture>
</p>

<p align="center">
  <a href="README.md">English</a> &nbsp;|&nbsp; <b>中文</b>
</p>

<p align="center">
  🌐 <a href="https://adventists-ai.github.io/duplexjev/#zh">项目主页</a> &nbsp;|&nbsp;
  📄 论文（arXiv，即将发布） &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/adventists-ai">模型权重</a> &nbsp;|&nbsp;
  📦 <a href="https://pypi.org/project/duplexjev/">PyPI</a> &nbsp;|&nbsp;
  📊 <a href="https://huggingface.co/datasets/adventists-ai/qa100">qa100</a> &nbsp;|&nbsp;
  📝 <a href="#10-引用">引用</a>
</p>

---

## 1. 简介

**DuplexJev** 把一个冻结的语音大模型变成面向全双工语音智能体的**类型化判断引擎**（产品名 **Speech-to-Decision**）。
现成 ASR 编码器的隐状态经连接器送入冻结的大模型；运行时声明的每个问题——*用户说完了吗？先说哪句垫话？说话的是谁？*——
都读成**单个 token 上的闭集概率分布**：不做 ASR 解码，也不做文本解码。同一通电话的多个问题、乃至多通电话的问题，
可以在同一次前向计算里一起完成。

```
音频 ──► 冻结的 ASR 编码器（Qwen3-ASR、MOSS-Transcribe、Whisper、SenseVoice 等）
            ├─ B：最后一层 ───────────────────────────────────┐
            └─ A：三层的交叉注意力融合 ───────────────────────┤ （零初始化，残差）
                                                              ▼
                          连接器（帧拼接 → 6.25 token/秒，MLP → d_LLM）
                                                              ▼
状态 + N 个类型化问题 ──► 冻结的大模型（0.6B – 32B），一次前向
                                                              ▼
            p(A|q1) … p(D|q1),  …,  p(A|qN) … p(D|qN)      —— 0 步解码
```

- **类型化单 token 读出。** 每个问题的选项用打乱的字母标注，答案取下一个 token 在这些字母上的 softmax。输出一定合法，`max p` 可作置信度。
- **不只听懂字，还听得出人。** 只用转写蒸馏训练的语音大模型听不出性别和情绪；直接监督那一个答案 token，两者都能到 90% 左右，内容理解只变动约 1 分。
- **前缀共享。** 上下文和音频只编码一次，所有问题后缀打包进同一行、用块对角掩码隔开；答案与逐个运行一致（仅 bf16 数值误差）。

## 2. 最新动态

- **2026-09-27** —— 新增 **16 个连接器**：MOSS-Transcribe-Diarize 和 Whisper-small 两种编码器，分别配 Qwen3-4B-2507、
  SmolLM3-3B、Falcon-H1-1.5B、Falcon-H1-3B；发布 [MOSS 编码器](https://huggingface.co/adventists-ai/MOSS-Transcribe-Diarize-Whisper-Encoder)
  和 [`duplexjev` 0.2.2](https://pypi.org/project/duplexjev/0.2.2/)（遇到 Falcon-H1 等带循环层的大模型自动改用 `mode="batch"`）。
  Qwen3-ASR-0.6B → **Qwen2.5-72B** 的连接器（A、B 两版）正在训练，结果稍后公布。
- **2026-09-25** —— 20 个小模型连接器：Qwen3-0.6B / 1.7B / 4B 配 Qwen3-ASR 和 SenseVoice-Small 编码器（另有 Whisper-small 配 Qwen3-1.7B）。
- **2026-09-24** —— 7 个 Qwen3-32B 连接器和 [`duplexjev` 0.2.1](https://pypi.org/project/duplexjev/0.2.1/)；论文投稿 ICASSP 2027。

即将推出：Speech-to-Decision 商用 API（2026-10-10）、开源全双工 Jev 对话流水线（2026-10-15）。

## 3. 模型

每个 checkpoint 都是**某一对编码器和大模型专用的连接器**：只包含训练好的连接器，冻结的编码器和大模型由
`Decider.from_pretrained("adventists-ai/<仓库名>")` 自动下载。它不能换用其他编码器或大模型，同系列的其他尺寸也不行。

### 排行榜

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

分数为论文口径的百分比（单 token 读出，单张 H200），见[评测基准](#6-评测基准)。CPU 列为一次判断（10 个问题、4.5 秒音频）
在 8 个 CPU 线程上的耗时，fp32、未量化；在单张 H200 上所有模型都在约 0.05–0.2 秒内完成。

**怎么选。**
- *服务器端，综合最好：* `DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B`（一张 80 GB 显卡）。
- *端侧 / 小显卡：* `DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B`；纯 CPU 用 `…-Qwen3-0.6B`，1 秒以内。
- *只做内容判断、需要商用：* 用不带 `-Para` 的内容版（Apache-2.0；性别和情绪为随机水平），例如
  `DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B`（qa100 90 分）。
- *Falcon-H1 系列* 需要 GPU（其中的 Mamba 层在 CPU 上没有快速实现）。

所有训练过的版本（只训内容、只训性别、只训情绪、各种编码器）都保留在 🤗 [Hugging Face](https://huggingface.co/adventists-ai)；
论文中 Qwen3-32B 各版本的完整对比见 [docs/paper_results.md](docs/paper_results.md)。

## 4. 快速上手

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

使用本地模型副本、多卡切分、其他 Ultravox 格式模型，以及包的各项保证：见 [docs/package.md](docs/package.md)。

## 5. 性能

同一通电话的 10 个判断，Qwen3-32B、单张 H200（vLLM，bf16）：单 token 读出 **92 毫秒**；常见的级联做法（先转写，
再让大模型生成 JSON）要 **1,567 毫秒，另加 ASR 411 毫秒**。在 0.25 秒的时限内，单 token 读出每秒能处理 17 个这样的事件，
级联做法一个也处理不了。完整表格见 [docs/paper_results.md](docs/paper_results.md)。

## 6. 评测基准

| 基准 | 任务 | 条数 | 链接 |
|---|---|---:|---|
| qa100 | 语音选择题，合成语音（中 + 英） | 100 | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| ZJU-ML | 浙大音频基准 v2.0.0 主语言部分：语音提问，一半为真人录音 | 100 | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| Easy-Turn | 四类话轮状态（说完 / 没说完 / 附和 / 等一下），零样本 | 800 | Easy-Turn 测试集（[arXiv:2509.23938](https://arxiv.org/abs/2509.23938)） |
| 性别 | 说话人性别，真人录音（AISHELL-1、Common Voice、LibriSpeech），中 + 英 | 800 | [`evaluation/`](evaluation) |
| 情绪 | 中性 / 高兴 / 生气 / 伤心，表演语音（ESD、CREMA-D），中 + 英 | 800 | [`evaluation/`](evaluation) |

**主语言** = qa100、ZJU-ML、Easy-Turn 三项平均；**副语言** = 性别、情绪两项平均；**总分** = 两者平均。
随机水平：四选一任务 25%，性别 50%。排行榜由 [`evaluation/build_leaderboard.py`](evaluation/build_leaderboard.py)
根据同目录下的结果文件生成。

## 7. 训练

编码器和大模型都冻结，只训练连接器；训练框架为打过补丁的 [Ultravox](https://github.com/fixie-ai/ultravox)。

1. **内容（R1–R2）。** 在 Ultravox v0.6 数据混合（WenetSpeech、GigaSpeech、Common Voice、CoVoST 2、People's Speech、
   LibriSpeech、MLS、MUSAN）上做转写蒸馏；数据切成 100 个互不重叠的包，R1、R2 各用一包。
2. **判断。** 在读出位置对单个选项字母做交叉熵。性别：AISHELL-1、LibriSpeech；情绪：ESD、CREMA-D。
3. **混合目标（`-Para`）。** 内容、性别、情绪样本一起训练：内容样本做蒸馏，判断样本做答案 token 交叉熵。

配方、数据链接和脚本：[`training/`](training)。情绪语料不再分发。

## 8. 仓库结构

| 路径 | 内容 |
|---|---|
| [`duplexjev/`](duplexjev) | 可安装的包：`Decider`、`Question`、命令行和按时间片批处理的服务 |
| [`duplexjev/research/`](duplexjev/research) | 论文代码：读出、问题约定、带融合的编码器 |
| [`tests/`](tests) | 等价性和批不变性测试 |
| [`examples/`](examples) | 快速上手、时间片批处理计时、服务客户端 |
| [`evaluation/`](evaluation) | 评测脚本、排行榜脚本和结果 |
| [`training/`](training) | Ultravox 配置和补丁、数据配方、100 包切分、编码器移植 |
| [`benchmarks/`](benchmarks) | 延迟、SLO 容量和前缀共享实验 |
| [`docs/`](docs) | 项目主页、[论文结果](docs/paper_results.md)、[包的细节](docs/package.md) |

论文代码里还留有我们集群上的路径，见 [docs/PATHS.md](docs/PATHS.md)。

## 9. 许可

代码：Apache-2.0（[LICENSE](LICENSE)）。连接器权重：Apache-2.0；用情绪数据训练的（`-Emotion-` 和 `-Para-`）为 CC BY-NC 4.0，
因为 ESD 仅限研究使用。qa100：CC-BY-4.0。冻结的编码器和大模型沿用各自的许可（Falcon-H1：TII Falcon License；
SenseVoice：FunASR Model License）。部分训练语料（如 WenetSpeech、CoVoST 2）有非商用条款。详见 [NOTICE](NOTICE)。

## 10. 引用

```bibtex
@inproceedings{jin2027duplexjev,
  title     = {Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen {LLM} Hear Beyond the Transcript},
  author    = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Song, Haigang and Pang, Zhikun and Zhang, Xiaowen},
  booktitle = {Submitted to IEEE ICASSP},
  year      = {2027}
}
```

## 11. 致谢

基于 [Ultravox](https://github.com/fixie-ai/ultravox)、[Qwen3](https://github.com/QwenLM/Qwen3)、Qwen3-ASR、
[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)、Whisper、SenseVoice、
SmolLM3 和 Falcon-H1 构建。感谢浙大团队提供[音频基准](https://github.com/Vsky-morigen/audio-gender-benchmark)。
Claude（Anthropic）协助编写代码。
问题和建议：[GitHub Issues](https://github.com/adventists-ai/duplexjev/issues)。
