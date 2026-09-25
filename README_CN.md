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
  📝 <a href="#9-引用">引用</a>
</p>

---

## 1. 简介

**DuplexJev** 把一个冻结的语音大模型变成面向全双工语音智能体的**类型化判断引擎**（产品名 **Speech-to-Decision**）。
现成 ASR 编码器的隐状态经连接器送入冻结的大模型；运行时声明的每个问题——*用户说完了吗？先说哪句垫话？说话的是谁？*——
都读成**单个 token 上的闭集概率分布**：不做 ASR 解码，也不做文本解码。同一通电话的多个问题、乃至多通电话的问题，
可以在同一次前向计算里一起完成。

论文：*Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen LLM Hear Beyond the
Transcript*（已投 ICASSP 2027）。

```
音频 ──► 冻结的 ASR 编码器（Qwen3-ASR-0.6B，12.5 Hz）
            ├─ B：最后一层 h18 ───────────────────────────────┐
            └─ A：交叉注意力融合  Q=h18, K=h14, V=h9 ─────────┤ （零初始化，残差）
                                                              ▼
                            投影器（每 2 帧拼接 → 6.25 token/秒，MLP → d_LLM）
                                                              ▼
状态 + N 个类型化问题 ──► 冻结的大模型（Qwen3-32B），一次前向
                                                              ▼
            p(A|q1) … p(D|q1),  …,  p(A|qN) … p(D|qN)      —— 0 步解码
```

- **类型化单 token 读出。** 每个问题的选项用打乱的字母标注，答案取下一个 token 在这些字母上的 softmax。输出一定合法，`max p` 可作置信度。
- **不只听懂字，还听得出人。** 用转写蒸馏训练的语音大模型听不出性别和情绪，因为它的"老师"只读转写文字、听不见声音。
  我们改为直接监督那一个答案 token（对选项字母做交叉熵），性别和情绪识别率都达到 90%，内容理解只降 1 分。
- **前缀共享。** 模板、对话上下文和音频只编码一次；所有问题后缀打包进同一行，用块对角 4-D 掩码隔开，位置编号从前缀长度重新开始。
  KV 缓存只占 `P + ΣL_i` 个位置，而不是 `N(P + L)`；答案与逐个运行一致（仅 bf16 数值误差）。
- **模块化。** 任意 ASR 编码器 + 小型可训练连接器（1780 万参数；A 版融合模块另加 320 万）+ 任意接受嵌入输入的冻结大模型。

## 2. 最新动态

- **2026-09-25** —— 发布 **14 个小模型连接器**（Qwen3-ASR 编码器配 Qwen3-0.6B / 1.7B / 4B，Whisper-small 配 Qwen3-1.7B），以及 6 个 SenseVoice-Small 编码器连接器（配 Qwen3-0.6B / 1.7B / 4B），各有内容版和性别 + 情绪版，见[小模型连接器](#小模型连接器端侧规模)。
- **2026-09-24** —— 在 🤗 [Hugging Face](https://huggingface.co/adventists-ai) 发布 **7 个连接器权重和 2 个编码器仓库**；发布 [`duplexjev` 0.2.1](https://pypi.org/project/duplexjev/0.2.1/)（音频改为在末尾补静音：在开头补会让情绪准确率最多低 6 分；`text_model` / `audio_model` 离线可用）。
- **2026-09** —— 论文投稿 ICASSP 2027；发布 `duplexjev` 包、研究代码、延迟基准和项目主页。
- **2026-10-10（计划）** —— Speech-to-Decision 商用 API 上线。
- **2026-10-15（计划）** —— 开源全双工 Jev 对话管线。

## 3. 模型下载

每个权重都是**专门连接一对模型的连接器**：只包含训练好的投影器（A 版另含融合模块），把冻结的 ASR 编码器接到冻结的大模型上。
换成别的编码器或大模型（包括同系列的其他尺寸）都不能用。`Decider.from_pretrained("adventists-ai/<仓库名>")` 会自动下载编码器和大模型。

所有连接器都使用冻结的 **Qwen3-ASR-0.6B** 编码器和冻结的 **[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)** 大模型。

| 仓库 | 连接器 | 训练内容 | 可训练参数 | 许可 |
|---|---|---|---:|---|
| 🤗 [DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B) | A · 交叉注意力融合 | 内容（R2） | 2110 万 | Apache-2.0 |
| 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B) | B · 最后一层 | 内容（R2），语音问答最好 | 1780 万 | Apache-2.0 |
| 🤗 [DuplexJev-A-Gender-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Gender-Qwen3-ASR-0.6B-Qwen3-32B) | A | + 说话人性别 | 2110 万 | Apache-2.0 |
| 🤗 [DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B) | B | + 说话人性别 | 1780 万 | Apache-2.0 |
| 🤗 [DuplexJev-A-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | A | + 情绪（四类） | 2110 万 | CC BY-NC 4.0 |
| 🤗 [DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-B-Emotion-Qwen3-ASR-0.6B-Qwen3-32B) | B | + 情绪（四类） | 1780 万 | CC BY-NC 4.0 |
| 🤗 [DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B](https://huggingface.co/adventists-ai/DuplexJev-A-Para-Qwen3-ASR-0.6B-Qwen3-32B) | A | 性别 + 情绪 + 内容，混合目标（副语言最好） | 2110 万 | CC BY-NC 4.0 |

编码器（自动下载）：🤗 [Qwen3-ASR-0.6B-Encoder](https://huggingface.co/adventists-ai/Qwen3-ASR-0.6B-Encoder)（B 版用）和
🤗 [Qwen3-ASR-0.6B-Encoder-XAttn](https://huggingface.co/adventists-ai/Qwen3-ASR-0.6B-Encoder-XAttn)（A 版用；权重相同，另含融合模块代码）。
两者都是 Qwen3-ASR-0.6B 原样的音频编码器，Apache-2.0。

情绪连接器用 ESD 训练，而 ESD 仅限研究使用，所以这三个权重仅供非商业研究使用。

### 小模型连接器（端侧规模）

同一套配方（R1 → R2 → MIX-KD，最后一层连接器 B）用在小的冻结大模型和三种编码器上。下表成绩为 MIX-KD 连接器在论文评测口径下的结果（%）；
CPU 为一次判断事件（10 个问题、4.5 秒音频）的耗时，fp32、未量化（一张 H200 GPU 上为 40–80 毫秒；SenseVoice 组合因前端逐段计算为 130–150 毫秒）。小模型即使读转写，
知识类问答也不强，适合做短判断和听说话人。模型卡里另列了 `duplexjev` 包的实测数字。

| 编码器 | 大模型 | 内容连接器（Apache-2.0） | + 性别与情绪，MIX-KD（CC BY-NC 4.0） | 可训练 | 总参数 | qa100（语音 / 读转写） | 性别 | 情绪 | CPU 8 线程 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR-0.6B | Qwen3-0.6B | 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-0.6B) | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-0.6B) | 9.4 M | 0.8 B | 38 / 45 | 89.4 | 89.2 | 0.9 s |
| Qwen3-ASR-0.6B | Qwen3-1.7B | 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-1.7B) | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-1.7B) | 11.5 M | 1.9 B | 59 / 66 | 84.8 | 89.1 | 2.1 s |
| Whisper-small | Qwen3-1.7B | 🤗 [DuplexJev-B-Whisper-small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Whisper-small-Qwen3-1.7B) | 🤗 [DuplexJev-B-Para-Whisper-small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Whisper-small-Qwen3-1.7B) | 29.4 M | 1.8 B | 48 / 66 | 78.8 | 62.4 | 2.4 s |
| Qwen3-ASR-0.6B | Qwen3-4B | 🤗 [DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-4B) | 🤗 [DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-Qwen3-ASR-0.6B-Qwen3-4B) | 12.6 M | 4.2 B | 72 / 82 | 89.4 | 91.9 | 5.4 s |
| SenseVoice-Small | Qwen3-0.6B | 🤗 [DuplexJev-B-SenseVoice-Small-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-SenseVoice-Small-Qwen3-0.6B) | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-0.6B) | 8.4 M | 0.8 B | 34 / 46 | 67.5 | 85.8 | 1.0 s |
| SenseVoice-Small | Qwen3-1.7B | 🤗 [DuplexJev-B-SenseVoice-Small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-SenseVoice-Small-Qwen3-1.7B) | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-1.7B) | 10.5 M | 2.0 B | 47 / 66 | 66.9 | 78.4 | 2.4 s |
| SenseVoice-Small | Qwen3-4B | 🤗 [DuplexJev-B-SenseVoice-Small-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-SenseVoice-Small-Qwen3-4B) | 🤗 [DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B](https://huggingface.co/adventists-ai/DuplexJev-B-Para-SenseVoice-Small-Qwen3-4B) | 11.5 M | 4.3 B | 61 / 82 | 76.8 | 89.0 | 6.1 s |

用 Qwen3-ASR 编码器时，连 Qwen3-0.6B 听性别和情绪的能力也和 Qwen3-32B 相当（89 / 89 对 90 / 90）。
SenseVoice-Small 保留了情绪信息（最高 89），但说话人性别弱一些（67–77）；Whisper-small 两者都更弱。SenseVoice 编码器（[`adventists-ai/SenseVoice-Small-Encoder`](https://huggingface.co/adventists-ai/SenseVoice-Small-Encoder)，不依赖 FunASR 的 SenseVoiceSmall 移植版）按 FunASR 模型许可再分发，需另装 `torchaudio`。

## 4. 快速上手

```bash
pip install "duplexjev[speech]>=0.2.1"      # 需要 HTTP 服务时装 [all]
```

**一段音频，多个问题**（默认用法）：

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Gender-Qwen3-ASR-0.6B-Qwen3-32B", device="auto")

# 中文语音：用中文提问、中文选项
d.decide("call_018.wav", [
    Question("turn", "用户说完了吗？", ["说完了", "还没说完"], lang="zh"),
    Question("gender", "说话人的性别是？", ["男性", "女性"], lang="zh"),
], lang="zh")
# {'turn': {'answer': '说完了', 'confidence': 0.97, 'probs': {...}}, 'gender': {...}}

# 英文语音
d.decide("call_017.wav", [Question("gender", "What is the perceived gender of the speaker?", ["female", "male"])])
```

请用音频的语言提问：连接器训练时用的是与语言匹配的提示（中文语音配中文问题和选项）。每个模型卡里列出了训练时的问法。

**多段音频，多个问题**（进阶）：每组问题用 `audio=<id>`、id 列表或 `"*"`（全部音频）指明问的是哪段音频，全部在一次批量计算里完成。

```python
d.decide_batch(
    {"car1": "a.wav", "car2": "b.wav", "car3": "c.wav"},
    [Question("turn", "Has the user finished the turn?", ["finished", "not finished"], audio="*"),
     Question("gender", "What is the perceived gender of the speaker?", ["female", "male"], audio=["car2", "car3"]),
     Question("human", "Does the user need a human agent?", ["yes", "no"], audio="car1")],
    context={"car1": "Driver asked to call home twice."})
```

**批量服务：** 同一个节拍内到达的请求在一次计算里回答。

```bash
duplexjev serve --model adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B --tick-ms 160
# POST /v1/decide        {"audio_b64": ..., "questions": [...]}
# POST /v1/decide_batch  {"audios": {"car1": ..., "car2": ...}, "questions": [{..., "audio": "car1"}]}
```

命令行：`duplexjev decide --model M call.wav --q "turn|Has the user finished?|finished,not finished"`。

### 选择模型

`Decider.from_pretrained(...)`（命令行 `--model`）接收 Hugging Face 仓库名或本地路径，指向一个 **Ultravox 格式的语音权重**：
ASR 编码器 + 为它训练的连接器，并写明训练时用的冻结大模型。所以只需要选一个名字，编码器随之而来，大模型按配置里的 id 下载。

```python
d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B")      # 全部自动下载
d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B",
                            text_model="/data/models/Qwen3-32B")                        # 用本地的大模型副本
d = Decider.from_pretrained("/data/models/my-checkpoint", device="auto")                # 本地路径，多卡切分
```

| 权重 | 编码器（内含） | 冻结大模型 | 状态 |
|---|---|---|---|
| `adventists-ai/DuplexJev-{A,B}[-Gender,-Emotion,-Para]-Qwen3-ASR-0.6B-Qwen3-32B` | Qwen3-ASR-0.6B（A 版另含融合） | Qwen3-32B | 已发布，0.2.1 实测 |
| `fixie-ai/ultravox-v0_6-qwen-3-32b` | Whisper-large-v3-turbo | Qwen3-32B | 已测（qa100 0.89） |
| `fixie-ai/ultravox-v0_6-gemma-3-27b` | Whisper-large-v3-turbo | Gemma-3-27B（需接受许可） | 格式相同 |

- **ASR 编码器不能单独换。** 连接器只配它训练时的编码器。`audio_model=` / `text_model=` 只用来指向*同一个*编码器或大模型的本地副本
  （离线机器、共享模型目录）；换成别的模型能加载，但答案没有意义。
- 只用编码器的隐状态，不做任何转写。
- `device="cuda:0"`（默认：第一张卡，bf16）或 `device="auto"` 把大模型切到多张卡上。32B 大模型 bf16 需要一张 80 GB 显卡。
- 支持 Hugging Face 标准环境变量：`HF_TOKEN`、`HF_HOME`、`HF_ENDPOINT`（镜像）、`HF_HUB_OFFLINE=1`（只用本地缓存）。

包的保证（`tests/` 中的单元测试，并在 Qwen3-32B 上核对）：

- **0 步解码。** 每个答案都是下一个 token 在选项字母上的 softmax；选项用打乱的字母标注（`n_perm` 可对多种顺序取平均）。
- **精确前缀共享**（默认 `mode="packed"`）：一段音频的上下文和音频只编码一次，所有问题打包进一行。fp32 下与逐题运行相差不超过 2e-5；bf16 下平均差 0.001–0.002，100 个答案里最多变 1 个。
- **批次无关。** 一条音频的答案不受同批其他音频影响（fp32 最大差 1e-5）。为此每段音频要对齐到整数个音频 token，包会在**末尾**补静音。
- **准确率核对。** 通过本包（0.2.1，默认提示）测试，每个已发布连接器与论文数字相差 0–5 分（见各模型卡），例如 A-Gender 性别 89.2（论文 89.4），A-Para 情绪 88.2（论文 90.0）。
- **速度（当前）。** 纯 PyTorch、一张 A800、bf16：64 通电话各 8 个问题共 9 秒（每通约 140 毫秒）。论文中的延迟数字基于 vLLM 引擎；包的 vLLM 后端在计划中。

Ultravox 格式的语音权重需要 transformers 4.51–4.55，`speech` 选项会自动安装。

更多示例见 [`examples/`](examples)。

## 5. 评测结果

**准确率**（%，论文表 2–3；单 token 读出，一张 H200）：

| 连接器 | qa100 | ZJU-ML | Easy-Turn | 性别（800 条真人） | ZJU 性别 | 情绪（800 条，四类） |
|---|---:|---:|---:|---:|---:|---:|
| 直接读标准转写 | 91 | – | – | – | – | – |
| A（R2） | 83 | 77 | 77.1 | 55 | – | 28 |
| B（R2） | **90** | 79 | 76.1 | 54 | – | 27 |
| A-Gender | 86 | 79 | – | 89.4 | 73 | – |
| B-Gender | 87 | 81 | – | 87.9 | 60 | – |
| A-Emotion | 84 | 71 | – | – | – | 71.8 |
| B-Emotion | 89 | 69 | – | – | – | 85.5 |
| A-Para（混合目标） | 82 | 61 | – | **89.9** | **90** | **90.0** |

qa100：100 道中英文语音选择题（题干为合成语音）。ZJU-ML：浙大音频基准 v2.0.0 的主语言部分，其中一半是真人录音的常识题；
情绪数据会让它下降 6–17 分，几乎全部落在这一半上。性别：800 条真人语音（AISHELL-1、Common Voice、LibriSpeech），瞎猜为 50%。
情绪：800 条表演语音（ESD、CREMA-D），中性 / 高兴 / 生气 / 伤心，瞎猜为 25%。

**延迟与容量**（一张 H200，bf16，Qwen3-32B；每个事件 10 个判断，30 条 2–12 秒真人语音，两种方法用同一个 vLLM 引擎）：

| 上下文 | 方法 | 解码步数 | 单个事件 | 0.25 秒内事件/秒 | 0.5 秒 | 2 秒 |
|---|---|---:|---:|---:|---:|---:|
| 无 | ASR → 大模型生成 JSON | 12.5 + 86 | 1,567 ms（另加 ASR 411 ms） | 0 | 0 | 16.9 |
| 无 | 单 token 读出 | 0 | 92 ms | 17.3 | 20.4 | 33.5 |
| 1.5k token | ASR → 大模型生成 JSON | 12.5 + 87 | 1,598 ms（另加 ASR） | 0 | 0 | 8.4 |
| 1.5k token | 单 token 读出 | 0 | 144 ms | 9.7 | 12.0 | 18.4 |

一台 8×H200 服务器（每卡一个引擎）约 0.1 秒回答 8 个事件（80 个判断）。

**前缀共享**（打包一行 vs 逐个调用）：1.5k token 上下文下 100 个问题便宜 7 倍，5k token 上下文下 50 个问题便宜 20 倍，40/40 条答案一致。

| 基准 | 任务 | 链接 |
|---|---|---|
| qa100 | 语音选择题，合成语音（中 + 英） | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| 浙大音频基准 · `gender` | 说话人性别（中 + 英，真人 + 合成） | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| 浙大音频基准 · `main_language` | 语音选择题，真人 + 合成 | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |

评测脚本：[`evaluation/`](evaluation)。

## 6. 训练

冻结 Qwen3-ASR-0.6B 编码器和 Qwen3-32B，只训练连接器，基于打过补丁的 [Ultravox](https://github.com/fixie-ai/ultravox)。

1. **内容（R1–R2）。** 转写蒸馏（让模型在听音频时的输出分布逼近读转写时的分布，token 级 KL），数据为 Ultravox v0.6 配方
   （WenetSpeech、GigaSpeech、Common Voice、CoVoST 2、People's Speech、LibriSpeech、MLS、MUSAN），切成 100 个互不重叠、各自保持官方配比的数据包；R1、R2 各用一个包。
2. **判断。** 答案 token 监督：对读出位置的选项字母做交叉熵。性别：AISHELL-1 和 LibriSpeech 的真人语音，标签来自语料自带的说话人信息。情绪：ESD 和 CREMA-D。
3. **混合目标（A-Para）。** 内容、性别、情绪样本一起训练：内容样本用蒸馏，判断样本用答案 token 交叉熵。

配方、数据链接和脚本见 [`training/`](training)。情绪语料不随仓库分发。

## 7. 仓库结构

| 路径 | 内容 |
|---|---|
| [`duplexjev/`](duplexjev) | 可安装的包：`Decider`、`Question`、命令行和节拍批量服务 |
| [`duplexjev/research/`](duplexjev/research) | 论文代码：读出、问题约定、带融合的编码器 |
| [`tests/`](tests) | 等价性与批次无关性测试 |
| [`examples/`](examples) | 快速上手、节拍批量计时、服务客户端 |
| [`evaluation/`](evaluation) | 基准测试脚本及各基准的获取方式 |
| [`training/`](training) | Ultravox 配置与补丁、数据配方、100 包切分、续写生成 |
| [`benchmarks/`](benchmarks) | 延迟、SLO 容量与前缀共享实验 |
| [`docs/`](docs) | 项目主页 |

论文代码里仍有我们集群上的路径，见 [docs/PATHS.md](docs/PATHS.md)。

## 8. 许可

代码：Apache-2.0（[LICENSE](LICENSE)）。模型权重：Apache-2.0，但用情绪数据训练的连接器（A-Emotion、B-Emotion、A-Para 以及所有 `-Para-` 小模型连接器）为 CC BY-NC 4.0，
因为 ESD 仅限研究使用。qa100：CC-BY-4.0。部分训练语料（如 WenetSpeech、CoVoST 2）仅限非商业使用，商用前请自行核对。
第三方模型和数据集保留各自的许可，见 [NOTICE](NOTICE)。

## 9. 引用

```bibtex
@inproceedings{jin2027duplexjev,
  title     = {Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen {LLM} Hear Beyond the Transcript},
  author    = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Song, Haigang and Pang, Zhikun and Zhang, Xiaowen},
  booktitle = {Submitted to IEEE ICASSP},
  year      = {2027}
}
```

## 10. 致谢

基于 [Ultravox](https://github.com/fixie-ai/ultravox)、[Qwen3](https://github.com/QwenLM/Qwen3) 和 Qwen3-ASR 构建。
感谢浙大团队的[音频基准](https://github.com/Vsky-morigen/audio-gender-benchmark)。Claude（Anthropic）协助编写了代码。
问题与反馈：[GitHub Issues](https://github.com/adventists-ai/duplexjev/issues)。
