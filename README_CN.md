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
  🎧 <a href="https://api.adventists.cn/duplexjev/">在线体验</a> &nbsp;|&nbsp;
  🌐 <a href="https://adventists-ai.github.io/duplexjev/#zh">项目主页</a> &nbsp;|&nbsp;
  📄 <a href="https://arxiv.org/abs/2610.02638">论文（arXiv:2610.02638）</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/adventists-ai/DuplexJev-32B-Turn">DuplexJev-32B-Turn</a> · <a href="https://huggingface.co/adventists-ai/DuplexJev-4B-Turn">4B-Turn</a> &nbsp;|&nbsp;
  📦 <a href="https://pypi.org/project/duplexjev/">PyPI</a> &nbsp;|&nbsp;
  📊 <a href="https://huggingface.co/datasets/adventists-ai/qa100">qa100</a> &nbsp;|&nbsp;
  📝 <a href="#10-引用">引用</a>
</p>

---

**DuplexJev 回答全双工语音智能体在每一轮都要问的闭集问题——*说完了吗？接话还是继续听？只是附和吗？情绪、性别、意图？*——
答案是一个"直接听音频"的大模型输出的一个 token：不解码，一次前向回答多个问题。**

| [DuplexJev-32B-Turn](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) | 成绩 | 参照 |
|---|---:|---|
| 话轮，Easy-Turn 测试集 | **95.3** | 专用 Easy-Turn 检测器 96.4 |
| 话轮，CoDeTT 中 / 英（零样本） | **69.2 / 70.0** | Qwen3-Omni 70.4 / 70.9；专用话轮模型 37.9–65.4 |
| 性别 / 情绪 | **91.5 / 91.1** | 只用转写训练的语音大模型：约 55 / 28（瞎猜） |
| 口语问答，VoiceBench OBQA / MMSU（题目用语音） | **85.5 / 72.1** | 同一大模型直接读文字转写：95.4 / 79.3 |
| 速度 | **10 个判断约 0.24 秒** | 单张 H200，一次前向；Easy-Turn 检测器 1 个判断 263 ms |

模型：**DuplexJev-32B-Turn**（一张 80 GB 显卡）和 **DuplexJev-4B-Turn**（约 10 GB）（[第 3 节](#3-模型)）。与专用检测器的完整对比见[第 5 节](#5-性能)。

## 1. 简介

**DuplexJev** 把一个冻结的语音大模型变成面向全双工语音智能体的**类型化判断引擎**（产品名 **Speech-to-Decision**）。
现成 ASR 编码器的隐状态经连接器送入冻结的大模型；运行时声明的每个问题——*用户说完了吗？先说哪句垫话？说话的是谁？*——
都读成**单个 token 上的闭集概率分布**：不做 ASR 解码，也不做文本解码。同一通电话的多个问题、乃至多通电话的问题，
可以在同一次前向计算里一起完成。所有发布的模型都可以直接用 vLLM 部署（[第 3 节](#3-模型)）；推荐
**[DuplexJev-32B-Turn](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn)**，显卡小用 **[DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn)**。

```
音频 ──► Qwen3-ASR-0.6B 编码器（冻结）
                 ▼  最后一层
          连接器：每 2 帧拼接 → 6.25 token/秒，MLP → d_LLM        ← 训练的部分（Turn 版另加大模型上的 rank 16 LoRA）
                 ▼
状态 + N 个类型化问题 ──► 冻结的大模型，一次前向
                          （DuplexJev-32B：Qwen3-VL-32B 的语言模型 · DuplexJev-4B：Qwen3-4B）
                 ▼
          p(A|q1) … p(D|q1),  …,  p(A|qN) … p(D|qN)      —— 0 步解码
```

- **类型化单 token 读出。** 每个问题的选项用打乱的字母标注，答案取下一个 token 在这些字母上的 softmax。输出一定合法，`max p` 可作置信度。
- **不只听懂字，还听得出人。** 只用转写蒸馏训练的语音大模型听不出性别和情绪；直接监督那一个答案 token，两者都能到 90% 左右，内容理解只变动约 1 分。
- **前缀共享。** 上下文和音频只编码一次，所有问题后缀打包进同一行、用块对角掩码隔开；答案与逐个运行一致（仅 bf16 数值误差）。

## 2. 最新动态

- **2026-10-05** —— 📄 论文已上 arXiv：[arXiv:2610.02638](https://arxiv.org/abs/2610.02638)。🤗 [**DuplexJev-32B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn)
  和 [**DuplexJev-4B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn) 发布：话轮升级版（连接器 + rank 16 的 LoRA，
  用公开话轮数据训练）。32B-Turn：Easy-Turn 95.3，CoDeTT 零样本 69.2 / 70.0，qa100、性别、情绪都略有提升。
  [研究笔记](research/2026-10-turn-taking-lora_zh.md)。
- **2026-10-03** —— 🤗 [**DuplexJev-Gemma-31B**](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B) 发布，第一个非 Qwen 模型：
  Qwen3-ASR-0.6B 编码器 + 连接器 + Gemma-4-31B-it（合并了 LoRA）。vLLM 下 qa100 97、性别 94.4、情绪 90.6（尚未做话轮训练）。
  需要 [`duplexjev-vllm`](https://pypi.org/project/duplexjev-vllm/) ≥ 0.2.0。
- **2026-10-01** —— 🎧 [**在线体验和试用 API**](https://api.adventists.cn/duplexjev/)（DuplexJev-4B）以及
  [`duplexjev` 0.3](https://pypi.org/project/duplexjev/)：`duplexjev quick call.wav` 对任意音频打印一张默认判断表
  （话轮状态、该怎么做、打断/抢话、意图、情绪、性别、语言、转人工）。
- **2026-09-30** —— 🤗 [**DuplexJev-32B**](https://huggingface.co/adventists-ai/DuplexJev-32B) 发布，成为新的主推模型：
  Qwen3-ASR-0.6B 编码器 + 连接器 + Qwen3-VL-32B 的语言模型，合在一个仓库，用 vLLM 部署。总分 87.7，是目前最高的：
  qa100 90、Easy-Turn 79.1、性别 89.4、情绪 90.6；口语 VoiceBench MMSU 70.8（之前的 32B 连接器为 56–60）。DuplexJev-4B 用同一配方更新（总分 75.9 → 78.6）。连接器 checkpoint 移到了[单独的页面](docs/connectors_zh.md)。
- **2026-09-28** —— 新开 [研究笔记](research) 专区。第一篇：[跨层融合对语音连接器有没有用？](research/2026-09-connector-a-vs-b_zh.md)
- **2026-09-28** —— 第一个完整模型 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) 和 vLLM 插件
  [`duplexjev-vllm`](https://pypi.org/project/duplexjev-vllm/) 发布。

即将推出：Speech-to-Decision 商用 API（2026-10-10）、开源全双工 Jev 对话流水线（2026-10-15）。

## 3. 模型

每个模型都是**一个包含全部组件的仓库**（音频编码器 + 训练好的连接器 + 大模型），装一个小插件就能用
[vLLM](https://github.com/vllm-project/vllm) 部署（见[第 4 节](#4-快速上手)）。

| 模型 | 适用 | 基座 | 参数 | 显存 | qa100 | ZJU-ML | Easy-Turn | 性别 | 情绪 | 总分 | 许可 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ⭐ 🤗 [**DuplexJev-32B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) | 服务器端，效果最好 | Qwen3-VL-32B（语言模型部分，加 rank 16 LoRA）+ Qwen3-ASR-0.6B 编码器 | 33.0 B | 一张 80 GB 显卡 | **96** | 87 | **95.3**\* | **91.5** | **91.1** | **92.2** | CC BY-NC 4.0 |
| 🤗 [DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn) | 小显卡、端侧 | Qwen3-4B（加 rank 16 LoRA）+ Qwen3-ASR-0.6B 编码器 | 4.2 B | 约 10 GB | 74 | 52 | 92.5\* | 89.4 | 92.0 | 80.0 | CC BY-NC 4.0 |

话轮升级前的 [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B)（总分 87.7）和 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B)（78.6）
仍可下载，但已被替代：Turn 版在我们跑的每项评测上都持平或更好。

分数为论文口径的百分比（单 token 读出）；总分 = 主语言（qa100、ZJU-ML、Easy-Turn）与副语言（性别、情绪）的平均，
见[评测基准](#6-评测基准)。\* 两个模型训练时都用了 Easy-Turn 的训练集（与测试集不重叠），所以 Easy-Turn 分数属于域内。两个模型在单张显卡上回答同一段音频的 10 个问题都在约 0.1–0.25 秒内。

**另有 🤗 [DuplexJev-Gemma-31B](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B)**（Gemma-4-31B-it + Qwen3-ASR-0.6B 编码器，
约 58 GB，一张 80 GB 显卡，CC BY-NC 4.0）。它还没做话轮训练；和 Turn 版一样，大模型带一个已合并的 LoRA。用 vLLM、按[第 4 节](#4-快速上手)的提问格式：
qa100 97、ZJU-ML 87、性别 94.4、情绪 90.6；Easy-Turn 68.0（零样本，未做话轮训练）。它对提问措辞敏感（论文口径措辞下性别 78.5、
情绪 60.9），所以没有放进上表；详见模型卡。

**怎么选。** 有 80 GB 显卡就用 `DuplexJev-32B-Turn`，显卡小就用 `DuplexJev-4B-Turn`。Turn 版可以直接替换原版（提示词、插件都一样），
我们跑的每项评测都持平或更好（在误差范围内）。32B 模型只用了 Qwen3-VL-32B 的语言模型部分，
目前输入是音频和文字（图像输入在计划中）。

> **研究用 checkpoint。** 这些模型背后还有几十个连接器 checkpoint，覆盖其他编码器（MOSS、Whisper、SenseVoice）、
> 大模型（Qwen3 0.6B–32B、Falcon-H1、SmolLM3）和两种连接器类型，配合 `duplexjev` PyTorch 包使用。
> 排行榜、分项成绩和包的用法见 **[docs/connectors_zh.md](docs/connectors_zh.md)**。

## 4. 快速上手

**不需要显卡**——直接调我们的试用 API（DuplexJev-4B；有限流，不保存音频）：

```bash
pip install duplexjev
duplexjev quick call_zh.wav --api https://api.adventists.cn/duplexjev --lang zh
```

```
判断       答案                置信度
-----------------------------------------
话轮状态   话说完了，可以接话  0.87
该怎么做   马上回应            0.91
...
8 个判断，80 ms
```

换成自己的题目：`--questions table.json`（`{"id", "text", "options"}` 的列表），或在 Python 里
`quick("call.wav", lang="zh", questions=[Question("product", "用户在问哪个产品？", ["手机", "电脑", "其他"], lang="zh")])`。
带录音功能的网页版在 [api.adventists.cn/duplexjev](https://api.adventists.cn/duplexjev/)。

**用自己的显卡**，vLLM 部署：

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-32B --max-model-len 4096      # 小显卡换成 adventists-ai/DuplexJev-4B
duplexjev quick call.wav --vllm http://localhost:8000/v1 --lang zh   # 用默认判断表问自己的服务
duplexjev gateway --vllm http://localhost:8000/v1 --port 8020        # 在前面套上这个网页 demo 和 HTTP API
```

也可以在 Python 里问自己的问题（OpenAI 兼容接口；每个问题一个请求，`max_tokens=1`）：

```python
from vllm_client import DuplexJevClient      # examples/vllm_client.py：只依赖 openai 和标准库

dj = DuplexJevClient("http://localhost:8000/v1")
dj.decide(open("call_018.wav", "rb").read(), {
    "turn":    ("用户说完了吗？", ["说完了", "还没说完"]),
    "gender":  ("说话人的性别是？", ["男性", "女性"]),
    "emotion": ("说话人当时的情绪状态是？", ["中性", "高兴", "生气", "伤心"]),
}, lang="zh")
# {'turn': {'answer': '说完了', 'confidence': 0.9, 'probs': {...}}, 'gender': {...}, 'emotion': {...}}
```

同一段音频的多个问题并发发送，在 vLLM 的前缀缓存里共用音频部分。用音频的语言提问（中文语音用中文问题和选项）。
提示词格式和插件说明见[模型卡](https://huggingface.co/adventists-ai/DuplexJev-32B)和 [`vllm_plugin/`](vllm_plugin)。
想用 PyTorch 包加载连接器 checkpoint，见 [docs/connectors_zh.md](docs/connectors_zh.md#pytorch-包)。

## 5. 性能

### 话轮：与专用检测器对比

DuplexJev-32B-Turn 的话轮判断达到专用检测器的水平；同一次前向里，它还能回答你声明的其他任意问题（意图、情绪、性别、
是否打断……）。全程不解码，同一段音频问 10 个问题和问 1 个的耗时差不多。

| 模型 | Easy-Turn 测试集（说完 / 没说完 / 附和 / 等一下） | Easy-Turn 总体 | CoDeTT 中 / 英 | 同一次调用的其他判断 | 延迟 |
|---|---|---:|---:|---|---|
| Smart Turn v2（95 MB） | 78.7 / 62.0 / – / – | – | – / 51.4（v3） | 无 | 27 ms |
| TEN Turn Detection（7 B） | 86.7 / 89.3 / – / 91.0 | – | – | 无 | 204 ms |
| Easy-Turn（0.85 B） | 96.3 / 97.7 / 91.0 / 98.0 | 96.4 | 37.9 / – | 无 | 263 ms |
| NAMO-Turn · FireRedChat | – | – | 59.5 / – · – / 65.4 | 无 | – |
| GPT-4o-audio · Qwen3-Omni | – | – | 66.6 / 71.9 · 70.4 / 70.9 | 有（靠生成文字） | 秒级 |
| **DuplexJev-32B-Turn**（本项目） | **98.7** / 94.3 / 88.0 / 95.0 | **95.3** | **69.2 / 70.0** | **有，任意多个，一次前向** | **10 个判断约 0.24 s**（单张 H200） |
| DuplexJev-4B-Turn（本项目） | 92.3 / 93.0 / 90.0 / 94.0 | 92.5 | 67.0 / 68.6 | 有 | 8 个判断约 0.08 s（单张 H100） |

- **Easy-Turn**：800 条测试集的准确率（%）。Easy-Turn 本身和我们的 Turn 模型都用过它的训练集。其他行取自 Easy-Turn
  论文（表 2，他们的硬件），延迟是单个判断的耗时。
- **CoDeTT**（[arXiv:2603.25434](https://arxiv.org/abs/2603.25434)）：1.8 万条，系统状态已知，4 个动作准确率平均。
  对我们是零样本（训练中没有 CoDeTT 数据）；其他行取自 CoDeTT 论文。我们的口径只给当前这句的音频、不给历史，
  官方口径还会播放之前的用户语音。
- 在我们自建的 TurnBench-dev 片段协议上（非官方榜单），Turn 模型 88.7（32B）/ 87.3（4B），升级前为 60.7 / 49.4。
- **其他能力没有变差。** 32B：qa100 90 → 96，性别 89.4 → 91.5，情绪 90.6 → 91.1，VoiceBench MMSU 70.8 → 72.1。
  做法（只用公开数据、一个 rank 16 的 LoRA，4B 和 32B 同一配方）见[研究笔记](research/2026-10-turn-taking-lora_zh.md)。

### 速度与口语知识

| | DuplexJev-32B(-Turn) | DuplexJev-4B(-Turn) |
|---|---|---|
| 一次判断（同一段音频 10 个问题，作为 10 个并发请求发给 vLLM OpenAI 服务） | 单张 H200 中位 **240 毫秒** | 默认 8 个问题约 **80 毫秒**（单张 H100，我们的试用 API） |
| 显存 | 一张 80 GB 显卡 | 约 10 GB |
| 口语知识问答 VoiceBench OBQA / MMSU（语音输入） | **83.7 / 70.8** | 47.3 / 41.3 |

作为参照，同一个大模型直接读文字是 95.4 / 79.3（Qwen3-VL-32B）。和常见的级联做法（先转写，再让大模型生成 JSON）相比，
单 token 读出完成 10 个判断要 92 毫秒，级联要 1,567 毫秒另加 ASR 411 毫秒（论文测量：Qwen3-32B、单张 H200、vLLM，10 个问题打包成一个请求）。
完整表格见 [docs/paper_results.md](docs/paper_results.md)。

## 6. 评测基准

| 基准 | 任务 | 条数 | 链接 |
|---|---|---:|---|
| qa100 | 语音选择题，合成语音（中 + 英） | 100 | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| ZJU-ML | 浙大音频基准 v2.0.0 主语言部分：语音提问，一半为真人录音 | 100 | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| Easy-Turn | 四类话轮状态（说完 / 没说完 / 附和 / 等一下）；完整模型用过它的训练集 | 800 | Easy-Turn 测试集（[arXiv:2509.23938](https://arxiv.org/abs/2509.23938)） |
| 性别 | 说话人性别，真人录音（AISHELL-1、Common Voice、LibriSpeech），中 + 英 | 800 | [`evaluation/`](evaluation) |
| 情绪 | 中性 / 高兴 / 生气 / 伤心，表演语音（ESD、CREMA-D），中 + 英 | 800 | [`evaluation/`](evaluation) |

**主语言** = qa100、ZJU-ML、Easy-Turn 三项平均；**副语言** = 性别、情绪两项平均；**总分** = 两者平均。
随机水平：四选一任务 25%，性别 50%。[连接器排行榜](docs/connectors_zh.md#排行榜)由 [`evaluation/build_leaderboard.py`](evaluation/build_leaderboard.py)
根据同目录下的结果文件生成。

## 7. 训练

编码器和大模型都冻结，只训练连接器；训练框架为打过补丁的 [Ultravox](https://github.com/fixie-ai/ultravox)。
两个完整模型都按同样的三步训练：

1. **内容，R1 和 R2。** 在 Ultravox v0.6 数据混合（WenetSpeech、GigaSpeech、Common Voice、CoVoST 2、People's Speech、
   LibriSpeech、MLS、MUSAN）上做转写蒸馏；数据切成 100 个互不重叠的包，R1、R2 各用一包。
2. **判断。** 在读出位置对单个选项字母做交叉熵：话轮状态（Easy-Turn 训练集）、性别（AISHELL-1、LibriSpeech）、
   情绪（ESD、CREMA-D）。
3. **混合训练。** 从 R2 出发，内容和判断样本一起训练（内容做蒸馏，判断做答案 token 交叉熵）；4,000 步，学习率 1e-4，
   内容样本加权（×10）、性别和情绪降权（×0.5），在学会话轮、性别、情绪的同时保住内容理解。

配方、数据链接和脚本：[`training/`](training)。情绪语料不再分发。[docs/connectors_zh.md](docs/connectors_zh.md) 里的研究用
checkpoint 用的是这套配方的早期版本。

## 8. 仓库结构

| 路径 | 内容 |
|---|---|
| [`vllm_plugin/`](vllm_plugin) | `duplexjev-vllm`：部署 DuplexJev-32B、DuplexJev-4B 和 DuplexJev-Gemma-31B 的 vLLM 插件 |
| [`duplexjev/`](duplexjev) | `duplexjev` 包：`quick()` 和默认判断表、vLLM 与 API 客户端、网页 demo 网关；加载连接器 checkpoint 的 PyTorch `Decider` |
| [`examples/`](examples) | [vLLM 客户端](examples/vllm_client.py)、快速上手、服务客户端 |
| [`evaluation/`](evaluation) | 评测脚本和结果 |
| [`training/`](training) | Ultravox 配置和补丁、数据配方、100 包切分、编码器移植 |
| [`research/`](research) | 研究笔记：实验和设计取舍，附数字和误差范围 |
| [`docs/`](docs) | 项目主页、[连接器 checkpoint](docs/connectors_zh.md)、[论文结果](docs/paper_results.md)、[包的细节](docs/package.md) |
| [`duplexjev/research/`](duplexjev/research)、[`tests/`](tests)、[`benchmarks/`](benchmarks) | 论文代码（读出、问题约定、融合编码器）、等价性测试、延迟和容量实验 |

论文代码里还留有我们集群上的路径，见 [docs/PATHS.md](docs/PATHS.md)。

## 9. 许可

代码：Apache-2.0（[LICENSE](LICENSE)）。完整模型（DuplexJev-32B、DuplexJev-4B、DuplexJev-Gemma-31B）：CC BY-NC 4.0，因为其中的连接器用情绪数据训练过。连接器权重：Apache-2.0；用情绪数据训练的（`-Emotion-` 和 `-Para-`）为 CC BY-NC 4.0，
因为 ESD 仅限研究使用。qa100：CC-BY-4.0。冻结的编码器和大模型沿用各自的许可（Falcon-H1：TII Falcon License；
SenseVoice：FunASR Model License）。部分训练语料（如 WenetSpeech、CoVoST 2）有非商用条款。详见 [NOTICE](NOTICE)。

## 10. 引用

```bibtex
@misc{jin2026duplexjev,
  title         = {Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen {LLM} Hear Beyond the Transcript},
  author        = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Song, Haigang and Pang, Zhikun and Zhang, Xiaowen},
  year          = {2026},
  eprint        = {2610.02638},
  archivePrefix = {arXiv},
  note          = {Submitted to IEEE ICASSP 2027}
}
```

## 11. 致谢

DuplexJev-32B 和 DuplexJev-4B 基于 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)、[Qwen3](https://github.com/QwenLM/Qwen3)
和 Qwen3-ASR（DuplexJev-Gemma-31B 基于 [Gemma 4](https://huggingface.co/google/gemma-4-31B-it)），用 [Ultravox](https://github.com/fixie-ai/ultravox) 训练，用 [vLLM](https://github.com/vllm-project/vllm) 部署。
研究用 checkpoint 还用到 [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)、Whisper、
SenseVoice、SmolLM3 和 Falcon-H1。感谢浙大团队提供[音频基准](https://github.com/Vsky-morigen/audio-gender-benchmark)。
Claude（Anthropic）协助编写代码。
问题和建议：[GitHub Issues](https://github.com/adventists-ai/duplexjev/issues)。
