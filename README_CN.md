<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.svg">
    <img src="docs/assets/banner_light.svg" alt="DuplexJev —— 不解码的批量语音判断" width="880">
  </picture>
</p>

<p align="center">
  <a href="README.md">English</a> &nbsp;|&nbsp; <b>中文</b>
</p>

<h1 align="center">别再先转写、再判断了。</h1>

<p align="center">
  <b>DuplexJev：语音智能体的“反射神经”。</b><br>
  用户说完了吗？我该不该停？他生气了吗？该接哪句垫词？<br>
  全部直接从原始语音读出，<b>一次前向</b>：<b>10 个判断 92 毫秒</b>，比 ASR → LLM 级联<b>快 21 倍</b>。<br>
  👁️ <b>新：它还看得见。</b><a href="https://huggingface.co/adventists-ai/DuplexJev-32B-Vision">DuplexJev-32B-Vision</a> 在同一次前向里把声音和画面对照着判断（预览版）。
</p>

<p align="center">
  <a href="https://api.adventists.cn/duplexjev/"><img src="https://img.shields.io/badge/%F0%9F%8E%A7%20%E5%9C%A8%E7%BA%BF%E4%BD%93%E9%AA%8C-%E4%B8%8D%E7%94%A8%E6%B3%A8%E5%86%8C-22D3A6?style=for-the-badge" alt="在线体验"></a>
  <a href="https://github.com/adventists-ai/duplexjev/stargazers"><img src="https://img.shields.io/badge/%E2%AD%90-%E7%82%B9%E4%B8%AA%E6%98%9F-181717?style=for-the-badge&logo=github" alt="点星"></a>
  <a href="integrations/pipecat"><img src="https://img.shields.io/badge/Pipecat-%E6%8F%92%E4%BB%B6-6c47ff?style=for-the-badge" alt="Pipecat 插件"></a>
  <a href="integrations/livekit"><img src="https://img.shields.io/badge/LiveKit-%E6%8F%92%E4%BB%B6-1f6feb?style=for-the-badge" alt="LiveKit 插件"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02638"><img src="https://img.shields.io/badge/arXiv-2610.02638-b31b1b" alt="arXiv"></a>
  <a href="https://huggingface.co/adventists-ai"><img src="https://img.shields.io/badge/%F0%9F%A4%97-%E5%BC%80%E6%94%BE%E6%9D%83%E9%87%8D%204B%20%C2%B7%2032B-yellow" alt="开放权重"></a>
  <a href="https://pypi.org/project/duplexjev/"><img src="https://img.shields.io/pypi/v/duplexjev" alt="PyPI"></a>
  <a href="https://huggingface.co/spaces/BitKnight/DuplexJev"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HF%20%E6%BC%94%E7%A4%BA-orange" alt="HF 演示"></a>
</p>

<p align="center"><img src="docs/assets/race.gif" alt="十个判断：DuplexJev 92 毫秒，ASR → LLM 1,978 毫秒，同一张 H200" width="820"></p>
<p align="center"><sub>▶ <a href="docs/assets/duplexjev_launch_zh.mp4">看 2 分半中文介绍视频（有声音）</a> · 画面里的答案都是模型真实输出</sub></p>

| 为什么换 | **DuplexJev** | 常规做法 |
|---|:---:|---|
| ⚡ 对一句话做十个判断（单张 H200） | **92 毫秒** | 1,978 毫秒——ASR → LLM → JSON，同卡同模型 |
| 💸 成本 | **N 个问题 ≈ 一次前向** · 4B 版**每百万次判断不到 0.5 美元** | 每个问题一次生成 |
| 📚 5k token 上下文上问 50 个问题 | **5.5 秒** | 111.5 秒（每题单独调用） |
| 👂 从声音听情绪（四分类） | **91** | 约 28（只用转写训练的语音大模型） |
| 🎯 话轮判断，CoDeTT 英文零样本 | **70.0** | 51.4（Smart Turn v3） |
| 🔊 重音、停顿、笑声、呼吸、叹气 | **听得到** | 转写里全丢了 |
| 👁️ 声音 + 画面，一次前向（新，预览） | 标红词 = 重读词 **94.1** · 表情 ↔ 语气 **93.5** | 只看图 50（瞎猜） |
| 🧩 新增一种判断 | **运行时加一行文字** | 重新采数据、重新训模型 |

**30 秒看到结果：**

```bash
pip install duplexjev
python -c "from duplexjev.remote import ApiClient; print(ApiClient('https://api.adventists.cn/duplexjev').decide('your.wav'))"
```

**自部署：**`vllm serve adventists-ai/DuplexJev-4B-Para`（约 10 GB 显存），配 `duplexjev-vllm` 插件（[第 4 节](#4-快速上手)）·
**Pipecat：**`pip install pipecat-duplexjev` · **LiveKit Agents：**`pip install livekit-plugins-duplexjev`（[integrations/livekit](integrations/livekit)）：基于音频的话轮检测，同一次前向顺带给出情绪和非语言声音。

> 🏢 **要用在产品里？** AI降临派的全双工对话接口已服务 **300 多款 AI 硬件、10 万多台终端**。
> 呼叫中心、车载助手、智能硬件、语音 Agent 平台：权重为 CC BY-NC 4.0，我们提供**商用授权和托管的
> Speech-to-Decision API**。联系 **jiejin@adventists.ai**。

⭐ **点个星**关注后续能力：图像 + 语音联合判断、开源全双工管线都在路上。

### DuplexJev 是一个 Speech-to-Decision（语音到判断）模型。

**输入：**一段音频，加上你需要判断的问题（每题带选项）。**输出：**每道题选中的选项和它的概率。所有问题在一次前向里一起回答，
不转写、不解码文字：10 个判断约 **0.1–0.24 秒**；单张 H200 在每个请求都不超过 0.25 秒的前提下，每秒可完成约 **170 个判断**
（[第 5 节](#5-性能)）。

它直接依据原始音频做判断，既用到**说了什么**（主语言信息），也用到**转写里没有的信息**——谁在说、怎么说（副语言信息）：

- **主语言能力：接近直接读文字的大模型。** 语音选择题 90 分，同一大模型读标准转写 91 分（qa100，论文）。
  DuplexJev-32B-Turn：qa100 96、ZJU-ML 87；VoiceBench OBQA / MMSU 85.5 / 72.1（读文字 95.4 / 79.3）。
- **副语言能力：感知很强。** 性别 91.5、情绪 91.1，而只用转写训练的语音大模型只有瞎猜水平（约 55 / 28）。
- **重音与停顿（新）。** DuplexJev-32B-Stress 能在没见过的中英文句子上听出重读的是哪个词（“**我**没说……”和“我没说她拿了**钱**”），
  中文 92 / 英文 88；中性语音 99% 能答“没有特别强调”；其他模型都是瞎猜水平（[研究笔记](research/2026-10-stress-understanding_zh.md)）。
- **常见语音判断任务：达到或超过专用模型。** 话轮状态（Easy-Turn 测试集）95.3（专用 Easy-Turn 检测器 96.4；TEN、Smart Turn
  只覆盖部分类别）。话轮动作（CoDeTT）零样本 69.2 / 70.0，超过所有专用话轮模型（37.9–65.4），与 Qwen3-Omni（70.4 / 70.9）持平。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench_turn_dark.svg">
    <img src="docs/assets/bench_turn_light.svg" alt="DuplexJev 与专用话轮检测器、语音大模型在 CoDeTT 和 Easy-Turn 上的话轮准确率对比" width="880">
  </picture>
</p>

**一段音频，八个判断。** 🔊 *"Wait, wait — stop. That's not the address I asked for."*（"等等——停，这不是我要去的地址。"助手正在说话时用户插进来；
[试听](docs/audio/ex1.wav)，合成语音）——DuplexJev-32B-Turn 一次调用，单张 H200 耗时 0.2 秒：

| 问题 | 答案 | 概率 |
|---|---|---:|
| 用户现在的话轮状态是？ | 让对方等一下 | 0.92 |
| 用户想做什么？ | 导航 | 1.00 |
| 先说哪句垫话？ | “稍等，” | 0.73 |
| 助手正在说话时听到这句，应该怎么做？ | 停下来听 | 1.00 |
| 说话人性别 | 男性 | 0.99 |
| 语言 | 英语 | 1.00 |
| 情绪 | 生气 | 1.00 |
| 助手自己能处理吗？ | 先向用户追问 | 0.51 |

<details>
<summary><b>听懂文字之外的信息：每种能力一个例子</b></summary>

DuplexJev-32B-Stress 在短音频上的真实输出（vLLM，两种选项顺序取平均）；带播放器的同一组卡片见
[项目主页](https://adventists-ai.github.io/duplexjev/#abilities)。重音与断句的真人示例音频来自 MSPB（CC BY 4.0）。最后三行（Para 版）给的是评测分数，不是示例。

| 能力 | 判断什么 | 示例 → 模型回答（DuplexJev-32B-Stress） |
|---|---|---|
| 话轮状态 | 用户是说完了、话说到一半、只是附和，还是让助手等一下？靠语调和节奏判断，不只看文字。 | 🔊 [帮我导航到最近的充电站。](docs/audio/ex4.wav) → **说完了** 100%<br>🔊 [Could you turn the air conditioning down a little and, um —](docs/audio/ex3.wav) → **没说完** 99%<br>🔊 [Mm-hmm, yeah.](docs/audio/ex2.wav) → **只是附和** 100%<br>🔊 [等一下，我想想……](docs/audio/ex5.wav) → **让对方等一下** 100% |
| 说话时被插话 | 助手正在说话时，分清用户是真的要打断，还是只在附和：该停下来听，还是继续说。 | 🔊 [Wait, wait — stop. That's not the address I asked for.](docs/audio/ex1.wav) → **停下来听** 100%<br>🔊 [Mm-hmm, yeah.](docs/audio/ex2.wav) → **继续说** 100% |
| 不理会旁人的声音 | 旁边有人聊天、电视声或噪声，既不该让助手开口，也不该让它停下。 | 🔊 [哎你晚上吃什么？要不我们去吃火锅吧。——行啊，那我先订个位子。](docs/audio/ab_b1.wav) → **不出声，继续等（这不是对助手说的）** 100% |
| 情绪 | 中性、高兴、生气或伤心，从声音里听出来，而不是从文字里猜。 | 🔊 [Wait, wait — stop. That's not the address I asked for.](docs/audio/ex1.wav) → **生气** 100%<br>🔊 [太好了，谢谢你！](docs/audio/ex6.wav) → **高兴** 93%<br>🔊 [帮我导航到最近的充电站。](docs/audio/ex4.wav) → **中性** 100% |
| 说话人性别 | 听出说话人的性别——转写里没有这个信息。 | 🔊 [帮我导航到最近的充电站。](docs/audio/ex4.wav) → **男性** 100%<br>🔊 [太好了，谢谢你！](docs/audio/ex6.wav) → **女性** 100% |
| 重音（强调哪个词） 🆕 | 同样的字，重读不同的词，意思就不同。模型能听出重读落在哪个词上，也能判断“没有特别强调”。 | 🔊 [**张昊**昨晚做烤肉。](docs/audio/ab_m1.wav) → **张昊** 100%<br>🔊 [张昊**昨晚**做烤肉。](docs/audio/ab_m2.wav) → **昨晚** 100%<br>🔊 [张昊昨晚**做烤肉**。](docs/audio/ab_m3.wav) → **做烤肉** 97%<br>🔊 [我明天去北京开会。](docs/audio/ab_n0.wav) → **没有特别强调哪个词** 96% |
| 停顿与断句 🆕 | 在哪里停顿会改变句子结构：是一样东西还是两样，谁对谁做了什么。 | 🔊 [我买了巧克力雪糕｜和果汁。](docs/audio/ab_q1.wav) → **两样（巧克力雪糕、果汁）** 96%<br>🔊 [我买了巧克力｜雪糕｜和果汁。](docs/audio/ab_q2.wav) → **三样（巧克力、雪糕、果汁）** 98% |
| 非语言声音 🆕（Para） | 笑声、呼吸、咳嗽、叹气、吸鼻子、清嗓子、打喷嚏、哭声，带“都没有”选项。 | 训练没见过的真人语音（NonverbalTTS 测试集），笑 / 呼吸 / 咳嗽 / 叹气 AUC：32B-Para 0.85 / 0.73 / 0.70 / 0.74，4B-Para 0.80 / 0.76 / 0.75 / 0.81（之前 0.40–0.73；AudioSet AST 检测器 0.71–0.85）。 |
| 说话方式 🆕（Para） | 耳语、大喊、语速很快或很慢、音调特别高或特别低，还是正常说话。 | EARS 留出说话人 7 选 1：69%（32B-Para）/ 67%（4B-Para），之前 45–55%。 |
| 是谁在说 🆕（Para） | 给一段声音样本，判断是不是同一个人。像人一样的粗略判断，远不如专用说话人模型，不能用来识别身份。 | AISHELL-1 同性别配对：EER 18%（32B-Para）/ 9%（4B-Para）；官方 VoxCeleb1-H 列表：40% / 37%（专用 x-vector 模型约 2–4%）。 |

</details>

<details>
<summary><b>常见判断任务的完整对比</b></summary>

| 任务 | 评测集 | DuplexJev | 专用模型和其他模型 |
|---|---|---:|---|
| 话轮状态：说完 / 没说完 / 附和 / 等一下 | Easy-Turn 测试集（800） | **95.3** | Easy-Turn 检测器 96.4 · TEN Turn Detection、Smart Turn v2 只覆盖 4 类中的 3 / 2 类 |
| 已知系统状态，该怎么做 | CoDeTT 中 / 英（1.8 万，零样本） | **69.2 / 70.0** | Easy-Turn 37.9（中）· Smart-Turn-v3 51.4（英）· NAMO-Turn 59.5（中）· FireRedChat 65.4（英）· GPT-4o-audio 66.6 / 71.9 · Qwen3-Omni 70.4 / 70.9 · Gemini-3-Pro 80.8 / 81.9 |
| 性别 | 800 条真实语音（AISHELL-1、LibriSpeech） | **91.5** | 只用转写训练的语音大模型：约 55（瞎猜） |
| 情绪（4 类） | 800 条（ESD、CREMA-D） | **91.1** | 只用转写训练的语音大模型：约 28（瞎猜） |
| 是不是同一个人 | 官方 VoxCeleb1-H 列表：不同人之间性别和国籍相同（2,000 个试次） | **EER 36.9%**（4B-Para）/ 40.3%（32B-Para）；研究 checkpoint‡ 29.6% / 31.2% | 只靠性别：瞎猜（50）；专用说话人模型（ECAPA-TDNN）：EER 约 2% |
| 笑 / 呼吸 / 咳嗽 / 叹气 | NonverbalTTS 测试集（真人语音，训练未见），AUC | **0.85 / 0.73 / 0.70 / 0.74**（32B-Para） | AudioSet AST 检测器 0.71 / 0.75 / 0.85 / 0.76 |
| 口语知识问答 | VoiceBench OBQA / MMSU | **85.5 / 72.1** | 同一大模型直接读文字：95.4 / 79.3 |

其他系统的数字取自各自论文（Easy-Turn 表 2、CoDeTT、Dynamic-SUPERB Phase-2）。Easy-Turn 对 Easy-Turn 检测器和我们的模型都是域内评测。
‡ 研究 checkpoint：只训练说话人验证（训练数据不含 VoxCeleb1 说话人），未发布；Para 版为了保住其他各项能力，在这一项上让了一些分，详见[研究笔记](research/2026-10-speaker-identity_zh.md)。随机配对里约一半的"不同人"性别也不同，只靠性别猜就能到 74%，所以只报困难配对。详见[第 5 节](#5-性能)。

</details>

**模型：**🤗 [**DuplexJev-32B-Para**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para)（一张 80 GB 显卡）·
🤗 [**DuplexJev-4B-Para**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)（约 10 GB），用 vLLM 部署（[第 4 节](#4-快速上手)）。
新增：还能听出**笑声、呼吸、咳嗽、叹气**，听出**怎么说**（耳语、大喊），并判断两段声音是不是**同一个人**——[试一试](#试试新能力)。

## 1. 简介

**DuplexJev** 把一个冻结的语音大模型变成面向全双工语音智能体的**类型化判断引擎**（产品名 **Speech-to-Decision**）。
现成 ASR 编码器的隐状态经连接器送入冻结的大模型；运行时声明的每个问题——*用户说完了吗？先说哪句垫话？说话的是谁？*——
都读成**单个 token 上的闭集概率分布**：不做 ASR 解码，也不做文本解码。同一通电话的多个问题、乃至多通电话的问题，
可以在同一次前向计算里一起完成。所有发布的模型都可以直接用 vLLM 部署（[第 3 节](#3-模型)）；推荐
**[DuplexJev-32B-Para](https://huggingface.co/adventists-ai/DuplexJev-32B-Para)**，显卡小用 **[DuplexJev-4B-Para](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)**。

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

- **2026-10-10** —— 🤗 [**DuplexJev-32B-Vision**](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision)（研究预览）：同一道题可以同时用上**听到的和看到的**。没见过的说话人的真实照片配他们自己的声音：64.4（只给图 50），其中开心的声音 81.7。`pip install "duplexjev[vision]>=0.5"`，或用 `duplexjev-vllm>=0.3` 通过 `vllm serve` 部署（每道题约 0.05 秒）。见[语音 + 图像一起判断](#语音--图像一起判断研究预览)。
- **2026-10-10** —— 🔬 研究预览：**图像 + 语音一起判断。** DuplexJev-32B-Para 的语言模型就是 Qwen3-VL 的，视觉编码器可以直接接回去；再加一层 LoRA，它能把听到的和看到的对起来（留出的合成测试集上：重读词 vs 标红词 94.1，语气 vs 表情 93.5，声音 vs 文字说明 97.5；只给图 = 50）。真实案例见[项目主页](https://adventists-ai.github.io/duplexjev/#abilities)。
- **2026-10-10** —— 🤗 [**DuplexJev-4B-Para v1.1**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)：同一个小模型现在也能**正常聊天**（包括多轮），判断类成绩与 v1.0 基本持平（LoRA 阶段加入对话回放、按行平均损失）。v1.0 仍可用 revision `v1.0` 下载。[研究笔记](research/2026-10-chat-replay_zh.md)。
- **2026-10-09** —— 🤗 [**DuplexJev-32B-Para**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) 和
  [**DuplexJev-4B-Para**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)：一个模型里副语言能力最全的版本。在话轮、性别、
  情绪、重音、停顿之外，还能听出**笑声、呼吸、咳嗽、叹气**（训练未见的真人语音上 AUC 0.70–0.85，之前 0.40–0.73），识别**说话方式**
  （耳语、大喊、快慢……），并像人一样粗略判断**是不是同一个人**。4B-Para 也是主语言最强的 4B（qa100 79，VoiceBench 57.8 / 41.8），
  现已成为 `duplexjev` 包（0.4）的默认模型。
- **2026-10-08** —— 🤗 [**DuplexJev-32B-Stress**](https://huggingface.co/adventists-ai/DuplexJev-32B-Stress)：在 DuplexJev-32B-Turn
  基础上**听得懂重音和停顿**。重读的是哪个词：中文留出句 92 / 英文 88；中性语音答“没有特别强调”99%；主语言、性别、情绪、话轮
  都在误差范围内持平。每种语音来源里都配了正负样本；“靠重音选意思”的题会拉低 VoiceBench，所以没有放进训练。同时发布
  🤗 [**DuplexJev-4B-Stress**](https://huggingface.co/adventists-ai/DuplexJev-4B-Stress)（重读词判断 95 / 95；VoiceBench 比 4B-Turn 低约 4 分）。
  [研究笔记](research/2026-10-stress-understanding_zh.md)。
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

即将推出：开源全双工 Jev 对话流水线（2026-10-20 前后）。商用授权与托管 Speech-to-Decision API：jiejin@adventists.ai。

## 3. 模型

每个模型都是**一个包含全部组件的仓库**（音频编码器 + 训练好的连接器 + 大模型），装一个小插件就能用
[vLLM](https://github.com/vllm-project/vllm) 部署（见[第 4 节](#4-快速上手)）。

| 模型 | 适用 | 基座 | 参数 | 显存 | qa100 | ZJU-ML | Easy-Turn | 性别 | 情绪 | 总分 | 许可 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ⭐ 🤗 [**DuplexJev-32B-Para**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) 🆕 | 服务器端，能力最全 | 同 32B-Turn（再加一个 rank 16 LoRA） | 33.0 B | 一张 80 GB 显卡 | 95 | 87 | 94.0\* | **92.1** | 89.5 | 91.4 | CC BY-NC 4.0 |
| ⭐ 🤗 [**DuplexJev-4B-Para**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) 🆕 | 小显卡，能力最全，**也能正常聊天**（v1.1）；`duplexjev` 默认模型 | 同 4B-Turn（再加一个 rank 16 LoRA） | 4.2 B | 约 10 GB | 77 | **58** | 91.0\* | 89.2 | 90.0 | **82.5** | CC BY-NC 4.0 |
| 🤗 [**DuplexJev-32B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) | 服务器端，效果最好 | Qwen3-VL-32B（语言模型部分，加 rank 16 LoRA）+ Qwen3-ASR-0.6B 编码器 | 33.0 B | 一张 80 GB 显卡 | **96** | 87 | **95.3**\* | **91.5** | **91.1** | **92.2** | CC BY-NC 4.0 |
| 🤗 [DuplexJev-32B-Stress](https://huggingface.co/adventists-ai/DuplexJev-32B-Stress) 🆕 | 服务器端，加重音与停顿 | 同 32B-Turn（再加一个 rank 16 LoRA） | 33.0 B | 一张 80 GB 显卡 | **96** | 87 | 94.4\* | **92.1** | **91.4** | — | CC BY-NC 4.0 |
| 🤗 [DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn) | 小显卡、端侧 | Qwen3-4B（加 rank 16 LoRA）+ Qwen3-ASR-0.6B 编码器 | 4.2 B | 约 10 GB | 74 | 52 | 92.5\* | 89.4 | 92.0 | 80.0 | CC BY-NC 4.0 |
| 🤗 [DuplexJev-4B-Stress](https://huggingface.co/adventists-ai/DuplexJev-4B-Stress) 🆕 | 小显卡，加重音与停顿 | 同 4B-Turn（再加一个 rank 16 LoRA） | 4.2 B | 约 10 GB | 73 | 51 | 91.8\* | 89.1 | 91.4 | — | CC BY-NC 4.0 |

**Para 版新增的能力**（AUC，0.5 为瞎猜；训练未见的真人语音：NonverbalTTS 测试集、第三方中文自然对话样本；说话人 EER 用 AISHELL-1 同性别配对）：

| | 笑 | 呼吸 | 咳嗽 | 叹气 | 对话中的笑 / 呼吸 | 说话方式（7 类） | 是否同一人 EER |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32B-Stress | 0.73 | 0.56 | 0.40 | 0.70 | 0.61 / 0.44 | 45% | — |
| **32B-Para** | **0.85** | **0.73** | **0.70** | **0.74** | **0.80 / 0.78** | **69%** | 18% |
| **4B-Para** v1.1 | 0.81 | 0.77 | 0.77 | 0.71 | 0.85 / 0.74 | 67% | ≈ v1.0 (9%)¹ |
| 4B-Para v1.0 | 0.80 | 0.76 | 0.75 | 0.81 | 0.84 / 0.84 | 67% | 9% |

¹ v1.1 没有在 AISHELL-1 配对上重测；在 VoxCeleb1 困难配对上 EER 为 42.0%（v1.0 为 41.8%）。

代价很小，模型卡上都列了（32B-Para：情绪比 32B-Turn 低 1.6、Easy-Turn 低 1.2；4B-Para：ZJU-ML 比 4B-Turn 低 2、Easy-Turn 低 1）。
说话人判断很粗略，像人一样，不能用来识别身份。

**DuplexJev-4B-Para v1.1（2026-10-10）** 加入了对话回放训练，同一份权重还能正常聊天。判断类成绩与 v1.0 相差 0.5–2.6 分（v1.0 仍可用 revision `v1.0` 下载），见[研究笔记](research/2026-10-chat-replay_zh.md)。32B 版正在训练。

Stress 版增加了重音和停顿理解（哪个词被重读：中文留出句 32B 92 / 4B 95，见[研究笔记](research/2026-10-stress-understanding_zh.md)）。32B-Stress 其余各项都与 32B-Turn 在误差内持平；**4B-Stress 用约 4 分 VoiceBench 换来这项能力**（42.0 / 38.2 对 46.4 / 41.7），所以小显卡默认仍推荐 4B-Turn。


话轮升级前的 [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B)（总分 87.7）和 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B)（78.6）
仍可下载，但已被替代：Turn 版在我们跑的每项评测上都持平或更好。

分数为论文口径的百分比（单 token 读出）；总分 = 主语言（qa100、ZJU-ML、Easy-Turn）与副语言（性别、情绪）的平均，
见[评测基准](#6-评测基准)。\* 两个模型训练时都用了 Easy-Turn 的训练集（与测试集不重叠），所以 Easy-Turn 分数属于域内。两个模型在单张显卡上回答同一段音频的 10 个问题都在约 0.1–0.25 秒内。

**另有 🤗 [DuplexJev-Gemma-31B](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B)**（Gemma-4-31B-it + Qwen3-ASR-0.6B 编码器，
约 58 GB，一张 80 GB 显卡，CC BY-NC 4.0）。它还没做话轮训练；和 Turn 版一样，大模型带一个已合并的 LoRA。用 vLLM、按[第 4 节](#4-快速上手)的提问格式：
qa100 97、ZJU-ML 87、性别 94.4、情绪 90.6；Easy-Turn 68.0（零样本，未做话轮训练）。它对提问措辞敏感（论文口径措辞下性别 78.5、
情绪 60.9），所以没有放进上表；详见模型卡。

**怎么选。** 想一个模型听全声音里的信息（非语言声音、说话方式、说话人、重音、情绪、性别、话轮）就用 **Para** 版：
80 GB 显卡用 `DuplexJev-32B-Para`，小显卡用 `DuplexJev-4B-Para`。最看重情绪和话轮的最后一分，用 `DuplexJev-32B-Turn` / `4B-Turn`；
重音和停顿是主要需求，用 `DuplexJev-32B-Stress` / `4B-Stress`。Turn 版可以直接替换原版（提示词、插件都一样），
我们跑的每项评测都持平或更好（在误差范围内）。32B 模型只用了 Qwen3-VL-32B 的语言模型部分，
输入是音频和文字。要同时输入语音和图像，用 [DuplexJev-32B-Vision](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision)（研究预览）。

> **研究用 checkpoint。** 这些模型背后还有几十个连接器 checkpoint，覆盖其他编码器（MOSS、Whisper、SenseVoice）、
> 大模型（Qwen3 0.6B–32B、Falcon-H1、SmolLM3）和两种连接器类型，配合 `duplexjev` PyTorch 包使用。
> 排行榜、分项成绩和包的用法见 **[docs/connectors_zh.md](docs/connectors_zh.md)**。

## 4. 快速上手

**不需要显卡**——直接调我们的试用 API（有限流，不保存音频）：

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
vllm serve adventists-ai/DuplexJev-4B-Para --max-model-len 4096   # 大显卡换成 adventists-ai/DuplexJev-32B-Para
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

### 试试新能力

用 Para 模型（`vllm serve adventists-ai/DuplexJev-4B-Para`，或本地 `pip install "duplexjev[speech]"`）时，`duplexjev quick`
的默认判断表多了两行：*非语言声音* 和 *说话方式*。也可以直接问（下面是模型训练时用的问法；记得保留“没有 / 都没有”选项）：

```python
dj.decide(clip_zh, {
    "sound":  ("除了说话，这段音频里还有哪种声音？", ["笑声", "呼吸声", "咳嗽", "叹气", "都没有"]),
    "laugh":  ("这段话里有没有笑声？", ["有", "没有"]),
    "style":  ("说话人是用什么方式在说话？", ["小声耳语", "大声喊着说", "正常音量说话"]),
    "stress": ("说话人是否重读（强调）了某个词？", ["张昊", "昨晚", "做烤肉", "没有强调任何词"]),
}, lang="zh")
```

是不是同一个人：把一段声音样本和新的语音拼成一个文件，中间隔约 0.8 秒静音，问
`"静音前是某人的一段声音样本，静音后是另一段语音。两段是同一个人吗？"`，选项 `["是不同的人", "是同一个人"]`
（英文：`"Before the silence is a voice sample of someone; after it is another clip. Are they the same speaker?"`，`["different speakers", "same speaker"]`）。
现成脚本：[`examples/para_abilities.py`](examples/para_abilities.py)。

### 语音 + 图像一起判断（研究预览）

[DuplexJev-32B-Vision](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision) 把 Qwen3-VL 的视觉编码器接回
DuplexJev-32B-Para 的语言模型前面，同一道题可以同时用上**听到的**和**看到的**（单张 80 GB 显卡）：

```python
# pip install "duplexjev[vision]>=0.5"
from duplexjev.vision import VisionDecider
vd = VisionDecider.from_pretrained("adventists-ai/DuplexJev-32B-Vision")
vd.decide(audio="clip.wav", image="face.jpg", questions=[
    {"id": "match", "text": "Does the face in the picture show the same emotion as the speaker's voice?", "options": ["Yes", "No"]}])
vd.chat(image="photo.jpg", text="图片里写了什么？")
```

也可以用 vLLM 部署（`pip install "duplexjev-vllm>=0.3"`），一张图配 3 秒语音，每道题约 0.05 秒：

```bash
vllm serve adventists-ai/DuplexJev-32B-Vision --max-model-len 8192 --limit-mm-per-prompt '{"image": 1, "audio": 1, "video": 0}'
python examples/vllm_vision_client.py clip.wav face.jpg
```

留出集成绩：重读词 vs 标红词 94.1，表情 vs 语气 93.5，声音 vs 文字说明 97.5（只给图：50）；没见过的说话人的真实照片配他们自己的声音 64.4，其中开心的声音 81.7。真实照片的案例见[项目主页](https://adventists-ai.github.io/duplexjev/#ab-vision)。

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
