# DuplexJev 全部发布记录

[← 返回 README](../README_CN.md) · [English](news.md)

- **2026-10-15** —— 🤗 [**DuplexJev-32B-Para v1.3**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) 发布：**听重音大幅提升**。与闭源模型同一批题：StressTest **71.5 → 86.5**（Gemini 3.8 Flash 89.0、Qwen3-Omni 70.0、GPT-audio-1.5 62.5），MSPB **68.0 → 77.3**（测过的模型里最高；Qwen3-Omni 71.9），由重音判断意思 90.5 → 93.0。加上 LISTEN，六个公开副语言基准合计（1,035 题，不含训练语料）**50.2**，Gemini 3.8 Flash 47.8、Qwen3-Omni 45.0。情绪略升，性别和 Easy-Turn 约低 1 分。v1.2 仍可用 revision `v1.2` 下载；[魔搭](https://www.modelscope.cn/models/AdventistsAI/DuplexJev-32B-Para)同步。
- **2026-10-15** —— 🏆 **对比闭源旗舰，包括实时模型。** 用同一批 LISTEN 题测了 Gemini 3.8 Live、GPT-realtime-2.1 和 GPT-audio-1.5：在测过的 7 个模型里，DuplexJev-32B-Para 的 LISTEN 纯音频四项平均最高（37.3；其次 Qwen3-Omni 33.9、Gemini 3.8 Flash 33.1；两个实时模型 30.9 和 25.6）。脚本和每道题的预测在 [`benchmarks/paraling_compare`](../benchmarks/paraling_compare)。LiveKit eot-bench 补齐其余 12 种语言（零样本：DuplexJev 只训练过中英文）：300 毫秒预算下，14 种语言的误打断全部少于 Smart Turn v3.2（[§5](../README_CN.md#实时话轮检测livekit-eot-bench-与-pipecat-smart-turn)）。
- **2026-10-14** —— 🤗 [**DuplexJev-4B-Para-LiveTurn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para-LiveTurn)：把 32B-Para v1.2 的话轮配方用到 4B 上（约 10 GB）。LiveKit eot-bench 最多等 300 ms 时的误打断：英文 **25.5%**、中文 **38.8%**（4B-Para v1.1 为 46.4% / 48.5%，Smart Turn v3.2 为 35.2% / 42.9%）。重音、VoiceBench、VocalSound 比 4B-Para 低几分，通用 4B 仍是 4B-Para。[魔搭](https://www.modelscope.cn/models/AdventistsAI/DuplexJev-4B-Para-LiveTurn)同步上线。
- **2026-10-13** —— 🤗 [**DuplexJev-32B-Para v1.2**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para)：新增**口音判断**（英文口音、中文方言区：4 选 1 从 39.6 → 56.0、25.7 → 48.2；是否标准普通话 46.6 → 91.3），**用户刚停顿时的话轮判断**明显变准：LiveKit [eot-bench](https://github.com/livekit/eot-bench) 上，最多等 300 ms 时的误打断从 29.8%（32B-Turn）降到 23.5%，像插件那样同时给模型助手上一句话时降到 **20.0%**（eot-bench 完整模式；Smart Turn v3.2 为 35.2%）。新版插件会自动带上这句话：`pipecat-duplexjev 0.3`、`livekit-plugins-duplexjev 0.2`、网关 `duplexjev 0.6`。[试用 API](https://api.adventists.cn/duplexjev/) 也已换成 32B-Para v1.2。v1.1 仍可用 revision `v1.1` 下载。详见[第 5 节](../README_CN.md#实时话轮检测livekit-eot-bench-与-pipecat-smart-turn)。
- **2026-10-12** —— 全部 10 个 DuplexJev 模型已上线 **[魔搭 ModelScope](https://www.modelscope.cn/organization/AdventistsAI)**（权重与 Hugging Face 一致，已逐文件核对 sha256），国内下载更快。
- **2026-10-12** —— 🤗 [**DuplexJev-32B-Para v1.1**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para)：现在是总分最高的 DuplexJev 模型（92.6，32B-Turn 为 92.2），非语言声音（咳嗽 AUC 0.70 → 0.85、呼吸 0.73 → 0.81）、说话方式（69% → 82%）和重音（DEBATE 87 → 93、StressPresso 84 → 96）都明显提升。提示词和插件不变；v1.0 仍可用 revision `v1.0` 下载。
- **2026-10-10** —— 🤗 [**DuplexJev-32B-Vision**](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision)（研究预览）：同一道题可以同时用上**听到的和看到的**。没见过的说话人的真实照片配他们自己的声音：64.4（只给图 50），其中开心的声音 81.7。`pip install "duplexjev[vision]>=0.5"`，或用 `duplexjev-vllm>=0.3` 通过 `vllm serve` 部署（每道题约 0.05 秒）。见[语音 + 图像一起判断](../README_CN.md#声音--画面联合判断研究预览)。
- **2026-10-10** —— 🔬 研究预览：**图像 + 语音一起判断。** DuplexJev-32B-Para 的语言模型就是 Qwen3-VL 的，视觉编码器可以直接接回去；再加一层 LoRA，它能把听到的和看到的对起来（留出的合成测试集上：重读词 vs 标红词 94.1，语气 vs 表情 93.5，声音 vs 文字说明 97.5；只给图 = 50）。真实案例见[项目主页](https://adventists-ai.github.io/duplexjev/#abilities)。
- **2026-10-10** —— 🤗 [**DuplexJev-4B-Para v1.1**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)：同一个小模型现在也能**正常聊天**（包括多轮），判断类成绩与 v1.0 基本持平（LoRA 阶段加入对话回放、按行平均损失）。v1.0 仍可用 revision `v1.0` 下载。[研究笔记](../research/2026-10-chat-replay_zh.md)。
- **2026-10-09** —— 🤗 [**DuplexJev-32B-Para**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) 和
  [**DuplexJev-4B-Para**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)：一个模型里副语言能力最全的版本。在话轮、性别、
  情绪、重音、停顿之外，还能听出**笑声、呼吸、咳嗽、叹气**（训练未见的真人语音上 AUC 0.70–0.85，之前 0.40–0.73），识别**说话方式**
  （耳语、大喊、快慢……），并像人一样粗略判断**是不是同一个人**。4B-Para 也是主语言最强的 4B（qa100 79，VoiceBench 57.8 / 41.8），
  现已成为 `duplexjev` 包（0.4）的默认模型。
- **2026-10-08** —— 🤗 [**DuplexJev-32B-Stress**](https://huggingface.co/adventists-ai/DuplexJev-32B-Stress)：在 DuplexJev-32B-Turn
  基础上**听得懂重音和停顿**。重读的是哪个词：中文留出句 92 / 英文 88；中性语音答“没有特别强调”99%；主语言、性别、情绪、话轮
  都在误差范围内持平。每种语音来源里都配了正负样本；“靠重音选意思”的题会拉低 VoiceBench，所以没有放进训练。同时发布
  🤗 [**DuplexJev-4B-Stress**](https://huggingface.co/adventists-ai/DuplexJev-4B-Stress)（重读词判断 95 / 95；VoiceBench 比 4B-Turn 低约 4 分）。
  [研究笔记](../research/2026-10-stress-understanding_zh.md)。
- **2026-10-05** —— 📄 论文已上 arXiv：[arXiv:2610.02638](https://arxiv.org/abs/2610.02638)。🤗 [**DuplexJev-32B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn)
  和 [**DuplexJev-4B-Turn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn) 发布：话轮升级版（连接器 + rank 16 的 LoRA，
  用公开话轮数据训练）。32B-Turn：Easy-Turn 95.3，CoDeTT 零样本 69.2 / 70.0，qa100、性别、情绪都略有提升。
  [研究笔记](../research/2026-10-turn-taking-lora_zh.md)。
- **2026-10-03** —— 🤗 [**DuplexJev-Gemma-31B**](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B) 发布，第一个非 Qwen 模型：
  Qwen3-ASR-0.6B 编码器 + 连接器 + Gemma-4-31B-it（合并了 LoRA）。vLLM 下 qa100 97、性别 94.4、情绪 90.6（尚未做话轮训练）。
  需要 [`duplexjev-vllm`](https://pypi.org/project/duplexjev-vllm/) ≥ 0.2.0。
- **2026-10-01** —— 🎧 [**在线体验和试用 API**](https://api.adventists.cn/duplexjev/)（DuplexJev-4B）以及
  [`duplexjev` 0.3](https://pypi.org/project/duplexjev/)：`duplexjev quick call.wav` 对任意音频打印一张默认判断表
  （话轮状态、该怎么做、打断/抢话、意图、情绪、性别、语言、转人工）。
- **2026-09-30** —— 🤗 [**DuplexJev-32B**](https://huggingface.co/adventists-ai/DuplexJev-32B) 发布，成为新的主推模型：
  Qwen3-ASR-0.6B 编码器 + 连接器 + Qwen3-VL-32B 的语言模型，合在一个仓库，用 vLLM 部署。总分 87.7，是目前最高的：
  qa100 90、Easy-Turn 79.1、性别 89.4、情绪 90.6；口语 VoiceBench MMSU 70.8（之前的 32B 连接器为 56–60）。DuplexJev-4B 用同一配方更新（总分 75.9 → 78.6）。连接器 checkpoint 移到了[单独的页面](connectors_zh.md)。
- **2026-09-28** —— 新开 [研究笔记](../research) 专区。第一篇：[跨层融合对语音连接器有没有用？](../research/2026-09-connector-a-vs-b_zh.md)
- **2026-09-28** —— 第一个完整模型 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) 和 vLLM 插件
  [`duplexjev-vllm`](https://pypi.org/project/duplexjev-vllm/) 发布。


即将推出：开源全双工 Jev 对话流水线（2026-10-20 前后）。商用授权与托管 Speech-to-Decision API：jiejin@adventists.ai。
