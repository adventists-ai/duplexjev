# DuplexJev 模型

[← 返回 README](../README_CN.md) · [English](models.md)

README 主推四个模型：**DuplexJev-32B-Para**、**DuplexJev-4B-Para**、**DuplexJev-4B-Para-LiveTurn** 和 **DuplexJev-32B-Vision**（预览）。
本页列出全部已发布模型、各版本成绩，以及我们自己各版本之间的对比。

## 全部已发布模型

| 模型 | 适用 | 基座 | 参数 | 显存 | qa100 | ZJU-ML | Easy-Turn | 性别 | 情绪 | 总分 | 许可 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ⭐ 🤗 [**DuplexJev-32B-Para**](https://huggingface.co/adventists-ai/DuplexJev-32B-Para) 🆕 v1.3 | 服务器端，能力最全（重音大幅提升，含口音），实时话轮最好，总分最高 | 同 32B-Turn（再加一个 rank 16 LoRA） | 33.0 B | 一张 80 GB 显卡 | 96 | 87 | 94.4\* | **93.8** | **92.1** | **92.7** | CC BY-NC 4.0 |
| ⭐ 🤗 [**DuplexJev-4B-Para**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para) 🆕 | 小显卡，能力最全，**也能正常聊天**（v1.1）；`duplexjev` 默认模型 | 同 4B-Turn（再加一个 rank 16 LoRA） | 4.2 B | 约 10 GB | 77 | **58** | 91.0\* | 89.2 | 90.0 | **82.5** | CC BY-NC 4.0 |
| 🤗 [**DuplexJev-4B-Para-LiveTurn**](https://huggingface.co/adventists-ai/DuplexJev-4B-Para-LiveTurn) 🆕 | 小显卡，给 LiveKit / Pipecat 语音 agent 做**实时话轮判断**（另加口音）；通用 4B 仍是 4B-Para | 同 4B-Para（加口音数据和停顿级话轮数据） | 4.2 B | 约 10 GB | 76 | 62 | 92.0\* | 89.5 | 90.5 | — | CC BY-NC 4.0 |
| 🤗 [DuplexJev-32B-Vision](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision)（预览） | 声音 + 画面一起判断 | 32B-Para + Qwen3-VL 视觉编码器 + 再加一个 LoRA | — | 一张 80 GB 显卡 | | | | | | | CC BY-NC 4.0 |
| 🤗 [DuplexJev-32B-Stress](https://huggingface.co/adventists-ai/DuplexJev-32B-Stress) 🆕 | 服务器端，加重音与停顿 | 同 32B-Turn（再加一个 rank 16 LoRA） | 33.0 B | 一张 80 GB 显卡 | **96** | 87 | 94.4\* | 92.1 | 91.4 | — | CC BY-NC 4.0 |
| 🤗 [DuplexJev-4B-Stress](https://huggingface.co/adventists-ai/DuplexJev-4B-Stress) 🆕 | 小显卡，加重音与停顿 | 同 4B-Turn（再加一个 rank 16 LoRA） | 4.2 B | 约 10 GB | 73 | 51 | 91.8\* | 89.1 | 91.4 | — | CC BY-NC 4.0 |

分数为论文口径的百分比（单 token 读出）；总分 = 主语言（qa100、ZJU-ML、Easy-Turn）与副语言（性别、情绪）的平均，
见[评测基准](../README_CN.md#评测基准)。\* 两个模型训练时都用了 Easy-Turn 的训练集（与测试集不重叠），所以 Easy-Turn 分数属于域内。两个模型在单张显卡上回答同一段音频的 10 个问题都在约 0.1–0.25 秒内。

**怎么选。** 想一个模型听全声音里的信息（非语言声音、说话方式、说话人、重音、情绪、性别、话轮）就用 **Para** 版：
80 GB 显卡用 `DuplexJev-32B-Para`，小显卡用 `DuplexJev-4B-Para`。实时话轮也是 `DuplexJev-32B-Para` v1.2 最好；小显卡上做实时话轮，用 `DuplexJev-4B-Para-LiveTurn`；
重音和停顿是主要需求，用 `DuplexJev-32B-Stress` / `4B-Stress`。所有模型的提示词和插件都一样。
32B 模型只用了 Qwen3-VL-32B 的语言模型部分，
输入是音频和文字。要同时输入语音和图像，用 [DuplexJev-32B-Vision](https://huggingface.co/adventists-ai/DuplexJev-32B-Vision)（研究预览）。

## Para 版新增能力（各版本）

**Para 版新增的能力**（AUC，0.5 为瞎猜；训练未见的真人语音：NonverbalTTS 测试集、第三方中文自然对话样本；说话人 EER 用 AISHELL-1 同性别配对）：

| | 笑 | 呼吸 | 咳嗽 | 叹气 | 对话中的笑 / 呼吸 | 说话方式（7 类） | 是否同一人 EER |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32B-Stress | 0.73 | 0.56 | 0.40 | 0.70 | 0.61 / 0.44 | 45% | — |
| **32B-Para** v1.2 | 0.87 | **0.81** | 0.84 | 0.75 | 0.80 / 0.56 | 79% | — |
| 32B-Para v1.1 | **0.88** | **0.81** | **0.85** | 0.80 | 0.83 / 0.65 | **82%** | ≈ v1.0（18%）² |
| 32B-Para v1.0 | 0.85 | 0.73 | 0.70 | 0.74 | 0.80 / 0.78 | 69% | 18% |
| **4B-Para** v1.1 | 0.81 | 0.77 | 0.77 | 0.71 | 0.85 / 0.74 | 67% | ≈ v1.0 (9%)¹ |
| 4B-Para v1.0 | 0.80 | 0.76 | 0.75 | 0.81 | 0.84 / 0.84 | 67% | 9% |

¹ v1.1 没有在 AISHELL-1 配对上重测；在 VoxCeleb1 困难配对上 EER 为 42.0%（v1.0 为 41.8%）。
² 32B-Para v1.1 没有在 AISHELL-1 配对上重测；在 VoxCeleb1 随机 / 困难配对上 EER 为 28.2% / 39.6%（v1.0 为 37.4% / 45.7%）。

代价很小，模型卡上都列了（32B-Para v1.2：情绪比 32B-Turn 低 1，英文重音、叹气、说话人配对比 v1.1 略低；4B-Para：ZJU-ML 比 4B-Turn 低 2、Easy-Turn 低 1）。
说话人判断很粗略，像人一样，不能用来识别身份。

**DuplexJev-4B-Para v1.1（2026-10-10）** 加入了对话回放训练，同一份权重还能正常聊天。判断类成绩与 v1.0 相差 0.5–2.6 分（v1.0 仍可用 revision `v1.0` 下载），见[研究笔记](../research/2026-10-chat-replay_zh.md)。32B 版正在训练。

Stress 版增加了重音和停顿理解（哪个词被重读：中文留出句 32B 92 / 4B 95，见[研究笔记](../research/2026-10-stress-understanding_zh.md)）。32B-Stress 其余各项都与 32B-Turn 在误差内持平；**4B-Stress 用约 4 分 VoiceBench 换来这项能力**（42.0 / 38.2 对 46.4 / 41.7），所以小显卡默认仍推荐 4B-Para。

## DuplexJev-32B-Para：v1.3 与 v1.2 对比

题目和测试方法与[闭源模型对比](../README_CN.md#与闭源模型对比含实时旗舰)相同。v1.3 听重音大幅提升；字面中性时的情绪 v1.2 略好。
v1.2 仍可用 revision `v1.2` 下载（`vllm serve adventists-ai/DuplexJev-32B-Para --revision v1.2`）。

| 测试 | DuplexJev-32B-Para v1.3 | v1.2 |
| --- | ---: | ---: |
| **公开基准合计**（LISTEN 四项、StressTest、MSPB，1,128 题） | **50.9** | 46.8 |
| **LISTEN 纯音频四项平均** | **37.8** | 37.3 |
| LISTEN 字面中性 | 43.0 | **45.0** |
| ……只算训练没见过的语料（TESS、SAVEE、RAVDESS） | 29.9 | **34.6** |
| LISTEN 完全没有字 | **26.0** | 25.0 |
| LISTEN 字面是反讽 | **44.0** | 43.0 |
| LISTEN 字面与语气一致 | 38.0 | 36.0 |
| 情绪 ESD + CREMA-D† | **92.0** | 91.0 |
| 哪个词重读，中 / 英† | **95.5 / 96.0** | 93.5 / 91.4 |
| 由重音判断意思 DEBATE† | **93.0** | 90.5 |
| StressTest | 86.5 | 71.5 |
| 说话方式 EARS† | 79.5 | **80.0** |
| MSPB（陈述 / 疑问等） | **77.3** | 68.0 |

† 测试集与训练语料同源（说话人或句子不重叠）。v1.3 其他变化：由重音判断意思 90.5 → 93.0；情绪略升；性别和 Easy-Turn 约低 1 分。LISTEN 四项平均全集 38.9（v1.2：38.4）。

## 实时话轮检测：各版本成绩

LiveKit eot-bench（验证集）与 Smart Turn v3.2 测试集；测试方法见 [README](../README_CN.md#实时话轮检测livekit-eot-bench-与-pipecat-smart-turn)。

| 模型 | eot-bench 英文：最多等 300 / 600 ms 的误打断 | eot-bench 中文 | Smart Turn v3.2 测试集准确率 英 / 中 |
|---|---:|---:|---:|
| LiveKit Turn Detector v1 | 9.9 / 4.5% | 20.6 / 8.7% | – |
| **DuplexJev-32B-Para v1.2**，带助手上一句话（插件设置） | **20.0 / 9.4%** | **37.7 / 12.2%** | – |
| **DuplexJev-32B-Para v1.2**，只用音频 | 23.5 / 11.1% | 40.0 / 12.8% | **94.0 / 91.1**° |
| DuplexJev-32B-Turn，只用音频 | 29.8 / 13.9% | 44.9 / 14.5% | 65.2 / 84.6 |
| **DuplexJev-4B-Para-LiveTurn**（约 10 GB），带助手上一句话（插件设置） | **25.5 / 12.3%** | **38.8 / 12.0%** | **93.3 / 89.6**° |
| DuplexJev-4B-Para v1.1，只用音频 | 46.4 / 17.7% | 48.5 / 15.6% | 67.9 / 78.9 |
| Smart Turn v3.2 | 35.2 / 14.8% | 42.9 / 13.2% | 94.7 / 90.5 |

## 与 Qwen3-Omni 逐项对比（v1.2，题量更大）

两个模型（DuplexJev-32B-Para v1.2、Qwen3-Omni-30B-A3B-Instruct，bf16，单张 H200）用同一批题、同样的提示词，选项轮换取平均，
取概率最高的选项字母。

| 能力 | 测试 | DuplexJev-32B-Para | Qwen3-Omni-30B |
|---|---|---:|---:|
| 情绪四分类 | ESD + CREMA-D（400）† | **90.5** | 45.2 |
| 情绪：字面中性 | LISTEN 中性文本（400）· 只算训练没见过的语料 TESS、SAVEE、RAVDESS（213） | **50.7** · **39.9** | 32.8 · 27.2 |
| 情绪：完全没有字 | LISTEN 副语言（400） | **26.0** | 18.2 |
| 情绪：字面与语气一致 | LISTEN 情绪一致（IEMOCAP、MELD、MSP-Podcast、MOSEI、OMG；400） | 34.8 | **43.0** |
| 情绪：字面是反讽 | LISTEN 情绪不一致（MUStARD，400） | 39.2 | 40.5 |
| 哪个词重读，中 / 英 | 留出句（400 / 187）† | **91.5 / 91.4** | 62.5 / 73.3 |
| 由重音判断意思 | DEBATE 留出句（400）† | **88.5** | 43.8 |
| 由重音判断意思 | StressTest（218） | **72.0** | 69.7 |
| 说话方式（耳语、大喊、快、慢……） | EARS 留出说话人（220）† | **78.6** | 49.5 |
| 笑 / 呼吸 / 咳嗽 / 叹气，AUC | NonverbalTTS 测试集（400） | 0.93 / **0.86** / 0.86 / 0.85 | **0.96** / 0.65 / **0.87** / **0.87** |

- † 测试集与我们的训练语料同源（说话人或句子不重叠）；Qwen3-Omni 为零样本。LISTEN、StressTest、NonverbalTTS 是公开基准；
  LISTEN 的语料除 ESD、CREMA-D 外都不在我们的情绪训练数据里。
- 这里的 NonverbalTTS AUC 是两个模型都答过的 400 题上的结果，所以和上面全测试集那一行不同。
- LISTEN 最多有 16 种情绪标签，DuplexJev 只训练过四种（中性、高兴、生气、伤心）。
- 读出方式核对：让 Qwen3-Omni 改为自由生成文字作答（600 题），分数变化 −9 到 +4，差距不是单 token 读出造成的。
- 在中文自然对话（MagicData 样例：笑声、呼吸、重音）上，两者持平。字面本身就带情绪时（LISTEN 情绪一致），Qwen3-Omni 领先：它读字，DuplexJev 听声音。

## 归档模型

已被 Para 版取代；仍可下载，新项目不建议使用。

| 模型 | 替代为 | qa100 | ZJU-ML | Easy-Turn | 性别 | 情绪 | 总分 |
|---|---|---:|---:|---:|---:|---:|---:|
| 🤗 [DuplexJev-32B-Turn](https://huggingface.co/adventists-ai/DuplexJev-32B-Turn) | 32B-Para | 96 | 87 | 95.3\* | 91.5 | 91.1 | 92.2 |
| 🤗 [DuplexJev-4B-Turn](https://huggingface.co/adventists-ai/DuplexJev-4B-Turn) | 4B-Para / 4B-Para-LiveTurn | 74 | 52 | 92.5\* | 89.4 | 92.0 | 80.0 |
| 🤗 [DuplexJev-32B](https://huggingface.co/adventists-ai/DuplexJev-32B) | 32B-Para | | | | | | 87.7 |
| 🤗 [DuplexJev-4B](https://huggingface.co/adventists-ai/DuplexJev-4B) | 4B-Para | | | | | | 78.6 |

Turn 版在个别单项上还略高一点（32B-Turn：qa100 +1、情绪 +1；4B-Turn：情绪 +2），但没有 Para 版的任何新能力（非语言声音、
说话方式、说话人、重音、口音）。DuplexJev-32B / 4B 是话轮升级前的版本（论文模型）。

🤗 [DuplexJev-Gemma-31B](https://huggingface.co/adventists-ai/DuplexJev-Gemma-31B)（Gemma-4-31B-it + Qwen3-ASR-0.6B 编码器，
约 58 GB）：没有做话轮训练。按[第 4 节](../README_CN.md#快速上手)的提问格式：qa100 97、ZJU-ML 87、性别 94.4、情绪 90.6，Easy-Turn 68.0（零样本）；
对提问措辞敏感（论文口径措辞下性别 78.5、情绪 60.9）。

## 研究用检查点（checkpoint）

> **研究用 checkpoint。** 这些模型背后还有几十个连接器 checkpoint，覆盖其他编码器（MOSS、Whisper、SenseVoice）、
> 大模型（Qwen3 0.6B–32B、Falcon-H1、SmolLM3）和两种连接器类型，配合 `duplexjev` PyTorch 包使用。
> 它们放在单独的组织 🤗 [adventists-ai-research](https://huggingface.co/adventists-ai-research)（2026-10-14 迁移，旧链接自动跳转）。
> 排行榜、分项成绩和包的用法见 **[docs/connectors_zh.md](connectors_zh.md)**。
