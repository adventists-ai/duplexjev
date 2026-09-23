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
  🤗 模型权重（即将发布） &nbsp;|&nbsp;
  📊 <a href="data/qa100">qa100 测试集</a> &nbsp;|&nbsp;
  📝 <a href="#引用">引用</a>
</p>

---

**DuplexJev** 把一个冻结的语音大模型变成全双工语音智能体的**类型化决策引擎**（产品名：**Speech-to-Decision**）。
现成语音识别编码器的隐状态经投影送入冻结的大模型；运行时声明的每一个问题（*用户说完了吗？先放哪句垫话？流程走到哪一步？*）
都被读成**一个 token 上的闭集概率分布**——不做语音转写的解码，也不做文本解码。同一通电话、甚至多通电话里的许多问题，共用一次前向计算。

> **论文：** *Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents*（ICASSP 2027 投稿）。
> **状态：** 研究版本整理中。`research/` 是论文实验所用的原始代码；整理好的可安装版本和模型权重随后发布。

## 亮点

- **0 步解码。** 在一张 H200 上，对一句话做 10 个判断只要 **92 毫秒**；同一引擎上“先转写、再让大模型生成 JSON”需要约 **2.0 秒**。
- **对话级延迟预算下的容量。** 0.5 秒预算内，级联方案一个事件都完不成；读出方案单卡每秒可处理 17–20 个事件。
- **长上下文不再成倍增加成本。** 精确的前缀共享：1.5k token 上下文上的 100 个问题便宜 7 倍，5k 上下文上的 50 个问题便宜 20 倍。
- **模块化。** 任意语音编码器 + 一个小投影器 + 任意冻结大模型；对编码器中间层做交叉注意力融合，可以把副语言信息带进来。

## 工作原理

```
音频 ──► 冻结的 ASR 编码器（Qwen3-ASR-0.6B，12.5 Hz）
            ├─ B：只用最后一层 h18 ─────────────────────────────┐
            └─ A：交叉注意力融合  Q=h18, K=h14, V=h9 ───────────┤ （零初始化，残差）
                                                              ▼
                        投影器（两帧拼一 → 6.25 token/秒，MLP → 大模型维度）
                                                              ▼
状态 + N 个类型化问题 ──► 冻结的大模型（Qwen3-32B），一次前向
                                                              ▼
            p(A|q1) … p(D|q1),  …,  p(A|qN) … p(D|qN)      —— 0 步解码
```

* **类型化单 token 读出。** 每个问题把选项放在随机排列的字母（A、B……）下，答案是下一个 token 在这些字母上的 softmax。输出永远合法，`max p` 就是置信度。
* **前缀共享。** 共享前缀（模板、对话上下文、音频）只编码一次；所有问题的后缀拼进同一行，用块对角 4-D 掩码隔开，位置编码从前缀长度重新开始。KV 缓存只需 `P + ΣL_i` 个位置，而不是 `N(P + L)`。答案与逐题单跑一致（仅有 bf16 精度噪声）。
* **模块化。** 任意语音编码器、一个小的可训练投影器（3990 万参数；A 版本另加 320 万融合参数）、任意接受嵌入输入的冻结大模型。编码器和大模型都不训练。

## 实测结果（1× H200，bf16，Qwen3-32B）

每个事件 10 个判断，30 段真实语音（2–12 秒），两种方法使用同一个 vLLM 引擎：

| 上下文 | 方法 | 解码步数 | 单个事件 | 0.25 秒内每秒事件数 | 0.5 秒 | 2 秒 |
|---|---|---:|---:|---:|---:|---:|
| 无 | 转写 → 大模型生成 JSON | 12.5 + 86 | 1,567 ms（另加转写 411 ms） | 0 | 0 | 16.9 |
| 无 | 单 token 读出 | 0 | 92 ms | 17.3 | 20.4 | 33.5 |
| 1.5k token | 转写 → 大模型生成 JSON | 12.5 + 87 | 1,598 ms（另加转写） | 0 | 0 | 8.4 |
| 1.5k token | 单 token 读出 | 0 | 144 ms | 9.7 | 12.0 | 18.4 |

前缀共享（单行打包）与逐题调用相比：1.5k token 上下文上的 100 个问题便宜 7 倍，5k 上下文上的 50 个问题便宜 20 倍；
在各种上下文长度下 40/40 答案一致。准确率表格（qa100、浙大性别基准、浙大主语言基准）将随最终 checkpoint 补上。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `research/readout/` | 类型化单 token 读出、选项字母渲染、决策协议、模型加载 |
| `research/encoder/` | 带交叉注意力融合的 Qwen3-ASR 编码器（A 版本）及 A/B 配置 |
| `research/cost/` | 前缀共享实验（M1–M3）、级联基线、同引擎延迟与延迟预算下的容量 |
| `research/eval/` | qa100 / 性别 / 浙大基准评测脚本 |
| `research/data_pipeline/` | Ultravox v0.6 训练混合数据的 100 包切分、校验、Qwen3-32B 续写库 |
| `research/train_configs/` | A、B 两个版本的训练配置（拼帧数 2） |
| `research/demo/` | 用发布的 checkpoint 生成项目主页示例 |
| `data/qa100/` | **qa100** 中英双语语音选择题测试集（CC-BY-4.0） |
| `docs/` | 项目主页（GitHub Pages） |

`research/` 按论文实验时的原样发布，仍含我们集群上的绝对路径，见 [research/PATHS.md](research/PATHS.md)。

## qa100

100 道四选一题目：中文 50 题 + 英文 50 题，每种语言各 25 道逻辑题、25 道知识题，正确答案在 A–D 上均衡分布。
题干用 VoxCPM2 合成语音，并用独立的语音识别模型逐条核对（归一化后完全一致，否则重新合成）；选项以文字给出。
三种条件使用同一个提示词：*音频*、*文本*（标准转写，上限）、*只给选项*（下限）。详见 [data/qa100/DATASHEET.md](data/qa100/DATASHEET.md)。

## 许可证

代码与模型权重：Apache-2.0（[LICENSE](LICENSE)）。qa100：CC-BY-4.0。第三方组件和数据集保留各自的许可证，见 [NOTICE](NOTICE)。

## 引用

```bibtex
@misc{jin2026duplexjev,
  title  = {Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents},
  author = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Pang, Zhikun and Zhang, Xiaowen},
  year   = {2026},
  note   = {ICASSP 2027 submission}
}
```

## 致谢

本项目基于 [Ultravox](https://github.com/fixie-ai/ultravox)（训练代码与投影器设计）、[Qwen3](https://github.com/QwenLM/Qwen3) 与 Qwen3-ASR，
以及浙江大学的 [audio-gender-benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark)。类型化单 token 决策沿用了 Jev 所推动的 “System-One” 思路。
