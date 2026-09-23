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
  📊 <a href="https://huggingface.co/datasets/adventists-ai/qa100">qa100</a> &nbsp;|&nbsp;
  📝 <a href="#9-引用">引用</a>
</p>

---

## 1. 简介

**DuplexJev** 把一个冻结的语音大模型变成面向全双工语音智能体的**类型化判断引擎**（产品名 **Speech-to-Decision**）。
现成 ASR 编码器的隐状态经投影送入冻结的大模型；运行时声明的每个问题——*用户说完了吗？先说哪句垫话？流程走到哪一步？*——
都读成**单个 token 上的闭集概率分布**：不做 ASR 解码，也不做文本解码。同一通电话的多个问题、乃至多通电话的问题，
可以在同一次前向计算里一起完成。

论文：*Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents*（已投 ICASSP 2027）。

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
- **前缀共享。** 模板、对话上下文和音频只编码一次；所有问题后缀打包进同一行，用块对角 4-D 掩码隔开，位置编号从前缀长度重新开始。
  KV 缓存只占 `P + ΣL_i` 个位置，而不是 `N(P + L)`；答案与逐个运行一致（仅 bf16 数值误差）。
- **模块化。** 任意 ASR 编码器 + 小型可训练投影器（3990 万参数；A 版融合模块另加 320 万）+ 任意接受嵌入输入的冻结大模型。

## 2. 最新动态

- **2026-09** —— 论文投稿 ICASSP 2027；在 PyPI 发布 [`duplexjev`](https://pypi.org/project/duplexjev/) 包（批量判断器、节拍服务）、研究代码、延迟基准和项目主页。
- **2026-10-10（计划）** —— Speech-to-Decision 商用 API 上线。
- **2026-10-15（计划）** —— 开源推理管线与模型权重。

## 3. 模型下载

每个权重都是**专门连接一对模型的适配器**：只包含训练好的投影器（A 版另含融合模块），用来把下表中冻结的 ASR 编码器
接到冻结的大模型上。换成别的编码器或大模型（包括同系列的其他尺寸）都不能用，需要重新训练适配器。

| 模型 | ASR 编码器（冻结） | 大模型（冻结） | 编码器读出 | 可训练参数 | 下载 |
|---|---|---|---|---:|---|
| DuplexJev-A | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 交叉注意力融合（h18 / h14 / h9） | 43.1 M | 🤗 `DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B` (即将发布) |
| DuplexJev-B | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 最后一层（h18） | 39.9 M | 🤗 `DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B` (即将发布) |

编码器和大模型请从原仓库下载，再加载适配器。适配器仓库统一命名为 `DuplexJev-<版本>-<ASR 编码器>-<大模型>`，以后接其他模型对时也按此命名。

## 4. 快速上手

```bash
pip install "duplexjev[all]"      # PyPI；只用文本模型可 pip install duplexjev
```

**任意开源大模型，输入转写文字**（不需要语音模型）：

```python
from duplexjev import Decider, Question

d = Decider.from_pretrained("Qwen/Qwen3-8B")
qs = [Question("turn", "用户这句话说完了吗？", ["说完了", "没说完"], lang="zh"),
      Question("intent", "用户想做什么？", ["空调", "媒体", "导航", "电话"], lang="zh")]
d.decide(["帮我把空调打开", "导航到"], qs)
# [{'turn': {'answer': '说完了', 'confidence': 0.97, 'probs': {...}}, 'intent': {...}}, {...}]
```

**语音模型，直接输入音频**（Ultravox 格式；DuplexJev 适配器同样加载）：

```python
d = Decider.from_pretrained("fixie-ai/ultravox-v0_6-qwen-3-32b")
d.decide(["call_017.wav", "call_018.wav"], qs)          # 所有通话、所有问题一次前向
```

**批量服务：** 同一个节拍内到达的所有请求合成一次前向完成。

```bash
duplexjev serve --model fixie-ai/ultravox-v0_6-qwen-3-32b --tick-ms 160
curl -s localhost:8000/v1/decide -H 'content-type: application/json' \
  -d '{"text": "打电话给", "lang": "zh", "questions": [{"id": "turn", "text": "说完了吗？", "options": ["说完了", "没说完"], "lang": "zh"}]}'
```

包的保证（`tests/` 有单元测试；已在 A800 上用 Qwen3-32B 和 Ultravox v0.6 实测）：

- **0 步解码。** 答案取下一个 token 在选项字母上的 softmax；选项用打乱的字母标注（`n_perm` 可对多种顺序取平均）。
- **精确的前缀共享**（默认 `mode="packed"`）：每个条目的上下文和音频只编码一次，所有问题打包进同一行、用块对角掩码隔开。Qwen3-32B 在 fp32 下，打包与逐题运行的概率差最大 2e-5；bf16 下平均差 0.001–0.002，100 题中答案最多改变 1 题。
- **批次无关。** 一个条目的答案不受同批其他条目影响（fp32 最大差 1e-5）。语音场景下这需要把每段音频对齐到整数个音频 token，包里已处理（否则 Ultravox 批量推理时，短音频最后一个音频 token 会混入填充）。bf16 下 GPU 内核随批形状变化，单独运行与合批运行在 100 题中有 1 题答案不同（文本和语音都是）。
- **准确率核对。** 经本包跑 qa100：Ultravox v0.6（音频）0.88–0.90，Qwen3-32B（转写文字）0.92。
- **当前速度。** 纯 PyTorch、单张 A800、bf16：64 路通话每路 8 个问题，一次 9 秒（每路约 140 ms）；服务端 48 个并发请求在一个节拍内答完。论文里的延迟数字用的是 vLLM 引擎；包的 vLLM 后端在路线图上。

语音模型需要 transformers 4.51–4.55（安装 `speech` 扩展会自动满足）；纯文本模型也支持更新的版本。

更多见 [`examples/`](examples)：节拍批量计时、服务端客户端、语音快速上手。

## 5. 评测结果

**延迟与容量**（1 张 H200，bf16，Qwen3-32B；每个事件 10 个判断，30 条 2–12 秒真实语音，两种方法用同一个 vLLM 引擎）：

| 上下文 | 方法 | 解码步数 | 单个事件 | 0.25 秒内 事件/秒 | 0.5 秒 | 2 秒 |
|---|---|---:|---:|---:|---:|---:|
| 无 | ASR → 大模型生成 JSON | 12.5 + 86 | 1,567 ms（另加 ASR 411 ms） | 0 | 0 | 16.9 |
| 无 | 单 token 读出 | 0 | 92 ms | 17.3 | 20.4 | 33.5 |
| 1.5k token | ASR → 大模型生成 JSON | 12.5 + 87 | 1,598 ms（另加 ASR） | 0 | 0 | 8.4 |
| 1.5k token | 单 token 读出 | 0 | 144 ms | 9.7 | 12.0 | 18.4 |

**前缀共享**（单行打包 vs 逐个调用）：1.5k token 上下文上 100 个问题省 7 倍，5k token 上下文上 50 个问题省 20 倍；40/40 条答案完全一致。

**准确率**将在最终权重发布时补充，评测集如下，详见 [`evaluation/`](evaluation)。

| 评测集 | 任务 | 链接 |
|---|---|---|
| qa100 | 语音四选一，合成语音（中 + 英） | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| 浙大 audio-gender-benchmark · `gender` | 说话人性别（中 + 英，真人 + TTS） | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| 浙大 audio-gender-benchmark · `main_language` | 语音四选一，真人语音 | [GitHub](https://github.com/Vsky-morigen/audio-gender-benchmark) |

## 6. 训练

编码器 Qwen3-ASR-0.6B 与大模型 Qwen3-32B 均冻结，只训练投影器（及融合模块）。训练基于打过补丁的
[Ultravox](https://github.com/fixie-ai/ultravox)，数据为 Ultravox v0.6 配方（WenetSpeech、GigaSpeech、Common Voice、
CoVoST 2、People's Speech、LibriSpeech、MLS、MUSAN），切成 100 个互不重叠、各自保持官方配比的训练包。
配方、数据链接与脚本见 [`training/`](training)。

## 7. 仓库结构

| 路径 | 内容 |
|---|---|
| [`duplexjev/`](duplexjev) | 可安装的包：`Decider`、`Question`、命令行与节拍批量服务 |
| [`duplexjev/research/`](duplexjev/research) | 论文代码：读出、问题契约、带融合的编码器 |
| [`tests/`](tests) | 一致性与批次无关性测试 |
| [`examples/`](examples) | 快速上手、节拍批量计时、服务端客户端 |
| [`evaluation/`](evaluation) | 评测脚本，以及各评测集的获取方式 |
| [`training/`](training) | Ultravox 配置与补丁、数据配方、100 包切分、续写生成 |
| [`benchmarks/`](benchmarks) | 延迟、SLO 容量与前缀共享实验 |
| [`docs/`](docs) | 项目主页 |

论文代码里仍有我们集群上的路径，见 [docs/PATHS.md](docs/PATHS.md)。

## 8. 许可证

代码与模型权重：Apache-2.0（[LICENSE](LICENSE)）。qa100：CC-BY-4.0。第三方模型与数据集沿用各自的许可证，见 [NOTICE](NOTICE)。

## 9. 引用

```bibtex
@misc{jin2026duplexjev,
  title  = {Batched Speech Decisions Without Decoding: A Modular Decision Sidecar for Full-Duplex Spoken Agents},
  author = {Jin, Jie and Ma, Ziyin and Yin, Min and Chen, Jinyu and Pang, Zhikun and Zhang, Xiaowen},
  year   = {2026},
  note   = {ICASSP 2027 submission}
}
```

## 10. 致谢

本项目基于 [Ultravox](https://github.com/fixie-ai/ultravox)、[Qwen3](https://github.com/QwenLM/Qwen3) 与 Qwen3-ASR。
感谢浙大团队提供 [audio-gender-benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark)。
问题与反馈请提 [GitHub Issues](https://github.com/adventists-ai/duplexjev/issues)。
