# Research notes · 研究笔记

Experiments and design decisions behind DuplexJev that did not fit in the paper: what we tried, what worked, what did
not. Each note states its setup, numbers with their noise level, and limitations; checkpoints and scripts are linked
so the results can be checked.

DuplexJev 背后那些没写进论文的实验和取舍：试过什么、哪些有用、哪些没用。每篇笔记写清实验设置、带误差范围的数字和局限，
并附上 checkpoint 和脚本链接，方便复核。

| date | note | 中文 |
|---|---|---|
| 2026-10-10 | [Keeping a speech-decision LLM able to talk: conversation replay and per-row loss](2026-10-chat-replay.md) | [让语音决策模型继续“会聊天”](2026-10-chat-replay_zh.md) |
| 2026-10-09 | [One model, more of the voice: non-verbal sounds, speaking style and speaker on top of stress](2026-10-para-abilities.md) | [一个模型听出更多：非语言声音、说话方式和说话人](2026-10-para-abilities_zh.md) |
| 2026-10-08 | [Teaching a speech-decision LLM to hear stress: data, two gates, negatives and the "source shortcut"](2026-10-stress-understanding.md) | [让语音判断大模型听懂重音](2026-10-stress-understanding_zh.md) |
| 2026-10-06 | [Can a speech-decision LLM tell who is speaking? Shortcuts, probes and ten ablations](2026-10-speaker-identity.md) | [语音判断大模型能听出“是谁在说”吗？](2026-10-speaker-identity_zh.md) |
| 2026-10-05 | [A rank-16 LoRA makes a speech-decision LLM a competitive turn-taking detector](2026-10-turn-taking-lora.md) | [一个 rank 16 的 LoRA，让话轮能力追上专用检测器](2026-10-turn-taking-lora_zh.md) |
| 2026-09-28 | [Does cross-layer fusion help a speech connector? A vs. B on Qwen3-32B and Qwen2.5-72B](2026-09-connector-a-vs-b.md) | [跨层融合对语音连接器有没有用？](2026-09-connector-a-vs-b_zh.md) |
