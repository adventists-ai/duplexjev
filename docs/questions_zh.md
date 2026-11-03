# 写自己的判断题

*[English](questions.md) · [返回 README](../README_CN.md)*

DuplexJev 可以对一段音频回答任意选择题，一次前向就出结果。默认判断表（话轮状态、情绪、性别等）覆盖常见场景；本页讲怎么写自己的题。

## 请求与返回

试用 API（免 key；每个 IP 每分钟 30 次；单个音频不超过 4 MB；服务端不保存音频）：

```
POST https://api.adventists.cn/duplexjev/v1/decide
Content-Type: application/json

{
  "audio_b64": "<音频文件的 base64>",
  "format": "wav",
  "lang": "zh",
  "questions": [
    {"id": "laugh_pos", "text": "笑声出现在哪里？",
     "options": ["说话前", "说话后", "边说边笑", "没有笑声"], "lang": "zh"},
    {"id": "stress", "text": "这句话里哪个词被重读？",
     "options": ["先", "核对", "金额", "发出去", "没有明显重读"], "lang": "zh"}
  ]
}
```

```
{"answers": {
   "laugh_pos": {"answer": "没有笑声", "confidence": 0.99, "probs": {"说话前": 0.005, ...}},
   "stress":    {"answer": "先",       "confidence": 0.93, "probs": {...}}},
 "ms": 98.0, "model": "DuplexJev-32B-Para"}
```

- **每道题都要带 `"lang"`**（`"zh"` 或 `"en"`），并且用音频的语言提问。没带 `lang` 的题会按英文模板拼提示词。
- `probs` 是每个选项的概率。以后想自己定阈值的话，可以存下来。
- 同一段音频的题放在一次请求里问：共用同一段音频，只算一次调用。
- 每题 2–10 个选项。选项给模型看时会打乱顺序，你列的顺序不影响结果。
- 支持 wav、mp3、flac、ogg、m4a，采样率不限。

自己部署时请求格式一样：`duplexjev gateway --vllm http://localhost:8000/v1` 提供同样的 `/v1/decide`（见 [README](../README_CN.md#快速上手)）。

## 出题要点

1. **一定要有“没有”选项**：没有笑声、没有明显重读、没有停顿。没有这个选项，什么都没发生时模型也只能硬选一个。
2. **问“在哪里”时，选项直接用这句话里的词**：“咳嗽出现在哪个词后面？医生 / 发痒 / 晚上 / 没有咳嗽”。手上有转写的话，从转写里取词。
3. **“有没有”和“在哪里”合成一道题**：说话前 / 说话后 / 边说边笑 / 没有，一题就能答完。
4. **问法要短，一题只问一件事。** 在题目里加额外说明（如“不要把拟声词当成笑声”）往往适得其反。
5. **选项之间要互斥。** 可能同时成立的，拆成两道题。

## 常用模板

| 想知道什么 | 问法 | 选项 |
|---|---|---|
| 非语言声音及位置 | 笑声出现在哪里？（咳嗽、叹气等同理） | 说话前 / 说话后 / 边说边笑 / 没有笑声 |
| 是哪种非语言声音 | 除了说话，这段音频里还有哪种声音？ | 笑声 / 呼吸声 / 咳嗽 / 叹气 / 都没有 |
| 重音 | 这句话里哪个词被重读？ | 句中的词 + 没有明显重读 |
| 拖音 | 哪个字被拉长了？ | 句中的字 + 没有 |
| 停顿位置 | 在哪个词后面停顿了？ | 句中的词 + 没有停顿 |
| 指定位置的事件 | 咳嗽出现在哪个词后面？ | 句中的词 + 没有咳嗽 |
| 说话方式 | 说话人是用什么方式在说话？ | 小声耳语 / 大声喊着说 / 正常音量说话 |

话轮、情绪等内置判断，可以从默认判断表开始改：`GET https://api.adventists.cn/duplexjev/v1/table?lang=zh`。

## Python

用我们的包（`pip install duplexjev`）：

```python
from duplexjev.remote import ApiClient

dj = ApiClient("https://api.adventists.cn/duplexjev")      # 也可以换成自己的网关
dj.decide("clip.wav", [
    {"id": "cough", "text": "咳嗽出现在哪个词后面？",
     "options": ["医生", "发痒", "晚上", "没有咳嗽"], "lang": "zh"},
], lang="zh")
# {'cough': {'answer': '发痒', 'confidence': ..., 'probs': {...}}}
```

只用标准库：

```python
import base64, json, urllib.request

def decide(path, questions, lang="zh", url="https://api.adventists.cn/duplexjev"):
    body = {"audio_b64": base64.b64encode(open(path, "rb").read()).decode(), "format": "wav",
            "lang": lang, "questions": [dict(q, lang=lang) for q in questions]}
    req = urllib.request.Request(url + "/v1/decide", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["answers"]
```

试用 API 有限流，服务器在国内。批量跑或做实时 agent，建议自己部署（见 [README](../README_CN.md#快速上手)）。
