"""The default decision table: ten judgments a voice agent needs about every user turn.

Turn state, gender, emotion, non-verbal sounds and speaking style use the wordings the DuplexJev models were trained
on (the last two need a Para model such as DuplexJev-4B-Para, the default); the others are read zero-shot by the LLM
from what it hears. Edit a copy of the table to ask your own questions:

    from duplexjev import default_table, Question
    table = default_table("zh") + [Question("product", "用户在问哪个产品？", ["手机", "电脑", "其他"], lang="zh")]
"""
from __future__ import annotations

from .question import Question

_EN = [
    ("turn", "Has the user finished speaking?",
     ["finished, the assistant can reply", "not finished, still talking", "just a backchannel, not taking the turn",
      "hesitating or asking to wait"]),
    ("reply", "What should the assistant do now?", ["reply now", "keep listening", "give a short acknowledgement"]),
    ("barge_in", "Is the user trying to interrupt or take over the conversation?", ["yes", "no"]),
    ("intent", "What is the user doing?",
     ["asking a question", "making a request", "chatting", "complaining", "confirming or agreeing", "declining or refusing"]),
    ("emotion", "What is the speaker's emotional state?", ["neutral", "happy", "angry", "sad"]),
    ("gender", "What is the perceived gender of the speaker?", ["female", "male"]),
    ("language", "Which language is the user speaking?", ["Chinese", "English", "other"]),
    ("human", "Does the user want to talk to a human agent?", ["yes", "no"]),
    ("sound", "Besides speech, which sound can be heard in this audio?",
     ["laughter", "breathing", "coughing", "a sigh", "None of these"]),
    ("style", "How is the speaker talking?",
     ["whispering", "speaking very loudly / shouting", "a normal speaking voice"]),
]

_ZH = [
    ("turn", "用户现在处于什么话轮状态？",
     ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"]),
    ("reply", "助手现在应该怎么做？", ["马上回应", "继续听", "简单应一声"]),
    ("barge_in", "用户是不是想打断或者抢话？", ["是", "不是"]),
    ("intent", "用户在做什么？", ["提问", "提要求或下指令", "闲聊", "抱怨或投诉", "确认或同意", "拒绝或否定"]),
    ("emotion", "说话人当时的情绪状态是？", ["中性", "高兴", "生气", "伤心"]),
    ("gender", "说话人的性别是？", ["男性", "女性"]),
    ("language", "用户说的是哪种语言？", ["中文", "英文", "其他"]),
    ("human", "用户是不是想转人工？", ["是", "不是"]),
    ("sound", "除了说话，这段音频里还有哪种声音？", ["笑声", "呼吸声", "咳嗽", "叹气", "都没有"]),
    ("style", "说话人是用什么方式在说话？", ["小声耳语", "大声喊着说", "正常音量说话"]),
]

LABELS = {
    "en": {"turn": "turn state", "reply": "what to do", "barge_in": "barge-in", "intent": "intent",
           "emotion": "emotion", "gender": "gender", "language": "language", "human": "wants a human",
           "sound": "non-verbal sound", "style": "speaking style"},
    "zh": {"turn": "话轮状态", "reply": "该怎么做", "barge_in": "打断/抢话", "intent": "意图",
           "emotion": "情绪", "gender": "性别", "language": "语言", "human": "转人工",
           "sound": "非语言声音", "style": "说话方式"},
}


def default_table(lang: str = "en") -> list[Question]:
    """The ten default questions in ``"en"`` or ``"zh"`` (ask in the language of the clip)."""
    rows = {"en": _EN, "zh": _ZH}[lang]
    return [Question(qid, text, opts, lang=lang) for qid, text, opts in rows]


def format_table(answers: dict, lang: str = "en", ms: float | None = None) -> str:
    """Render ``{id: {answer, confidence, probs}}`` as a plain-text table."""
    names = LABELS.get(lang, {})
    rows = [(names.get(k, k), v["answer"], f"{v['confidence']:.2f}") for k, v in answers.items()]
    head = ("判断", "答案", "置信度") if lang == "zh" else ("decision", "answer", "confidence")
    w0 = max(_width(r[0]) for r in rows + [head])
    w1 = max(_width(r[1]) for r in rows + [head])
    line = lambda a, b, c: f"{_pad(a, w0)}  {_pad(b, w1)}  {c}"  # noqa: E731
    out = [line(*head), "-" * (w0 + w1 + 14)] + [line(*r) for r in rows]
    if ms is not None:
        out.append(("{n} 个判断，{ms:.0f} ms" if lang == "zh" else "{n} decisions in {ms:.0f} ms").format(n=len(rows), ms=ms))
    return "\n".join(out)


def _width(s: str) -> int:  # CJK characters take two columns in a terminal
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)


def _pad(s: str, w: int) -> str:
    return s + " " * (w - _width(s))
