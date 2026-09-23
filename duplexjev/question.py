"""Typed closed-set questions and how they are rendered into a prompt suffix."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

TEMPLATES = {
    "en": {
        "question": "Question: {q}",
        "options": "Options:",
        "instruction": "Answer with only the letter of the correct option.",
    },
    "zh": {
        "question": "问题：{q}",
        "options": "选项：",
        "instruction": "请只回答正确选项的字母。",
    },
}


@dataclass(frozen=True)
class Question:
    """One runtime-declared decision.

    Args:
        id: key used in the result (e.g. ``"turn"``).
        text: the question as the model should read it.
        options: 2–26 answer options; the result is a probability for each.
        lang: prompt language for the fixed words (``"en"`` or ``"zh"``).
        audio: for ``Decider.decide_batch``: the clip id this group is about, a list of ids, or ``"*"`` for all
            clips. Ignored by ``Decider.decide`` (one clip).
    """

    id: str
    text: str
    options: Sequence[str]
    lang: str = "en"
    audio: Any = None
    _options: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        opts = tuple(str(o) for o in self.options)
        if not self.id:
            raise ValueError("Question.id must be non-empty")
        if not self.text:
            raise ValueError(f"Question {self.id!r}: text must be non-empty")
        if not 2 <= len(opts) <= len(LETTERS):
            raise ValueError(f"Question {self.id!r}: needs 2..{len(LETTERS)} options, got {len(opts)}")
        if len(set(opts)) != len(opts):
            raise ValueError(f"Question {self.id!r}: options must be distinct")
        if self.lang not in TEMPLATES:
            raise ValueError(f"Question {self.id!r}: lang must be one of {sorted(TEMPLATES)}")
        if isinstance(self.audio, list):
            object.__setattr__(self, "audio", tuple(self.audio))
        object.__setattr__(self, "_options", opts)
        object.__setattr__(self, "options", opts)

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(id=d["id"], text=d["text"], options=d["options"], lang=d.get("lang", "en"), audio=d.get("audio"))

    def to_dict(self) -> dict:
        d = {"id": self.id, "text": self.text, "options": list(self.options), "lang": self.lang}
        if self.audio is not None:
            d["audio"] = list(self.audio) if isinstance(self.audio, tuple) else self.audio
        return d

    def for_audio(self, audio) -> "Question":
        """A copy of this option group bound to another clip id."""
        return Question(self.id, self.text, self.options, self.lang, audio)

    # ------------------------------------------------------------------ rendering
    def permutation(self, seed: int = 0) -> list[int]:
        """Deterministic option order for this question: letter i shows option perm[i].

        Options are shown under permuted letters so that no option is tied to a fixed position. The order depends
        only on (question id, options, seed), so results are reproducible and identical across batch layouts.
        """
        key = "\x1f".join([self.id, "\x1e".join(self.options), str(seed)]).encode()
        rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))
        perm = list(range(len(self.options)))
        rng.shuffle(perm)
        return perm

    def render(self, perm: Sequence[int]) -> str:
        t = TEMPLATES[self.lang]
        lines = [t["question"].format(q=self.text), "", t["options"]]
        lines += [f"{LETTERS[i]}. {self.options[j]}" for i, j in enumerate(perm)]
        lines += ["", t["instruction"]]
        return "\n".join(lines)

    def letters(self) -> str:
        return LETTERS[: len(self.options)]


def as_questions(qs, unique: bool = True) -> list[Question]:
    out = [q if isinstance(q, Question) else Question.from_dict(q) for q in qs]
    ids = [q.id for q in out]
    if unique and len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question ids: {ids}")
    return out
