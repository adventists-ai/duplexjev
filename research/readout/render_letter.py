"""协议 S：把 NanoJev 题渲染成「只回一个字母」的 suffix。

硬规则（来自 NanoJev TYPESAFE_CONTRACT + 本线 plan）：

- transport qid / item id 不得出现在模型输入里
- option 名和描述都是语义输入，字母只是闭集槽
- 字母↔option 由 item.id + qid + seed 确定性置换，排除位置捷径
- 每个字母必须是 tokenizer 里恰好 1 个 token，且互不相同
"""

from __future__ import annotations

import hashlib
import re
import string
from typing import Any

from .contract import canonical_state_json, validate_question

AUDIO_PH = "<|audio|>"
CHOICE_LETTERS = list(string.ascii_uppercase)  # A..Z，实际最多 8
SCORE_DIGITS = list("123456789")
BOOL_LETTERS = ("N", "Y")  # false, true 的默认槽；仍会随 seed 对调


def _rng_indices(n: int, salt: str) -> list[int]:
    h = hashlib.sha256(salt.encode("utf-8")).digest()
    # 简单 Fisher–Yates，字节流当随机源，不依赖 random 全局状态
    idx = list(range(n))
    pos = 0
    for i in range(n - 1, 0, -1):
        pos = (pos + 1) % len(h)
        j = h[pos] % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
        h = hashlib.sha256(h).digest()
    return idx


def assign_slots(option_ids: list[str], salt: str, alphabet: list[str]) -> dict[str, str]:
    if len(option_ids) > len(alphabet):
        raise ValueError(f"选项数 {len(option_ids)} 超过字母表 {len(alphabet)}")
    order = _rng_indices(len(option_ids), salt)
    shuffled = [option_ids[i] for i in order]
    return {alphabet[i]: shuffled[i] for i in range(len(shuffled))}


def invert(slot_to_opt: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in slot_to_opt.items()}


def gate_single_tokens(tokenizer, tokens: list[str], loc: str) -> dict[str, int]:
    """启动闸门：每个标签恰好 1 token，且 id 互异。"""
    ids = {}
    for t in tokens:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"{loc}: 标签 {t!r} 不是单 token，encode={enc}")
        ids[t] = int(enc[0])
    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"{loc}: 标签首 token 有碰撞: {ids}")
    return ids


def option_ids_of(question: dict) -> list[str]:
    typ = question["type"]
    if typ == "choice":
        return list(question["criteria"])
    if typ == "boolean":
        return ["false", "true"]
    return [str(i) for i in range(len(question["criteria"]))]


def option_text(question: dict, oid: str) -> str:
    typ = question["type"]
    if typ == "choice":
        return question["criteria"][oid]
    if typ == "boolean":
        crit = question.get("criteria") or {}
        default = {
            "false": "The proposition is false.",
            "true": "The proposition is true.",
        }
        return crit.get(oid, default[oid])
    return question["criteria"][int(oid)]


def render_question_suffix(question: dict, slot_to_opt: dict[str, str]) -> str:
    """不含 state、不含 qid。音频占位符由外层拼。"""
    validate_question("q", question, "render")
    lines = [
        f"Question type: {question['type']}",
        "Question:",
        question["instructions"].strip(),
    ]
    for slot, oid in slot_to_opt.items():
        lines.append(f"{slot} = {oid} — {option_text(question, oid)}")
    if question["type"] == "score":
        lines.append("只输出一个数字。")
    elif question["type"] == "boolean":
        lines.append("只输出一个字母。")
    else:
        lines.append("只输出一个字母。")
    return "\n".join(lines)


def render_user_text(state_obj: Any, question: dict, slot_to_opt: dict[str, str], with_audio: bool) -> str:
    """完整 user 文本。qid 不出现。"""
    body = [
        f"State:\n{canonical_state_json(state_obj)}",
        render_question_suffix(question, slot_to_opt),
    ]
    text = "\n\n".join(body)
    if with_audio:
        return f"{AUDIO_PH}\n{text}"
    return text


def contains_forbidden(text: str, *needles: str) -> list[str]:
    """短运输 id（如 qid=`q`）按完整 token 查，避免撞上 required 里的字母。"""
    hit = []
    for n in needles:
        if not n:
            continue
        if len(n) <= 2:
            pat = re.compile(
                rf"(?<![A-Za-z0-9_\u4e00-\u9fff]){re.escape(n)}(?![A-Za-z0-9_\u4e00-\u9fff])"
            )
            if pat.search(text):
                hit.append(n)
        elif n in text:
            hit.append(n)
    return hit


def gold_slot(question: dict, gold: Any, slot_to_opt: dict[str, str]) -> str:
    if question["type"] == "boolean":
        if gold in (True, "true"):
            oid = "true"
        elif gold in (False, "false"):
            oid = "false"
        else:
            raise ValueError(f"boolean gold 非法: {gold!r}")
    elif question["type"] == "score":
        oid = str(int(gold))
    else:
        oid = str(gold)
    inv = invert(slot_to_opt)
    if oid not in inv:
        raise ValueError(f"gold {oid!r} 不在当前槽位映射 {slot_to_opt}")
    return inv[oid]


def slots_for_question(item_id: str, qid: str, question: dict, seed: int = 0) -> dict[str, str]:
    oids = option_ids_of(question)
    salt = f"{item_id}\t{qid}\t{seed}"
    if question["type"] == "score":
        alphabet = SCORE_DIGITS
    elif question["type"] == "boolean":
        alphabet = list(BOOL_LETTERS)
    else:
        alphabet = CHOICE_LETTERS[:8]
    return assign_slots(oids, salt, alphabet)


def assert_qid_absent(text: str, item_id: str, qid: str) -> None:
    hit = contains_forbidden(text, item_id, qid)
    if hit:
        raise RuntimeError(f"模型输入泄漏了运输 id: {hit}；文本=\n{text}")
