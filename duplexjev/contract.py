"""NanoJev 形状的请求校验 + 本线 100 题条目外壳。

校验规则对齐 TianyuCodings/NanoJev scripts/predict_toy_decisions.py
的 validate_request（2026-09-21 核过）：

- 载荷必须是 {"states": [...]}
- 每个 state 只含 id / state / questions
- 题型 choice | boolean | score
- Choice criteria：2–255 个 option_id → 非空描述
- Boolean criteria：可选，只能有 false/true
- Score criteria：2–10 条有序描述
- transport id / qid 不得当作模型输入（由 renderer 保证，这里只校验存在）

本线在 NanoJev state 外包一层音频/因果/gold 字段，见 validate_item。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "jev-qwen-b0-v1"
FAMILIES = ("speech_state", "policy_action", "noul", "sanity", "content", "gender")
QUESTION_TYPES = ("choice", "boolean", "score")


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def canonical_state_json(state: Any) -> str:
    """S_t 的稳定序列化。不用 Python str(dict)（NanoJev 自己记了这个坑）。"""
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state 字符串不得为空")
        return state
    if isinstance(state, (dict, list)):
        if state == {} or state == []:
            raise ValueError("state 对象/数组不得为空")
        return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raise ValueError("state 须为字符串、JSON 对象或数组")


def validate_question(qid: str, question: Any, loc: str) -> dict:
    if not nonempty_text(qid) or not isinstance(question, dict):
        raise ValueError(f"{loc} question ID 必须是非空字符串，内容必须为对象")
    extra = set(question) - {"type", "instructions", "criteria"}
    if extra:
        raise ValueError(f"{loc}:{qid} 含不支持的 question 字段: {sorted(extra)}")
    typ = question.get("type")
    if typ == "noul":
        typ = "boolean"
        question = dict(question)
        question["type"] = "boolean"
    if typ not in QUESTION_TYPES or not nonempty_text(question.get("instructions")):
        raise ValueError(f"{loc}:{qid} 题型或 instructions 无效")
    if typ == "boolean":
        if "criteria" in question:
            criteria = question["criteria"]
            if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                raise ValueError("Boolean criteria 只能是含 false 和/或 true 键的对象")
            if not all(nonempty_text(v) for v in criteria.values()):
                raise ValueError("Boolean criterion 必须是非空字符串")
    elif typ == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError("Choice criteria 必须是含 2–255 项的对象")
        if not all(nonempty_text(k) and nonempty_text(v) for k, v in criteria.items()):
            raise ValueError("Choice 候选 ID 和语义描述必须是非空字符串")
    else:
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score criteria 必须是含 2–10 项的有序数组")
        if not all(nonempty_text(v) for v in criteria):
            raise ValueError("Score 等级描述必须是非空字符串")
    return question


def validate_state(state: Any) -> dict:
    if not isinstance(state, dict) or set(state) != {"id", "state", "questions"}:
        raise ValueError("每个 state 项必须只包含 id、state、questions")
    if not nonempty_text(state["id"]):
        raise ValueError("state id 必须是非空字符串")
    canonical_state_json(state["state"])
    questions = state["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions 必须是非空对象")
    for qid, q in questions.items():
        validate_question(qid, q, state["id"])
    return state


def validate_request(payload: Any) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"states"}:
        raise ValueError('输入必须为且仅为 {"states": [...]}')
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("states 必须是非空数组")
    seen = set()
    out = []
    for st in states:
        st = validate_state(st)
        if st["id"] in seen:
            raise ValueError(f"重复的 state id: {st['id']}")
        seen.add(st["id"])
        out.append(st)
    return out


def _gold_ok(question: dict, gold: Any) -> bool:
    typ = question["type"]
    if typ == "choice":
        return isinstance(gold, str) and gold in question["criteria"]
    if typ == "boolean":
        return gold in (True, False, "true", "false")
    if isinstance(gold, int) and not isinstance(gold, bool):
        return 0 <= gold < len(question["criteria"])
    if isinstance(gold, str) and gold.isdigit():
        return 0 <= int(gold) < len(question["criteria"])
    return False


def validate_item(item: Any) -> dict:
    """100 题 / 脚手架条目。state 字段本身必须是合法 NanoJev state。"""
    if not isinstance(item, dict):
        raise ValueError("item 必须是对象")
    need = {
        "id",
        "family",
        "source",
        "split",
        "audio",
        "sample_rate",
        "t_ms",
        "gold",
        "soft_dist",
        "state",
    }
    missing = need - set(item)
    extra = set(item) - need - {"note", "text", "task", "lang", "bench_id"}
    if missing:
        raise ValueError(f"{item.get('id','?')} 缺字段: {sorted(missing)}")
    if extra:
        raise ValueError(f"{item.get('id','?')} 多了未登记字段: {sorted(extra)}")
    if not nonempty_text(item["id"]):
        raise ValueError("item.id 必须是非空字符串")
    if item["family"] not in FAMILIES:
        raise ValueError(f"{item['id']} family 必须是 {FAMILIES}")
    if not nonempty_text(item["audio"]):
        raise ValueError(f"{item['id']} audio 路径不得为空")
    if item["sample_rate"] != 16000:
        raise ValueError(f"{item['id']} sample_rate 必须是 16000")
    if type(item["t_ms"]) is not int or item["t_ms"] < 0:
        raise ValueError(f"{item['id']} t_ms 必须是非负整数")
    st = validate_state(item["state"])
    if st["id"] != item["id"]:
        raise ValueError(f"{item['id']} state.id 必须与 item.id 相同")
    if len(st["questions"]) != 1:
        raise ValueError(f"{item['id']} 第一版每条只允许一道题")
    qid, q = next(iter(st["questions"].items()))
    if not _gold_ok(q, item["gold"]):
        raise ValueError(f"{item['id']} gold={item['gold']!r} 不在题 {qid} 的答案空间里")
    if item["soft_dist"] is not None:
        if not isinstance(item["soft_dist"], dict) or not item["soft_dist"]:
            raise ValueError(f"{item['id']} soft_dist 必须是非空对象或 null")
    return item


def validate_pack(doc: Any) -> dict:
    if not isinstance(doc, dict) or "items" not in doc:
        raise ValueError("题包必须是含 items 数组的对象")
    items = doc["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("items 必须是非空数组")
    seen = set()
    out = []
    for it in items:
        it = validate_item(it)
        if it["id"] in seen:
            raise ValueError(f"重复的 item id: {it['id']}")
        seen.add(it["id"])
        out.append(it)
    return {"schema_version": doc.get("schema_version", SCHEMA_VERSION), "items": out}


def as_nanojev_request(items: list[dict]) -> dict:
    return {"states": [it["state"] for it in items]}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(blob)


def empty_execution() -> dict:
    """与 NanoJev 响应对齐的延迟/前向计数。未跑推理时全是占位。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "states": 0,
        "questions": 0,
        "candidate_paths": 0,
        "forward_passes": 0,
        "autoregressive_decode_steps": 0,
        "prefix_sharing": False,
        "temperature": {"value": 1.0, "fitted_by_this_command": False},
        "t_audio_encode_ms": None,
        "t_prefill_shared_ms": None,
        "t_suffix_batch_ms": None,
        "t_read_ms": None,
        "questions_per_s": None,
        "peak_mem_bytes": None,
    }
