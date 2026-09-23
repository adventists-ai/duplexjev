#!/usr/bin/env python3
"""无 GPU、无 tokenizer 的契约与渲染单测。

覆盖 NanoJev 文档里那几条隔离规则，以及本线 100 题外壳。
不加载 checkpoint，不访问网络。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from jev_qwen.contract import (  # noqa: E402
    as_nanojev_request,
    sha256_json,
    validate_item,
    validate_pack,
    validate_request,
)
from jev_qwen.render_letter import (  # noqa: E402
    assert_qid_absent,
    gold_slot,
    render_user_text,
    slots_for_question,
)
from jev_qwen.render_path import path_texts  # noqa: E402


def sample_item(**kw):
    item = {
        "id": "q001",
        "family": "speech_state",
        "source": "easyturn-heldout",
        "split": "probe100",
        "audio": "wavs/q001.wav",
        "sample_rate": 16000,
        "t_ms": 1840,
        "gold": "incomplete",
        "soft_dist": None,
        "state": {
            "id": "q001",
            "state": {"agent_speaking": False, "agent_played_ms": 0, "overlap": False},
            "questions": {
                "speech_state": {
                    "type": "choice",
                    "instructions": "根据截至当前时刻的用户发声，判断话轮状态。只根据已听到的证据。",
                    "criteria": {
                        "complete": "话已说完，可以接话",
                        "incomplete": "句子没说完，还在继续",
                        "backchannel": "简短附和，不抢话",
                        "wait": "还在组织，或明确要求先停一下",
                    },
                }
            },
        },
    }
    item.update(kw)
    return item


class ContractTests(unittest.TestCase):
    def test_sample_item_valid(self):
        it = validate_item(sample_item())
        self.assertEqual(it["gold"], "incomplete")
        req = as_nanojev_request([it])
        validate_request(req)

    def test_reject_qid_as_gold_space_miss(self):
        it = sample_item(gold="B")
        with self.assertRaises(ValueError):
            validate_item(it)

    def test_reject_two_questions(self):
        it = sample_item()
        it["state"]["questions"]["extra"] = {
            "type": "boolean",
            "instructions": "用户是否在抢话？",
        }
        with self.assertRaises(ValueError):
            validate_item(it)

    def test_pack_sha_stable(self):
        a = sha256_json(validate_pack({"items": [sample_item()]}))
        b = sha256_json(validate_pack({"items": [sample_item()]}))
        self.assertEqual(a, b)

    def test_boolean_and_score(self):
        b = sample_item(
            id="q002",
            family="noul",
            gold="true",
            state={
                "id": "q002",
                "state": {"agent_speaking": True, "agent_played_ms": 400, "overlap": True},
                "questions": {
                    "is_taking_floor": {
                        "type": "boolean",
                        "instructions": "用户是否在抢话？",
                        "criteria": {
                            "false": "没有夺取话语权",
                            "true": "正在夺取话语权",
                        },
                    }
                },
            },
        )
        validate_item(b)
        s = sample_item(
            id="q003",
            family="policy_action",
            gold=3,
            state={
                "id": "q003",
                "state": {"agent_speaking": True, "agent_played_ms": 800, "overlap": True},
                "questions": {
                    "takeover_strength": {
                        "type": "score",
                        "instructions": "用户抢话的强度",
                        "criteria": ["几乎没有", "弱", "中", "强", "立刻必须停"],
                    }
                },
            },
        )
        validate_item(s)


class RenderTests(unittest.TestCase):
    def test_qid_not_in_prompt(self):
        it = sample_item()
        qid, q = next(iter(it["state"]["questions"].items()))
        slots = slots_for_question(it["id"], qid, q, seed=0)
        text = render_user_text(it["state"]["state"], q, slots, with_audio=True)
        assert_qid_absent(text, it["id"], qid)
        self.assertIn("<|audio|>", text)
        self.assertIn("incomplete", text)
        self.assertTrue(any(f"{s} = " in text for s in slots))

    def test_permutation_changes_letter_not_options(self):
        it = sample_item()
        qid, q = next(iter(it["state"]["questions"].items()))
        a = slots_for_question(it["id"], qid, q, seed=0)
        b = slots_for_question(it["id"], qid, q, seed=1)
        self.assertEqual(set(a.values()), set(q["criteria"]))
        self.assertEqual(set(b.values()), set(q["criteria"]))
        # 同一题换 seed，映射应不同（极小概率哈希撞车，那时这个断言才失败）
        self.assertNotEqual(a, b)
        self.assertEqual(gold_slot(q, "incomplete", a) in a, True)

    def test_path_count_equals_options(self):
        it = sample_item()
        q = next(iter(it["state"]["questions"].values()))
        paths = path_texts(it["state"]["state"], q)
        self.assertEqual(len(paths), 4)
        joined = "\n".join(t for _, t in paths)
        self.assertNotIn("speech_state", joined)
        self.assertNotIn("q001", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
