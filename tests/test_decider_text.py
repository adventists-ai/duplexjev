"""Packed prefix sharing must give the same answers as one row per question, for any batch composition."""
import pytest

from duplexjev import Question

Q = [
    Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
    Question("intent", "What does the user want?", ["climate", "media", "navigation", "phone", "chit-chat"]),
    Question("filler", "Which filler fits?", ["Sure —", "One moment —", "(stay silent)"]),
]
ITEMS = [
    "Turn on the air conditioning please.",
    {"text": "Can you navigate to the", "context": "Driver is on the highway."},
    {"text": "帮我打电话给妈妈", "lang": "zh"},
]


def maxdiff(a, b):
    return max(abs(a[i][q]["probs"][o] - b[i][q]["probs"][o]) for i in range(len(a)) for q in a[i] for o in a[i][q]["probs"])


def test_packed_equals_batch(tiny_decider):
    p = tiny_decider.decide(ITEMS, Q, mode="packed")
    b = tiny_decider.decide(ITEMS, Q, mode="batch")
    assert maxdiff(p, b) < 1e-4
    assert tiny_decider.last_stats["decode_steps"] == 0


def test_batch_composition_does_not_change_answers(tiny_decider):
    together = tiny_decider.decide(ITEMS, Q)
    alone = [tiny_decider.decide([it], Q)[0] for it in ITEMS]
    assert maxdiff(together, alone) < 1e-4
    chunked = tiny_decider.decide(ITEMS, Q, max_items=2)
    assert maxdiff(together, chunked) < 1e-4


def test_output_shape_and_probabilities(tiny_decider):
    r = tiny_decider.decide(ITEMS, Q, n_perm=3)
    assert len(r) == len(ITEMS)
    for res in r:
        assert set(res) == {"turn", "intent", "filler"}
        for q in Q:
            probs = res[q.id]["probs"]
            assert list(probs) == list(q.options)
            assert abs(sum(probs.values()) - 1) < 1e-4
            assert res[q.id]["answer"] == max(probs, key=probs.get)


def test_per_item_questions(tiny_decider):
    items = [{"text": "hello", "questions": [Q[0]]}, {"text": "play jazz", "questions": [Q[1], Q[2]]}]
    r = tiny_decider.decide(items)
    assert set(r[0]) == {"turn"} and set(r[1]) == {"intent", "filler"}


def test_audio_rejected_by_text_model(tiny_decider):
    import numpy as np

    with pytest.raises(ValueError):
        tiny_decider.decide([np.zeros(16000, dtype="float32")], Q)


def test_token_budget_chunks_give_same_answers(tiny_decider):
    whole = tiny_decider.decide(ITEMS, Q)
    small = tiny_decider.decide(ITEMS, Q, max_tokens=300)
    assert tiny_decider.last_stats["batches"] >= 2
    assert maxdiff(whole, small) < 1e-4
