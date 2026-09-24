"""Decider on a tiny speech checkpoint: API shape, packed == one row per question, batch invariance."""
import pytest

from duplexjev import Question

TURN = Question("turn", "Has the user finished the turn?", ["finished", "not finished"])
GENDER = Question("gender", "Speaker gender", ["female", "male"])
INTENT = Question("intent", "What does the user want?", ["climate", "media", "navigation", "phone", "chit-chat"])
QS = [TURN, GENDER, INTENT]


def maxdiff(a, b):
    return max(abs(a[q]["probs"][o] - b[q]["probs"][o]) for q in a for o in a[q]["probs"])


def test_decide_one_clip(decider, clips):
    r = decider.decide(clips[0], QS)
    assert set(r) == {"turn", "gender", "intent"}
    for q in QS:
        assert list(r[q.id]["probs"]) == list(q.options)
        assert abs(sum(r[q.id]["probs"].values()) - 1) < 1e-4
        assert r[q.id]["answer"] == max(r[q.id]["probs"], key=r[q.id]["probs"].get)
    assert decider.last_stats["decode_steps"] == 0
    single = decider.decide(clips[0], TURN)  # a single option group is accepted too
    assert set(single) == {"turn"}


def test_packed_equals_one_row_per_question(decider, clips):
    for c in clips:
        assert maxdiff(decider.decide(c, QS), decider.decide(c, QS, mode="batch")) < 1e-4


def test_decide_batch_routes_groups_to_clips(decider, clips):
    qs = [TURN.for_audio("*"), GENDER.for_audio(["b", "c"]), INTENT.for_audio("a")]
    r = decider.decide_batch({"a": clips[0], "b": clips[1], "c": clips[2]}, qs)
    assert set(r) == {"a", "b", "c"}
    assert set(r["a"]) == {"turn", "intent"} and set(r["b"]) == {"turn", "gender"} and set(r["c"]) == {"turn", "gender"}
    assert decider.last_stats["clips"] == 3 and decider.last_stats["questions"] == 6


def test_batch_invariance(decider, clips):
    together = decider.decide_batch(clips, [q.for_audio("*") for q in QS])
    for i, c in enumerate(clips):
        assert maxdiff(together[i], decider.decide(c, QS)) < 1e-3
    chunked = decider.decide_batch(clips, [q.for_audio("*") for q in QS], max_tokens=200)
    assert decider.last_stats["batches"] >= 2
    for i in range(len(clips)):
        assert maxdiff(together[i], chunked[i]) < 1e-3


def test_decide_batch_needs_audio_ids(decider, clips):
    with pytest.raises(ValueError):
        decider.decide_batch(clips, [TURN])  # no audio=
    with pytest.raises(ValueError):
        decider.decide_batch(clips, [TURN.for_audio(7)])  # unknown clip


def test_text_models_are_rejected():
    from duplexjev import Decider

    with pytest.raises(ValueError):
        Decider.from_pretrained("Qwen/Qwen3-0.6B", device="cpu")


def test_align_pads_silence_at_the_end(decider):
    import numpy as np

    a = np.linspace(-1, 1, 16000 + 123, dtype=np.float32)
    b = decider._align(a)
    hop = 160 * int(getattr(decider.processor, "encoder_ds_factor", 2)) * int(getattr(decider.processor, "stack_factor", 8))
    assert len(b) % hop == 0 and len(b) - len(a) < hop
    assert np.array_equal(b[: len(a)], a) and not b[len(a):].any()
