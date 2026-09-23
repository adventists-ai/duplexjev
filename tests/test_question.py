import pytest

from duplexjev import Question


def test_permutation_is_deterministic_and_complete():
    q = Question("intent", "What does the user want?", ["a", "b", "c", "d", "e"])
    p = q.permutation(0)
    assert sorted(p) == list(range(5))
    assert p == Question("intent", "What does the user want?", ["a", "b", "c", "d", "e"]).permutation(0)
    assert any(q.permutation(k) != p for k in range(1, 6))


def test_render_lists_every_option_under_a_letter():
    q = Question("turn", "Finished?", ["yes", "no"], lang="zh")
    text = q.render([1, 0])
    assert "A. no" in text and "B. yes" in text and text.startswith("问题：Finished?")


@pytest.mark.parametrize("kw", [
    dict(id="", text="x", options=["a", "b"]),
    dict(id="q", text="", options=["a", "b"]),
    dict(id="q", text="x", options=["a"]),
    dict(id="q", text="x", options=["a", "a"]),
    dict(id="q", text="x", options=["a", "b"], lang="fr"),
])
def test_invalid_questions_are_rejected(kw):
    with pytest.raises(ValueError):
        Question(**kw)


def test_audio_binding_round_trip():
    q = Question("gender", "Speaker gender", ["female", "male"], audio=["car1", "car2"])
    assert q.audio == ("car1", "car2")
    assert Question.from_dict(q.to_dict()) == q
    assert q.for_audio("*").audio == "*" and q.for_audio(None).audio is None
