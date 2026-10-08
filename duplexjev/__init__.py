"""DuplexJev: Jev-style typed decisions about speech, with zero decode steps.

    from duplexjev import quick
    quick("call.wav", api="https://<host>/duplexjev")   # default table of 8 decisions, printed as a table

    from duplexjev import Decider, Question
    d = Decider.from_pretrained("adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B")
    d.decide("call.wav", [Question("turn", "Has the user finished?", ["finished", "not finished"])])
    d.decide_batch({"car1": "a.wav", "car2": "b.wav"},
                   [Question("turn", "Has the user finished?", ["finished", "not finished"], audio="*"),
                    Question("gender", "Speaker gender", ["female", "male"], audio="car2")])
"""
from .question import Question
from .quick import quick
from .table import default_table, format_table

__all__ = ["Decider", "Question", "load_audio", "quick", "default_table", "format_table"]
__version__ = "0.4.0"


def __getattr__(name):  # torch is only imported when the local engine is used
    if name in ("Decider", "load_audio"):
        from . import decider

        return getattr(decider, name)
    raise AttributeError(name)
