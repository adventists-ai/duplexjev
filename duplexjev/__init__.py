"""DuplexJev: Jev-style typed decisions about speech, with zero decode steps.

    from duplexjev import Decider, Question
    d = Decider.from_pretrained("adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B")
    d.decide("call.wav", [Question("turn", "Has the user finished?", ["finished", "not finished"])])
    d.decide_batch({"car1": "a.wav", "car2": "b.wav"},
                   [Question("turn", "Has the user finished?", ["finished", "not finished"], audio="*"),
                    Question("gender", "Speaker gender", ["female", "male"], audio="car2")])
"""
from .decider import Decider, load_audio
from .question import Question

__all__ = ["Decider", "Question", "load_audio"]
__version__ = "0.2.0"
