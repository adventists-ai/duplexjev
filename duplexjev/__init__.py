"""DuplexJev: batched typed speech decisions without decoding.

    from duplexjev import Decider, Question
    d = Decider.from_pretrained("Qwen/Qwen3-0.6B")          # any causal LM (text), or a speech checkpoint
    d.decide(["I want to turn on the"], [Question("turn", "Has the user finished?", ["finished", "not finished"])])
"""
from .decider import Decider, load_audio
from .question import Question

__all__ = ["Decider", "Question", "load_audio"]
__version__ = "0.1.0"
