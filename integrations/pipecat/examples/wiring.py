"""Minimal Pipecat (>= 1.0) wiring: DuplexJev decides when the user's turn is over.

Use it in place of the default smart-turn model in any Pipecat bot. Put the aggregator pair in your pipeline as usual:
    Pipeline([transport.input(), stt, aggregators.user(), llm, tts, transport.output(), aggregators.assistant()])
"""
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_duplexjev import DuplexJevTurnAnalyzer


def on_answers(answers):
    # Same forward pass, free extra signals: route an angry caller, react to laughter, ...
    print({k: v["answer"] for k, v in answers.items()})


turn = DuplexJevTurnAnalyzer(
    url="http://localhost:8420",  # your `duplexjev gateway`; omit to use the free hosted trial API
    lang="en",
    extra_questions=["emotion", "sound"],
    on_answers=on_answers,
)

context = LLMContext()
aggregators = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_turn_strategies=UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn)]),
    ),
)
