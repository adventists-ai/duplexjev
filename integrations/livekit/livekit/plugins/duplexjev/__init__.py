"""DuplexJev audio turn detection for LiveKit Agents.

    from livekit.plugins import duplexjev, silero
    session = AgentSession(turn_detection=duplexjev.TurnDetector(), vad=silero.VAD.load(), ...)
"""

from livekit.agents import Plugin

from .log import logger
from .turn_detector import (
    DEFAULT_URL,
    PRESETS,
    REPLY_QUESTION,
    TURN_QUESTION,
    TurnDetector,
    TurnDetectorStream,
)
from .version import __version__

__all__ = [
    "TurnDetector",
    "TurnDetectorStream",
    "DEFAULT_URL",
    "PRESETS",
    "TURN_QUESTION",
    "REPLY_QUESTION",
    "__version__",
]


class DuplexJevPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DuplexJevPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
