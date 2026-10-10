"""A LiveKit voice agent whose end-of-turn decisions come from DuplexJev.

    pip install livekit-plugins-duplexjev "livekit-agents[silero]>=1.8"
    export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
    python examples/agent.py console      # or: dev / start

STT, LLM and TTS go through LiveKit Inference here; swap in any plugins you like. DuplexJev only replaces the
turn detector. Set DUPLEXJEV_API_URL to your own gateway for real-time use (the default hosted trial API is
rate limited and served from mainland China).
"""
import logging

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import duplexjev, silero

logger = logging.getLogger("duplexjev-agent")
server = AgentServer()


def log_answers(answers: dict) -> None:
    logger.info("duplexjev %s", {k: v["answer"] for k, v in answers.items()})


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2",
        vad=silero.VAD.load(),  # required: LiveKit asks the turn detector after ~200 ms of VAD silence
        turn_handling={
            "turn_detection": duplexjev.TurnDetector(
                extra_questions=["emotion", "sound"],  # answered in the same request
                on_answers=log_answers,
            ),
        },
    )
    await session.start(agent=Agent(instructions="You are a friendly, concise voice assistant."), room=ctx.room)
    await session.generate_reply(instructions="Greet the user.")


if __name__ == "__main__":
    cli.run_app(server)
