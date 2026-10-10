"""Feed real clips through the analyzer the way Pipecat does (20 ms frames, then 200 ms of silence)."""
import asyncio
import sys
import time
import wave

import numpy as np

from pipecat_duplexjev import DuplexJevTurnAnalyzer


def load16k(path):
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    sr = w.getframerate()
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)


async def main(paths):
    a = DuplexJevTurnAnalyzer(sample_rate=16000, lang="en", extra_questions=["emotion"])
    a.set_sample_rate(16000)  # done by the transport in a real pipeline
    for p in paths:
        x = load16k(p)
        frame = 320
        for i in range(0, len(x), frame):
            a.append_audio(x[i:i + frame].tobytes(), is_speech=True)
        for _ in range(10):
            a.append_audio(np.zeros(frame, np.int16).tobytes(), is_speech=False)
        t = time.time()
        state, m = await a.analyze_end_of_turn()
        print(f"{p}: {state.name:10s} p(finished)={m.probability:.2f}  "
              f"turn={a.last_answers['turn']['answer']!r} emotion={a.last_answers['emotion']['answer']} "
              f"server {a.last_server_ms} ms, round trip {1000 * (time.time() - t):.0f} ms")
        a.clear()


asyncio.run(main(sys.argv[1:]))
