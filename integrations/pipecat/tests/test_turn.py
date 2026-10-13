import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from pipecat_duplexjev import DuplexJevTurnAnalyzer
from pipecat_duplexjev.turn import _wav16k


class Fake(BaseHTTPRequestHandler):
    p_finished = 0.9
    other = 1
    last = None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.last = body
        opts = body["questions"][0]["options"]
        probs = {o: 0.0 for o in opts}
        probs[opts[0]] = Fake.p_finished
        probs[opts[Fake.other]] = 1 - Fake.p_finished
        ans = {"turn": {"answer": max(probs, key=probs.get), "confidence": max(probs.values()), "probs": probs}}
        for q in body["questions"][1:]:
            ans[q["id"]] = {"answer": q["options"][0], "confidence": 1.0, "probs": {q["options"][0]: 1.0}}
        out = json.dumps({"answers": ans, "ms": 42.0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    s = HTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def test_wav16k_resamples():
    wav = _wav16k(np.zeros(48000, np.float32), 48000)
    assert wav[:4] == b"RIFF" and abs(len(wav) - 44 - 32000) <= 2


def test_complete_and_incomplete(server):
    a = DuplexJevTurnAnalyzer(url=server, sample_rate=16000, extra_questions=["emotion"])
    Fake.p_finished = 0.9
    r = a._predict_endpoint(np.zeros(16000, np.float32))
    assert r == {"prediction": 1, "probability": 0.9}
    assert a.last_answers["emotion"]["answer"] == "neutral" and a.last_server_ms == 42.0
    assert [q["id"] for q in Fake.last["questions"]] == ["turn", "emotion"]
    Fake.p_finished = 0.2
    assert a._predict_endpoint(np.zeros(16000, np.float32))["prediction"] == 0


def test_server_down_does_not_raise():
    a = DuplexJevTurnAnalyzer(url="http://127.0.0.1:9", sample_rate=16000)
    assert a._predict_endpoint(np.zeros(1600, np.float32)) == {"prediction": 0, "probability": 0.0}


def test_assistant_said_is_sent(server):
    a = DuplexJevTurnAnalyzer(url=server, sample_rate=16000)
    a._predict_endpoint(np.zeros(16000, np.float32))
    assert "assistant_said" not in Fake.last
    a.set_assistant_said("  What   time works for you? " + "x" * 400)
    a._predict_endpoint(np.zeros(16000, np.float32))
    assert len(Fake.last["assistant_said"]) == 300 and Fake.last["assistant_said"].endswith("x")


def test_context_tracker():
    import asyncio
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TextFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat_duplexjev import DuplexJevContextTracker

    a = DuplexJevTurnAnalyzer(url="http://127.0.0.1:9", sample_rate=16000)
    t = DuplexJevContextTracker(a)
    sent = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        sent.append(frame)

    t.push_frame = push

    async def run():
        for f in [LLMFullResponseStartFrame(), TextFrame("Sure, "), TextFrame("what time works?"),
                  LLMFullResponseEndFrame()]:
            await t.process_frame(f, FrameDirection.DOWNSTREAM)

    asyncio.run(run())
    assert a.assistant_said == "Sure, what time works?" and len(sent) == 4


def test_act_readout(server):
    a = DuplexJevTurnAnalyzer(url=server, sample_rate=16000, readout="act")
    Fake.p_finished, Fake.other = 0.7, 1
    r = a._predict_endpoint(np.zeros(16000, np.float32))
    assert Fake.last["questions"][0]["text"] == "What should the assistant do now?"
    assert r["prediction"] == 1 and abs(r["probability"] - 0.7) < 1e-9 and a.last_turn_class == "reply"


def test_hesitation_extends_silence_limit(server):
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
    a = DuplexJevTurnAnalyzer(url=server, sample_rate=16000, wait_stop_secs=6.0)
    base = a._stop_ms
    Fake.p_finished, Fake.other = 0.1, 3  # mostly "hesitating or asking to wait"
    r = a._predict_endpoint(np.zeros(16000, np.float32))
    assert r["prediction"] == 0 and a.last_turn_class == "hesitating" and a._stop_ms == 6000
    a._clear(EndOfTurnState.COMPLETE)
    assert a._stop_ms == base
    Fake.p_finished, Fake.other = 0.1, 1  # "not finished": normal limit
    a._predict_endpoint(np.zeros(16000, np.float32))
    assert a.last_turn_class == "not_finished" and a._stop_ms == base
