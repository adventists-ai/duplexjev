# Stand-in for `duplexjev gateway` (no GPU): P(finished) from how quiet the last 300 ms are. Only for plumbing tests.
import base64, io, json, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        w = wave.open(io.BytesIO(base64.b64decode(d["audio_b64"])))
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        tail = x[-4800:]; r = float(np.sqrt(np.mean(tail ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-6))
        p1 = float(np.clip(1 - r, 0.01, 0.99)); q = d["questions"][0]; o = q["options"]
        probs = {o[0]: p1, **{k: (1 - p1) / (len(o) - 1) for k in o[1:]}}
        body = json.dumps({"answers": {q["id"]: {"answer": max(probs, key=probs.get), "probs": probs}}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
ThreadingHTTPServer(("127.0.0.1", 8420), H).serve_forever()
