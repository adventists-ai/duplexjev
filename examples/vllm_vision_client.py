"""Ask closed-set questions about a speech clip and a picture together, through vLLM serving DuplexJev-32B-Vision.

Server (one 80 GB GPU):
    pip install "vllm[audio]>=0.29" "duplexjev-vllm>=0.3"
    vllm serve adventists-ai/DuplexJev-32B-Vision --max-model-len 8192 \
        --limit-mm-per-prompt '{"image": 1, "audio": 1, "video": 0}'

Client (this file, standard library only):
    python vllm_vision_client.py clip.wav face.jpg

Each question is one request with max_tokens=1; the answer is the next-token distribution over the option letters,
averaged over cyclic option orders (as in training). Audio or image may be left out.
"""
import base64
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "adventists-ai/DuplexJev-32B-Vision"
LETTERS = "ABCDEFGHIJ"
LETTER_IDS = [32 + i for i in range(10)]  # "A".."J" in the Qwen3 tokenizer
INSTR = "Answer with only the letter of the correct option."


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _content(body, audio_b64=None, image_b64=None, image_mime="image/jpeg"):
    head = "The user said: <|audio|>" if audio_b64 else ""
    if image_b64:
        parts = [{"type": "text", "text": (head + "\n\n" if head else "") + "Picture: "},
                 {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                 {"type": "text", "text": "\n\n" + body}]
    else:
        parts = [{"type": "text", "text": (head + "\n\n" if head else "") + body}]
    if audio_b64:
        parts.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}})
    return parts


def _one(question, options, audio_b64, image_b64, shift):
    n = len(options)
    opts = [options[(j + shift) % n] for j in range(n)]
    body = question + "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + "\n\n" + INSTR
    req = {"model": MODEL, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
           "allowed_token_ids": LETTER_IDS[:n],
           # The model was trained with Qwen3's empty <think></think> block.
           "chat_template_kwargs": {"enable_thinking": False},
           "messages": [{"role": "user", "content": _content(body, audio_b64, image_b64)}]}
    r = urllib.request.Request(URL, json.dumps(req).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(r, timeout=120))
    # top_logprobs is reported before allowed_token_ids is applied: keep only the letters and renormalise.
    top = {t["token"]: t["logprob"] for t in out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    lg = [top.get(LETTERS[i], -1e9) for i in range(n)]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    return {opts[i]: e[i] / sum(e) for i in range(n)}


def decide(questions, audio=None, image=None, n_perm=2):
    """questions: [{"id", "text", "options"}] -> {id: {"answer", "confidence", "probs"}}"""
    a = _b64(audio) if audio else None
    im = _b64(image) if image else None
    jobs = [(q, s) for q in questions for s in range(min(n_perm, len(q["options"])))]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda j: _one(j[0]["text"], j[0]["options"], a, im, j[1]), jobs))
    out = {}
    for q in questions:
        ps = [p for (qq, _), p in zip(jobs, res) if qq is q]
        probs = {o: sum(p[o] for p in ps) / len(ps) for o in q["options"]}
        best = max(probs, key=probs.get)
        out[q["id"]] = {"answer": best, "confidence": probs[best], "probs": probs}
    return out


if __name__ == "__main__":
    audio, image = sys.argv[1], sys.argv[2]
    qs = [{"id": "match", "text": "Does the face in the picture show the same emotion as the speaker's voice?",
           "options": ["Yes", "No"]},
          {"id": "face", "text": "What emotion does the face in the picture show?",
           "options": ["happy", "angry", "sad", "neutral"]}]
    for k, v in decide(qs, audio, image).items():
        print(k, v["answer"], {o: round(p, 3) for o, p in v["probs"].items()})
