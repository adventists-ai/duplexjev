# Writing your own questions

*[中文](questions_zh.md) · [back to README](../README.md)*

DuplexJev answers any multiple-choice question about a clip in one forward pass. The default table (turn state,
emotion, gender, ...) covers the common cases; this page is about writing your own.

## Request and response

Trial API (no key, 30 requests per minute per IP, clips up to 4 MB, audio is not stored):

```
POST https://api.adventists.cn/duplexjev/v1/decide
Content-Type: application/json

{
  "audio_b64": "<base64 of the audio file>",
  "format": "wav",
  "lang": "en",
  "questions": [
    {"id": "laugh",  "text": "Where is the laughter?",
     "options": ["before speaking", "after speaking", "while speaking", "no laughter"], "lang": "en"},
    {"id": "stress", "text": "Which word is stressed?",
     "options": ["wait", "stop", "address", "asked", "no clear stress"], "lang": "en"}
  ]
}
```

```
{"answers": {
   "laugh":  {"answer": "no laughter", "confidence": 0.96, "probs": {"before speaking": 0.01, ...}},
   "stress": {"answer": "stop",        "confidence": 0.98, "probs": {...}}},
 "ms": 98.0, "model": "DuplexJev-32B-Para"}
```

- **Put `"lang"` on every question** (`"en"` or `"zh"`), and ask in the language of the clip. A question without
  `lang` is rendered with the English template.
- `probs` gives a probability for every option. Keep it if you want to set your own thresholds later.
- Ask all questions about one clip in one request; they share the audio and count as one call.
- 2–10 options per question. Options are shown to the model in a shuffled order, so their order in your list does
  not matter.
- wav, mp3, flac, ogg and m4a are accepted, at any sample rate.

The same request works against your own server: `duplexjev gateway --vllm http://localhost:8000/v1` exposes the same
`/v1/decide` (see the [README](../README.md#quick-start)).

## Rules of thumb

1. **Always include a "none" option**: *no laughter*, *no clear stress*, *no pause*. Without it the model has to pick
   something even when nothing happened.
2. **For "where" questions, use the words of the utterance as options**: *"After which word does the speaker cough?
   doctor / itchy / night / no cough"*. If you have a transcript, build the options from it.
3. **Ask "whether" and "where" in one question**: *before speaking / after speaking / while speaking / none* answers both.
4. **Keep the question short, and ask one thing per question.** Extra instructions in the question ("do not count
   onomatopoeia as laughter") tend to hurt rather than help.
5. **Options should be mutually exclusive.** If two can be true at once, split them into two questions.

## Templates

| What you want to know | Question | Options |
|---|---|---|
| Non-verbal sound and its position | Where is the laughter? (cough, sigh, ...) | before speaking / after speaking / while speaking / no laughter |
| Which non-verbal sound | Besides speech, which sound can be heard in this audio? | laughter / breathing / coughing / a sigh / none of these |
| Stress | Which word is stressed? | words of the utterance + no clear stress |
| Lengthened sound | Which word is drawn out? | words of the utterance + none |
| Pause position | After which word does the speaker pause? | words of the utterance + no pause |
| Event after a given word | After which word does the speaker cough? | words of the utterance + no cough |
| Speaking style | How is the speaker talking? | whispering / speaking very loudly / a normal speaking voice |

For turn-taking, emotion and the other built-in decisions, start from the default table
(`GET https://api.adventists.cn/duplexjev/v1/table?lang=en`) and edit it.

## Python

With the package (`pip install duplexjev`):

```python
from duplexjev.remote import ApiClient

dj = ApiClient("https://api.adventists.cn/duplexjev")      # or your own gateway
dj.decide("clip.wav", [
    {"id": "cough", "text": "After which word does the speaker cough?",
     "options": ["doctor", "itchy", "night", "no cough"], "lang": "en"},
], lang="en")
# {'cough': {'answer': 'itchy', 'confidence': ..., 'probs': {...}}}
```

Standard library only:

```python
import base64, json, urllib.request

def decide(path, questions, lang="en", url="https://api.adventists.cn/duplexjev"):
    body = {"audio_b64": base64.b64encode(open(path, "rb").read()).decode(), "format": "wav",
            "lang": lang, "questions": [dict(q, lang=lang) for q in questions]}
    req = urllib.request.Request(url + "/v1/decide", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["answers"]
```

The trial API is rate-limited and hosted in China. For batch jobs or real-time agents, deploy your own server
([README](../README.md#quick-start)).
